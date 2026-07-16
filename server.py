
import asyncio
import importlib
import json
from collections import deque
from datetime import datetime, timezone
import os
import shutil
import sys
import threading

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import db
import strategy_base
import EMACrossShortTest
import KalmanFilter
import ExtendedKalmanFilter
import AccelerationKalmanFilter
import EMAlgoTest
import EMAlgoNonLinTest
import run_backtest

BASE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(BASE, 'reports')
os.makedirs(REPORTS, exist_ok=True)

# ---- server log ring buffer -------------------------------------------------
# Everything printed by the server and its worker threads (backtests, sync,
# report builds) is teed into this buffer and served at /api/logs so the
# dashboard's log window can show it (warnings about excluded symbols, run
# tracebacks, etc.) without shell access.
_LOG_BUF: deque = deque(maxlen=4000)     # (id, 'HH:MM:SS', line)
_log_lock = threading.Lock()
_log_seq = 0
# access-log lines from the dashboard's own polling would flood the window
# (the log poll would log itself every 2s) — drop them at capture time
_LOG_SKIP = ('/api/logs', '/api/status', '/api/sync/status',
             '/api/report/status', '/api/results')


class _LogTee:
    """File-like wrapper: pass writes through and collect complete lines."""
    def __init__(self, orig):
        self._orig = orig
        self._part = ''

    def write(self, s):
        try:
            self._orig.write(s)
        except Exception:
            pass
        global _log_seq
        with _log_lock:
            self._part += s
            while '\n' in self._part:
                line, self._part = self._part.split('\n', 1)
                if line.strip() and not any(p in line for p in _LOG_SKIP):
                    _log_seq += 1
                    _LOG_BUF.append((_log_seq,
                                     datetime.now().strftime('%H:%M:%S'),
                                     line.rstrip()))

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def isatty(self):
        return False


sys.stdout = _LogTee(sys.stdout)
sys.stderr = _LogTee(sys.stderr)

app = FastAPI(title='Backtrader Dashboard')
app.mount('/reports', StaticFiles(directory=REPORTS), name='reports')
# Vendored JS (lightweight-charts) so the dashboard needs no internet/CDN.
app.mount('/vendor', StaticFiles(directory=os.path.join(BASE, 'vendor')),
          name='vendor')
# KF/EKF/EM comparison report (build_report.py output) on the same server.
REPORT_OUT = os.path.join(BASE, 'report_out')
os.makedirs(REPORT_OUT, exist_ok=True)
app.mount('/report', StaticFiles(directory=REPORT_OUT, html=True), name='report')

_run_thread: threading.Thread | None = None
_report_thread: threading.Thread | None = None
_report_state: dict = {'state': 'idle'}
_sync_thread: threading.Thread | None = None
_sync_state: dict = {'state': 'idle'}
_console_proc = None                    # persistent console_worker.py subprocess
_console_lock = threading.Lock()


def _read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@app.get('/')
def index():
    return FileResponse(os.path.join(BASE, 'dashboard.html'))


@app.get('/api/results')
def results():
    data = _read_json(run_backtest.RESULTS_PATH)
    if data is None:
        return JSONResponse({'error': 'no results yet — run a backtest'},
                            status_code=404)
    return data


@app.get('/api/status')
def status():
    running = _run_thread is not None and _run_thread.is_alive()
    data = _read_json(run_backtest.STATUS_PATH) or {'state': 'idle'}
    # If the runner crashed hard before writing an error status, don't report
    # a stale 'running' state forever.
    if data.get('state') == 'running' and not running:
        data['state'] = 'error'
        data.setdefault('error', 'runner stopped unexpectedly')
    data['thread_alive'] = running
    return data


@app.get('/api/strategies')
def strategies():
    """Strategy picker data: every registered strategy with its tuned default
    params, plus the module default selection and backtest days."""
    return {
        'default': run_backtest.STRATEGY.__name__,
        'days': run_backtest.BACKTEST_DAYS,
        'timeframe': '1m',
        'timeframes': ['1m', '1h'],
        'strategies': {name: params
                       for name, (_cls, params) in run_backtest.STRATEGIES.items()},
    }


@app.post('/api/run')
async def run(request: Request):
    global run_backtest
    if _run_thread is not None and _run_thread.is_alive():
        return JSONResponse({'error': 'backtest already running'},
                            status_code=409)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    err = _reload_modules()
    if err:
        return JSONResponse({'error': err}, status_code=500)
    return _start_run(payload)


def _reload_modules():
    """Reload every project module from disk so code/param edits (including the
    STRATEGIES registry defaults) take effect without restarting the server.
    Order is dependency-first: base classes and leaf strategy modules before
    run_backtest (which does `from <module> import <Strategy>`); reloading it
    first would capture a stale class, then reloading the strategy module in
    place rebinds that class's globals — mixing old methods with new functions.
    Returns None on success or an error string."""
    global db, strategy_base, EMACrossShortTest, KalmanFilter, EMAlgoTest, EMAlgoNonLinTest, run_backtest
    global ExtendedKalmanFilter, AccelerationKalmanFilter
    try:
        db = importlib.reload(db)
        strategy_base = importlib.reload(strategy_base)
        EMACrossShortTest = importlib.reload(EMACrossShortTest)
        KalmanFilter = importlib.reload(KalmanFilter)
        EMAlgoTest = importlib.reload(EMAlgoTest)
        EMAlgoNonLinTest = importlib.reload(EMAlgoNonLinTest)
        ExtendedKalmanFilter = importlib.reload(ExtendedKalmanFilter)
        AccelerationKalmanFilter = importlib.reload(AccelerationKalmanFilter)
        run_backtest = importlib.reload(run_backtest)
    except Exception as e:
        return 'reload failed: %s' % e
    return None


@app.post('/api/reload')
def reload_modules():
    """Reload strategy modules + run_backtest so edits to code or the STRATEGIES
    registry (default params like EMNonLinTest's `dead`) appear in the picker
    without having to run a backtest first."""
    if _run_thread is not None and _run_thread.is_alive():
        return JSONResponse({'error': 'cannot reload while a backtest is running'},
                            status_code=409)
    err = _reload_modules()
    if err:
        return JSONResponse({'error': err}, status_code=500)
    return {'reloaded': True, 'strategies': list(run_backtest.STRATEGIES.keys())}


def _start_run(payload=None):
    global _run_thread
    payload = payload or {}
    kwargs = {'strategy': payload.get('strategy'),
              'params': payload.get('params'),
              'days': payload.get('days'),
              'timeframe': payload.get('timeframe')}
    _run_thread = threading.Thread(target=run_backtest.run, kwargs=kwargs,
                                   daemon=True)
    _run_thread.start()
    return {'started': True}


@app.post('/api/report')
async def build_comparison_report(request: Request):
    """Rebuild the KF/EKF/EM plotly comparison (build_report.py) in a thread;
    the result is served from /report/. Poll /api/report/status."""
    global _report_thread
    if _report_thread is not None and _report_thread.is_alive():
        return JSONResponse({'error': 'report already building'}, status_code=409)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    symbol = payload.get('symbol') or None
    # [{'name': <registry name>, 'params': {...}}, ...]; None -> report default
    selections = payload.get('strategies') or None
    if selections is not None and not isinstance(selections, list):
        selections = None
    # 'compare old tests' mode: list of archived run tags to compare instead
    tags = payload.get('tags') or None
    if tags is not None and not isinstance(tags, list):
        tags = None
    timeframe = payload.get('timeframe') if payload.get('timeframe') in ('1m', '1h') else '1m'
    try:
        days = int(payload.get('days') or 0) or None   # None -> full history
    except (TypeError, ValueError):
        days = None

    def _work():
        import time as _time
        _report_state.update(state='building', started=_time.time(), error=None)
        try:
            import build_report
            build_report = importlib.reload(build_report)
            if tags:
                build_report.build_from_folders(tags)
            else:
                # symbol=None/'' -> portfolio report over ALL symbols
                build_report.build(symbol, selections=selections,
                                   timeframe=timeframe, days=days)
            _report_state.update(state='done',
                                 elapsed=round(_time.time() - _report_state['started'], 1))
        except Exception as e:
            _report_state.update(state='error', error=str(e))

    _report_thread = threading.Thread(target=_work, daemon=True)
    _report_thread.start()
    return {'started': True}


@app.get('/api/report/status')
def report_status():
    st = dict(_report_state)
    st['thread_alive'] = _report_thread is not None and _report_thread.is_alive()
    if st.get('state') == 'building' and not st['thread_alive']:
        st['state'] = 'error'
        st.setdefault('error', 'report thread stopped unexpectedly')
    st['ready'] = os.path.exists(os.path.join(REPORT_OUT, 'index.html'))
    return st


def _console_localonly(request):
    """The console executes arbitrary Python — never serve it off-box. If the
    server is ever bound to 0.0.0.0 (e.g. to share the report), this endpoint
    stays localhost-only."""
    host = getattr(request.client, 'host', '')
    return host in ('127.0.0.1', '::1', 'localhost')


def _console_start():
    global _console_proc
    import subprocess
    _console_proc = subprocess.Popen(
        [sys.executable, os.path.join(BASE, 'console_worker.py')],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, encoding='utf-8', cwd=BASE)
    banner = json.loads(_console_proc.stdout.readline())
    return banner


@app.post('/api/console')
async def console_exec(request: Request):
    """Run Python in the persistent console worker. Body: {"code": "..."}.
    Blocks until the code finishes (notebook semantics); use /api/console/restart
    to kill a runaway command."""
    if not _console_localonly(request):
        return JSONResponse({'error': 'console is localhost-only'}, status_code=403)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    code = (payload or {}).get('code', '')
    if not isinstance(code, str) or not code.strip():
        return {'out': '', 'err': None}
    def _exchange():
        global _console_proc
        with _console_lock:
            if _console_proc is None or _console_proc.poll() is not None:
                banner = _console_start()
                prefix = banner.get('out', '') + '\n'
            else:
                prefix = ''
            _console_proc.stdin.write(json.dumps({'code': code}) + '\n')
            _console_proc.stdin.flush()
            reply = json.loads(_console_proc.stdout.readline())
        reply['out'] = prefix + reply.get('out', '')
        return reply

    try:
        # off the event loop: a long-running command must not freeze the app
        return await asyncio.to_thread(_exchange)
    except Exception as e:
        return JSONResponse({'error': 'console worker failed: %s' % e},
                            status_code=500)


@app.post('/api/console/restart')
async def console_restart(request: Request):
    """Kill the worker (unsticks runaway code, clears all variables)."""
    if not _console_localonly(request):
        return JSONResponse({'error': 'console is localhost-only'}, status_code=403)
    def _do_restart():
        global _console_proc
        with _console_lock:
            if _console_proc is not None and _console_proc.poll() is None:
                _console_proc.kill()
            _console_proc = None
            return _console_start()

    try:
        banner = await asyncio.to_thread(_do_restart)
    except Exception as e:
        return JSONResponse({'error': 'restart failed: %s' % e}, status_code=500)
    return {'restarted': True, 'out': banner.get('out', '')}


def _read_positions_by_symbol(run_dir):
    """Group a run's archived position_log.csv by symbol (the full detail the
    Positions table shows). Empty dict if the file is missing."""
    import csv as _csv
    path = os.path.join(run_dir, 'position_log.csv')
    out = {}
    if not os.path.exists(path):
        return out
    numeric = {'entry_price', 'exit_price', 'size', 'pnl', 'pnlcomm'}
    with open(path, newline='', encoding='utf-8') as f:
        for row in _csv.DictReader(f):
            d = {}
            for k, v in row.items():
                if k in numeric and v not in (None, ''):
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
                elif k == 'bars_held' and v not in (None, ''):
                    try:
                        d[k] = int(float(v))
                    except ValueError:
                        d[k] = v
                else:
                    d[k] = v
            out.setdefault(row.get('symbol', '?'), []).append(d)
    return out


@app.get('/api/testruns')
def list_test_runs():
    """Archived backtest folders (reports/test_data/<tag>/) that carry a
    run.json snapshot — the ones the dashboard can reload / the report can
    compare. Newest first."""
    tdir = run_backtest.TEST_DATA_DIR
    out = []
    if os.path.isdir(tdir):
        for tag in sorted(os.listdir(tdir), reverse=True):
            snap = _read_json(os.path.join(tdir, tag, 'run.json'))
            if not snap:
                continue
            p, s = snap.get('params', {}), snap.get('summary', {})
            out.append({'tag': tag, 'generated': snap.get('generated'),
                        'strategy': p.get('strategy'), 'timeframe': p.get('timeframe', '1m'),
                        'pnl': s.get('pnl'), 'return_pct': s.get('return_pct'),
                        'trades': s.get('trades_closed'), 'symbols': s.get('symbols'),
                        'window_start': p.get('window_start'), 'window_end': p.get('window_end')})
    return {'runs': out}


@app.get('/api/testrun/{tag}')
def load_test_run(tag: str):
    """Load one archived run for the dashboard: its run.json snapshot plus the
    per-symbol positions from position_log.csv. No candle/indicator chart data
    (not stored) — the UI shows PnL + positions only for loaded runs."""
    if not tag.replace('-', '').isalnum():           # tag is a timestamp folder
        return JSONResponse({'error': 'bad tag'}, status_code=400)
    run_dir = os.path.join(run_backtest.TEST_DATA_DIR, tag)
    snap = _read_json(os.path.join(run_dir, 'run.json'))
    if not snap:
        return JSONResponse({'error': 'no such archived run (needs run.json)'},
                            status_code=404)
    pos = _read_positions_by_symbol(run_dir)
    snap['charts'] = []                              # no candles for loaded runs
    snap['positionsBySymbol'] = pos
    snap['loaded_from'] = tag
    return snap


@app.get('/api/runs')
def runs_history(limit: int = 500):
    """Past backtest runs from Postgres (falls back cleanly when DB is down).
    `limit` caps the rows returned (db.list_runs defaulted to 50, which is why
    the History tab used to stop at the 50th run)."""
    try:
        import db
        conn = db.get_conn()
        out = db.list_runs(conn, limit=max(1, min(int(limit), 5000)))
        conn.close()
        return {'runs': out}
    except Exception as e:
        return JSONResponse({'error': 'database unavailable: %s' % e},
                            status_code=503)


@app.post('/api/sync')
async def sync_data(request: Request):
    """Incremental market-data sync — the exact routine launch.bat runs
    (fetch_binance_csv.main): asks postgres what's missing over the last N
    days, downloads only the gaps, upserts. Poll /api/sync/status."""
    global _sync_thread
    if _sync_thread is not None and _sync_thread.is_alive():
        return JSONResponse({'error': 'sync already running'}, status_code=409)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    try:
        days = max(1, min(int((payload or {}).get('days', 180)), 2000))
    except (TypeError, ValueError):
        days = 180
    interval = (payload or {}).get('interval')
    interval = interval if interval in ('1m', '1h') else '1m'

    def _work():
        import time as _time
        _sync_state.update(state='syncing', days=days, interval=interval,
                           started=_time.time(), error=None)
        try:
            import fetch_binance_csv
            fetch_binance_csv = importlib.reload(fetch_binance_csv)
            stats = fetch_binance_csv.main(days, interval) or {}
            _sync_state.update(state='done', **stats)
        except Exception as e:
            _sync_state.update(state='error', error=str(e))

    _sync_thread = threading.Thread(target=_work, daemon=True)
    _sync_thread.start()
    return {'started': True, 'days': days}


@app.get('/api/sync/status')
def sync_status():
    st = dict(_sync_state)
    st['thread_alive'] = _sync_thread is not None and _sync_thread.is_alive()
    if st.get('state') == 'syncing' and not st['thread_alive']:
        st['state'] = 'error'
        st.setdefault('error', 'sync thread stopped unexpectedly')
    return st


# ---- saved per-strategy / per-timeframe parameter configs ----------------
CONFIG_STORE = os.path.join(BASE, 'saved_configs.json')
_config_lock = threading.Lock()


def _load_configs():
    try:
        with open(CONFIG_STORE, encoding='utf-8') as f:
            d = json.load(f)
        return d if isinstance(d, list) else []
    except (OSError, ValueError):
        return []


def _write_configs(items):
    with open(CONFIG_STORE, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=1)


@app.get('/api/configs')
def list_configs():
    """All saved configs: [{id, strategy, timeframe, name, params, notes,
    default, saved}]. `default` is per (strategy, timeframe)."""
    with _config_lock:
        return {'configs': _load_configs()}


@app.post('/api/configs')
async def modify_configs(request: Request):
    """action=save {strategy,timeframe,name,params,notes} | default {id} |
    notes {id,notes} | delete {id}."""
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    action = payload.get('action')
    with _config_lock:
        items = _load_configs()
        if action == 'save':
            strategy = str(payload.get('strategy') or '')
            timeframe = payload.get('timeframe') if payload.get('timeframe') in ('1m', '1h') else '1m'
            params = payload.get('params')
            if not strategy or not isinstance(params, dict):
                return JSONResponse({'error': 'strategy and params required'}, status_code=400)
            cid = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-') + strategy
            first_for_pair = not any(c['strategy'] == strategy and c['timeframe'] == timeframe
                                     for c in items)
            items.append({'id': cid, 'strategy': strategy, 'timeframe': timeframe,
                          'name': str(payload.get('name') or '')[:80] or cid,
                          'params': params,
                          'notes': str(payload.get('notes') or '')[:2000],
                          'default': first_for_pair,   # first save = default
                          'saved': datetime.now(timezone.utc).isoformat(timespec='seconds')})
        elif action == 'default':
            cid = payload.get('id')
            hit = next((c for c in items if c['id'] == cid), None)
            if hit is None:
                return JSONResponse({'error': 'unknown config id'}, status_code=404)
            for c in items:
                if c['strategy'] == hit['strategy'] and c['timeframe'] == hit['timeframe']:
                    c['default'] = (c['id'] == cid)
        elif action == 'notes':
            hit = next((c for c in items if c['id'] == payload.get('id')), None)
            if hit is None:
                return JSONResponse({'error': 'unknown config id'}, status_code=404)
            hit['notes'] = str(payload.get('notes') or '')[:2000]
        elif action == 'delete':
            items = [c for c in items if c['id'] != payload.get('id')]
        else:
            return JSONResponse({'error': 'unknown action'}, status_code=400)
        _write_configs(items)
        return {'ok': True, 'configs': items}


@app.get('/api/logs')
def get_logs(since: int = 0, limit: int = 800):
    """Server log lines with id > `since` (tail `limit`). The dashboard's log
    window polls this incrementally."""
    with _log_lock:
        lines = [{'id': i, 't': t, 'line': l}
                 for (i, t, l) in _LOG_BUF if i > since]
    lines = lines[-max(1, min(limit, 2000)):]
    return {'lines': lines,
            'last': lines[-1]['id'] if lines else since}


@app.get('/api/data/status')
def data_status(interval: str = '1m'):
    """Per-symbol data health for `interval`: first/last stored bar, age of the
    last bar, and an integrity count (expected bars over the stored span vs
    actual — crypto trades 24/7 so any shortfall is a real hole)."""
    interval = interval if interval in ('1m', '1h') else '1m'
    bar_s = 3600 if interval == '1h' else 60
    try:
        import db
        conn = db.get_conn()
    except Exception as e:
        return JSONResponse({'error': 'db unavailable: %s' % e}, status_code=503)
    try:
        rows = db.per_symbol_span(conn, interval)
    finally:
        conn.close()
    now = datetime.now(timezone.utc)
    out = []
    for sym, mn, mx, cnt in rows:
        expected = int((mx - mn).total_seconds() // bar_s) + 1
        out.append({'symbol': sym,
                    'first': mn.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M'),
                    'last': mx.astimezone(timezone.utc).strftime('%Y-%m-%d %H:%M'),
                    'age_min': round((now - mx).total_seconds() / 60.0, 1),
                    'bars': cnt,
                    'expected': expected,
                    'missing': max(0, expected - cnt)})
    return {'interval': interval, 'generated': now.strftime('%Y-%m-%d %H:%M'),
            'symbols': out}


@app.get('/api/download/pyfolio')
def download_pyfolio():
    if not os.path.isdir(run_backtest.TEST_DATA_DIR):
        return JSONResponse({'error': 'no report yet'}, status_code=404)
    zip_base = os.path.join(REPORTS, 'test_data_report')
    shutil.make_archive(zip_base, 'zip', run_backtest.TEST_DATA_DIR)
    return FileResponse(zip_base + '.zip', filename='test_data_report.zip')


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)
