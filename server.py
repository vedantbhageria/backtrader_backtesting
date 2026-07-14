
import importlib
import json
import os
import shutil
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
    # Reload every project module from disk so edits are picked up without a
    # server restart. Order is dependency-first: base classes and leaf modules
    # before the strategies that import them, and run_backtest last (it imports
    # the strategies). Missing a module here means the button silently runs
    # stale code — the reason edits to strategy_base/KalmanFilter didn't apply.
    global db, strategy_base, EMACrossShortTest, KalmanFilter, EMAlgoTest, EMAlgoNonLinTest, run_backtest
    global ExtendedKalmanFilter, AccelerationKalmanFilter
    try:
        db = importlib.reload(db)
        strategy_base = importlib.reload(strategy_base)
        EMACrossShortTest = importlib.reload(EMACrossShortTest)
        KalmanFilter = importlib.reload(KalmanFilter)
        EMAlgoTest = importlib.reload(EMAlgoTest)
        EMAlgoNonLinTest = importlib.reload(EMAlgoNonLinTest)
        # Leaf strategy modules MUST reload before run_backtest, which does
        # `from <module> import <Strategy>`. If run_backtest reloads first it
        # captures a stale class, then reloading the strategy module in place
        # rebinds that class's globals — mixing old methods with new functions
        # (e.g. old f_func + new estimatefuture -> arg-count crash).
        ExtendedKalmanFilter = importlib.reload(ExtendedKalmanFilter)
        AccelerationKalmanFilter = importlib.reload(AccelerationKalmanFilter)
        run_backtest = importlib.reload(run_backtest)
    except Exception as e:
        return JSONResponse({'error': 'reload failed: %s' % e}, status_code=500)
    return _start_run(payload)


def _start_run(payload=None):
    global _run_thread
    payload = payload or {}
    kwargs = {'strategy': payload.get('strategy'),
              'params': payload.get('params'),
              'days': payload.get('days')}
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

    def _work():
        import time as _time
        _report_state.update(state='building', started=_time.time(), error=None)
        try:
            import build_report
            build_report = importlib.reload(build_report)
            build_report.build(symbol or build_report.SYMBOL_PREF,
                               selections=selections)
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

    def _work():
        import time as _time
        _sync_state.update(state='syncing', days=days,
                           started=_time.time(), error=None)
        try:
            import fetch_binance_csv
            fetch_binance_csv = importlib.reload(fetch_binance_csv)
            stats = fetch_binance_csv.main(days) or {}
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


@app.get('/api/download/pyfolio')
def download_pyfolio():
    if not os.path.isdir(run_backtest.TEST_DATA_DIR):
        return JSONResponse({'error': 'no report yet'}, status_code=404)
    zip_base = os.path.join(REPORTS, 'test_data_report')
    shutil.make_archive(zip_base, 'zip', run_backtest.TEST_DATA_DIR)
    return FileResponse(zip_base + '.zip', filename='test_data_report.zip')


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)
