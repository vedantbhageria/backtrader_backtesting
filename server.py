
import importlib
import json
import os
import shutil
import threading

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import db
import strategy_base
import EMACrossShortTest
import KalmanFilter
import ExtendedKalmanFilter
import AccelerationKalmanFilter
import EMAlgoTest
import run_backtest

BASE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(BASE, 'reports')
os.makedirs(REPORTS, exist_ok=True)

app = FastAPI(title='Backtrader Dashboard')
app.mount('/reports', StaticFiles(directory=REPORTS), name='reports')
# Vendored JS (lightweight-charts) so the dashboard needs no internet/CDN.
app.mount('/vendor', StaticFiles(directory=os.path.join(BASE, 'vendor')),
          name='vendor')

_run_thread: threading.Thread | None = None


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


@app.post('/api/run')
def run():
    global run_backtest
    if _run_thread is not None and _run_thread.is_alive():
        return JSONResponse({'error': 'backtest already running'},
                            status_code=409)
    # Reload every project module from disk so edits are picked up without a
    # server restart. Order is dependency-first: base classes and leaf modules
    # before the strategies that import them, and run_backtest last (it imports
    # the strategies). Missing a module here means the button silently runs
    # stale code — the reason edits to strategy_base/KalmanFilter didn't apply.
    global db, strategy_base, EMACrossShortTest, KalmanFilter, EMAlgoTest, run_backtest
    global ExtendedKalmanFilter, AccelerationKalmanFilter
    try:
        db = importlib.reload(db)
        strategy_base = importlib.reload(strategy_base)
        EMACrossShortTest = importlib.reload(EMACrossShortTest)
        KalmanFilter = importlib.reload(KalmanFilter)
        EMAlgoTest = importlib.reload(EMAlgoTest)
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
    return _start_run()


def _start_run():
    global _run_thread
    _run_thread = threading.Thread(target=run_backtest.run, daemon=True)
    _run_thread.start()
    return {'started': True}


@app.get('/api/runs')
def runs_history():
    """Past backtest runs from Postgres (falls back cleanly when DB is down)."""
    try:
        import db
        conn = db.get_conn()
        out = db.list_runs(conn)
        conn.close()
        return {'runs': out}
    except Exception as e:
        return JSONResponse({'error': 'database unavailable: %s' % e},
                            status_code=503)


@app.get('/api/download/pyfolio')
def download_pyfolio():
    if not os.path.isdir(run_backtest.TEST_DATA_DIR):
        return JSONResponse({'error': 'no report yet'}, status_code=404)
    zip_base = os.path.join(REPORTS, 'test_data_report')
    shutil.make_archive(zip_base, 'zip', run_backtest.TEST_DATA_DIR)
    return FileResponse(zip_base + '.zip', filename='test_data_report.zip')


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)
