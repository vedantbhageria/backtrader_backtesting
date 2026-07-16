"""Child-process entry point for one backtest job.

    python backend/job_runner.py <job_dir>

Reads <job_dir>/job.json ({strategy, params, days, timeframe}), redirects all
of run_backtest's output paths into the job dir, and runs the backtest. The
parent (server.py's job manager) watches <job_dir>/status.json — the same
status file format the single-run flow always used — plus the process exit
code. Each job is a separate OS process, so N jobs genuinely run in parallel
(numba/GIL and all).
"""
import json
import os
import sys

import _paths  # noqa: F401


def main():
    job_dir = os.path.abspath(sys.argv[1])
    with open(os.path.join(job_dir, 'job.json'), encoding='utf-8') as f:
        job = json.load(f)

    import run_backtest as RB
    # Redirect every artifact into the job dir. The test_data archive and the
    # postgres run-history row keep their global locations on purpose — those
    # are shared, append-only stores.
    RB.REPORTS = job_dir
    RB.STATUS_PATH = os.path.join(job_dir, 'status.json')
    RB.RESULTS_PATH = os.path.join(job_dir, 'results.json')
    RB.CHARTDATA_DIR = os.path.join(job_dir, 'chartdata')

    RB.run(strategy=job.get('strategy'), params=job.get('params'),
           days=job.get('days'), timeframe=job.get('timeframe'))

    # exit code mirrors the final state so the parent can catch hard crashes
    try:
        with open(RB.STATUS_PATH, encoding='utf-8') as f:
            ok = json.load(f).get('state') == 'done'
    except Exception:
        ok = False
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
