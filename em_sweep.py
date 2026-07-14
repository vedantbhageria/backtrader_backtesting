"""Lightweight parameter sweep for EMTest (EMAlgoTest.py).

Reuses whatever CSVs are already in datas/. Runs a representative subset of
symbols per config for speed, ranks by PnL. Validate the winner on the full
universe via the server afterward. Throwaway tuning harness, not part of the app.
"""
import glob
import os
import sys
import time

import backtrader as bt

from EMAlgoTest import EMTest

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'datas')
STARTING_CASH = 100_000.0
LEVERAGE = 10
COMMISSION = 0.0002

STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ONLY = sys.argv[2] if len(sys.argv) > 2 else None


def load_feeds(cerebro, stride):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, '*-1m.csv')))
    paths = paths[::stride] if stride > 1 else paths
    for path in paths:
        sym = os.path.basename(path)[:-len('-1m.csv')]
        cerebro.adddata(bt.feeds.GenericCSVData(
            dataname=path, dtformat='%Y-%m-%d %H:%M:%S',
            timeframe=bt.TimeFrame.Minutes, compression=1,
            datetime=0, open=1, high=2, low=3, close=4, volume=5,
            openinterest=-1, name=sym))
    return len(paths)


def run_cfg(params, stride):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(STARTING_CASH)
    cerebro.broker.setcommission(commission=COMMISSION, leverage=float(LEVERAGE))
    n = load_feeds(cerebro, stride)
    cerebro.addstrategy(EMTest, **params)
    strat = cerebro.run()[0]
    end = cerebro.broker.getvalue()
    tl = strat.trade_log
    wins = sum(1 for t in tl if t['pnlcomm'] > 0)
    return dict(pnl=end - STARTING_CASH, trades=len(tl),
                win=(100.0 * wins / len(tl)) if tl else 0.0, n=n)


BASE_P = dict(trade_usd=2000.0, k=2, warmup=180, reversion=False,
              q_level=0.1e-3, q_vel=0.1e-6,
              em_window=720, em_interval=720, em_iters=5)

CONFIGS = {
    'base':              {},
    'reversion=True':    dict(reversion=True),
    'k=1.5':             dict(k=1.5),
    'k=2.5':              dict(k=2.5),
    'k=3':               dict(k=3.0),
    'em_window=360':     dict(em_window=360, em_interval=360),
    'em_window=1440':    dict(em_window=1440, em_interval=1440),
    'em_interval=180':   dict(em_interval=180),   # refit more often than window
    'em_interval=2880':  dict(em_interval=2880),  # refit less often
    'em_iters=2':        dict(em_iters=2),
    'em_iters=10':       dict(em_iters=10),
    'warmup=360':        dict(warmup=360),
}

if __name__ == '__main__':
    results = {}
    items = ([(ONLY, CONFIGS[ONLY])] if ONLY else list(CONFIGS.items()))
    print('sweep: stride=%d (subset), %d configs' % (STRIDE, len(items)), flush=True)
    for name, over in items:
        p = dict(BASE_P); p.update(over)
        t0 = time.time()
        try:
            r = run_cfg(p, STRIDE)
            r['secs'] = round(time.time() - t0, 1)
            results[name] = r
            print('%-20s pnl=%+9.1f  trades=%6d  win=%4.1f%%  n=%d  (%ss)'
                  % (name, r['pnl'], r['trades'], r['win'], r['n'], r['secs']),
                  flush=True)
        except Exception as e:
            print('%-20s FAILED: %s' % (name, e), flush=True)

    print('\n=== ranked by pnl (subset, stride=%d) ===' % STRIDE, flush=True)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]['pnl']):
        print('%-20s pnl=%+9.1f  trades=%6d  win=%4.1f%%'
              % (name, r['pnl'], r['trades'], r['win']), flush=True)
