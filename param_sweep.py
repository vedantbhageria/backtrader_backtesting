
import glob
import os
import sys
import time

import backtrader as bt

from ExtendedKalmanFilter import ExtendedKalmanTest

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'datas')
STARTING_CASH = 100_000.0
LEVERAGE = 10
COMMISSION = 0.0002

STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 4   # every Nth symbol
ONLY = sys.argv[2] if len(sys.argv) > 2 else None        # run a single config


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
    cerebro.addstrategy(ExtendedKalmanTest, **params)
    strat = cerebro.run()[0]
    end = cerebro.broker.getvalue()
    tl = strat.trade_log
    wins = sum(1 for t in tl if t['pnlcomm'] > 0)
    return dict(pnl=end - STARTING_CASH, trades=len(tl),
                win=(100.0 * wins / len(tl)) if tl else 0.0, n=n)


BASE_P = dict(trade_usd=2000.0, k=2, a=7, warmup=180, reversion=False,
              q_level=0.1e-3, q_vel=0.1e-6,
              k_exit=0.0, min_hold=1, cost_mult=0.0, drag_m=0.0003,
              c_d_window=180, trend_bias=False)

CONFIGS = {
    'clean_base(dz off)':  {},
    'min_hold=3':          dict(min_hold=3),
    'min_hold=5':          dict(min_hold=5),
    'min_hold=10':         dict(min_hold=10),
    'min_hold=20':         dict(min_hold=20),
    'drag_m=1e-4':         dict(drag_m=0.0001),
    'drag_m=1e-3':         dict(drag_m=0.001),
    'drag_m=3e-3':         dict(drag_m=0.003),
    'drag_off':            dict(drag_m=1e12),
    'k=1.5':               dict(k=1.5),
    'k=2.5':               dict(k=2.5),
    'k=3':                 dict(k=3.0),
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
