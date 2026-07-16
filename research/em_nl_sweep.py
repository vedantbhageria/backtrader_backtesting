"""Sweep EMNonLinTest: validate d_max (drag clamp) and the EM window.

Besides PnL it counts BLOW-UPS -- bars where the drag runaway drove the
prediction to an impossible value (pred < 0, or |pred/price - 1| > 0.5).
That's the -20M spike, quantified: a good d_max should give 0 blow-ups.
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, 'backend'), os.path.join(_ROOT, 'strategies')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glob
import os
import sys
import time

import numpy as np
import backtrader as bt

from EMAlgoNonLinTest import EMNonLinTest

BASE = _ROOT
DATA_DIR = os.path.join(BASE, 'datas')
START_CASH, COMMISSION, LEVERAGE = 100_000.0, 0.0002, 10.0
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def load_feeds(cerebro, stride):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, '*-1m.csv')))[::stride]
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
    cerebro.broker.setcash(START_CASH)
    cerebro.broker.setcommission(commission=COMMISSION, leverage=float(LEVERAGE))
    n = load_feeds(cerebro, stride)
    cerebro.addstrategy(EMNonLinTest, diag=True, **params)
    strat = cerebro.run()[0]
    end = cerebro.broker.getvalue()
    tl = strat.trade_log
    wins = sum(1 for t in tl if t['pnlcomm'] > 0)
    # blow-up scan across every symbol's recorded diagnostics
    blow = 0
    worst = 0.0
    for sym, rows in strat._diag.items():
        for rec in rows:
            pred, price = rec['pred'], rec['price']
            if price <= 0:
                continue
            rel = pred / price - 1.0
            if pred < 0 or abs(rel) > 0.5:
                blow += 1
            if abs(rel) > abs(worst):
                worst = rel
    return dict(pnl=end - START_CASH, trades=len(tl),
                win=(100.0 * wins / len(tl)) if tl else 0.0,
                blow=blow, worst=worst, n=n)


BASE_P = dict(trade_usd=2000.0, k=2.5, warmup=180, reversion=False,
              q_level=0.1e-3, q_vel=0.1e-6, q_acc=0.1e-9, d0=0.0,
              d_max=10.0, em_window=720, em_interval=1440, em_iters=3, dead=300)

CONFIGS = {
    'd_max=1e6 (old)':  dict(d_max=1e6),
    'd_max=50':         dict(d_max=50.0),
    'd_max=20':         dict(d_max=20.0),
    'd_max=10 (base)':  {},
    'd_max=5':          dict(d_max=5.0),
    'd_max=2':          dict(d_max=2.0),
    'em_window=360':    dict(em_window=360),
    'em_window=1440':   dict(em_window=1440),
    'em_interval=720':  dict(em_interval=720),
    'em_interval=2880': dict(em_interval=2880),
    'em_iters=2':       dict(em_iters=2),
}

if __name__ == '__main__':
    print('EMNonLinTest sweep: stride=%d subset, %d configs\n' % (STRIDE, len(CONFIGS)),
          flush=True)
    results = {}
    for name, over in CONFIGS.items():
        p = dict(BASE_P); p.update(over)
        t0 = time.time()
        try:
            r = run_cfg(p, STRIDE)
            results[name] = r
            print('%-18s pnl=%+9.1f trades=%6d win=%4.1f%% blowups=%5d worst=%+.2f n=%d (%ss)'
                  % (name, r['pnl'], r['trades'], r['win'], r['blow'], r['worst'],
                     r['n'], round(time.time() - t0, 1)), flush=True)
        except Exception as e:
            print('%-18s FAILED: %s' % (name, e), flush=True)

    print('\n=== ranked by pnl (blow-ups matter more than pnl) ===', flush=True)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]['pnl']):
        print('%-18s pnl=%+9.1f  blowups=%d' % (name, r['pnl'], r['blow']), flush=True)
