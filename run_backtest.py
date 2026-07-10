
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import glob
import json
import math
import os
import shutil
import time
import traceback
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use('Agg')  # headless: pyfolio still renders its tear sheet to a file
import matplotlib.pyplot as plt

import backtrader as bt

from EMACrossShortTest import EMACrossShortTest
from KalmanFilter import KalmanTest

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'datas')
REPORTS = os.path.join(BASE, 'reports')
CHARTDATA_DIR = os.path.join(REPORTS, 'chartdata')    # per-symbol interactive chart data (JSON)
TEST_DATA_DIR = os.path.join(REPORTS, 'test_data')   # per-run report archive
PYFOLIO_DIR = TEST_DATA_DIR   # back-compat alias (server.py references it)
STATUS_PATH = os.path.join(REPORTS, 'status.json')
RESULTS_PATH = os.path.join(REPORTS, 'results.json')

STARTING_CASH = 100_000.0
LEVERAGE = 10
COMMISSION = 0.0002   # per-side, as a fraction of notional (0.0002 = 0.02%)

# ---- risk metrics -------------------------------------------------------
# Bars are 1-minute and crypto trades 24/7, so a year is 365*24*60 minutes.
# Sharpe/Sortino annualize a per-period ratio by sqrt(periods per year):
#     mean(r - rf) / std(r - rf, ddof=1) * sqrt(MINUTES_PER_YEAR)
# rf is a PER-MINUTE rate (RISK_FREE_ANNUAL / MINUTES_PER_YEAR), matching how
# empyrical subtracts risk_free from every return observation.
MINUTES_PER_YEAR = 365 * 24 * 60
RISK_FREE_ANNUAL = 0.06   

# ---- strategy selection -------------------------------------------------
# Swap the strategy by changing these two lines. Any PortfolioStrategy
# subclass works (see strategy_base.py); its build_chart_lines() drives the
# dashboard overlays, and STRATEGY_PARAMS is recorded in results.json.
STRATEGY = KalmanTest

STRATEGY_PARAMS = dict(
    trade_usd=2000.0,
    k=1.5,
    a=7,
    warmup=180,
    reversion=False,
    q_level = 0.1e-3,
    q_vel = 0.1e-6,
)
"""STRATEGY_PARAMS = dict(
    fast_ema_period =  60,
    slow_ema_period =  120,
)"""

# EMA alternative:
# STRATEGY = EMACrossShortTest
# STRATEGY_PARAMS = dict(trade_usd=2000.0, fast_ema_period=15, slow_ema_period=30)

BACKTEST_DAYS = 7
BACKTEST_END = None   # 'YYYY-MM-DD HH:MM' (UTC), or None = latest available bar


def _status(**kw):
    os.makedirs(REPORTS, exist_ok=True)
    kw['ts'] = time.time()
    with open(STATUS_PATH, 'w', encoding='utf-8') as f:
        json.dump(kw, f)


def _prepare_csvs_from_db():
    try:
        import db
        conn = db.get_conn()
    except Exception as e:
        print('[backtest] DB unavailable (%s) — using existing CSVs in datas/' % e)
        return None
    try:
        mn, mx = db.bars_span(conn)
        if mn is None:
            raise ValueError('database has no bars — run fetch_binance_csv.py first')
        naive = lambda t: t.astimezone(timezone.utc).replace(tzinfo=None)
        avail_first, avail_last = naive(mn), naive(mx)

        if BACKTEST_END:
            end = datetime.strptime(BACKTEST_END, '%Y-%m-%d %H:%M')
        else:
            end = avail_last
        start = end - timedelta(days=BACKTEST_DAYS)

        tol = timedelta(hours=1)
        if start < avail_first - tol or end > avail_last + tol:
            raise ValueError(
                '%s -> %s UTC period not present in data '
                '(available %s -> %s), aborting run'
                % (start, end, avail_first, avail_last))

        os.makedirs(DATA_DIR, exist_ok=True)
        for old in glob.glob(os.path.join(DATA_DIR, '*.csv')):
            os.remove(old)
        start_aw = start.replace(tzinfo=timezone.utc)
        end_aw = end.replace(tzinfo=timezone.utc)
        expected = int((end - start).total_seconds() // 60)   # ~bars in a full window

        n_sym, short, missing = 0, [], []
        for sym in db.symbols(conn):
            path = os.path.join(DATA_DIR, '%s-1m.csv' % sym)
            n = db.export_bars_csv(conn, sym, start_aw, end_aw, path)
            if n == 0:
                missing.append(sym)
                continue
            n_sym += 1
            if n < expected - 5:              # small slack for boundary/minor gaps
                short.append((sym, n))
        print('[backtest] exported %d symbols from postgres for the window '
              '(expected ~%d bars each)' % (n_sym, expected))
        if missing:
            print('[backtest] WARNING: %d symbol(s) have NO data in the window, '
                  'excluded from the run: %s' % (len(missing), ', '.join(missing)))
        if short:
            short.sort(key=lambda x: x[1])
            preview = ', '.join('%s(%d)' % (s, n) for s, n in short[:12])
            more = '' if len(short) <= 12 else ' ... +%d more' % (len(short) - 12)
            print('[backtest] NOTE: %d symbol(s) enter late / have gaps '
                  '(bars present, expected ~%d): %s%s'
                  % (len(short), expected, preview, more))
        return start, end
    finally:
        conn.close()


def _csv_span(path):
    """(first_dt, last_dt) of a bar CSV without parsing the whole file."""
    with open(path, 'rb') as f:
        f.readline()                              # header
        first = f.readline().split(b',')[0].decode()
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 4096))
        last = f.read().splitlines()[-1].split(b',')[0].decode()
    fmt = '%Y-%m-%d %H:%M:%S'
    return datetime.strptime(first, fmt), datetime.strptime(last, fmt)


def _resolve_window():
    """Backtest [start, end] from config + what the CSVs actually cover.

    Returns (start, end) or raises ValueError with the abort message when the
    requested period isn't present in the data.
    """
    spans = []
    for p in glob.glob(os.path.join(DATA_DIR, '*-1m.csv')):
        try:
            spans.append(_csv_span(p))
        except Exception:
            pass                                   # empty/corrupt file
    if not spans:
        raise ValueError('no usable CSVs in datas/ — run fetch_binance_csv.py first')
    avail_first = min(s[0] for s in spans)
    avail_last = max(s[1] for s in spans)

    if BACKTEST_END:
        end = datetime.strptime(BACKTEST_END, '%Y-%m-%d %H:%M')
    else:
        end = avail_last
    start = end - timedelta(days=BACKTEST_DAYS)

    tol = timedelta(hours=1)   # first/last bars may sit just inside the edge
    if start < avail_first - tol or end > avail_last + tol:
        raise ValueError(
            '%s -> %s UTC period not present in data '
            '(available %s -> %s), aborting run'
            % (start, end, avail_first, avail_last))
    return start, end


def _load_feeds(cerebro, fromdate, todate):
    symbols = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, '*-1m.csv'))):
        sym = os.path.basename(path)[:-len('-1m.csv')]
        data = bt.feeds.GenericCSVData(
            dataname=path,
            dtformat='%Y-%m-%d %H:%M:%S',
            timeframe=bt.TimeFrame.Minutes,
            compression=1,
            datetime=0, open=1, high=2, low=3, close=4, volume=5,
            openinterest=-1,
            name=sym,
            fromdate=fromdate,
            todate=todate,
        )
        cerebro.adddata(data)
        symbols.append(sym)
    return symbols


def _epoch(dt):
    """Naive UTC datetime -> integer epoch seconds (lightweight-charts time)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def _resample_returns(rets, bars_per_period=60*24):

    out = []
    acc = 1.0
    count = 0
    for r in rets:
        if r != r:          # skip NaN
            continue
        acc *= (1.0 + r)
        count += 1
        if count == bars_per_period:
            out.append(acc - 1.0)
            acc = 1.0
            count = 0
    return out

def _risk_metrics(rets):

    import pandas as pd

    r_minute = [x for x in rets if x == x]        # drop NaN
    r = _resample_returns(r_minute)
    print(r)
    n = len(r)
    if n < 2:
        return {}
    T = MINUTES_PER_YEAR/60 *1/24

    mu = sum(r) / n
    var = sum((x - mu) ** 2 for x in r) / (n - 1)      # ddof=1, matches empyrical
    sd = math.sqrt(var)

    cum = 1.0
    for x in r_minute:
        cum *= (1.0 + x)
    cum -= 1.0

    arith = mu * T
    try:
        cagr = (1.0 + cum) ** (T / n) - 1.0 if cum > -1.0 else float('nan')
    except OverflowError:
        cagr = float('inf')

    #log sharpe calculation
    import numpy as np
    log_returns = np.log(1+np.array(r))
    log_risk_free_rate = np.log(RISK_FREE_ANNUAL + 1)

    log_returns_annualised = np.mean(log_returns) * T
    ann_vol_log = np.std(log_returns, ddof = 1) * np.sqrt(T)

    ann_vol_arithmetic = sd * math.sqrt(T)

    out = {'cumulative_return': cum,
           'annual_return_cagr': cagr,
           'annual_volatility_returns': ann_vol_arithmetic,
           'annual_volatility_log_returns':ann_vol_log,
           'sharpe_arithmetic': None,
           'sharpe_log_returns': None}
    if ann_vol_arithmetic > 0:
        out['sharpe_arithmetic'] = (arith - RISK_FREE_ANNUAL) / ann_vol_arithmetic
        out['sharpe_log_returns'] = (log_returns_annualised - log_risk_free_rate) / ann_vol_log
    return out


def _dump_chartdata(strat):
    """Write per-symbol chart JSON: candles + fill markers + whatever overlay
    lines the strategy exposes via build_chart_lines(). Strategy-agnostic —
    a new strategy only has to return its own lines."""
    os.makedirs(CHARTDATA_DIR, exist_ok=True)
    saved = []
    total = len(strat.datas)

    try:
        lines_by_symbol = strat.build_chart_lines() or {}
    except Exception as e:
        print('[backtest] build_chart_lines failed (%s) — charts get no overlays' % e)
        lines_by_symbol = {}

    # Positions grouped per symbol (oldest first) for the per-stock table.
    pos_by_symbol = {}
    for t in strat.trade_log:
        pos_by_symbol.setdefault(t['symbol'], []).append(t)
    for lst in pos_by_symbol.values():
        lst.sort(key=lambda t: t.get('entry_dt') or '')

    for i, d in enumerate(strat.datas):
        times = [_epoch(bt.num2date(v)) for v in d.datetime.array]
        o, h = list(d.open.array), list(d.high.array)
        low, c = list(d.low.array), list(d.close.array)

        candles = [{'time': t, 'open': o[j], 'high': h[j],
                    'low': low[j], 'close': c[j]}
                   for j, t in enumerate(times)]

        markers = []
        for e in strat.executed[d._name]:
            buy = e['side'] == 'BUY'
            markers.append({
                'time': _epoch(datetime.fromisoformat(e['dt'])),
                'position': 'belowBar' if buy else 'aboveBar',
                'color': '#26a69a' if buy else '#ef5350',
                'shape': 'arrowUp' if buy else 'arrowDown',
                'text': e['side'],
            })
        markers.sort(key=lambda m: m['time'])

        cd = {'symbol': d._name,
              'candles': candles,
              'lines': lines_by_symbol.get(d._name, []),
              'markers': markers,
              'positions': pos_by_symbol.get(d._name, [])}
        with open(os.path.join(CHARTDATA_DIR, '%s.json' % d._name), 'w',
                  encoding='utf-8') as f:
            json.dump(cd, f)
        saved.append(d._name)

        if (i + 1) % 5 == 0 or (i + 1) == total:
            _status(state='running', phase='charts', done=i + 1, total=total)
    return saved


_POS_COLS = ['side', 'size', 'entry_signal_dt', 'entry_dt',
             'exit_signal_dt', 'exit_dt', 'entry_price',
             'exit_price', 'bars_held', 'pnl', 'pnlcomm']


def _dump_position_logs(strat, out_dir):
    """Write the position log into the per-run report folder:
      <run>/position_log.csv        — all symbols, one row per closed position
      <run>/positions/<SYMBOL>.csv  — one file per stock
    Returns the number of positions written."""
    import csv as _csv
    by_symbol = {}
    for t in strat.trade_log:
        by_symbol.setdefault(t['symbol'], []).append(t)
    for lst in by_symbol.values():
        lst.sort(key=lambda t: t.get('entry_dt') or '')

    # per-symbol files
    pos_dir = os.path.join(out_dir, 'positions')
    os.makedirs(pos_dir, exist_ok=True)
    for sym, rows in by_symbol.items():
        with open(os.path.join(pos_dir, '%s.csv' % sym), 'w',
                  newline='', encoding='utf-8') as f:
            w = _csv.DictWriter(f, fieldnames=_POS_COLS, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)

    # combined file (symbol-prefixed), grouped by symbol then entry time
    combined = sorted(strat.trade_log,
                      key=lambda t: (t['symbol'], t.get('entry_dt') or ''))
    with open(os.path.join(out_dir, 'position_log.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = _csv.DictWriter(f, fieldnames=['symbol'] + _POS_COLS,
                            extrasaction='ignore')
        w.writeheader()
        w.writerows(combined)
    return len(strat.trade_log)


def _dump_run_config(out_dir, strat, bt_start, bt_end, n_symbols):
    import csv as _csv
    rows = [('strategy', STRATEGY.__name__)]
    eff = {k: getattr(strat.params, k) for k in strat.params._getkeys()}
    for k, v in eff.items():
        rows.append(('param.%s' % k, v))
    rows += [
        ('broker.starting_cash', STARTING_CASH),
        ('broker.leverage', LEVERAGE),
        ('broker.commission', COMMISSION),
        ('broker.commission_pct_per_side', '%.4f%%' % (COMMISSION * 100)),
        ('risk_free_annual', RISK_FREE_ANNUAL),
        ('window_start', bt_start.isoformat()),
        ('window_end', bt_end.isoformat()),
        ('backtest_days', BACKTEST_DAYS),
        ('symbols', n_symbols),
    ]
    """with open(os.path.join(out_dir, 'config.csv'), 'w',
              newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        w.writerow(['setting', 'value'])
        w.writerows(rows)"""
    return rows


def _write_report_xlsx(out_dir, config_rows, stats, run_tag):
    try:
        import pandas as pd
    except Exception as e:
        print('[backtest] xlsx skipped (%s)' % e)
        return
    cfg = pd.DataFrame(config_rows, columns=['setting', 'value'])
    ps = pd.Series(stats or {}, dtype=object).rename('value')
    ps.index.name = 'metric'
    path = os.path.join(out_dir, 'report.xlsx')
    with pd.ExcelWriter(path, engine='openpyxl') as xl:
        cfg.to_excel(xl, sheet_name=str(run_tag), index=False)
        ps.to_excel(xl, sheet_name=str(run_tag), startrow=len(cfg) + 2)

    path = os.path.join(REPORTS, 'historical_reports.xlsx')
    with pd.ExcelWriter(path, engine = "openpyxl", mode="a", if_sheet_exists="overlay") as xl:
        cfg.to_excel(xl, sheet_name=str(run_tag), index=False)
        ps.to_excel(xl, sheet_name=str(run_tag), startrow=len(cfg) + 2)

def _pyfolio_report(strat, out_dir):
    """Export pyfolio items + tear sheet into out_dir (one folder per run,
    e.g. reports/pyfolio/20260707-153000/ — old runs are never overwritten)."""
    os.makedirs(out_dir, exist_ok=True)
    images, stats, error = [], {}, None

    pyf = strat.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev = pyf.get_pf_items()

    # Always dump the raw items so they can be analyzed elsewhere.
    returns.to_csv(os.path.join(out_dir, 'returns_pyfolio.csv'))
    try:
        transactions.to_csv(os.path.join(out_dir, 'transactions.csv'))
    except Exception:
        pass

    try:
        import pandas as pd
        import pyfolio as pf

        try:
            import empyrical as ep
            MPY = MINUTES_PER_YEAR       # minutes per year (crypto trades 24/7)
            rf = RISK_FREE_ANNUAL / MPY  # per-observation rate (returns are 1m)
            r = returns.dropna()

            rm = _risk_metrics(list(r.values))
            raw = {
                'Cumulative return': rm.get('cumulative_return'),
                'Annualised return (CAGR)': rm.get('annual_return_cagr'),
                'Annual volatility (Return)': rm.get('annual_volatility_returns'),
                'Annual volatility (Log Return)': rm.get('annual_volatility_log_returns'),
                'Sharpe ratio (Returns)': rm.get('sharpe_arithmetic'),
                'Sharpe ratio (log returns)': rm.get('sharpe_log_returns'),

                'Sortino ratio':     ep.sortino_ratio(r, annualization=MPY,
                                                      required_return=rf),
                'Calmar ratio':      ep.calmar_ratio(r, annualization=MPY),
                'Max drawdown':      ep.max_drawdown(r),
                'Stability':         ep.stability_of_timeseries(r),
                'Tail ratio':        ep.tail_ratio(r),
                'Daily VaR (5%)':    ep.value_at_risk(r),
                'Skew':              float(r.skew()),
                'Kurtosis':          float(r.kurtosis()),
            }
            stats = {k: (None if v is None or (isinstance(v, float) and v != v)
                         else round(float(v), 6))
                     for k, v in raw.items()}

            # Trade counters (integers, kept out of the float rounding above).
            wins = sum(1 for t in strat.trade_log if t['pnlcomm'] > 0)
            n_trades = len(strat.trade_log)
            stats['Total trades'] = n_trades
            stats['Winning trades'] = wins
            stats['Losing trades'] = n_trades - wins
            stats['Win rate %'] = (round(100.0 * wins / n_trades, 2)
                                   if n_trades else None)

            """(pd.Series(stats, dtype=object).rename('value')
             .to_csv(os.path.join(out_dir, 'perf_stats.csv'),
                     index_label='metric'))"""
        except Exception as e:
            error = 'stats: %s' % e

        try:
            plt.close('all')
            fig = pf.create_returns_tear_sheet(returns, return_fig=True)
            fig.savefig(os.path.join(out_dir, 'returns_tear_sheet.png'),
                        dpi=90, bbox_inches='tight')
            images.append('returns_tear_sheet.png')
            plt.close('all')
        except Exception as e:
            error = (error + ' | ' if error else '') + 'returns_tear_sheet: %s' % e
    except ImportError:
        error = 'pyfolio not installed (pip install pyfolio-reloaded)'

    return images, stats, error


def _dget(node, *keys, default=0):
    cur = node
    for k in keys:
        try:
            cur = cur[k]
        except (KeyError, TypeError, IndexError):
            return default
    return cur if cur or cur == 0 else default


def run():
    started = time.time()

    try:
        _status(state='running', phase='loading')
        os.makedirs(REPORTS, exist_ok=True)
        shutil.rmtree(CHARTDATA_DIR, ignore_errors=True)
        for stale in ('trades.csv', 'positions.csv'):
            try:
                os.remove(os.path.join(REPORTS, stale))
            except OSError:
                pass

        try:
            window = _prepare_csvs_from_db()
            if window is None:
                window = _resolve_window()
            bt_start, bt_end = window
        except ValueError as e:
            print('[backtest] %s' % e)
            _status(state='error', error=str(e))
            return
        print('[backtest] window: %s -> %s UTC' % (bt_start, bt_end))

        cerebro = bt.Cerebro()  # stdstats=True -> BuySell arrows on the plots
        cerebro.broker.setcash(STARTING_CASH)
        cerebro.broker.setcommission(commission=COMMISSION, leverage=float(LEVERAGE))

        symbols = _load_feeds(cerebro, bt_start, bt_end)
        if not symbols:
            _status(state='error',
                    error='no CSVs in datas/ — run fetch_binance_csv.py first')
            return
        print('[backtest] %d feeds loaded' % len(symbols))

#--------------------------------------------------------------------------------------
        cerebro.addstrategy(STRATEGY, **STRATEGY_PARAMS)
#---------------------------------------------------------------------------------------


        cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio',
                            timeframe=bt.TimeFrame.Minutes, compression=1)
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='dd')

        _status(state='running', phase='engine', total=len(symbols))

        strat = cerebro.run()[0]
        end_value = cerebro.broker.getvalue()
        print('[backtest] engine done, end value %.2f' % end_value)

        ta = strat.analyzers.trades.get_analysis()
        dd = strat.analyzers.dd.get_analysis()
        n_closed = _dget(ta, 'total', 'closed')
        n_won = _dget(ta, 'won', 'total')
        n_lost = _dget(ta, 'lost', 'total')

        # Risk metrics from the minute-level equity curve. Same helper the
        # pyfolio stats table uses, so the cards and the table always agree.
        eq = [v for _, v in strat.equity]
        rets = [0] + ([(b - a) / a for a, b in zip(eq, eq[1:]) if a] if len(eq) > 2 else [])

        rm = _risk_metrics(rets)
        rnd = lambda k, d=4: (None if rm.get(k) is None or rm.get(k) != rm.get(k)
                              else round(rm[k], d))

        summary = {
            'start_cash': STARTING_CASH,
            'end_value': round(end_value, 2),
            'pnl': round(end_value - STARTING_CASH, 2),
            'return_pct': round((end_value / STARTING_CASH - 1) * 100, 4),
            'trades_closed': n_closed,
            'won': n_won,
            'lost': n_lost,
            'win_rate_pct': round(100.0 * n_won / n_closed, 2) if n_closed else None,
            'max_drawdown_pct': round(_dget(dd, 'max', 'drawdown', default=0.0), 4),
            'annual_return_cagr': rnd('annual_return_cagr'),
            'annual_volatility_returns': rnd('annual_volatility_returns'),
            'annual_volatility_log_returns': rnd('annual_volatility_log_returns'),
            'sharpe_arithmetic': rnd('sharpe_arithmetic'),
            'sharpe_log_returns': rnd('sharpe_log_returns'),
            'symbols': len(symbols),
        }

        # ---- per-symbol -------------------------------------------------
        per_symbol = {}
        for t in strat.trade_log:
            s = per_symbol.setdefault(t['symbol'],
                                      {'trades': 0, 'pnl': 0.0, 'won': 0})
            s['trades'] += 1
            s['pnl'] += t['pnlcomm']
            s['won'] += 1 if t['pnlcomm'] > 0 else 0
        for sym in symbols:
            per_symbol.setdefault(sym, {'trades': 0, 'pnl': 0.0, 'won': 0})
        for s in per_symbol.values():
            s['pnl'] = round(s['pnl'], 4)

        # ---- chart data (candles + EMAs + fills for interactive charts) -
        _status(state='running', phase='charts', done=0, total=len(symbols))
        charts = _dump_chartdata(strat)
        print('[backtest] chart data written for %d symbols' % len(charts))

        # ---- per-run report archive (pyfolio + position logs) -----------
        _status(state='running', phase='pyfolio')
        run_tag = time.strftime('%Y%m%d-%H%M%S')
        run_dir = os.path.join(TEST_DATA_DIR, run_tag)
        pf_images, pf_stats, pf_error = _pyfolio_report(strat, run_dir)
        import pandas as pd
        n_pos = _dump_position_logs(strat, run_dir)
        config_rows = _dump_run_config(run_dir, strat, bt_start, bt_end, len(symbols))
        _write_report_xlsx(run_dir, config_rows, pf_stats, run_tag)   # config + perf_stats
        print('[backtest] report done (%s) -> test_data/%s (%d positions)'
              % (pf_error or 'ok', run_tag, n_pos))

        # ---- results.json ------------------------------------------------
        equity = strat.equity
        if len(equity) > 3000:  # thin for the dashboard
            step = len(equity) // 3000 + 1
            equity = equity[::step] + [strat.equity[-1]]

        # Full position log to CSV (one row per closed position, grouped by
        # symbol then entry time) + capped list in the JSON.
        import csv as _csv
        cols = ['symbol'] + _POS_COLS
        positions = sorted(strat.trade_log,
                           key=lambda t: (t['symbol'], t.get('entry_dt') or ''))
        with open(os.path.join(REPORTS, 'positions.csv'), 'w',
                  newline='', encoding='utf-8') as f:
            w = _csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(positions)

        results = {
            'generated': datetime.now(timezone.utc).isoformat(),
            'params': dict(STRATEGY_PARAMS,
                           strategy=STRATEGY.__name__,
                           leverage=LEVERAGE,
                           window_start=bt_start.isoformat(),
                           window_end=bt_end.isoformat()),
            'summary': summary,
            'per_symbol': per_symbol,
            'charts': charts,
            'equity': equity,
            'trades': strat.trade_log[-2000:],
            'pyfolio': {'images': pf_images, 'stats': pf_stats,
                        'error': pf_error,
                        'dir': 'reports/test_data/%s' % run_tag},
        }

        _status(state='running', phase='database')
        try:
            import db
            conn = db.get_conn()
            db.init_schema(conn)
            run_id = db.save_run(conn, results,
                                 strat.trade_log, strat.equity)
            conn.close()
            results['db'] = {'saved': True, 'run_id': run_id}
            print('[backtest] saved to postgres as run %d' % run_id)
        except Exception as e:
            results['db'] = {'saved': False, 'error': str(e)}
            print('[backtest] postgres save failed (file outputs intact): %s' % e)

        with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f)

        _status(state='done', elapsed=round(time.time() - started, 1),
                pnl=summary['pnl'], trades=n_closed)
        print('[backtest] done in %.1fs, pnl %.2f, %d trades'
              % (time.time() - started, summary['pnl'], n_closed))
    except Exception as e:
        print('[backtest] FAILED: %s\n%s' % (e, traceback.format_exc()))
        _status(state='error', error=str(e))


if __name__ == '__main__':
    run()
