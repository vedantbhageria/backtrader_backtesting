"""Persistent Python REPL worker behind the dashboard's Console tab.

Protocol: one JSON object per line on stdin ({"code": "..."}), one JSON
object per line on stdout ({"out": "...", "err": "..."|null}). User code runs
in ONE persistent namespace (variables survive between commands, like a
notebook) with stdout/stderr captured; the last expression's repr is echoed
like a REPL. Runs as a subprocess so runaway user code can be killed/restarted
without touching the dashboard server.

Preloaded helpers (see _help_text) wrap the generated artifacts:
reports/chartdata/<SYM>.json, reports/positions.csv, reports/results.json.
"""
import ast
import glob
import io
import json
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr

BASE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(BASE, 'reports')
CHARTDATA = os.path.join(REPORTS, 'chartdata')

import numpy as np
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)


def results():
    """reports/results.json as a dict (summary, params, per_symbol, ...)."""
    with open(os.path.join(REPORTS, 'results.json'), encoding='utf-8') as f:
        return json.load(f)


def symbols():
    """Symbols that have chart data from the last run."""
    return sorted(os.path.basename(p)[:-5]
                  for p in glob.glob(os.path.join(CHARTDATA, '*.json')))


def chartdata(sym):
    """Raw chartdata dict for a symbol: candles, lines, markers, positions."""
    with open(os.path.join(CHARTDATA, '%s.json' % sym.upper()),
              encoding='utf-8') as f:
        return json.load(f)


def candles(sym):
    """OHLCV candles as a DataFrame indexed by UTC time."""
    df = pd.DataFrame(chartdata(sym)['candles'])
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    return df.set_index('time')


def lines(sym):
    """Strategy overlay lines: {name: DataFrame(time, value)}."""
    out = {}
    for L in chartdata(sym).get('lines', []):
        df = pd.DataFrame(L['points'])
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df = df.set_index('time')
        out[L['name']] = df
    return out


def markers(sym):
    """Buy/sell fill markers as a DataFrame."""
    df = pd.DataFrame(chartdata(sym).get('markers', []))
    if not df.empty:
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    return df


def positions(sym=None):
    """Closed positions (reports/positions.csv); optionally one symbol."""
    df = pd.read_csv(os.path.join(REPORTS, 'positions.csv'))
    if sym:
        df = df[df['symbol'] == sym.upper()].reset_index(drop=True)
    return df


def equity():
    """Account equity curve from results.json as a DataFrame."""
    eq = results().get('equity', [])
    df = pd.DataFrame(eq, columns=['time', 'value'])
    df['time'] = pd.to_datetime(df['time'])
    return df.set_index('time')


def _help_text():
    return (
        "data helpers (all read the LAST run's artifacts):\n"
        "  results()        results.json dict: summary, params, per_symbol...\n"
        "  symbols()        symbols with chart data\n"
        "  candles(sym)     OHLCV DataFrame indexed by UTC time\n"
        "  lines(sym)       strategy overlay lines {name: DataFrame}\n"
        "  markers(sym)     buy/sell fills DataFrame\n"
        "  positions(sym=None)  closed positions DataFrame\n"
        "  equity()         account equity curve DataFrame\n"
        "  chartdata(sym)   the raw chartdata JSON dict\n"
        "preloaded: np, pd, json, os, glob   paths: BASE, REPORTS, CHARTDATA\n"
        "variables persist between commands (notebook-style)."
    )


def helpdata():
    print(_help_text())


NAMESPACE = {
    'np': np, 'pd': pd, 'json': json, 'os': os, 'glob': glob,
    'BASE': BASE, 'REPORTS': REPORTS, 'CHARTDATA': CHARTDATA,
    'results': results, 'symbols': symbols, 'chartdata': chartdata,
    'candles': candles, 'lines': lines, 'markers': markers,
    'positions': positions, 'equity': equity, 'helpdata': helpdata,
}


def run_code(code):
    """Exec `code` in the persistent namespace; REPL-style last-expression
    echo. Returns (captured_output, error_or_None)."""
    buf = io.StringIO()
    try:
        tree = ast.parse(code, mode='exec')
    except SyntaxError:
        return '', traceback.format_exc(limit=0)
    # split a trailing expression so its value gets echoed like a REPL
    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = ast.Expression(tree.body.pop(-1).value)
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            if tree.body:
                exec(compile(tree, '<console>', 'exec'), NAMESPACE)
            if last_expr is not None:
                val = eval(compile(last_expr, '<console>', 'eval'), NAMESPACE)
                if val is not None:
                    NAMESPACE['_'] = val
                    print(repr(val) if not isinstance(
                        val, (pd.DataFrame, pd.Series)) else val)
    except Exception:
        return buf.getvalue(), traceback.format_exc()
    return buf.getvalue(), None


def main():
    out = sys.stdout                      # protocol channel
    sys.stdout = sys.__stdout__           # (kept as-is; code output is captured)
    banner = ('python %s · backtrader console\n' % sys.version.split()[0]
              + _help_text())
    out.write(json.dumps({'out': banner, 'err': None}) + '\n')
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            code = json.loads(line).get('code', '')
        except ValueError:
            continue
        o, e = run_code(code)
        out.write(json.dumps({'out': o, 'err': e}) + '\n')
        out.flush()


if __name__ == '__main__':
    main()
