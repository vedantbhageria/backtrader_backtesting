
import glob
import html
import json
import os

import numpy as np
import pandas as pd
import backtrader as bt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import _paths
import symbol_classes
import pnl_stats
DATA_DIR = os.path.join(_paths.ROOT, 'datas')
OUT_DIR = os.path.join(_paths.ROOT, 'report_out')
SYMBOL_PREF = 'ETHUSDT'
START_CASH, COMMISSION, LEVERAGE = 100_000.0, 0.0002, 10.0
DS = 7000    # downsample the series to this many points (SVG Scatter)

DEFAULT_SELECTION = [
    {'name': 'KalmanTest'},
    {'name': 'ExtendedKalmanTest'},
    {'name': 'EMTest'},
]
PALETTE = ['#4c9be8', '#e8834c', '#5cc98a', '#b58cf0', '#e5566a', '#4fc6c0',
           '#d29922', '#ff7eb6', '#8b96a5', '#58d68d', '#c39bd3', '#5dade2']

LABELS = {'P00': 'Var(position)', 'P11': 'Var(velocity)', 'P22': 'Var(accel)',
          'P01': 'Cov(pos, vel)', 'P02': 'Cov(pos, accel)', 'P12': 'Cov(vel, accel)'}


def pick_symbol(prefer, data_dir=DATA_DIR, suffix='1m'):
    paths = sorted(glob.glob(os.path.join(data_dir, '*-%s.csv' % suffix)))
    if not paths:
        raise ValueError('no %s CSVs in %s — run a %s backtest or sync first'
                         % (suffix, data_dir, suffix))
    hit = [p for p in paths if os.path.basename(p).startswith('%s-%s' % (prefer, suffix))]
    p = hit[0] if hit else paths[0]
    return os.path.basename(p)[:-len('-%s.csv' % suffix)], p


def _window_start(mn, mx, days):
    """Start of the export window: last `days` before mx (None -> everything)."""
    if not days:
        return mn
    from datetime import timedelta
    return max(mn, mx - timedelta(days=days))


def _ensure_symbol_csv(prefer, timeframe, days=None):
    """Export the chosen symbol's history for `timeframe` from postgres so the
    report reflects the database rather than whatever short window a prior run
    happened to leave in datas/. (None = everything).
    fallback: existing csvs."""
    import run_backtest
    data_dir, suffix, _ = run_backtest._tf_cfg(timeframe)
    os.makedirs(data_dir, exist_ok=True)
    try:
        import db
        conn = db.get_conn()
    except Exception as e:
        print('[report] DB unavailable (%s) — using existing %s CSVs' % (e, suffix))
        return pick_symbol(prefer, data_dir, suffix)
    try:
        syms = db.symbols(conn, timeframe)
        sym = (next((s for s in syms if s == prefer), None)
               or next((s for s in syms if s.startswith(prefer)), None)
               or (syms[0] if syms else None))
        if not sym:
            raise ValueError('database has no %s bars' % timeframe)
        mn, mx = db.bars_span(conn, timeframe)
        start = _window_start(mn, mx, days)
        path = os.path.join(data_dir, '%s-%s.csv' % (sym, suffix))
        n = db.export_bars_csv(conn, sym, start, mx, path, interval=timeframe)
        print('[report] exported %s %s bars for %s (%s -> %s)'
              % (n, timeframe, sym, start, mx))
        return sym, path
    finally:
        conn.close()


def _ensure_all_csvs(timeframe, days=None):
    """same as above but for all symbols together"""
    import run_backtest
    data_dir, suffix, _ = run_backtest._tf_cfg(timeframe)
    os.makedirs(data_dir, exist_ok=True)
    existing = lambda: [(os.path.basename(p)[:-len('-%s.csv' % suffix)], p)
                        for p in sorted(glob.glob(os.path.join(data_dir, '*-%s.csv' % suffix)))]
    try:
        import db
        conn = db.get_conn()
    except Exception as e:
        print('[report] DB unavailable (%s): using existing %s CSVs' % (e, suffix))
        return existing()
    try:
        syms = db.symbols(conn, timeframe)
        if not syms:
            raise ValueError('database has no %s bars' % timeframe)
        mn, mx = db.bars_span(conn, timeframe)
        start = _window_start(mn, mx, days)
        feeds = []
        for sym in syms:
            path = os.path.join(data_dir, '%s-%s.csv' % (sym, suffix))
            if db.export_bars_csv(conn, sym, start, mx, path, interval=timeframe):
                feeds.append((sym, path))
        print('[report] exported %d symbols (%s bars, %s -> %s)'
              % (len(feeds), timeframe, start, mx))
        return feeds
    finally:
        conn.close()


def run_strategy(label, Strat, params, feeds, color, compression=1):

    cer = bt.Cerebro()
    cer.broker.setcash(START_CASH)
    cer.broker.setcommission(commission=COMMISSION, leverage=LEVERAGE)
    for sym, path in feeds:
        cer.adddata(bt.feeds.GenericCSVData(
            dataname=path, dtformat='%Y-%m-%d %H:%M:%S',
            timeframe=bt.TimeFrame.Minutes, compression=compression,
            datetime=0, open=1, high=2, low=3, close=4, volume=5,
            openinterest=-1, name=sym))
    cer.addstrategy(Strat, **dict(params, diag=(len(feeds) == 1)))
    strat = cer.run()[0]
    end = cer.broker.getvalue()
    rows = strat._diag.get(feeds[0][0], []) if len(feeds) == 1 else []
    if rows:                       # strategies without filter internals (e.g.
        diag = pd.DataFrame(rows)  # EMACross) still get equity/PnL panels
        diag['dt'] = pd.to_datetime(diag['t'], unit='s', utc=True)
        diag = diag.set_index('dt')
    else:
        diag = None
    eq = pd.DataFrame(strat.equity, columns=['dt', 'value'])
    eq['dt'] = pd.to_datetime(eq['dt']); eq = eq.set_index('dt')
    tl = strat.trade_log
    wins = sum(1 for t in tl if t['pnlcomm'] > 0)
    rms = None
    if diag is not None:
        innov = diag['innov'].to_numpy(float)
        rms = round(float(np.sqrt(np.mean(innov ** 2))), 4)
    # 'trades' = closed position lifecycles ("Positions" in the dashboard);
    # 'trades_total' = individual order fills (entries + scale-in adds +
    # exits) — the two only diverge when a strategy adds to an open position.
    stats = dict(end_value=round(end, 2), pnl=round(end - START_CASH, 2),
                 trades=len(tl), trades_total=sum(len(v) for v in strat.executed.values()),
                 wins=wins,
                 win_rate=round(100.0 * wins / len(tl), 2) if tl else 0.0,
                 innov_rms=rms)
    return dict(name=label, diag=diag, equity=eq, stats=stats, color=color)


def ds(index, y):
    y = np.asarray(y, float)
    step = max(1, len(y) // DS)
    return index[::step], y[::step]


LAYOUT = dict(template='plotly_dark', hovermode='x',
              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,29,43,0.6)',
              margin=dict(l=60, r=20, t=40, b=40),
              legend=dict(orientation='h', y=1.06, x=0),
              font=dict(family='system-ui, sans-serif', size=12),
              hoverlabel=dict(bgcolor='#1b2536', bordercolor='#37c1d1',
                              font=dict(color='#e7edf5', size=12)))


def fig_pred(results):
    fig = make_subplots(rows=len(results), cols=1, vertical_spacing=0.07,
                        subplot_titles=[r['name'] for r in results])
    for i, res in enumerate(results, 1):
        d = res['diag']
        x, price = ds(d.index, d['price'].values)
        _, pred = ds(d.index, d['pred'].values)
        _, up = ds(d.index, d['upper'].values)
        _, lo = ds(d.index, d['lower'].values)
        show = (i == 1)
        fig.add_trace(go.Scatter(x=x, y=up, line=dict(width=0), hoverinfo='skip',
                                 showlegend=False, name='upper'), row=i, col=1)
        fig.add_trace(go.Scatter(x=x, y=lo, fill='tonexty', line=dict(width=0),
                                 fillcolor='rgba(120,165,215,0.40)', hoverinfo='skip',
                                 name='±kσ band', showlegend=show), row=i, col=1)
        fig.add_trace(go.Scatter(x=x, y=price, line=dict(color='#9aa4b8', width=1),
                                 name='price', showlegend=show,
                                 hovertemplate='price %{y:.2f}<extra></extra>'), row=i, col=1)
        fig.add_trace(go.Scatter(x=x, y=pred, line=dict(color='#e8590c', width=1.2),
                                 name='prediction', showlegend=show,
                                 hovertemplate='pred %{y:.2f}<extra></extra>'), row=i, col=1)
        fig.update_yaxes(title_text='price', row=i, col=1)
    fig.update_layout(height=300 * len(results), **LAYOUT)
    return fig


def fig_innov(results):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14,
                        subplot_titles=['Innovation (price − prediction)',
                                        'Innovation uncertainty √S'])
    for res in results:
        d = res['diag']
        x, inn = ds(d.index, d['innov'].values)
        _, s = ds(d.index, np.sqrt(d['S'].values))
        fig.add_trace(go.Scatter(x=x, y=inn, line=dict(width=0.9, color=res['color']),
                                 name='%s · RMS %.4g' % (res['name'], res['stats']['innov_rms']),
                                 hovertemplate='%{y:.3f}<extra></extra>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=s, line=dict(width=0.9, color=res['color']),
                                 name=res['name'], showlegend=False,
                                 hovertemplate='√S %{y:.3f}<extra></extra>'), row=2, col=1)
    fig.add_hline(y=0, row=1, col=1, line=dict(color='gray', width=0.5))
    fig.update_layout(height=560, **LAYOUT)
    return fig


def fig_P(res):
    d = res['diag']
    terms = [c for c in d.columns if c.startswith('P')]
    cols = 3
    rows = int(np.ceil(len(terms) / cols))
    fig = make_subplots(rows=rows, cols=cols, vertical_spacing=0.16, horizontal_spacing=0.08,
                        subplot_titles=['%s · %s' % (c, LABELS.get(c, c)) for c in terms])
    for i, c in enumerate(terms):
        r, cc = i // cols + 1, i % cols + 1
        x, y = ds(d.index, d[c].values)
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='#e5566a', width=1),
                                 name=c, showlegend=False,
                                 hovertemplate='%{y:.4g}<extra></extra>'), row=r, col=cc)
        tail = d[c].iloc[200:].to_numpy(float)
        lo, hi = float(np.nanmin(tail)), float(np.nanmax(tail))
        pad = (hi - lo) * 0.2 or (abs(hi) * 0.2) or 1e-9
        fig.update_yaxes(range=[lo - pad, hi + pad], row=r, col=cc)
    fig.update_layout(height=300 * rows, **{k: v for k, v in LAYOUT.items() if k != 'hovermode'},
                      hovermode='closest')
    return fig


def fig_equity(results):
    fig = go.Figure()
    for res in results:
        e = res['equity']
        x, y = ds(e.index, e['value'].values)
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=res['color'], width=1.3),
                                 name='%s · PnL %+.0f' % (res['name'], res['stats']['pnl']),
                                 hovertemplate='%{y:.0f}<extra></extra>'))
    fig.add_hline(y=START_CASH, line=dict(color='gray', dash='dash', width=0.6))
    fig.update_layout(height=400, **LAYOUT)
    fig.update_yaxes(title_text='account value ($)')
    return fig


def fig_winpnl(results):
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Win rate (%)', 'PnL ($)'])
    names = [r['name'] for r in results]
    bar_c = [r['color'] for r in results]
    fig.add_trace(go.Bar(x=names, y=[r['stats']['win_rate'] for r in results],
                         marker_color=bar_c, showlegend=False,
                         hovertemplate='%{y:.2f}%<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=[r['stats']['pnl'] for r in results],
                         marker_color=bar_c, showlegend=False,
                         hovertemplate='$%{y:.2f}<extra></extra>'), row=1, col=2)
    fig.update_layout(height=360, template='plotly_dark',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(22,29,43,0.6)',
                      margin=dict(l=50, r=20, t=40, b=30))
    return fig


_SUMMARY_LABELS = {'trades': 'positions', 'trades_total': 'trades'}


def summary_table(results):
    cols = ['end_value', 'pnl', 'trades', 'trades_total', 'wins', 'win_rate']
    head = ''.join('<th>%s</th>' % _SUMMARY_LABELS.get(c, c) for c in cols)
    body = ''
    for res in results:
        s = res['stats']
        cells = ''
        for c in cols:
            v = s[c]
            # only pnl has a meaningful sign — leave counts/rates neutral
            cls = ' pos' if c == 'pnl' and isinstance(v, (int, float)) and v > 0 \
                else ' neg' if c == 'pnl' and isinstance(v, (int, float)) and v < 0 else ''
            cells += '<td class="%s">%s</td>' % (cls.strip(), '&mdash;' if v is None else v)
        body += '<tr><th class="rowh">%s</th>%s</tr>' % (res['name'], cells)
    return '<table><thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table>' % (head, body)

_PSTAT_ROWS = [
    ('End value (USDT)',        'end_value',                    lambda v: '%.2f' % v),
    ('PnL (USDT)',              'pnl',                          lambda v: '%+.2f' % v),
    ('Return %',                'return_pct',                   lambda v: '%+.3f%%' % v),
    ('Positions',                'trades_closed',                lambda v: '%d' % v),
    ('Trades',                  'trades_total',                 lambda v: '%d' % v),
    ('Win rate %',              'win_rate_pct',                 lambda v: '%.1f%%' % v),
    ('Max drawdown %',          'max_drawdown_pct',             lambda v: '%.2f%%' % v),
    ('Ann. return (CAGR)',      'annual_return_cagr',           lambda v: '%+.2f%%' % (100*v)),
    ('Ann. volatility (ret)',   'annual_volatility_returns',    lambda v: '%.2f%%' % (100*v)),
    ('Ann. volatility (log)',   'annual_volatility_log_returns', lambda v: '%.2f%%' % (100*v)),
    ('Sharpe (arithmetic)',     'sharpe_arithmetic',            lambda v: '%.4f' % v),
    ('Sharpe (log)',            'sharpe_log_returns',           lambda v: '%.4f' % v),
]


def stats_table(results):
    """Portfolio-stats comparison (metrics as rows, runs as columns) for
    results that carry a full run summary (archived-run compares)."""
    runs = [r for r in results if r.get('summary')]
    if not runs:
        return ''
    head = ''.join('<th>%s</th>' % r['name'] for r in runs)
    body = ''
    for label, key, fmt in _PSTAT_ROWS:
        cells = ''
        for r in runs:
            v = r['summary'].get(key)
            missing = v is None or v != v
            cls = '' if missing or not isinstance(v, (int, float)) else (' pos' if v > 0 else ' neg' if v < 0 else '')
            try:
                cells += '<td class="%s">%s</td>' % (cls.strip(), '&mdash;' if missing else fmt(v))
            except (TypeError, ValueError):
                cells += '<td>%s</td>' % v
        body += '<tr><th class="rowh">%s</th>%s</tr>' % (label, cells)
    return ('<h2 id="portfolio-stats">Portfolio stats</h2>'
            '<table class="statstable"><thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table>'
            % (head, body))


def _sym_compare_html(runs):
    """Interactive per-symbol compare for runs that carry job chartdata: a
    symbol dropdown drives a per-run stats table (PnL/positions/win rate/
    Sharpe/max DD from the position logs) plus one subplot row per run
    (price + every stored indicator/prediction line, disclaimer:
    downsampled)."""
    runs = [r for r in runs if r.get('symdata')]
    if not runs:
        return ''
    syms = sorted(set().union(*[set(r['symdata']) for r in runs],
                              *[set(r.get('symstats') or {}) for r in runs]))
    if not syms:
        return ''
    # is there ANY additional (diag) series to toggle? if not, say so instead
    # of showing a checkbox that silently does nothing
    has_addl = any(L.get('additional')
                   for r in runs for e in r['symdata'].values()
                   for L in e.get('lines', []))
    addl_ctl = ("""
  <label style="margin-left:14px;font-size:13px;color:#8a97ad;cursor:pointer">
    <input type="checkbox" id="symaddl"> show additional (diag) data
  </label>""" if has_addl else """
  <label style="margin-left:14px;font-size:13px;color:#5a657a" title="none of the compared runs recorded per-bar filter internals — run with 'Record additional data' checked to get them">
    <input type="checkbox" id="symaddl" disabled> show additional (diag) data (none recorded)
  </label>""")
    payload = json.dumps({r['name']: r['symdata'] for r in runs})
    statspayload = json.dumps({r['name']: (r.get('symstats') or {}) for r in runs})
    meta = json.dumps([{'name': r['name'], 'color': r['color']} for r in runs])
    options = ''.join('<option>%s</option>' % s for s in syms)
    run_checks = ''.join(
        '<label style="margin-right:12px;font-size:12px;color:%s;cursor:pointer">'
        '<input type="checkbox" class="symruncb" data-run="%s" checked> %s</label>'
        % (r['color'], html.escape(r['name'], quote=True), html.escape(r['name']))
        for r in runs)
    return """
<h2>Per-symbol compare</h2>
<p class="note">Stats from the full position logs · one chart panel per run · drag to zoom.</p>
<p>
  <select id="symsel" style="background:#1b2536;color:#e7edf5;border:1px solid #243048;
   border-radius:6px;padding:6px 10px;font-size:14px">%s</select>%s
</p>
<p id="symrunpicker">%s</p>
<div id="symstatstbl"></div>
<div id="symcmp"></div>
<script>
const SYMDATA = %s;
const SYMSTATS = %s;
const SYMRUNS = %s;
const SYMMETRICS = [['Realised','pnl',2],['Unrealised','unrealised',2],['Total','total',2],
                    ['Positions','trades',0],['Win %%','win_rate',1],
                    ['Sharpe','sharpe',3],['Max DD','max_dd',2]];
function symStatsTable(sym) {
  const shown = new Set([...document.querySelectorAll('.symruncb:checked')].map(el => el.dataset.run));
  const runs = SYMRUNS.filter(r => shown.has(r.name));
  const rows = runs.map(r => (SYMSTATS[r.name] || {})[sym] || null);
  if (!rows.some(Boolean)) { document.getElementById('symstatstbl').innerHTML = ''; return; }
  const fmtc = (v, dp) => v == null ? '\\u2014' : Number(v).toFixed(dp);
  const head = '<tr><th></th>' + runs.map(r => '<th>' + r.name + '</th>').join('') + '</tr>';
  const body = SYMMETRICS.map(([label, key, dp]) => {
    const vals = rows.map(r => r ? r[key] : null);
    const nums = vals.filter(v => typeof v === 'number');
    let bestV = null, worstV = null;
    if (nums.length > 1 && new Set(nums).size > 1) { bestV = Math.max(...nums); worstV = Math.min(...nums); }
    const cells = vals.map(v => {
      const cls = v == null ? '' : v === bestV ? 'pos' : v === worstV ? 'neg' : '';
      return '<td class="' + cls + '">' + fmtc(v, dp) + '</td>';
    }).join('');
    return '<tr><th class="rowh">' + label + '</th>' + cells + '</tr>';
  }).join('');
  document.getElementById('symstatstbl').innerHTML =
    '<table><thead>' + head + '</thead><tbody>' + body + '</tbody></table>';
}
function drawSym(sym) {
  symStatsTable(sym);
  const showAddl = document.getElementById('symaddl').checked;
  const shown = new Set([...document.querySelectorAll('.symruncb:checked')].map(el => el.dataset.run));
  const traces = [], names = [];
  SYMRUNS.filter(run => shown.has(run.name)).forEach(run => {
    const d = (SYMDATA[run.name] || {})[sym];
    if (!d) return;
    names.push(run.name);
    const i = names.length;
    const xa = i === 1 ? 'x' : 'x' + i, ya = i === 1 ? 'y' : 'y' + i;
    const T = a => a.map(t => new Date(t * 1000));
    traces.push({x: T(d.price.t), y: d.price.v, name: 'price', xaxis: xa, yaxis: ya,
                 line: {color: '#8a97ad', width: 1}, showlegend: i === 1});
    d.lines.forEach(L => {
      if (L.additional && !showAddl) return;
      traces.push({x: T(L.t), y: L.v, name: L.additional ? L.name + ' (additional)' : L.name,
                   xaxis: xa, yaxis: ya, showlegend: i === 1,
                   line: {color: L.color, width: 1, dash: L.additional ? 'dot' : 'solid'}});
    });
  });
  const n = Math.max(names.length, 1);
  const ann = names.map((t, i) => ({text: t, showarrow: false, xref: 'paper',
    x: 0, xanchor: 'left', yref: 'paper', y: 1 - i / n, yanchor: 'bottom',
    font: {size: 12, color: '#8a97ad'}}));
  Plotly.react('symcmp', traces, {
    grid: {rows: n, columns: 1, pattern: 'independent'},
    height: Math.max(300, 270 * n), annotations: ann,
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(22,29,43,0.6)',
    font: {color: '#e7edf5', size: 12}, hovermode: 'x',
    margin: {l: 60, r: 20, t: 30, b: 30},
    legend: {orientation: 'h', y: 1.05},
  }, {responsive: true, scrollZoom: true});
}
document.getElementById('symsel').onchange = e => drawSym(e.target.value);
document.getElementById('symaddl').onchange = () => drawSym(document.getElementById('symsel').value);
document.querySelectorAll('.symruncb').forEach(cb =>
  cb.onchange = () => drawSym(document.getElementById('symsel').value));
drawSym(document.getElementById('symsel').value);
</script>""" % (options, addl_ctl, run_checks, payload, statspayload, meta)


_CLASS_METRICS = [('Realised', 'pnl', 2), ('Unrealised', 'unrealised', 2),
                  ('Total', 'total', 2), ('Positions', 'trades', 0),
                  ('Win %', 'win_rate', 1), ('Sharpe', 'sharpe', 3),
                  ('Max DD', 'max_dd', 2)]


def _class_compare_html(runs):
    """Per-class P&L compare (job runs only — needs position logs): a class
    dropdown drives one table, same metrics/coloring as the per-symbol
    contribution tables in the dashboard, with every symbol of that class
    pooled into one trade-level line."""
    runs = [r for r in runs if r.get('classdata')]
    if not runs:
        return ''
    classes = sorted(set().union(*[set(r['classdata']) for r in runs]))
    if not classes:
        return ''
    payload = json.dumps({r['name']: r['classdata'] for r in runs})
    names = json.dumps([r['name'] for r in runs])
    metrics = json.dumps(_CLASS_METRICS)
    options = ''.join('<option>%s</option>' % html.escape(c) for c in classes)
    return """
<h2>By class</h2>
<p class="note">Every symbol of a sector/category (Layer 1, DeFi, Meme, ...) pooled into one
trade-level line, so win rate/Sharpe/max DD are computed correctly rather than averaged
across the class's symbols. Job runs only (needs position logs).</p>
<p>
  <select id="classsel" style="background:#1b2536;color:#e7edf5;border:1px solid #243048;
   border-radius:6px;padding:6px 10px;font-size:14px">%s</select>
</p>
<div id="classcmp"></div>
<script>
const CLASSDATA = %s;
const CLASSRUNS = %s;
const CLASSMETRICS = %s;
function fmtc(v, dp) { return v == null ? '\\u2014' : Number(v).toFixed(dp); }
function drawClass(cls) {
  const rows = CLASSRUNS.map(name => (CLASSDATA[name] || {})[cls] || null);
  const symsUnion = [...new Set(rows.filter(Boolean).flatMap(r => r.symbols))].sort();
  let head = '<tr><th></th>' + CLASSRUNS.map(n => '<th>' + n + '</th>').join('') + '</tr>';
  let body = CLASSMETRICS.map(([label, key, dp]) => {
    const vals = rows.map(r => r ? r[key] : null);
    const nums = vals.filter(v => typeof v === 'number');
    let bestV = null, worstV = null;
    if (nums.length > 1 && new Set(nums).size > 1) {
      bestV = Math.max(...nums); worstV = Math.min(...nums);
    }
    const cells = vals.map(v => {
      const cls2 = v == null ? '' : v === bestV ? 'pos' : v === worstV ? 'neg' : '';
      return '<td class="' + cls2 + '">' + fmtc(v, dp) + '</td>';
    }).join('');
    return '<tr><th class="rowh">' + label + '</th>' + cells + '</tr>';
  }).join('');
  document.getElementById('classcmp').innerHTML =
    '<p class="note">' + symsUnion.length + ' symbols: ' + symsUnion.join(', ') + '</p>' +
    '<table><thead>' + head + '</thead><tbody>' + body + '</tbody></table>';
}
document.getElementById('classsel').onchange = e => drawClass(e.target.value);
drawClass(document.getElementById('classsel').value);
</script>""" % (options, payload, names, metrics)


def div(fig, first=False):
    return pio.to_html(fig, full_html=False,
                       include_plotlyjs=('inline' if first else False),
                       config={'displayModeBar': True, 'responsive': True, 'scrollZoom': True})


def build(symbol=SYMBOL_PREF, selections=None, timeframe='1m', days=None):
    import run_backtest
    timeframe = timeframe if timeframe in ('1m', '1h') else '1m'
    _, _, comp = run_backtest._tf_cfg(timeframe)
    if symbol:                                  # single-symbol: full internals
        sym, path = _ensure_symbol_csv(symbol, timeframe, days)
        feeds, subtitle = [(sym, path)], sym
    else:                                       # no symbol -> whole portfolio
        feeds = _ensure_all_csvs(timeframe, days)
        if not feeds:
            raise ValueError('no %s data available' % timeframe)
        subtitle = 'all symbols (%d)' % len(feeds)
    print('[report] %s  timeframe: %s' % (subtitle, timeframe))
    results = []
    for i, sel in enumerate(selections or DEFAULT_SELECTION):
        name = sel.get('name')
        if name not in run_backtest.STRATEGIES:
            print('[report] skipping unknown strategy %r' % name)
            continue
        cls, params, _days = run_backtest._resolve_run_config(
            name, sel.get('params'), None)
        color = PALETTE[i % len(PALETTE)]
        print('[report] running %s ...' % name)
        res = run_strategy(name, cls, params, feeds, color, comp)
        res['params'] = dict(params, timeframe=timeframe)   # for the disclaimer
        results.append(res)
    if not results:
        raise ValueError('no valid strategies selected')
    print('[report] building interactive figures ...')
    _render_page(results, subtitle=subtitle)


def _render_page(results, subtitle):
    with_diag = [r for r in results if r.get('diag') is not None]
    ref = with_diag[0]['diag'] if with_diag else results[0]['equity']
    span = '%s → %s' % (ref.index[0].strftime('%Y-%m-%d'),
                        ref.index[-1].strftime('%Y-%m-%d'))

    # parameters as a proper comparison table (param rows x run columns);
    # cells where runs differ are highlighted
    _skip = {'window_start', 'window_end'}
    keys = sorted({k for r in results for k in (r.get('params') or {})} - _skip)
    if keys:
        head = ''.join('<th>%s</th>' % r['name'] for r in results)
        body = ''
        for k in keys:
            vals = [(r.get('params') or {}).get(k) for r in results]
            differ = len({json.dumps(v, default=str) for v in vals}) > 1
            cells = ''.join(
                '<td%s>%s</td>' % (' style="color:#e3b341;font-weight:600"' if differ else '',
                                   '&mdash;' if v is None else v)
                for v in vals)
            body += '<tr><th class="rowh">%s</th>%s</tr>' % (k, cells)
        cfg_lines = ('<h2>Parameters</h2>'
                     '<table><thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table>'
                     '<p class="note">highlighted values differ between runs</p>'
                     % (head, body))
    else:
        cfg_lines = ''

    # timeframe disclaimer (from whichever runs recorded one)
    tfs = sorted({str((r.get('params') or {}).get('timeframe', '1m')) for r in results})
    tf_note = '<p class="note">Bars: %s.</p>' % ', '.join(tfs)

    # (label, html) — label is used for the jump-to-section index; sections
    # without a real heading (disclaimers, the params table's own <h2> is
    # inline in cfg_lines) pass '' and are skipped by the index.
    parts = [
        ('Summary', summary_table(results)),
        ('Portfolio stats', stats_table(results)),   # full compare (archived runs)
        ('', tf_note),
        ('Parameters', cfg_lines),
        ('', '<p class="note">Hover for values · drag to zoom · double-click to reset.</p>'),
        ('Win rate & PnL', '<h2>Win rate &amp; PnL</h2>' + div(fig_winpnl(results), first=True)),
        ('Equity curve', '<h2>Equity curve</h2>' + div(fig_equity(results))),
        ('By class', _class_compare_html(results)),   # jobs with position logs only; '' otherwise
        ('Per-symbol compare', _sym_compare_html(results)),   # jobs w/ chartdata only; '' otherwise
    ]
    if with_diag:
        parts += [
            ('Prediction & confidence bands',
             '<h2>Prediction &amp; confidence bands</h2>'
             '<p class="note">Drag-select to zoom into the prediction line and band.</p>'
             + div(fig_pred(with_diag))),
            ('Innovation & uncertainty',
             '<h2>Innovation &amp; its uncertainty</h2>'
             '<p class="note">Prediction error, and the filter\'s expected error (√S).</p>'
             + div(fig_innov(with_diag))),
            ('Covariance matrix P',
             '<h2>Covariance matrix P</h2>'
             '<p class="note">Post-warmup steady state · autoscale to see the initial transient.</p>'),
        ]
        for res in with_diag:
            parts.append(('', '<h3>%s</h3>' % res['name'] + div(fig_P(res))))

    names = ' vs '.join(r['name'] for r in results)
    # each logical block gets its own bordered card (and, if labeled, an
    # anchor id for the jump-to-section index) so sections read as distinct
    # panels instead of one continuous scroll
    sections, toc = [], []
    for i, (label, p) in enumerate(parts):
        if not p:
            continue
        sec_id = 'sec-%d' % i
        sections.append('<section class="rpt-sec" id="%s">%s</section>' % (sec_id, p))
        if label:
            toc.append('<a href="#%s">%s</a>' % (sec_id, html.escape(label)))
    body = '\n'.join(sections)
    toc_html = '<nav class="rpt-toc"><b>Jump to:</b> ' + ' &nbsp;·&nbsp; '.join(toc) + '</nav>' if toc else ''
    page_html = TEMPLATE.format(symbol=subtitle, span=span, names=names,
                                bars=len(ref), body=body, toc=toc_html)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'index.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(page_html)
    print('[report] wrote %s (%.1f MB)' % (out, os.path.getsize(out) / 1e6))


SYM_DS = 300   # points per series in the interactive symbol compare


MAX_SYMDATA_SYMBOLS = 40   # per run, in the per-symbol compare panel


def _job_symdata(chart_dir, per_symbol=None, max_symbols=MAX_SYMDATA_SYMBOLS):
    paths = sorted(glob.glob(os.path.join(chart_dir, '*.json')))
    if len(paths) > max_symbols:
        if per_symbol:
            def _rank(p):
                s = per_symbol.get(os.path.basename(p)[:-5]) or {}
                return abs(s.get('pnl') or 0) + (s.get('trades') or 0)
            paths = sorted(paths, key=_rank, reverse=True)[:max_symbols]
        else:
            paths = paths[:max_symbols]
    out = {}
    for p in paths:
        sym = os.path.basename(p)[:-5]
        try:
            with open(p, encoding='utf-8') as f:
                cd = json.load(f)
        except (OSError, ValueError):
            continue
        candles = cd.get('candles') or []
        if len(candles) < 2:
            continue
        st = max(1, len(candles) // SYM_DS)
        entry = {'price': {'t': [c['time'] for c in candles[::st]],
                           'v': [c['close'] for c in candles[::st]]},
                 'lines': []}
        for L in (cd.get('lines') or []):
            pts = L.get('points') or []
            if len(pts) < 2:
                continue
            st = max(1, len(pts) // SYM_DS)
            entry['lines'].append({'name': L.get('name', ''),
                                   'color': L.get('color', '#58a6ff'),
                                   'additional': L.get('kind') == 'additional',
                                   't': [q['time'] for q in pts[::st]],
                                   'v': [q['value'] for q in pts[::st]]})
        out[sym] = entry
    return out


def _read_unrealised(pdir):
    """{symbol: unrealised_mtm} from the run's unrealised.json sidecar (one
    level up from the positions/ dir). Empty when absent."""
    path = os.path.join(os.path.dirname(pdir.rstrip(r'\/')), 'unrealised.json')
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def _job_symstats(pdir):
    """{symbol: {trades, pnl, unrealised, total, win_rate, sharpe, max_dd}}
    for one job, from its positions/<SYMBOL>.csv logs (+ unrealised.json for
    open-position MTM) — same trade-level math the dashboard's per-symbol
    contribution table uses. `total` = realised pnl + unrealised."""
    if not os.path.isdir(pdir):
        return None
    by_sym = pnl_stats.read_position_pnls(pdir)
    unreal = _read_unrealised(pdir)
    if not by_sym and not unreal:
        return None
    out = {}
    for sym in set(by_sym) | set(unreal):
        st = pnl_stats.pnl_stats(by_sym.get(sym, []))
        st['unrealised'] = round(float(unreal.get(sym, 0.0)), 4)
        st['total'] = round((st.get('pnl') or 0.0) + st['unrealised'], 4)
        out[sym] = st
    return out


def _job_classdata(pdir):
    """{class: {trades, pnl, unrealised, total, win_rate, sharpe, max_dd,
    symbols}} for one job, from its positions/<SYMBOL>.csv logs (+
    unrealised.json), pooling symbols by sector tag."""
    if not os.path.isdir(pdir):
        return None
    by_sym = pnl_stats.read_position_pnls(pdir)
    unreal = _read_unrealised(pdir)
    if not by_sym and not unreal:
        return None
    by_class = {}
    class_syms = {}
    class_unreal = {}
    for sym in set(by_sym) | set(unreal):
        cls = symbol_classes.classify(sym)
        by_class.setdefault(cls, []).extend(by_sym.get(sym, []))
        class_syms.setdefault(cls, []).append(sym)
        class_unreal[cls] = class_unreal.get(cls, 0.0) + float(unreal.get(sym, 0.0))
    out = {}
    for cls, pnls in by_class.items():
        st = dict(symbols=sorted(class_syms[cls]), **pnl_stats.pnl_stats(pnls))
        st['unrealised'] = round(class_unreal.get(cls, 0.0), 4)
        st['total'] = round((st.get('pnl') or 0.0) + st['unrealised'], 4)
        out[cls] = st
    return out


def build_from_folders(tags):
    import run_backtest
    import _paths
    tdir = run_backtest.TEST_DATA_DIR
    jdir = os.path.join(_paths.ROOT, 'reports', 'jobs')
    results = []
    for i, tag in enumerate(tags or []):
        if str(tag).startswith('job:'):
            jid = tag[4:]
            snap = None
            try:
                with open(os.path.join(jdir, jid, 'results.json'), encoding='utf-8') as f:
                    snap = json.load(f)
            except (OSError, ValueError):
                print('[report] job %r has no results.json — skipped' % jid)
                continue
            symdata = _job_symdata(os.path.join(jdir, jid, 'chartdata'),
                                   per_symbol=snap.get('per_symbol'))
            label_tag = jid
            classdata = _job_classdata(os.path.join(jdir, jid, 'positions'))
            symstats = _job_symstats(os.path.join(jdir, jid, 'positions'))
        else:
            snap_path = os.path.join(tdir, tag, 'run.json')
            if not os.path.exists(snap_path):
                print('[report] archived run %r has no run.json — skipped' % tag)
                continue
            with open(snap_path, encoding='utf-8') as f:
                snap = json.load(f)
            symdata = None
            label_tag = tag
            classdata = None   # archived runs don't keep per-symbol position logs
            symstats = None
        p, s = snap.get('params', {}), snap.get('summary', {})
        eq = pd.DataFrame(snap.get('equity', []), columns=['dt', 'value'])
        if eq.empty:
            continue
        eq['dt'] = pd.to_datetime(eq['dt']); eq = eq.set_index('dt')
        stats = dict(end_value=s.get('end_value'), pnl=s.get('pnl'),
                     trades=s.get('trades_closed'), trades_total=s.get('trades_total'),
                     wins=s.get('won'),
                     win_rate=s.get('win_rate_pct'), innov_rms=None)
        # prefer the run's friendly name (defaults to the run tag/job id
        # itself when none was given, so this is never blank)
        label = '%s · %s' % (p.get('strategy', '?'), snap.get('name') or label_tag)
        results.append(dict(name=label, diag=None, equity=eq, stats=stats,
                            summary=s,                # full stats comparison
                            symdata=symdata,          # job chartdata (or None)
                            classdata=classdata,      # per-class pnl stats (or None)
                            symstats=symstats,        # per-symbol pnl stats (or None)
                            color=PALETTE[i % len(PALETTE)], params=p))
    if not results:
        raise ValueError('no runs with usable snapshots selected')
    print('[report] comparing %d run(s) ...' % len(results))
    _render_page(results, subtitle='saved backtests')


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{names} · {symbol}</title>
<style>
  :root {{ --bg:#0f1420; --ink:#e7edf5; --muted:#8a97ad; --line:#243048; --accent:#37c1d1;
           --panel:#151c2c; --pos:#3fb950; --neg:#f85149; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; line-height:1.5; }}
  .wrap {{ max-width:1120px; margin:0 auto; padding:40px 24px 80px; }}
  header p {{ color:var(--muted); margin:.2em 0; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.85rem; }}
  h1 {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:1.7rem; letter-spacing:-.02em; margin:0 0 4px; }}
  h2 {{ font-size:1.05rem; margin:0 0 6px; padding-bottom:6px; border-bottom:1px solid var(--line); }}
  h3 {{ font-size:.92rem; color:var(--muted); margin:22px 0 4px; font-weight:600; }}
  table {{ border-collapse:collapse; margin:18px 0 6px; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.9rem; }}
  th,td {{ padding:8px 16px; text-align:right; border-bottom:1px solid var(--line); }}
  thead th {{ color:var(--accent); font-weight:600; }}
  .rowh {{ text-align:left; color:var(--ink); }}
  tbody td {{ font-variant-numeric:tabular-nums; }}
  .note {{ color:var(--muted); font-size:.9rem; max-width:76ch; margin:6px 0 10px; }}
  .note b {{ color:var(--ink); font-weight:600; }}
  .plotly-graph-div {{ margin:0 0 6px; }}
  .pos {{ color:var(--pos); font-weight:600; }}
  .neg {{ color:var(--neg); font-weight:600; }}
  .rpt-sec {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
             padding:20px 24px; margin:18px 0; }}
  .rpt-sec:empty {{ display:none; }}
  .lookup-bar {{ position:sticky; top:0; z-index:10; background:var(--bg);
                padding:14px 0 10px; margin-bottom:4px; }}
  .lookup-bar input {{ width:100%; max-width:420px; background:var(--panel);
                       color:var(--ink); border:1px solid var(--line); border-radius:8px;
                       padding:9px 14px; font-size:.92rem; }}
  .lookup-bar input:focus {{ outline:1px solid var(--accent); }}
  .lookup-hint {{ color:var(--muted); font-size:.8rem; margin:6px 0 0; }}
  .rpt-sec.rpt-dim {{ opacity:.25; }}
  .rpt-toc {{ font-size:.85rem; color:var(--muted); margin:0 0 14px; line-height:1.9; }}
  .rpt-toc a {{ color:var(--accent); text-decoration:none; }}
  .rpt-toc a:hover {{ text-decoration:underline; }}
</style></head>
<body><div class="wrap">
<header>
  <h1>{names}</h1>
  <p>symbol {symbol} · {bars} bars · {span}</p>
</header>
<div class="lookup-bar">
  <input type="search" id="rptlookup" placeholder="Search this report (section titles, params, symbols)…" autocomplete="off">
  <p class="lookup-hint" id="rptlookuphint"></p>
</div>
{toc}
{body}
<script>
(function() {{
  const input = document.getElementById('rptlookup');
  const hint = document.getElementById('rptlookuphint');
  const secs = Array.from(document.querySelectorAll('.rpt-sec'));
  function run() {{
    const q = input.value.trim().toLowerCase();
    if (!q) {{ secs.forEach(s => s.classList.remove('rpt-dim')); hint.textContent = ''; return; }}
    let hitSecs = 0, firstHit = null;
    secs.forEach(sec => {{
      const heading = sec.querySelector('h2,h3');
      const text = sec.textContent.toLowerCase();
      const match = text.includes(q);
      sec.classList.toggle('rpt-dim', !match);
      if (match) {{
        hitSecs++;
        if (heading && heading.textContent.toLowerCase().includes(q) && !firstHit) firstHit = sec;
        if (!firstHit) firstHit = sec;
      }}
    }});
    hint.textContent = hitSecs ? `${{hitSecs}} matching section(s)` : 'no matches';
    if (firstHit) firstHit.scrollIntoView({{behavior: 'smooth', block: 'start'}});
  }}
  let t;
  input.addEventListener('input', () => {{ clearTimeout(t); t = setTimeout(run, 150); }});
}})();
</script>
</div></body></html>"""


if __name__ == '__main__':
    build()
