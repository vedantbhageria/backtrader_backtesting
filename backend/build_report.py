"""Run the KF / EKF / EM-Kalman single-symbol comparison and bake an INTERACTIVE
Plotly dashboard into one self-contained static HTML (plotly.js inlined — hover,
zoom, pan; no server logic, no CDN, no sockets). Serve the output folder:

    python build_report.py
    python -m http.server 8095 -d report_out
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import backtrader as bt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import _paths
DATA_DIR = os.path.join(_paths.ROOT, 'datas')
OUT_DIR = os.path.join(_paths.ROOT, 'report_out')
SYMBOL_PREF = 'ETHUSDT'
START_CASH, COMMISSION, LEVERAGE = 100_000.0, 0.0002, 10.0
DS = 7000    # downsample the series to ~this many points (SVG Scatter)

# Strategy classes + tuned defaults come from run_backtest.STRATEGIES (the
# same registry the dashboard's run picker uses), so the report accepts ANY
# registered strategy with per-strategy param overrides.
DEFAULT_SELECTION = [
    {'name': 'KalmanTest'},
    {'name': 'ExtendedKalmanTest'},
    {'name': 'EMTest'},
]
PALETTE = ['#4c9be8', '#e8834c', '#5cc98a', '#b58cf0', '#e5566a', '#4fc6c0']

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
    happened to leave in datas/. `days` limits the window (None = everything).
    Falls back to an existing CSV when the DB is unavailable."""
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
    """Export EVERY symbol's history for `timeframe` from postgres (for the
    all-symbols portfolio report). `days` limits the window (None = all).
    Falls back to whatever CSVs exist when the DB is unavailable. Returns
    [(sym, path), ...]."""
    import run_backtest
    data_dir, suffix, _ = run_backtest._tf_cfg(timeframe)
    os.makedirs(data_dir, exist_ok=True)
    existing = lambda: [(os.path.basename(p)[:-len('-%s.csv' % suffix)], p)
                        for p in sorted(glob.glob(os.path.join(data_dir, '*-%s.csv' % suffix)))]
    try:
        import db
        conn = db.get_conn()
    except Exception as e:
        print('[report] DB unavailable (%s) — using existing %s CSVs' % (e, suffix))
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
    """feeds: [(sym, path), ...]. A single feed keeps the filter internals
    (diag) for the prediction/innovation/P charts; multiple feeds run the whole
    portfolio and report equity/PnL only (diag=None)."""
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
    stats = dict(end_value=round(end, 2), pnl=round(end - START_CASH, 2),
                 trades=len(tl), wins=wins,
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


def summary_table(results):
    cols = ['end_value', 'pnl', 'trades', 'wins', 'win_rate', 'innov_rms']
    head = ''.join('<th>%s</th>' % c for c in cols)
    body = ''
    for res in results:
        s = res['stats']
        body += '<tr><th class="rowh">%s</th>%s</tr>' % (
            res['name'],
            ''.join('<td>%s</td>' % ('&mdash;' if s[c] is None else s[c])
                    for c in cols))
    return '<table><thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table>' % (head, body)


# (metric label, summary key, formatter) — mirrors the dashboard Account cards
_PSTAT_ROWS = [
    ('End value (USDT)',        'end_value',                    lambda v: '%.2f' % v),
    ('PnL (USDT)',              'pnl',                          lambda v: '%+.2f' % v),
    ('Return %',                'return_pct',                   lambda v: '%+.3f%%' % v),
    ('Trades',                  'trades_closed',                lambda v: '%d' % v),
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
            try:
                cells += '<td>%s</td>' % ('&mdash;' if v is None or v != v else fmt(v))
            except (TypeError, ValueError):
                cells += '<td>%s</td>' % v
        body += '<tr><th class="rowh">%s</th>%s</tr>' % (label, cells)
    return ('<h2>Portfolio stats</h2>'
            '<table><thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table>'
            % (head, body))


def _sym_compare_html(runs):
    """Interactive per-symbol compare for runs that carry job chartdata: a
    symbol dropdown drives one subplot row per run (price + every stored
    indicator/prediction line, downsampled). Pure client-side — the report
    stays a single self-contained file."""
    runs = [r for r in runs if r.get('symdata')]
    if not runs:
        return ''
    syms = sorted(set().union(*[set(r['symdata']) for r in runs]))
    if not syms:
        return ''
    payload = json.dumps({r['name']: r['symdata'] for r in runs})
    meta = json.dumps([{'name': r['name'], 'color': r['color']} for r in runs])
    options = ''.join('<option>%s</option>' % s for s in syms)
    return """
<h2>Per-symbol compare</h2>
<p class="note">Pick a symbol — one panel per run: price plus every stored
indicator/prediction line (downsampled to ~%d points; drag to zoom).</p>
<p><select id="symsel" style="background:#1b2536;color:#e7edf5;border:1px solid #243048;
   border-radius:6px;padding:6px 10px;font-size:14px">%s</select></p>
<div id="symcmp"></div>
<script>
const SYMDATA = %s;
const SYMRUNS = %s;
function drawSym(sym) {
  const traces = [], names = [];
  SYMRUNS.forEach(run => {
    const d = (SYMDATA[run.name] || {})[sym];
    if (!d) return;
    names.push(run.name);
    const i = names.length;
    const xa = i === 1 ? 'x' : 'x' + i, ya = i === 1 ? 'y' : 'y' + i;
    const T = a => a.map(t => new Date(t * 1000));
    traces.push({x: T(d.price.t), y: d.price.v, name: 'price', xaxis: xa, yaxis: ya,
                 line: {color: '#8a97ad', width: 1}, showlegend: i === 1});
    d.lines.forEach(L => traces.push({x: T(L.t), y: L.v, name: L.name, xaxis: xa,
                 yaxis: ya, line: {color: L.color, width: 1}, showlegend: i === 1}));
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
drawSym(document.getElementById('symsel').value);
</script>""" % (SYM_DS, options, payload, meta)


def div(fig, first=False):
    return pio.to_html(fig, full_html=False,
                       include_plotlyjs=('inline' if first else False),
                       config={'displayModeBar': True, 'responsive': True, 'scrollZoom': True})


def build(symbol=SYMBOL_PREF, selections=None, timeframe='1m', days=None):
    """selections: [{'name': <registry name>, 'params': {...overrides}}, ...]
    (None -> DEFAULT_SELECTION). Classes and tuned defaults resolve through
    run_backtest.STRATEGIES; unknown params are dropped / type-coerced by the
    same _resolve_run_config the run API uses. `timeframe` ('1m'/'1h') selects
    which bars the report runs on; `days` limits the window (None = all)."""
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
    """Assemble + write the report from a results list. Strategies with filter
    internals (diag not None) get the prediction/innovation/P charts; ones
    without (archived-run comparisons) show only summary/PnL/equity."""
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
    tf_note = ('<p class="note">📊 <b>Bars:</b> %s. All PnL, win-rate and risk '
               'figures are computed on these bars.</p>' % ', '.join(tfs))

    parts = [
        summary_table(results),
        stats_table(results),   # full portfolio-stats compare (archived runs)
        tf_note,
        cfg_lines,
        '<p class="note">Interactive: <b>hover</b> for values, <b>drag</b> to zoom, '
        'double-click to reset. Series are downsampled to ~%d points for the browser.</p>' % DS,
        '<h2>Win rate &amp; PnL</h2>' + div(fig_winpnl(results), first=True),
        '<h2>Equity curve</h2>' + div(fig_equity(results)),
        _sym_compare_html(results),   # jobs with chartdata only; '' otherwise
    ]
    if with_diag:
        parts += [
            '<h2>Prediction &amp; confidence bands</h2>'
            '<p class="note">Price &amp; prediction overlap at full zoom (the filter tracks '
            'price) — drag-select a small region to see the prediction line and '
            '±kσ band separate.</p>'
            + div(fig_pred(with_diag)),
            '<h2>Innovation &amp; its uncertainty</h2>'
            '<p class="note"><b>RMS</b> = √mean(innov²), the typical prediction '
            'error (price units). Lower panel is √S, the std-dev the filter expects '
            'each bar.</p>'
            + div(fig_innov(with_diag)),
            '<h2>Covariance matrix P</h2>'
            '<p class="note">Default view is the post-warmup <b>steady state</b> (P converges '
            'to a fixed covariance via the Riccati recursion; EM variants jump when they '
            'refit Q, R). Hover reads the exact value; autoscale (modebar) to see the '
            'initial transient. Diagonal = variances (≥0); off-diagonal covariances '
            'may be negative.</p>',
        ]
        for res in with_diag:
            parts.append('<h3>%s</h3>' % res['name'] + div(fig_P(res)))

    names = ' vs '.join(r['name'] for r in results)
    html = TEMPLATE.format(symbol=subtitle, span=span, names=names,
                           bars=len(ref), body='\n'.join(parts))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'index.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print('[report] wrote %s (%.1f MB)' % (out, os.path.getsize(out) / 1e6))


SYM_DS = 300   # points per series in the interactive symbol compare


def _job_symdata(chart_dir):
    """Downsampled per-symbol series from a job's chartdata:
    {sym: {price: {t, v}, lines: [{name, color, t, v}]}}. Every series keeps
    its own time axis (lines start after warmup / have re-warm gaps)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(chart_dir, '*.json'))):
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
            if L.get('kind') == 'additional':
                continue                # diag series live in the dashboard view
            pts = L.get('points') or []
            if len(pts) < 2:
                continue
            st = max(1, len(pts) // SYM_DS)
            entry['lines'].append({'name': L.get('name', ''),
                                   'color': L.get('color', '#58a6ff'),
                                   't': [q['time'] for q in pts[::st]],
                                   'v': [q['value'] for q in pts[::st]]})
        out[sym] = entry
    return out


def build_from_folders(tags):
    """Compare past backtests. Tags may be archive folders
    (reports/test_data/<tag>/run.json — equity/PnL only) or JOBS
    ('job:<id>', reports/jobs/<id>/ — comprehensive: their chartdata feeds an
    interactive per-symbol prediction compare). Anything a run didn't store is
    simply omitted for that run."""
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
            symdata = _job_symdata(os.path.join(jdir, jid, 'chartdata'))
            label_tag = jid
        else:
            snap_path = os.path.join(tdir, tag, 'run.json')
            if not os.path.exists(snap_path):
                print('[report] archived run %r has no run.json — skipped' % tag)
                continue
            with open(snap_path, encoding='utf-8') as f:
                snap = json.load(f)
            symdata = None
            label_tag = tag
        p, s = snap.get('params', {}), snap.get('summary', {})
        eq = pd.DataFrame(snap.get('equity', []), columns=['dt', 'value'])
        if eq.empty:
            continue
        eq['dt'] = pd.to_datetime(eq['dt']); eq = eq.set_index('dt')
        stats = dict(end_value=s.get('end_value'), pnl=s.get('pnl'),
                     trades=s.get('trades_closed'), wins=s.get('won'),
                     win_rate=s.get('win_rate_pct'), innov_rms=None)
        label = '%s · %s' % (p.get('strategy', '?'), label_tag)
        results.append(dict(name=label, diag=None, equity=eq, stats=stats,
                            summary=s,                # full stats comparison
                            symdata=symdata,          # job chartdata (or None)
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
  :root {{ --bg:#0f1420; --ink:#e7edf5; --muted:#8a97ad; --line:#243048; --accent:#37c1d1; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; line-height:1.5; }}
  .wrap {{ max-width:1120px; margin:0 auto; padding:40px 24px 80px; }}
  header p {{ color:var(--muted); margin:.2em 0; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.85rem; }}
  h1 {{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:1.7rem; letter-spacing:-.02em; margin:0 0 4px; }}
  h2 {{ font-size:1.05rem; margin:40px 0 6px; padding-bottom:6px; border-bottom:1px solid var(--line); }}
  h3 {{ font-size:.92rem; color:var(--muted); margin:22px 0 4px; font-weight:600; }}
  table {{ border-collapse:collapse; margin:18px 0 6px; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.9rem; }}
  th,td {{ padding:8px 16px; text-align:right; border-bottom:1px solid var(--line); }}
  thead th {{ color:var(--accent); font-weight:600; }}
  .rowh {{ text-align:left; color:var(--ink); }}
  tbody td {{ font-variant-numeric:tabular-nums; }}
  .note {{ color:var(--muted); font-size:.9rem; max-width:76ch; margin:6px 0 10px; }}
  .note b {{ color:var(--ink); font-weight:600; }}
  .plotly-graph-div {{ margin:0 0 6px; }}
</style></head>
<body><div class="wrap">
<header>
  <h1>{names}</h1>
  <p>symbol {symbol} · {bars} bars · {span}</p>
</header>
{body}
</div></body></html>"""


if __name__ == '__main__':
    build()
