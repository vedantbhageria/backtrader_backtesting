"""Run the KF / EKF / EM-Kalman single-symbol comparison and bake an INTERACTIVE
Plotly dashboard into one self-contained static HTML (plotly.js inlined — hover,
zoom, pan; no server logic, no CDN, no sockets). Serve the output folder:

    python build_report.py
    python -m http.server 8095 -d report_out
"""
import glob
import os

import numpy as np
import pandas as pd
import backtrader as bt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from KalmanFilter import KalmanTest
from ExtendedKalmanFilter import ExtendedKalmanTest
from EMAlgoTest import EMTest

DATA_DIR = 'datas'
OUT_DIR = 'report_out'
SYMBOL_PREF = 'ETHUSDT'
START_CASH, COMMISSION, LEVERAGE = 100_000.0, 0.0002, 10.0
DS = 7000    # downsample the series to ~this many points (SVG Scatter)

# label, class, params, colour
STRATS = [
    ('Kalman', KalmanTest,
     dict(k=2, warmup=180, q_level=0.1e-3, q_vel=0.1e-6), '#4c9be8'),
    ('Extended Kalman', ExtendedKalmanTest,
     dict(k=2, warmup=180, q_level=0.1e-3, q_vel=0.1e-6,
          k_exit=0.0, min_hold=1, cost_mult=0.0, drag_m=0.001,
          trend_bias=False, c_d_window=180), '#e8834c'),
    ('EM-Kalman', EMTest,
     dict(k=2.5, warmup=180, q_level=0.1e-3, q_vel=0.1e-6,
          em_window=720, em_interval=2880, em_iters=2), '#5cc98a'),
]

LABELS = {'P00': 'Var(position)', 'P11': 'Var(velocity)', 'P22': 'Var(accel)',
          'P01': 'Cov(pos, vel)', 'P02': 'Cov(pos, accel)', 'P12': 'Cov(vel, accel)'}


def pick_symbol(prefer):
    paths = sorted(glob.glob(os.path.join(DATA_DIR, '*-1m.csv')))
    hit = [p for p in paths if os.path.basename(p).startswith(prefer + '-1m')]
    p = hit[0] if hit else paths[0]
    return os.path.basename(p)[:-len('-1m.csv')], p


def run_strategy(label, Strat, params, path, sym, color):
    cer = bt.Cerebro()
    cer.broker.setcash(START_CASH)
    cer.broker.setcommission(commission=COMMISSION, leverage=LEVERAGE)
    cer.adddata(bt.feeds.GenericCSVData(
        dataname=path, dtformat='%Y-%m-%d %H:%M:%S',
        timeframe=bt.TimeFrame.Minutes, compression=1,
        datetime=0, open=1, high=2, low=3, close=4, volume=5,
        openinterest=-1, name=sym))
    cer.addstrategy(Strat, **dict(params, diag=True))
    strat = cer.run()[0]
    end = cer.broker.getvalue()
    diag = pd.DataFrame(strat._diag[sym])
    diag['dt'] = pd.to_datetime(diag['t'], unit='s', utc=True)
    diag = diag.set_index('dt')
    eq = pd.DataFrame(strat.equity, columns=['dt', 'value'])
    eq['dt'] = pd.to_datetime(eq['dt']); eq = eq.set_index('dt')
    tl = strat.trade_log
    wins = sum(1 for t in tl if t['pnlcomm'] > 0)
    innov = diag['innov'].to_numpy(float)
    stats = dict(end_value=round(end, 2), pnl=round(end - START_CASH, 2),
                 trades=len(tl), wins=wins,
                 win_rate=round(100.0 * wins / len(tl), 2) if tl else 0.0,
                 innov_rms=round(float(np.sqrt(np.mean(innov ** 2))), 4))
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
            res['name'], ''.join('<td>%s</td>' % s[c] for c in cols))
    return '<table><thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table>' % (head, body)


def div(fig, first=False):
    return pio.to_html(fig, full_html=False,
                       include_plotlyjs=('inline' if first else False),
                       config={'displayModeBar': True, 'responsive': True, 'scrollZoom': True})


def build(symbol=SYMBOL_PREF):
    sym, path = pick_symbol(symbol)
    print('[report] symbol:', sym)
    results = []
    for label, Strat, params, color in STRATS:
        print('[report] running %s ...' % label)
        results.append(run_strategy(label, Strat, params, path, sym, color))
    print('[report] building interactive figures ...')

    span = '%s → %s' % (results[0]['diag'].index[0].strftime('%Y-%m-%d'),
                        results[0]['diag'].index[-1].strftime('%Y-%m-%d'))
    parts = [
        summary_table(results),
        '<p class="note">Interactive: <b>hover</b> for values, <b>drag</b> to zoom, '
        'double-click to reset. Series are downsampled to ~%d points for the browser.</p>' % DS,
        '<h2>Win rate &amp; PnL</h2>' + div(fig_winpnl(results), first=True),
        '<h2>Equity curve</h2>' + div(fig_equity(results)),
        '<h2>Prediction &amp; confidence bands</h2>'
        '<p class="note">Price &amp; prediction overlap at full zoom (the filter tracks price) — '
        'drag-select a small region to see the prediction line and ±kσ band separate.</p>'
        + div(fig_pred(results)),
        '<h2>Innovation &amp; its uncertainty</h2>'
        '<p class="note"><b>RMS</b> = √mean(innov²), the typical prediction error (price units). '
        'Lower panel is √S, the std-dev the filter expects each bar.</p>'
        + div(fig_innov(results)),
        '<h2>Covariance matrix P</h2>'
        '<p class="note">Default view is the post-warmup <b>steady state</b> (P converges to a '
        'fixed covariance via the Riccati recursion; EM-Kalman\'s jumps when it refits Q, R). '
        'Hover reads the exact value; autoscale (modebar) to see the initial transient. '
        'Diagonal = variances (≥0); off-diagonal covariances may be negative.</p>',
    ]
    for res in results:
        parts.append('<h3>%s</h3>' % res['name'] + div(fig_P(res)))

    html = TEMPLATE.format(symbol=sym, span=span, bars=len(results[0]['diag']),
                           body='\n'.join(parts))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'index.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print('[report] wrote %s (%.1f MB)' % (out, os.path.getsize(out) / 1e6))


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KF vs EKF vs EM · {symbol}</title>
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
  <h1>Kalman vs Extended Kalman vs EM-Kalman</h1>
  <p>symbol {symbol} · {bars} bars · {span}</p>
</header>
{body}
</div></body></html>"""


if __name__ == '__main__':
    build()
