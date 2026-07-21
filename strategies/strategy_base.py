
from __future__ import (absolute_import, division, print_function,unicode_literals)

from datetime import timezone

import backtrader as bt


def epoch(dt):
    """Naive-UTC datetime -> integer epoch seconds (lightweight-charts time)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


class PortfolioStrategy(bt.Strategy):

    params = (
        ('trade_usd', 2000.0),
        ('printlog', False),
        ('diag', False),      # record per-bar Kalman internals into self._diag
        # ---- trailing exit (opt-in) -----------------------------------
        # A ratcheting percent stop, independent of on_bar()'s own signal:
        # long  -> stop sits trail_pct BELOW price, only ever moves UP.
        # short -> stop sits trail_pct ABOVE price, only ever moves DOWN.
        # Price moving favorably drags the stop along; price moving back
        # against the position leaves the stop frozen at its best level
        # until price recovers back through that frozen level, at which
        # point it resumes trailing.
        ('use_trail', False),
        ('trail_pct', 0.05),
    )

    def setup(self):
        pass

    def on_bar(self, d, price):

        return 0

    def build_chart_lines(self):
        """{symbol: [{'name', 'color', 'points': [{'time', 'value'}]}]}"""
        return {}


    def log(self, txt, dt=None, doprint=False):
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.datetime(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.orders = {}       # data -> pending order
        self._last_len = {}    # data -> last seen bar count
        # Recorded for the dashboard / exports.
        self.executed = {}     # symbol -> [{dt, side, price, size}]
        self.trade_log = []    # one entry per CLOSED position (open->close)
        self.equity = []       # [(iso_dt, account_value)]
        self._open_trades = {}  # trade.ref -> opening info (set on justopened)
        self._order_sig = {}   # order.ref -> signal-bar datetime (when decided)
        self._fill_sig = {}    # data -> signal dt of the fill being processed now
        self._trail_exit_price = {}   # data -> stop price of an in-flight trail-exit close
        self._diag = {}        # symbol -> [per-bar Kalman internals] (when p.diag)
        self._trail_stop = {}  # data -> current ratcheted trailing-stop price

        for d in self.datas:
            self._last_len[d] = 0
            self.executed[d._name] = []

        self.setup()

    def _diag_record(self, d, price, y_pred, innov, S, band, st):
        """Record one bar of internals for offline analysis (notebook).
        No-op unless the `diag` param is on, so live/dashboard runs are
        unaffected. Always captures prediction + bands, innovation and its
        variance S. Kalman/EM-family strategies additionally carry a state
        vector X and covariance P (set inside their _step) — captured as
        x0../P00.. when present. Non-Kalman strategies (e.g. LSTM) have no
        X/P concept, so those columns are simply omitted rather than raising."""
        if not self.params.diag:
            return
        rec = {'t': self.bar_epoch(d), 'price': price, 'pred': y_pred,
               'upper': y_pred + band, 'lower': y_pred - band,
               'innov': innov, 'S': S, 'band': band}
        X, P = st.get('X'), st.get('P')    # a-posteriori (set inside _step), if any
        if X is not None and P is not None:
            n = X.shape[0]
            for i in range(n):
                rec['x%d' % i] = float(X[i, 0])
            for i in range(n):
                for j in range(i, n):          # upper triangle (P is symmetric)
                    rec['P%d%d' % (i, j)] = float(P[i, j])
        self._diag.setdefault(d._name, []).append(rec)

    def bar_epoch(self, d):
        """Epoch seconds of the current bar on feed `d` (for chart points)."""
        return epoch(d.datetime.datetime(0))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        d = order.data
        if order.status == order.Completed:
            side = 'BUY' if order.isbuy() else 'SELL'
            # signal_dt = the bar the decision was made on (one bar before the
            # fill). notify_order fires just before notify_trade for this fill,
            # so stash it for the trade log to read.
            sig_dt = self._order_sig.pop(order.ref, None)
            self._fill_sig[d] = sig_dt
            self.executed[d._name].append({
                'signal_dt': sig_dt,                                  # decided
                'dt': bt.num2date(order.executed.dt).isoformat(),     # filled
                'side': side,
                'price': order.executed.price,
                'size': order.executed.size,
            })
            self.log('%s %s EXECUTED, Price: %.6f, Size: %.6f' %
                     (d._name, side, order.executed.price, order.executed.size))
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('%s Order Canceled/Margin/Rejected' % d._name)
            self._trail_exit_price.pop(d, None)   # that close never happened
        self.orders.pop(d, None)

    def notify_trade(self, trade):

        if trade.justopened:
            # Capture the opening side/price/size while they're still known
            # (at close, trade.size is 0).
            self._open_trades[trade.ref] = {
                'entry_signal_dt': self._fill_sig.get(trade.data),
                'entry_dt': bt.num2date(trade.dtopen).isoformat(),
                'entry_price': trade.price,
                'size': trade.size,            # signed: >0 long, <0 short
            }
        if trade.isclosed:
            o = self._open_trades.pop(trade.ref, {})
            size = o.get('size', trade.size) or trade.size
            entry = o.get('entry_price', trade.price)
            # exit price backed out of gross pnl: pnl = (exit - entry) * size
            exit_price = (entry + trade.pnl / size) if size else None
            trail_price = self._trail_exit_price.pop(trade.data, None)
            self.trade_log.append({
                'symbol': trade.data._name,
                'side': 'LONG' if size > 0 else 'SHORT',
                'size': abs(size),
                'entry_signal_dt': o.get('entry_signal_dt'),
                'entry_dt': o.get('entry_dt'),
                'exit_signal_dt': self._fill_sig.get(trade.data),
                'exit_dt': bt.num2date(trade.dtclose).isoformat(),
                'entry_price': round(entry, 8) if entry is not None else None,
                'exit_price': round(exit_price, 8) if exit_price is not None else None,
                'bars_held': trade.barlen,
                'pnl': round(trade.pnl, 6),          # gross
                'pnlcomm': round(trade.pnlcomm, 6),  # net of commission
                'exit_reason': 'trail' if trail_price is not None else 'signal',
                'trail_stop_price': round(trail_price, 8) if trail_price is not None else None,
            })
            self.log('%s %s CLOSED, size %.6f, GROSS %.2f, NET %.2f' %
                     (trade.data._name, 'LONG' if size > 0 else 'SHORT',
                      abs(size), trade.pnl, trade.pnlcomm))

    def prenext(self):
        # Backtrader delays next() until EVERY data has delivered its first
        # bar, so one late-listed symbol (e.g. GWEIUSDT, listed months into
        # the window) would idle the WHOLE portfolio until its listing date.
        # next() already guards per-data readiness (len(d) / _last_len), so
        # simply run the same logic during the pre-period.
        self.next()

    def next(self):
        ref = next((d for d in self.datas if len(d)), None)
        if ref is None:
            return
        self.equity.append((
            ref.datetime.datetime(0).isoformat(),
            round(self.broker.getvalue(), 4),
        ))

        for d in self.datas:

            if len(d) == 0 or len(d) == self._last_len[d]:
                continue
            self._last_len[d] = len(d)

            price = d.close[0]
            if price <= 0:
                continue

            # Always advance the model, even if an order is in flight.
            result = self.on_bar(d, price)

            if d in self.orders:      # pending order on this instrument -> wait
                continue

            # Normalize on_bar's return: scalar signal, (signal, size), or
            # (signal, size, stop_pct). stop_pct is a SIGNED percent (whole
            # number, not a fraction — e.g. +2 or -2 means 2%) requesting a
            # trailing exit for the position on THIS bar. The sign picks
            # which side of price the stop sits on: negative -> the
            # protective side (below price for longs, above for shorts);
            # positive -> the opposite side (above for longs, below for
            # shorts). Omit it (2-tuple/scalar) to fall back to the run's
            # use_trail/trail_pct params (protective side, trail_pct is a
            # fraction there, e.g. 0.05 == 5%).
            stop_pct = None
            if isinstance(result, (tuple, list)):
                signal = result[0]
                size = result[1] if len(result) > 1 else None
                if len(result) > 2 and result[2] is not None:
                    stop_pct = float(result[2]) / 100.0
            else:
                signal, size = result, None

            pos_size = self.getposition(d).size
            if pos_size == 0:
                self._trail_stop.pop(d, None)   # flat -> reset for the next trade
            else:
                signed_pct = stop_pct if stop_pct is not None else (
                    -self.params.trail_pct if self.params.use_trail else None)
                if signed_pct:
                    below = (pos_size > 0) != (signed_pct > 0)   # XOR: which side of price
                    pct_abs = abs(signed_pct)
                    cur = self._trail_stop.get(d)
                    if below:                            # stop ratchets UP only, hit on price falling to it
                        candidate = price * (1 - pct_abs)
                        new_stop = candidate if cur is None else max(cur, candidate)
                        hit = price <= new_stop
                    else:                                 # stop ratchets DOWN only, hit on price rising to it
                        candidate = price * (1 + pct_abs)
                        new_stop = candidate if cur is None else min(cur, candidate)
                        hit = price >= new_stop
                    self._trail_stop[d] = new_stop
                    if hit:
                        order = self.order_target_size(data=d, target=0)
                        if order is not None:
                            self.orders[d] = order
                            self._order_sig[order.ref] = d.datetime.datetime(0).isoformat()
                            self._trail_exit_price[d] = new_stop
                        self._trail_stop.pop(d, None)
                        continue   # trailing exit overrides this bar's on_bar signal
                else:
                    self._trail_stop.pop(d, None)

            if signal != signal:      # NaN signal (e.g. indicator not ready)
                continue

            # Explicit flatten: on_bar returns (0, 0) -> go to cash. A bare
            # scalar 0 still means "hold / no change" (backward compatible).
            if signal == 0 and size == 0:
                if self.getposition(d).size != 0:
                    order = self.order_target_size(data=d, target=0)
                    if order is not None:
                        self.orders[d] = order
                        self._order_sig[order.ref] = d.datetime.datetime(0).isoformat()
                continue

            if not signal:
                continue

            magnitude = size if size is not None else (self.params.trade_usd / price)
            target = signal * magnitude
            order = self.order_target_size(data=d, target=target)
            if order is not None:
                self.orders[d] = order
                # remember the bar we DECIDED on (fills next bar) so the log
                # can show signal-time vs fill-time.
                self._order_sig[order.ref] = d.datetime.datetime(0).isoformat()

    def stop(self):
        self.log('%s ending value %.2f'
                 % (type(self).__name__, self.broker.getvalue()), doprint=True)
