"""Shared framework for multi-symbol portfolio strategies.

To add a new strategy: subclass PortfolioStrategy and implement three hooks.

    class MyStrategy(PortfolioStrategy):
        params = (('my_param', 42),)   # backtrader merges base params in

        def setup(self):
            # called once; build per-feed indicators/state (self.datas is ready)
            self.ind = {d: bt.indicators.RSI(d) for d in self.datas}

        def on_bar(self, d, price):
            # called once per new bar per feed (even while an order is pending,
            # so stateful filters stay in sync). Return the desired position:
            #   +1 -> long trade_usd notional,  -1 -> short,  0 -> no change
            # or (signal, size) to set your own size in units:
            #   return (1, 500)  -> long 500 units, ignoring trade_usd
            return 1 if self.ind[d][0] < 30 else (-1 if self.ind[d][0] > 70 else 0)

        def build_chart_lines(self):
            # overlay lines for the dashboard chart, per symbol
            return {d._name: [{'name': 'RSI', 'color': '#58a6ff',
                               'points': [{'time': t, 'value': v}, ...]}]
                    for d in self.datas}

Everything else — order tracking, fills, trade log, equity curve, notional
sizing, stop-and-reverse mechanics — is handled here, and run_backtest.py
consumes it generically. Register the strategy in run_backtest.py's STRATEGY /
STRATEGY_PARAMS config to run it.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from datetime import timezone

import backtrader as bt


def epoch(dt):
    """Naive-UTC datetime -> integer epoch seconds (lightweight-charts time)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


class PortfolioStrategy(bt.Strategy):
    """Multi-feed, single-account base. One position per symbol, sized to
    ``trade_usd`` of notional, stop-and-reverse on signal change."""

    params = (
        ('trade_usd', 2000.0),
        ('printlog', False),
    )

    # ---- hooks for subclasses -----------------------------------------
    def setup(self):
        pass

    def on_bar(self, d, price):
        """Return the desired position for feed `d`. Two forms:

            return signal            # +1 long / -1 short / 0 no-change
                                     # sized to trade_usd notional (default)

            return (signal, size)    # size = position magnitude in UNITS,
                                     # overriding trade_usd. e.g. (1, 500) ->
                                     # long 500 units; (-1, 500) -> short 500.
                                     # size=None falls back to the notional.

        Called on every new bar for every feed (even while an order is
        pending), so stateful models stay in sync. The signal is ignored while
        an order is pending.
        """
        return 0

    def build_chart_lines(self):
        """{symbol: [{'name', 'color', 'points': [{'time', 'value'}]}]}"""
        return {}

    # ---- machinery (shared by every strategy) --------------------------
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

        for d in self.datas:
            self._last_len[d] = 0
            self.executed[d._name] = []

        self.setup()

    def bar_epoch(self, d):
        """Epoch seconds of the current bar on feed `d` (for chart points)."""
        return epoch(d.datetime.datetime(0))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        d = order.data
        if order.status == order.Completed:
            side = 'BUY' if order.isbuy() else 'SELL'
            self.executed[d._name].append({
                'dt': bt.num2date(order.executed.dt).isoformat(),
                'side': side,
                'price': order.executed.price,
                'size': order.executed.size,
            })
            self.log('%s %s EXECUTED, Price: %.6f, Size: %.6f' %
                     (d._name, side, order.executed.price, order.executed.size))
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('%s Order Canceled/Margin/Rejected' % d._name)
        self.orders.pop(d, None)

    def notify_trade(self, trade):
        # A backtrader "trade" IS a position: it opens when you go from flat to
        # a side, and closes when you return to flat. A stop-and-reverse fill
        # therefore closes one position (long) and opens the next (short) — two
        # entries in the log, exactly as expected.
        if trade.justopened:
            # Capture the opening side/price/size while they're still known
            # (at close, trade.size is 0).
            self._open_trades[trade.ref] = {
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
            self.trade_log.append({
                'symbol': trade.data._name,
                'side': 'LONG' if size > 0 else 'SHORT',
                'size': abs(size),
                'entry_dt': o.get('entry_dt'),
                'exit_dt': bt.num2date(trade.dtclose).isoformat(),
                'entry_price': round(entry, 8) if entry is not None else None,
                'exit_price': round(exit_price, 8) if exit_price is not None else None,
                'bars_held': trade.barlen,
                'pnl': round(trade.pnl, 6),          # gross
                'pnlcomm': round(trade.pnlcomm, 6),  # net of commission
            })
            self.log('%s %s CLOSED, size %.6f, GROSS %.2f, NET %.2f' %
                     (trade.data._name, 'LONG' if size > 0 else 'SHORT',
                      abs(size), trade.pnl, trade.pnlcomm))

    def next(self):
        self.equity.append((
            self.datas[0].datetime.datetime(0).isoformat(),
            round(self.broker.getvalue(), 4),
        ))

        for d in self.datas:
            # Only act when this feed delivered a new bar (feeds can start at
            # different times / have gaps).
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

            # on_bar may return either a direction (+1/-1/0) or a
            # (direction, size) pair, where `size` is the position magnitude in
            # units — overriding the default trade_usd notional. `size=None`
            # falls back to the notional. Returning just an int is unchanged.
            if isinstance(result, (tuple, list)):
                signal, size = result
            else:
                signal, size = result, None
            if not signal:
                continue

            magnitude = size if size is not None else (self.params.trade_usd / price)
            target = signal * magnitude
            order = self.order_target_size(data=d, target=target)
            if order is not None:
                self.orders[d] = order

    def stop(self):
        self.log('%s ending value %.2f'
                 % (type(self).__name__, self.broker.getvalue()), doprint=True)
