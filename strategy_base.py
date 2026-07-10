
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from datetime import timezone

import backtrader as bt


def epoch(dt):
    """Naive-UTC datetime -> integer epoch seconds (lightweight-charts time)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


class PortfolioStrategy(bt.Strategy):

    params = (
        ('trade_usd', 2000.0),
        ('printlog', False),
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
                # remember the bar we DECIDED on (fills next bar) so the log
                # can show signal-time vs fill-time.
                self._order_sig[order.ref] = d.datetime.datetime(0).isoformat()

    def stop(self):
        self.log('%s ending value %.2f'
                 % (type(self).__name__, self.broker.getvalue()), doprint=True)
