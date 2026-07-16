from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import sys

import backtrader as bt

from strategy_base import PortfolioStrategy, epoch


class EMACrossShortTest(PortfolioStrategy):
    """EMA crossover, stop-and-reverse, inverted variant.

    Bearish cross (fast below slow) -> LONG; bullish cross -> SHORT.
    """

    # backtrader merges these with the base class params (trade_usd, printlog).
    params = (
        ('fast_ema_period', 15),
        ('slow_ema_period', 30),
    )

    def setup(self):
        self.fast, self.slow, self.cross = {}, {}, {}
        for d in self.datas:
            self.fast[d] = bt.indicators.ExponentialMovingAverage(
                d, period=self.params.fast_ema_period)
            self.slow[d] = bt.indicators.ExponentialMovingAverage(
                d, period=self.params.slow_ema_period)
            # +1 fast crosses above slow (bull), -1 fast crosses below (bear)
            self.cross[d] = bt.indicators.CrossOver(self.fast[d], self.slow[d])

    def on_bar(self, d, price):
        cross = self.cross[d][0]
        if cross < 0:
            self.log('%s BEAR CROSS -> reverse LONG, %.6f' % (d._name, price))
            return 1        # bearish cross -> go LONG (inverted variant)
        if cross > 0:
            self.log('%s BULL CROSS -> reverse SHORT, %.6f' % (d._name, price))
            return -1       # bullish cross -> go SHORT
        return 0

    def build_chart_lines(self):
        fp, sp = self.params.fast_ema_period, self.params.slow_ema_period
        out = {}
        for d in self.datas:
            times = [epoch(bt.num2date(v)) for v in d.datetime.array]
            fast = list(self.fast[d].array)
            slow = list(self.slow[d].array)
            out[d._name] = [
                {'name': 'EMA %d' % fp, 'color': '#58a6ff',
                 'points': [{'time': t, 'value': round(fast[j], 8)}
                            for j, t in enumerate(times)
                            if j < len(fast) and fast[j] == fast[j]]},  # skip NaN
                {'name': 'EMA %d' % sp, 'color': '#ff9800',
                 'points': [{'time': t, 'value': round(slow[j], 8)}
                            for j, t in enumerate(times)
                            if j < len(slow) and slow[j] == slow[j]]},
            ]
        return out

    def stop(self):
        self.log('(fast %d / slow %d) Ending Value %.2f'
                 % (self.params.fast_ema_period, self.params.slow_ema_period,
                    self.broker.getvalue()), doprint=True)


if __name__ == '__main__':
    # Quick single-symbol run; use run_backtest.py for the full portfolio.
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EMACrossShortTest, printlog=True)

    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    symbol = 'BTCUSDT'
    datapath = os.path.join(modpath, 'datas/%s-1m.csv' % symbol)
    data = bt.feeds.GenericCSVData(
        dataname=datapath, dtformat='%Y-%m-%d %H:%M:%S',
        timeframe=bt.TimeFrame.Minutes, compression=1,
        datetime=0, open=1, high=2, low=3, close=4, volume=5, openinterest=-1,
        name=symbol)
    cerebro.adddata(data)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0, leverage=10.0)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
