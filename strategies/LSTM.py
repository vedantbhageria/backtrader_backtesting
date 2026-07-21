from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import sys
from collections import deque
from functools import partial

import backtrader as bt
import numpy as np
import jax
import jax.numpy as jnp

from strategy_base import PortfolioStrategy


# ---- hand-rolled single-layer LSTM cell, in plain JAX ops --------------
# A fused primitive (like torch.nn.LSTM/LSTMCell) has no vmap batching rule,
# so every symbol's online-training step would have to run as its own
# sequential Python/dispatch call — that per-call overhead (not the actual
# math) was the original bottleneck (torch, sequential: ~234 symbol-steps/sec,
# 359.6s for a real 30-day/100-symbol backtest). Built from plain
# matmul/sigmoid/tanh, this DOES vmap+jit cleanly, so all symbols' training
# steps run as ONE fused, batched XLA call per bar, each still with its own
# independent weights — see next()/_batch_online_step below. Verified on the
# same real 30-day/100-symbol backtest: 148.3s, a 2.4x wall-clock speedup
# (isolated micro-benchmarks of just the JIT'd call suggested far more, but
# per-bar Python/array-construction overhead — rebuilding each symbol's
# rolling window from a deque every bar, plus per-symbol bookkeeping —
# dominates the rest; that part doesn't disappear just because the model
# math got faster).

def _init_cell_params(key, hidden_size):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        'W_ih': jax.random.normal(k1, (4 * hidden_size, 1)) * 0.1,
        'W_hh': jax.random.normal(k2, (4 * hidden_size, hidden_size)) * 0.1,
        'b': jnp.zeros(4 * hidden_size),
        'W_fc': jax.random.normal(k3, (1, hidden_size)) * 0.1,
        'b_fc': jnp.zeros(1),
    }


def _forward_single(params, x, hidden_size):
    """x: (seq_len, 1) normalized prices -> scalar normalized prediction."""
    h0 = jnp.zeros(hidden_size)
    c0 = jnp.zeros(hidden_size)

    def step(carry, xt):
        h, c = carry
        gates = params['W_ih'] @ xt + params['W_hh'] @ h + params['b']
        i, f, g, o = jnp.split(gates, 4)
        i, f, o = jax.nn.sigmoid(i), jax.nn.sigmoid(f), jax.nn.sigmoid(o)
        g = jnp.tanh(g)
        c = f * c + i * g
        h = o * jnp.tanh(c)
        return (h, c), None

    (h, _), _ = jax.lax.scan(step, (h0, c0), x)
    return (params['W_fc'] @ h + params['b_fc'])[0]


def _loss_single(params, x, y, hidden_size):
    pred = _forward_single(params, x, hidden_size)
    return (pred - y) ** 2


def _bias_correct(x, b, step):
    """x / (1 - b**step), with `step` either a scalar (single-symbol
    pretraining) or a (B,) per-symbol array (batched online step — each
    symbol has its own step count since they become ready at different
    bars). In the (B,) case, reshape to (B,1,1,...) so it broadcasts against
    x's leading batch dim regardless of that leaf's own shape (params has
    leaves of several different ranks: W_ih, b, b_fc, ...)."""
    denom = 1 - b ** step
    if jnp.ndim(step) > 0:
        denom = denom.reshape(denom.shape + (1,) * (x.ndim - denom.ndim))
    return x / denom


def _adam_step(params, grads, m, v, step, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Bias-corrected Adam, applied as plain pytree tensor ops — same math
    torch.optim.Adam uses. `step` (1-indexed) drives bias correction; works
    identically whether params/m/v carry a leading per-symbol batch
    dimension (vmapped online step) or not (single-symbol pretraining)."""
    m = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1 - b2) * g ** 2, v, grads)
    mhat = jax.tree.map(lambda m: _bias_correct(m, b1, step), m)
    vhat = jax.tree.map(lambda v: _bias_correct(v, b2, step), v)
    params = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
                          params, mhat, vhat)
    return params, m, v


def _zeros_like_params(params):
    return jax.tree.map(jnp.zeros_like, params)


class LSTMTest(PortfolioStrategy):
    """Drop-in replacement for KalmanTest: same warmup -> predict -> innovation-band
    signal shape, but the point estimate + adaptation come from an online-finetuned
    LSTM instead of a linear-Gaussian Kalman filter.

    Every symbol trains its OWN independent LSTM (own weights, own Adam state) —
    same as before — but the per-bar online-training step for every symbol that's
    past warmup is batched into a single jit+vmap'd JAX call in next(), instead of
    each symbol's step running as its own sequential Python/torch call. Warmup
    pretraining (_init_model, once per symbol) stays a per-symbol JAX call — it's
    a much smaller share of total runtime and happens at different bars per symbol
    (late-listed symbols warm up later), so batching it isn't worth the complexity.
    """

    params = (
        ('k', 2.0),             # threshold in std devs of the innovation
        ('a', 7.0),             # kept for parity with KalmanTest (unused here)
        ('warmup', 60),         # bars collected before the initial fit
        ('seq_len', 20),        # lookback window fed to the LSTM each step
        ('hidden_size', 16),
        ('num_layers', 1),      # kept for parity with the old torch version
                                 # (unused — the hand-rolled cell is single-layer)
        ('lr', 1e-3),
        ('online_steps', 1),    # gradient steps taken per bar on the newest sample
        ('vol_window', 50),     # rolling window used to estimate innovation std (S)
        ('init_epochs', 150),   # epochs used to pretrain on the warmup window
        ('reversion', True),    # True: fade the deviation (default — matches
                                 # every other strategy's convention); False:
                                 # follow it (breakout)
        ('seed', 42),
        # ---- position management (ported from EMNonLinTest, each a toggle) --
        ('long_only', False),   # True: no short ENTRIES — an opposite (short)
                                 # signal flattens an open long to cash instead
                                 # of reversing into a short.
        ('max_hold', 0),        # time stop (bars). 0 = off. Flatten any position
                                 # that has been open >= max_hold bars; the clock
                                 # resets on every entry/flip (a reversal is a
                                 # new trade).
        ('dead', 0),            # settling period (bars). 0 = off. Suppress trade
                                 # ENTRIES for the first `dead` bars after this
                                 # symbol goes live, so the freshly-pretrained
                                 # model adapts online a bit before it's trusted.
                                 # (max_hold flattens still apply during it.)
        ('exit_band', 0.0),     # dead-zone exit. 0 = off. When |innovation| <
                                 # exit_band * band (price has reverted close to
                                 # the model's prediction, so the edge is gone),
                                 # exit to CASH instead of holding. This is the
                                 # main lever against "always in the market": at
                                 # 0 the strategy never voluntarily flattens and
                                 # only ever flips long<->short.
    )

    def setup(self):
        self._key = jax.random.PRNGKey(self.params.seed)
        self.models = {}
        self._chart = {}
        self._bar_cache = {}   # d -> (innov, S, y_pred, band), filled once per bar
        hs = self.params.hidden_size
        seq_len = self.params.seq_len

        # jit-compiled ONCE per run and reused for every symbol/every bar —
        # shapes must never change after the first call, or XLA silently
        # recompiles from scratch (this was the actual cause of a run taking
        # LONGER than the original torch version: the online-step batch used
        # to be sized to however many symbols were "ready" that bar, which
        # grows over the warmup period — every new size retraced the whole
        # LSTM+scan graph, dozens of times over a run).
        self._fwd_single = jax.jit(partial(_forward_single, hidden_size=hs))
        self._grad_batch = jax.jit(
            jax.vmap(jax.grad(partial(_loss_single, hidden_size=hs)),
                     in_axes=(0, 0, 0)))
        self._fwd_batch = jax.jit(
            jax.vmap(partial(_forward_single, hidden_size=hs), in_axes=(0, 0)))
        n_windows = self.params.warmup - seq_len   # constant: every symbol's
                                                    # pretrain window count
        if n_windows > 0:
            self._fwd_pretrain = jax.jit(jax.vmap(self._fwd_single, in_axes=(None, 0)))
            self._grad_pretrain = jax.jit(jax.grad(
                lambda p, X, Y: jnp.mean((self._fwd_pretrain(p, X) - Y) ** 2)))

        # Fixed batch size for the ENTIRE run = every symbol, always — the
        # online step below runs this every bar regardless of which symbols
        # are actually past warmup yet. Rows are fully independent models
        # (own weights, own gradients — vmap never mixes them), so "training"
        # a not-yet-ready symbol's placeholder row is harmless: it gets
        # completely overwritten the moment that symbol's real _init_model
        # runs. This is what keeps the batch shape — and therefore the JIT
        # cache — constant for the whole backtest.
        #
        # The model state (params/m/v) is kept as ONE PERSISTENT stacked JAX
        # pytree (self._P/_M/_V, leading dim = symbol slot) for the ENTIRE
        # run, rather than 100 separate per-symbol pytrees that get
        # re-stacked and unstacked every bar. That re-stack/unstack (up to
        # ~500 tiny eager JAX ops per bar: 5 leaves x up to 100 symbols) was
        # the actual remaining bottleneck — it grows as more symbols pass
        # warmup, which is exactly the slowdown-over-time this was chasing.
        # Everything else per symbol (warmup buffer, normalization stats,
        # last prediction, innovation history) is cheap Python/numpy and
        # stays in self.models[d] as before.
        self._all_ds = list(self.datas)
        self._slot = {d: i for i, d in enumerate(self._all_ds)}
        n = len(self._all_ds)
        init_keys = jax.random.split(self._key, n + 1)
        self._key = init_keys[0]
        self._P = jax.vmap(lambda k: _init_cell_params(k, hs))(init_keys[1:])
        self._M = _zeros_like_params(self._P)
        self._V = _zeros_like_params(self._P)
        self._adam_steps = np.zeros(n, dtype=np.int64)

        for d in self._all_ds:
            self.models[d] = {
                'warm': [],
                'ready': False,
                'mu': 0.0, 'sigma': 1.0,                    # placeholder until _init_model
                'buf': deque([0.0] * seq_len, maxlen=seq_len),   # pre-seeded: always full
                'innovs': deque(maxlen=self.params.vol_window),
                'last_pred': 0.0,
                # position management bookkeeping (see on_bar)
                'pos_sign': 0,        # -1/0/+1 sign of the position held last bar
                'bars_in_pos': 0,     # bars the current position has been open
                'bars_since_ready': 0,  # bars since this symbol went live (for `dead`)
            }
            self._chart[d._name] = []

    def _normalize(self, st, price):
        return (price - st['mu']) / st['sigma']

    def _denormalize(self, st, z):
        return z * st['sigma'] + st['mu']

    def _init_model(self, d, st):
        """One-time per-symbol pretrain on the warmup window (sequential — not
        the hot path). Same overlapping-windows fit the torch version did.
        Reuses self._fwd_pretrain/_grad_pretrain (jit-compiled ONCE in setup,
        same window-count shape for every symbol) — writes the result into
        this symbol's SLOT of the persistent stacked self._P/_M/_V (replacing
        the random init that slot was seeded with in setup), rather than
        keeping a separate per-symbol pytree."""
        i = self._slot[d]
        buf = np.array(st['warm'], dtype=float)
        st['mu'] = float(buf.mean())
        st['sigma'] = float(buf.std())
        if st['sigma'] < 1e-8:
            st['sigma'] = 1.0

        norm = (buf - st['mu']) / st['sigma']
        seq_len = self.params.seq_len

        self._key, subkey = jax.random.split(self._key)
        p = _init_cell_params(subkey, self.params.hidden_size)
        m, v = _zeros_like_params(p), _zeros_like_params(p)
        step = 0

        if len(norm) > seq_len:
            X, Y = [], []
            for j in range(len(norm) - seq_len):
                X.append(norm[j:j + seq_len])
                Y.append(norm[j + seq_len])
            X = jnp.asarray(np.array(X), dtype=jnp.float32)[..., None]   # (n, seq_len, 1)
            Y = jnp.asarray(np.array(Y), dtype=jnp.float32)              # (n,)

            for _ in range(self.params.init_epochs):
                step += 1
                g = self._grad_pretrain(p, X, Y)
                p, m, v = _adam_step(p, g, m, v, step, self.params.lr)

        self._P = jax.tree.map(lambda whole, new: whole.at[i].set(new), self._P, p)
        self._M = jax.tree.map(lambda whole, new: whole.at[i].set(new), self._M, m)
        self._V = jax.tree.map(lambda whole, new: whole.at[i].set(new), self._V, v)
        self._adam_steps[i] = step

        for val in norm[-seq_len:]:
            st['buf'].append(float(val))
        st['ready'] = True

        seq = jnp.asarray(np.array(st['buf']), dtype=jnp.float32)[:, None]
        st['last_pred'] = float(self._fwd_single(p, seq))

    def _batch_online_step(self):
        """Once per bar: batch the online-training + next-bar-prediction step
        across EVERY symbol (self._all_ds — fixed set, fixed order, fixed
        count for the whole run), not just the ones currently past warmup.

        This always runs at the SAME batch size, which is the whole point:
        each row is an independent model (own weights, own gradients — vmap
        never mixes rows), so "training" a not-yet-ready symbol's still-zero
        placeholder row is harmless — it gets fully overwritten by that
        symbol's own _init_model() the moment it's actually ready. Results
        are only PERSISTED into each symbol's Python-side bookkeeping (and
        only enter self._bar_cache, which on_bar reads) for symbols that are
        ready AND have a fresh bar this tick; everything else's output this
        bar is simply discarded. Keeping the batch shape constant is what
        lets XLA compile the graph exactly ONCE for the entire backtest
        instead of retracing every time the ready-count changes.

        self._P/_M/_V are kept as ONE PERSISTENT stacked pytree across the
        whole run (never re-stacked from / unstacked into 100 separate
        per-symbol pytrees) — that stack/unstack, growing every bar as more
        symbols pass warmup, was the actual remaining bottleneck."""
        ds = self._all_ds
        sts = [self.models[d] for d in ds]
        prices = [d.close[0] if len(d) else 0.0 for d in ds]
        zs = [self._normalize(st, p) for st, p in zip(sts, prices)]

        # 1) innovation from the prediction made LAST bar (before this bar's update)
        y_preds = [self._denormalize(st, st['last_pred']) for st in sts]
        innovs = [p - yp for p, yp in zip(prices, y_preds)]

        # 2) batched online-training step, directly on the persistent stacked state
        X = jnp.stack([jnp.asarray(np.array(st['buf']), dtype=jnp.float32)[:, None]
                       for st in sts])                       # (N, seq_len, 1)
        Y = jnp.asarray(zs, dtype=jnp.float32)                # (N,)
        for i in range(self.params.online_steps):
            self._adam_steps = self._adam_steps + 1
            steps = jnp.asarray(self._adam_steps)
            G = self._grad_batch(self._P, X, Y)
            self._P, self._M, self._V = _adam_step(
                self._P, G, self._M, self._V, steps, self.params.lr)

        # 3) roll each symbol's window forward, batched prediction for next bar
        for st, z in zip(sts, zs):
            st['buf'].append(float(z))
        Xnext = jnp.stack([jnp.asarray(np.array(st['buf']), dtype=jnp.float32)[:, None]
                           for st in sts])
        preds = np.asarray(self._fwd_batch(self._P, Xnext))   # ONE host sync for all N

        # 4) persist — ONLY for symbols that are ready and have a fresh bar
        # this tick; a not-yet-ready symbol's placeholder result is simply
        # discarded here (its real state got set by _init_model instead)
        for i, (d, st) in enumerate(zip(ds, sts)):
            if not st['ready']:
                continue
            if len(d) == 0 or len(d) == self._last_len.get(d, 0):
                continue    # no fresh bar for this symbol this tick

            st['last_pred'] = float(preds[i])

            innov, y_pred, price = innovs[i], y_preds[i], prices[i]
            st['innovs'].append(innov)
            if len(st['innovs']) >= 5:
                S = float(np.var(st['innovs']))
            else:
                S = (0.01 * price) ** 2
            band = self.params.k * (S ** 0.5)
            self._chart[d._name].append((self.bar_epoch(d), round(y_pred, 8),
                                         round(y_pred + band, 8),
                                         round(y_pred - band, 8)))
            self._diag_record(d, price, y_pred, innov, S, band, st)
            self._bar_cache[d] = (innov, S, y_pred, band)

    def next(self):
        self._batch_online_step()
        super().next()

    def prenext(self):
        self.next()

    def on_bar(self, d, price):
        st = self.models[d]

        # --- max_hold clock: how long the CURRENT position has been open.
        # Resets whenever the position's sign changes (flat->open, open->flat,
        # or a flip — a reversal is a new trade). Once a position is "overdue"
        # (open >= max_hold bars), EVERY exit path below flattens to cash
        # instead of holding. Runs on every bar, even during warmup, so a
        # position opened just before a re-warm still ages correctly. ---
        overdue = False
        pos = self.getposition(d).size
        sign = 1 if pos > 0 else (-1 if pos < 0 else 0)
        if sign != st['pos_sign']:
            st['bars_in_pos'] = 0
        st['pos_sign'] = sign
        if sign != 0:
            st['bars_in_pos'] += 1
            if self.params.max_hold and st['bars_in_pos'] >= self.params.max_hold:
                overdue = True

        # Warmup: collect prices until we can fit the initial model. Kept
        # sequential/per-symbol — pretraining happens once, at whatever bar
        # THIS symbol individually reaches `warmup` (late-listed symbols
        # reach it later than the rest), so there's no shared batch to build.
        if not st['ready']:
            st['warm'].append(price)
            if len(st['warm']) >= self.params.warmup:
                self._init_model(d, st)
            return (0, 0) if overdue else 0

        cached = self._bar_cache.pop(d, None)
        if cached is None:
            return (0, 0) if overdue else 0   # buffer wasn't full yet this bar
        innov, S, y_pred, band = cached
        if band <= 0:
            return (0, 0) if overdue else 0

        # `dead` settling period: for the first `dead` bars after this symbol
        # went live, suppress trade ENTRIES (the freshly-pretrained model is
        # still adapting online). Position management (max_hold) still applies.
        st['bars_since_ready'] += 1
        if st['bars_since_ready'] <= self.params.dead:
            return (0, 0) if overdue else 0

        if overdue:
            return (0, 0)

        # exit_band dead-zone: |innovation| has fallen back INSIDE
        # exit_band * band, i.e. price reverted close to the model's
        # prediction — the edge is gone, so exit to CASH rather than keep
        # holding. Off (0) by default, in which case the strategy never
        # voluntarily flattens and only ever flips long<->short (the original
        # "always in the market" behavior).
        if self.params.exit_band and abs(innov) < self.params.exit_band * band:
            if pos != 0:
                return (0, 0)
            return 0

        # innov = price - prediction. FOLLOW (reversion=False): trade in the
        # direction of the surprise (price broke ABOVE the band -> LONG).
        # REVERSION (reversion=True, default): fade it. Same convention as
        # every other strategy in this project — keep it consistent.
        long_sig = innov > band      # price above the upper band
        short_sig = innov < -band    # price below the lower band

        if self.params.reversion:
            long_sig, short_sig = short_sig, long_sig

        if long_sig:
            if pos > 0:
                return 0
            self.log('%s breakout ABOVE +band (innov %.6f, band %.6f) -> LONG'
                     % (d._name, innov, band))
            return 1
        if short_sig:
            if self.params.long_only:
                # no short ENTRIES — but an opposite signal still flattens an
                # open long (otherwise a long could only exit via exit_band/
                # max_hold/trailing-stop)
                if pos > 0:
                    self.log('%s short signal -> flatten long (long_only)' % d._name)
                    return (0, 0)
                return 0
            if pos < 0:
                return 0
            self.log('%s breakdown BELOW -band (innov %.6f, band %.6f) -> SHORT'
                     % (d._name, innov, band))
            return -1

        return 0

    def build_chart_lines(self):
        k = self.params.k
        out = {}
        for d in self.datas:
            rows = self._chart[d._name]
            out[d._name] = [
                {'name': 'LSTM', 'color': '#58a6ff',
                 'points': [{'time': t, 'value': e} for t, e, u, l in rows]},
                {'name': '+%g sigma' % k, 'color': '#8b949e',
                 'points': [{'time': t, 'value': u} for t, e, u, l in rows]},
                {'name': '-%g sigma' % k, 'color': '#8b949e',
                 'points': [{'time': t, 'value': l} for t, e, u, l in rows]},
            ]
        return out

    def stop(self):
        self.log('(k=%g warmup=%d seq_len=%d hidden=%d) Ending Value %.2f'
                 % (self.params.k, self.params.warmup, self.params.seq_len,
                    self.params.hidden_size, self.broker.getvalue()), doprint=True)


if __name__ == '__main__':
    # Quick single-symbol run; use run_backtest.py for the full portfolio.
    cerebro = bt.Cerebro()
    cerebro.addstrategy(LSTMTest, printlog=True)

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
