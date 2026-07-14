"""Extended (nonlinear) EM on the drag-EKF model.

Model (3-state, from ExtendedKalmanFilter.py, with a CONSTANT drag coeff d):

    x = [position, velocity, accel]
    f(x) = F3 @ x - d * [0, 0, v|v| / z]^T      (quadratic drag on accel,
    z_t   = x[0] + noise                          normalised by last price z)

Plain EM needs a closed-form E-step, which only exists for linear-Gaussian
models. The standard workaround ("EKS-EM") linearises:

  E-step:  extended Kalman filter + extended RTS smoother + lag-one
           covariance, all using the per-step Jacobian F_t = df/dx.
  M-step:  expectations of the nonlinear residual (x_t - f(x_{t-1})) are
           taken under a first-order expansion of f around the smoothed
           x_s[t-1].  Given d, the ML updates of Q and R stay closed-form.
           The drag coefficient d has no closed form in general -- BUT the
           expected complete-data log-likelihood is exactly QUADRATIC in d
           (f is linear in d), so the gradient equation dJ/dd = 0 solves in
           one Newton step: d* = -beta / (2*alpha).  Gradient descent would
           crawl to the same point; we jump there directly.  (If the drag
           model ever becomes non-quadratic in its parameters, swap the
           one-step solve in _extended_em_fit for a real GD loop -- the
           gradient is assembled from the same M0/M1/M2 statistics.)

Because of the linearisation, the monotone-likelihood guarantee of exact EM
is LOST: loglik should trend up but may wiggle.  Validated instead by (a)
exact reduction to the linear 3-state EM machinery at d=0, and (b) recovery
of a known drag coefficient from synthetic data.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import sys
from collections import deque

import backtrader as bt
import numpy as np

from strategy_base import PortfolioStrategy

_F3 = np.array([[1.0, 1.0, 0.5],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0]])          # constant-acceleration transition
_H3 = np.array([[1.0, 0.0, 0.0]])          # measure position only

_LOG2PI = float(np.log(2.0 * np.pi))

try:
    from numba import njit
    _NUMBA = True
except ImportError:                     # graceful fallback: no-op decorator
    _NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def _inv3(A):
    """Explicit 3x3 inverse via the adjugate (np.linalg.inv routes tiny
    matrices through LAPACK -- pure overhead at this size)."""
    a, b, c = A[0, 0], A[0, 1], A[0, 2]
    d, e, f = A[1, 0], A[1, 1], A[1, 2]
    g, h, i = A[2, 0], A[2, 1], A[2, 2]
    A11 = e*i - f*h; A12 = c*h - b*i; A13 = b*f - c*e
    A21 = f*g - d*i; A22 = a*i - c*g; A23 = c*d - a*f
    A31 = d*h - e*g; A32 = b*g - a*h; A33 = a*e - b*d
    det = a*A11 + b*A21 + c*A31
    out = np.empty((3, 3))
    out[0, 0] = A11/det; out[0, 1] = A12/det; out[0, 2] = A13/det
    out[1, 0] = A21/det; out[1, 1] = A22/det; out[1, 2] = A23/det
    out[2, 0] = A31/det; out[2, 1] = A32/det; out[2, 2] = A33/det
    return out


@njit(cache=True)
def _ekf_forward(Z, W, x0, P0, Q, r, d):
    """Extended Kalman forward pass.  H = [1,0,0], scalar R = r.
    W[t] = 1/normaliser for step t (the previous price), so the drag term is
    d * v|v| * W[t].  Also returns the per-step Jacobians (the smoother and
    lag-one recursions need them)."""
    T = Z.shape[0]
    X_f = np.empty((T, 3, 1)); P_f = np.empty((T, 3, 3))
    X_p = np.empty((T, 3, 1)); P_p = np.empty((T, 3, 3))
    Fj = np.empty((T, 3, 3))
    x, P = x0.copy(), P0.copy()
    K = np.zeros((3, 1))
    loglik = 0.0
    for t in range(T):
        v = x[1, 0]
        x_pred = _F3 @ x
        x_pred[2, 0] -= d * v * abs(v) * W[t]          # quadratic drag on accel
        F = _F3.copy()
        F[2, 1] -= d * 2.0 * abs(v) * W[t]             # d(drag)/dv
        P_pred = F @ P @ F.T + Q
        S = P_pred[0, 0] + r                           # H P_pred H^T + R
        innov = Z[t] - x_pred[0, 0]
        K = P_pred[:, :1] / S                          # P_pred H^T S^-1
        x = x_pred + K * innov
        P = P_pred - K * P_pred[0, :]                  # (I - K H) P_pred
        P = 0.5 * (P + P.T)
        X_p[t] = x_pred; P_p[t] = P_pred
        X_f[t] = x; P_f[t] = P
        Fj[t] = F
        loglik += -0.5 * (np.log(S) + innov*innov/S + _LOG2PI)
    return X_f, P_f, X_p, P_p, Fj, K, loglik


@njit(cache=True)
def _ekf_smoother(X_f, P_f, X_p, P_p, Fj):
    """Extended RTS backward pass -- the linear recursion with the per-step
    Jacobian standing in for F (Fj[t+1] is the transition INTO t+1)."""
    T = X_f.shape[0]
    X_s = X_f.copy(); P_s = P_f.copy()
    J = np.zeros((T, 3, 3))
    for t in range(T - 2, -1, -1):
        J[t] = P_f[t] @ Fj[t + 1].T @ _inv3(P_p[t + 1])
        X_s[t] = X_f[t] + J[t] @ (X_s[t + 1] - X_p[t + 1])
        Pt = P_f[t] + J[t] @ (P_s[t + 1] - P_p[t + 1]) @ J[t].T
        P_s[t] = 0.5 * (Pt + Pt.T)
    return X_s, P_s, J


@njit(cache=True)
def _ekf_lag(P_f, J, K_last, Fj, P0, P_p0):
    """Lag-one covariance smoother, Jacobian version of the exact linear
    recursion (Fj[t+1] = transition t -> t+1)."""
    T = P_f.shape[0]
    P_lag = np.zeros((T, 3, 3))
    J_prior = P0 @ Fj[0].T @ _inv3(P_p0)
    IKH = np.eye(3)
    IKH[:, 0] -= K_last[:, 0]                          # I - K H, H = [1,0,0]
    if T >= 2:
        P_lag[T - 1] = IKH @ Fj[T - 1] @ P_f[T - 2]
        for t in range(T - 2, 0, -1):
            P_lag[t] = P_f[t] @ J[t - 1].T \
                + J[t] @ (P_lag[t + 1] - Fj[t + 1] @ P_f[t]) @ J[t - 1].T
        P_lag[0] = P_f[0] @ J_prior.T \
            + J[0] @ (P_lag[1] - Fj[1] @ P_f[0]) @ J_prior.T
    else:
        P_lag[0] = IKH @ Fj[0] @ P0
    return P_lag, J_prior


@njit(cache=True)
def _ekf_mstats(Z, W, X_s, P_s, P_lag, x0_s, P0_s):
    """M-step sufficient statistics.  The transition residual, linearised
    around the smoothed x_s[t-1], is affine in the drag coefficient:

        x_t - f_d(x_{t-1})  ~  a_t + d*g_t   with   Ftil(d) = F3 - d*G_t

    so its expected outer-product sum is a QUADRATIC matrix polynomial
    M(d) = M0 + d*M1 + d^2*M2.  Everything the (Q, d) coordinate updates and
    the analytic d-gradient need is in (M0, M1, M2); r_sum feeds R."""
    T = Z.shape[0]
    M0 = np.zeros((3, 3)); M1 = np.zeros((3, 3)); M2 = np.zeros((3, 3))
    r_sum = 0.0
    for t in range(T):
        if t == 0:
            xp, Pp = x0_s, P0_s
        else:
            xp, Pp = X_s[t - 1], P_s[t - 1]
        v = xp[1, 0]
        g = np.zeros((3, 1)); g[2, 0] = v * abs(v) * W[t]      # +d*g in residual
        G = np.zeros((3, 3)); G[2, 1] = 2.0 * abs(v) * W[t]    # Ftil = F3 - d*G
        a = X_s[t] - _F3 @ xp
        Plag = P_lag[t]
        # E[(x_t - f)(x_t - f)^T] = (a + d g)(a + d g)^T + P_s[t]
        #   - Ftil Plag^T - Plag Ftil^T + Ftil Pp Ftil^T
        M0 += a @ a.T + P_s[t] \
            - _F3 @ Plag.T - Plag @ _F3.T + _F3 @ Pp @ _F3.T
        M1 += a @ g.T + g @ a.T \
            + G @ Plag.T + Plag @ G.T - G @ Pp @ _F3.T - _F3 @ Pp @ G.T
        M2 += g @ g.T + G @ Pp @ G.T
        innov = Z[t] - X_s[t][0, 0]
        r_sum += innov * innov + P_s[t][0, 0]
    return M0, M1, M2, r_sum


def _extended_em_fit(prices, x0, P0, Q0, r0, d0, n_iter=5, r_floor=1e-12,
                     d_min=0.0, d_max=None, coord_iters=3):
    """EKS-EM: learn Q (3x3), R (scalar) and the drag coefficient d.

    Per outer iteration: one extended E-step, then coordinate updates on the
    SAME expected complete-data log-likelihood:
      d | Q :  J(d) = tr(Q^-1 (M0 + d M1 + d^2 M2)) is quadratic, so the
               gradient equation  dJ/dd = tr(Q^-1 M1) + 2d tr(Q^-1 M2) = 0
               is solved exactly in one Newton step (see module docstring --
               gradient descent would converge to exactly this point).
      Q | d :  Q = M(d)/T, then eigenvalue-floored.
      R      :  closed form, as in the linear EM.
    Returns (Q, r, d, x_last, P_last, loglik_hist).
    """
    Z = np.ascontiguousarray(prices, dtype=float)
    T = len(Z)
    # W[t] = 1/(price used to normalise the drag at step t) = previous close
    W = np.empty(T)
    W[0] = 1.0 / Z[0]
    W[1:] = 1.0 / Z[:-1]

    x0 = np.ascontiguousarray(x0, dtype=float)
    P0 = np.ascontiguousarray(P0, dtype=float)
    Q = np.ascontiguousarray(Q0, dtype=float)
    r = float(r0)
    d = float(d0)
    loglik_hist = []

    for _ in range(n_iter):
        X_f, P_f, X_p, P_p, Fj, K_last, ll = _ekf_forward(Z, W, x0, P0, Q, r, d)
        if not np.isfinite(ll):
            # diverged (degenerate window / unstable linearisation): bail out
            # with the last finite parameters rather than crashing the refit.
            break
        loglik_hist.append(ll)
        X_s, P_s, J = _ekf_smoother(X_f, P_f, X_p, P_p, Fj)
        P_lag, J_prior = _ekf_lag(P_f, J, K_last, Fj, P0, P_p[0])

        # prior's smoothed posterior (same boundary treatment as linear EM)
        x0_s = x0 + J_prior @ (X_s[0] - X_p[0])
        P0_s = P0 + J_prior @ (P_s[0] - P_p[0]) @ J_prior.T
        P0_s = 0.5 * (P0_s + P0_s.T)

        M0, M1, M2, r_sum = _ekf_mstats(Z, W, X_s, P_s, P_lag, x0_s, P0_s)

        # coordinate ascent on (d, Q); M0/M1/M2 are fixed within this E-step
        for _c in range(coord_iters):
            Qi = np.linalg.inv(Q)
            alpha = float(np.trace(Qi @ M2))          # J(d) = alpha d^2 + beta d + const
            beta = float(np.trace(Qi @ M1))
            if alpha > 0.0:
                d = -beta / (2.0 * alpha)             # exact gradient root
            d = min(max(d, d_min), d_max) if d_max is not None else max(d, d_min)
            Qn = (M0 + d * M1 + d * d * M2) / T
            Qn = 0.5 * (Qn + Qn.T)
            w, v = np.linalg.eigh(Qn)
            Q = np.ascontiguousarray((v * np.clip(w, r_floor, None)) @ v.T)

        r = max(float(r_sum / T), r_floor)

    X_f, P_f, _, _, _, _, ll = _ekf_forward(Z, W, x0, P0, Q, r, d)
    loglik_hist.append(ll)
    return Q, r, d, X_f[-1].copy(), P_f[-1].copy(), loglik_hist


@njit(cache=True)
def _ekf_step_one(x, P, z, z_prev, Q, r, d):
    """One live EKF predict+update (same math as one _ekf_forward step).
    Returns (x_new, P_new, x_pred_level, innov, S)."""
    w = 1.0 / z_prev
    v = x[1, 0]
    x_pred = _F3 @ x
    x_pred[2, 0] -= d * v * abs(v) * w
    F = _F3.copy()
    F[2, 1] -= d * 2.0 * abs(v) * w
    P_pred = F @ P @ F.T + Q
    S = P_pred[0, 0] + r
    innov = z - x_pred[0, 0]
    K = P_pred[:, :1] / S
    x_new = x_pred + K * innov
    P_new = P_pred - K * P_pred[0, :]
    P_new = 0.5 * (P_new + P_new.T)
    return x_new, P_new, x_pred[0, 0], innov, S


class EMNonLinTest(PortfolioStrategy):
    """Drag-EKF band strategy whose Q, R AND drag coefficient are re-learned
    from recent data by extended EM (instead of ExtendedKalmanTest's
    hand-built vol/autocorr drag heuristic)."""

    params = (
        ('k', 2.0),             # threshold in std devs of the innovation
        ('warmup', 180),        # bars used for the initial polyfit seed
        ('q_level', 0.1e-3),    # INITIAL Q seeds (relative to est. R) --
        ('q_vel', 0.1e-6),      #   EM re-learns Q, R, d after warmup
        ('q_acc', 0.1e-9),
        ('d0', 0.0),            # initial drag coefficient
        ('d_max', 1e6),         # clamp for the learned drag (>=0 enforced)
        ('reversion', False),   # True: fade the deviation; False: follow it
        ('em_window', 720),     # bars of rolling history the EM refit uses
        ('em_interval', 1440),  # refit every this many bars
        ('em_iters', 3),        # outer EM iterations per refit
    )

    def setup(self):
        self.kf = {}
        self._chart = {}
        for d in self.datas:
            self.kf[d] = {'ready': False, 'warm': [],
                          'X': None, 'P': None, 'Xf': None,
                          'Q': None, 'R': None, 'd': float(self.params.d0),
                          'z_prev': None,
                          'pbuf': deque(maxlen=self.params.em_window),
                          'bars_since_em': 0}
            self._chart[d._name] = []

    def _init_filter(self, st):
        buf = np.array(st['warm'], dtype=float)
        idx = np.arange(len(buf))
        # quadratic fit -> position, velocity AND acceleration seeds
        c2, c1, c0 = np.polyfit(idx, buf, 2)
        resid = buf - (c2*idx**2 + c1*idx + c0)
        R = float(np.var(resid))
        R = max(R, (buf[-1] * 1e-6) ** 2, 1e-12)
        st['R'] = R
        st['Q'] = np.diag([R * self.params.q_level,
                           R * self.params.q_vel,
                           R * self.params.q_acc])
        n = len(buf) - 1
        X0 = np.array([[buf[-1]], [2.0*c2*n + c1], [2.0*c2]])
        P0 = np.diag([R, R, R]).astype(float)
        st['X'], st['P'] = X0, P0
        st['z_prev'] = float(buf[-1])
        st['Xf'] = None                     # no forecast until the first step
        st['pbuf'].extend(buf.tolist())
        st['ready'] = True

    def _refit_em(self, d, st):
        prices = list(st['pbuf'])
        x0 = np.array([[prices[0]], [0.0], [0.0]])
        P0 = st['P']                        # current uncertainty as the prior
        Q, r, drag, x_last, P_last, ll = _extended_em_fit(
            prices, x0, P0, st['Q'], st['R'], st['d'],
            n_iter=self.params.em_iters, d_max=self.params.d_max)
        # Never hand a sick fit to the live filter: the drag model is
        # generatively unstable, so a bad window can produce non-finite or
        # indefinite results. Keep the old parameters in that case, and
        # PSD-floor the handed-off covariance either way.
        healthy = (np.isfinite(Q).all() and np.isfinite(P_last).all()
                   and np.isfinite(x_last).all()
                   and np.isfinite(r) and np.isfinite(drag)
                   and (not ll or np.isfinite(ll[-1])))
        if not healthy:
            self.log('%s EEM refit DISCARDED (non-finite result) — keeping '
                     'previous Q/R/d' % d._name, doprint=True)
            return
        w, v = np.linalg.eigh(0.5 * (P_last + P_last.T))
        P_last = (v * np.clip(w, 1e-12, None)) @ v.T
        self.log('%s EEM refit: Q=diag(%.3g,%.3g,%.3g) R=%.3g d=%.4g  '
                 'loglik %.1f -> %.1f'
                 % (d._name, Q[0, 0], Q[1, 1], Q[2, 2], r, drag,
                    ll[0], ll[-1]))
        st['Q'], st['R'], st['d'] = Q, r, drag
        st['X'], st['P'] = x_last, P_last
        st['z_prev'] = float(prices[-1])

    def _step(self, st, price):
        x, P, y_pred, innov, S = _ekf_step_one(
            st['X'], st['P'], price, st['z_prev'],
            st['Q'], st['R'], st['d'])
        # Health check: S <= 0 or non-finite means the covariance lost
        # positive-definiteness (numerical blow-up of the unstable drag
        # dynamics). Signal the caller to self-heal instead of trading on a
        # complex-valued band.
        if not (np.isfinite(S) and S > 0.0
                and np.isfinite(innov) and np.isfinite(x).all()):
            return None
        st['X'], st['P'] = x, P
        st['z_prev'] = float(price)
        return float(innov), float(S), float(y_pred)

    def on_bar(self, d, price):
        st = self.kf[d]

        if not st['ready']:
            st['warm'].append(price)
            if len(st['warm']) >= self.params.warmup:
                self._init_filter(st)
            return 0

        st['pbuf'].append(price)
        st['bars_since_em'] += 1
        if (st['bars_since_em'] >= self.params.em_interval
                and len(st['pbuf']) >= self.params.em_window):
            self._refit_em(d, st)
            st['bars_since_em'] = 0

        step = self._step(st, price)
        if step is None:
            # Filter state corrupted (indefinite covariance) — re-warm this
            # symbol's filter from scratch; the drag resets to d0 and EM will
            # relearn it at the next refit.
            self.log('%s filter diverged — re-warming from scratch'
                     % d._name, doprint=True)
            st.update(ready=False, warm=[], d=float(self.params.d0),
                      bars_since_em=0)
            return 0
        innov, S, y_pred = step
        band = self.params.k * (S ** 0.5)
        self._chart[d._name].append((self.bar_epoch(d), round(y_pred, 8),
                                     round(y_pred + band, 8),
                                     round(y_pred - band, 8)))
        self._diag_record(d, price, y_pred, innov, S, band, st)
        if band <= 0:
            return 0

        long_sig = innov < -band
        short_sig = innov > band

        if self.params.reversion:
            long_sig, short_sig = short_sig, long_sig

        pos_size = self.getposition(d).size

        if long_sig:
            if pos_size > 0:
                return 0
            self.log('%s breakout ABOVE +band (innov %.6f, band %.6f) -> LONG'
                     % (d._name, innov, band))
            return 1
        if short_sig:
            if pos_size < 0:
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
                {'name': 'EEM-Kalman', 'color': '#58a6ff',
                 'points': [{'time': t, 'value': e} for t, e, u, l in rows]},
                {'name': '+%g sigma' % k, 'color': '#8b949e',
                 'points': [{'time': t, 'value': u} for t, e, u, l in rows]},
                {'name': '-%g sigma' % k, 'color': '#8b949e',
                 'points': [{'time': t, 'value': l} for t, e, u, l in rows]},
            ]
        return out

    def stop(self):
        self.log('(k=%g warmup=%d em_window=%d) Ending Value %.2f'
                 % (self.params.k, self.params.warmup, self.params.em_window,
                    self.broker.getvalue()), doprint=True)


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(EMNonLinTest, printlog=True)

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
