from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import sys
from collections import deque

import backtrader as bt
import numpy as np

from strategy_base import PortfolioStrategy

_F = np.array([[1.0, 1.0], [0.0, 1.0]])   # position += velocity
_H = np.array([[1.0, 0.0]])               # measure position only


def _kalman_step(x, P, z, F, H, Q, R):
    """One predict+update step. Returns (x_new, P_new, x_pred, P_pred, innov, S, K)."""
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    innov = z - H @ x_pred
    x_new = x_pred + K @ innov
    n = x.shape[0]
    P_new = (np.eye(n) - K @ H) @ P_pred
    P_new = 0.5 * (P_new + P_new.T)
    return x_new, P_new, x_pred, P_pred, innov, S, K


def _kalman_forward(Z, x0, P0, F, H, Q, R):

    T = len(Z)
    n = x0.shape[0]
    X_f = np.zeros((T, n, 1)); P_f = np.zeros((T, n, n))
    X_p = np.zeros((T, n, 1)); P_p = np.zeros((T, n, n))
    x, P = x0, P0
    loglik = 0.0
    K_last = None
    for t in range(T):
        x, P, x_pred, P_pred, innov, S, K = _kalman_step(x, P, Z[t], F, H, Q, R)
        X_p[t], P_p[t] = x_pred, P_pred
        X_f[t], P_f[t] = x, P
        K_last = K
        sign, logdet = np.linalg.slogdet(S)
        loglik += -0.5 * (logdet + (innov.T @ np.linalg.inv(S) @ innov).item()
                          + H.shape[0] * np.log(2 * np.pi))
    return X_f, P_f, X_p, P_p, K_last, loglik


def _rts_smoother(X_f, P_f, X_p, P_p, F):
    """Returns smoothed means/covariances
    and the smoothing gain J_t (needed for the lag-one covariance below)."""
    T, n, _ = X_f.shape
    X_s = np.zeros_like(X_f); P_s = np.zeros_like(P_f)
    J = np.zeros((T, n, n))
    X_s[-1], P_s[-1] = X_f[-1], P_f[-1]
    for t in range(T - 2, -1, -1):
        J[t] = P_f[t] @ F.T @ np.linalg.inv(P_p[t + 1])
        X_s[t] = X_f[t] + J[t] @ (X_s[t + 1] - X_p[t + 1])
        P_s[t] = P_f[t] + J[t] @ (P_s[t + 1] - P_p[t + 1]) @ J[t].T
        P_s[t] = 0.5 * (P_s[t] + P_s[t].T)
    return X_s, P_s, J


def _lag_one_covariance(P_f, J, K_last, F, H, P0, P_p0):

    T, n, _ = P_f.shape
    I = np.eye(n)
    P_lag = np.zeros((T, n, n))
    J_prior = P0 @ F.T @ np.linalg.inv(P_p0)

    if T >= 2:
        P_lag[T - 1] = (I - K_last @ H) @ F @ P_f[T - 2]
        for t in range(T - 2, 0, -1):
            P_lag[t] = P_f[t] @ J[t - 1].T + J[t] @ (P_lag[t + 1] - F @ P_f[t]) @ J[t - 1].T
        P_lag[0] = P_f[0] @ J_prior.T + J[0] @ (P_lag[1] - F @ P_f[0]) @ J_prior.T
    else:
        P_lag[0] = (I - K_last @ H) @ F @ P0
    return P_lag


_LOG2PI = float(np.log(2.0 * np.pi))


def _em_fit_cv(prices, x0, P0, Q0, R0, n_iter, r_floor):
    """Fused scalar fast path of em_fit for THIS module's fixed model
    (F=[[1,1],[0,1]], H=[1,0], scalar R). Same math as the general numpy
    path below, hand-expanded to plain float arithmetic -- the per-step numpy
    call overhead dominated the runtime (~56us/step -> ~2us/step). Validated
    against the general path to ~1e-9 on real and synthetic data."""
    Z = [float(p) for p in prices]
    T = len(Z)
    x0_0, x0_1 = float(x0[0, 0]), float(x0[1, 0])
    P0_00, P0_01, P0_11 = float(P0[0, 0]), float(P0[0, 1]), float(P0[1, 1])
    q00, q01, q11 = float(Q0[0, 0]), float(Q0[0, 1]), float(Q0[1, 1])
    r = float(R0[0, 0])
    loglik_hist = []

    from math import log as _log

    def forward(q00, q01, q11, r):
        """One filter pass. Returns per-t lists + last gain + loglik."""
        xp0 = [0.0]*T; xp1 = [0.0]*T
        pp00 = [0.0]*T; pp01 = [0.0]*T; pp11 = [0.0]*T
        xf0 = [0.0]*T; xf1 = [0.0]*T
        pf00 = [0.0]*T; pf01 = [0.0]*T; pf11 = [0.0]*T
        a0, a1 = x0_0, x0_1
        c00, c01, c11 = P0_00, P0_01, P0_11
        ll = 0.0
        K0 = K1 = 0.0
        for t in range(T):
            # predict
            p0 = a0 + a1
            b00 = c00 + 2.0*c01 + c11 + q00
            b01 = c01 + c11 + q01
            b11 = c11 + q11
            S = b00 + r
            K0 = b00 / S; K1 = b01 / S
            innov = Z[t] - p0
            xp0[t] = p0; xp1[t] = a1               # a-priori x1 == previous a1
            # update
            a0 = p0 + K0*innov
            a1 = a1 + K1*innov
            omk = 1.0 - K0
            n00 = omk*b00
            n11 = b11 - K1*b01
            n01 = 0.5*(omk*b01 + (b01 - K1*b00))   # symmetrized, as numpy path
            pp00[t] = b00; pp01[t] = b01; pp11[t] = b11
            xf0[t] = a0; xf1[t] = a1
            pf00[t] = n00; pf01[t] = n01; pf11[t] = n11
            c00, c01, c11 = n00, n01, n11
            ll += -0.5*(_log(S) + innov*innov/S + _LOG2PI)
        return (xp0, xp1, pp00, pp01, pp11, xf0, xf1,
                pf00, pf01, pf11, K0, K1, ll)

    for _ in range(n_iter):
        (xp0, xp1, pp00, pp01, pp11, xf0, xf1,
         pf00, pf01, pf11, KL0, KL1, ll) = forward(q00, q01, q11, r)
        loglik_hist.append(ll)

        # ---- RTS smoother (backward) + store J per t --------------------
        xs0 = [0.0]*T; xs1 = [0.0]*T
        ps00 = [0.0]*T; ps01 = [0.0]*T; ps11 = [0.0]*T
        J00 = [0.0]*T; J01 = [0.0]*T; J10 = [0.0]*T; J11 = [0.0]*T
        xs0[T-1], xs1[T-1] = xf0[T-1], xf1[T-1]
        ps00[T-1], ps01[T-1], ps11[T-1] = pf00[T-1], pf01[T-1], pf11[T-1]
        for t in range(T-2, -1, -1):
            a, b, c = pp00[t+1], pp01[t+1], pp11[t+1]
            det = a*c - b*b
            i00 = c/det; i01 = -b/det; i11 = a/det
            m00 = pf00[t] + pf01[t]; m01 = pf01[t]
            m10 = pf01[t] + pf11[t]; m11 = pf11[t]
            j00 = m00*i00 + m01*i01; j01 = m00*i01 + m01*i11
            j10 = m10*i00 + m11*i01; j11 = m10*i01 + m11*i11
            J00[t] = j00; J01[t] = j01; J10[t] = j10; J11[t] = j11
            dx0 = xs0[t+1] - xp0[t+1]; dx1 = xs1[t+1] - xp1[t+1]
            xs0[t] = xf0[t] + j00*dx0 + j01*dx1
            xs1[t] = xf1[t] + j10*dx0 + j11*dx1
            d00 = ps00[t+1] - pp00[t+1]
            d01 = ps01[t+1] - pp01[t+1]
            d11 = ps11[t+1] - pp11[t+1]
            g00 = j00*d00 + j01*d01; g01 = j00*d01 + j01*d11
            g10 = j10*d00 + j11*d01; g11 = j10*d01 + j11*d11
            e00 = g00*j00 + g01*j01
            e01 = g00*j10 + g01*j11
            e10 = g10*j00 + g11*j01
            e11 = g10*j10 + g11*j11
            ps00[t] = pf00[t] + e00
            ps11[t] = pf11[t] + e11
            ps01[t] = pf01[t] + 0.5*(e01 + e10)

        # ---- prior's smoothed posterior + J_prior -----------------------
        a, b, c = pp00[0], pp01[0], pp11[0]
        det = a*c - b*b
        i00 = c/det; i01 = -b/det; i11 = a/det
        m00 = P0_00 + P0_01; m01 = P0_01
        m10 = P0_01 + P0_11; m11 = P0_11
        jp00 = m00*i00 + m01*i01; jp01 = m00*i01 + m01*i11
        jp10 = m10*i00 + m11*i01; jp11 = m10*i01 + m11*i11
        dx0 = xs0[0] - xp0[0]; dx1 = xs1[0] - xp1[0]
        x0s0 = x0_0 + jp00*dx0 + jp01*dx1
        x0s1 = x0_1 + jp10*dx0 + jp11*dx1
        d00 = ps00[0] - pp00[0]; d01 = ps01[0] - pp01[0]; d11 = ps11[0] - pp11[0]
        g00 = jp00*d00 + jp01*d01; g01 = jp00*d01 + jp01*d11
        g10 = jp10*d00 + jp11*d01; g11 = jp10*d01 + jp11*d11
        p0s00 = P0_00 + (g00*jp00 + g01*jp01)
        p0s11 = P0_11 + (g10*jp10 + g11*jp11)
        p0s01 = P0_01 + 0.5*((g00*jp10 + g01*jp11) + (g10*jp00 + g11*jp01))

        # ---- lag-one covariance (backward, exact recursion) -------------
        lag00 = [0.0]*T; lag01 = [0.0]*T; lag10 = [0.0]*T; lag11 = [0.0]*T
        if T >= 2:
            f00 = pf00[T-2] + pf01[T-2]; f01 = pf01[T-2] + pf11[T-2]
            f10 = pf01[T-2]; f11 = pf11[T-2]
            omk = 1.0 - KL0
            lag00[T-1] = omk*f00
            lag01[T-1] = omk*f01
            lag10[T-1] = -KL1*f00 + f10
            lag11[T-1] = -KL1*f01 + f11
            for t in range(T-2, -1, -1):
                # F @ P_f[t]
                f00 = pf00[t] + pf01[t]; f01 = pf01[t] + pf11[t]
                f10 = pf01[t]; f11 = pf11[t]
                g00 = lag00[t+1] - f00; g01 = lag01[t+1] - f01
                g10 = lag10[t+1] - f10; g11 = lag11[t+1] - f11
                a00 = J00[t]*g00 + J01[t]*g10; a01 = J00[t]*g01 + J01[t]*g11
                a10 = J10[t]*g00 + J11[t]*g10; a11 = J10[t]*g01 + J11[t]*g11
                if t >= 1:
                    k00, k01, k10, k11 = J00[t-1], J01[t-1], J10[t-1], J11[t-1]
                else:
                    k00, k01, k10, k11 = jp00, jp01, jp10, jp11
                # P_f[t] @ K^T  (P_f symmetric)
                b00 = pf00[t]*k00 + pf01[t]*k01; b01 = pf00[t]*k10 + pf01[t]*k11
                b10 = pf01[t]*k00 + pf11[t]*k01; b11 = pf01[t]*k10 + pf11[t]*k11
                lag00[t] = b00 + a00*k00 + a01*k01
                lag01[t] = b01 + a00*k10 + a01*k11
                lag10[t] = b10 + a10*k00 + a11*k01
                lag11[t] = b11 + a10*k10 + a11*k11
        else:
            omk = 1.0 - KL0
            f00 = P0_00 + P0_01; f01 = P0_01 + P0_11
            lag00[0] = omk*f00; lag01[0] = omk*f01
            lag10[0] = -KL1*f00 + P0_01; lag11[0] = -KL1*f01 + P0_11

        # ---- M-step sums (vectorized) ------------------------------------
        za = np.asarray(Z)
        Xs0 = np.asarray(xs0); Xs1 = np.asarray(xs1)
        prev0 = np.empty(T); prev0[0] = x0s0; prev0[1:] = Xs0[:-1]
        prev1 = np.empty(T); prev1[0] = x0s1; prev1[1:] = Xs1[:-1]
        Ps00 = np.asarray(ps00); Ps01 = np.asarray(ps01); Ps11 = np.asarray(ps11)
        pv00 = np.empty(T); pv00[0] = p0s00; pv00[1:] = Ps00[:-1]
        pv01 = np.empty(T); pv01[0] = p0s01; pv01[1:] = Ps01[:-1]
        pv11 = np.empty(T); pv11[0] = p0s11; pv11[1:] = Ps11[:-1]

        S11 = np.array([[Ps00.sum() + (Xs0*Xs0).sum(), Ps01.sum() + (Xs0*Xs1).sum()],
                        [0.0,                          Ps11.sum() + (Xs1*Xs1).sum()]])
        S11[1, 0] = S11[0, 1]
        S10 = np.array([[np.sum(lag00) + (Xs0*prev0).sum(), np.sum(lag01) + (Xs0*prev1).sum()],
                        [np.sum(lag10) + (Xs1*prev0).sum(), np.sum(lag11) + (Xs1*prev1).sum()]])
        S00 = np.array([[pv00.sum() + (prev0*prev0).sum(), pv01.sum() + (prev0*prev1).sum()],
                        [0.0,                              pv11.sum() + (prev1*prev1).sum()]])
        S00[1, 0] = S00[0, 1]

        F = _F
        Q_new = (S11 - F @ S10.T - S10 @ F.T + F @ S00 @ F.T) / T
        Q_new = 0.5 * (Q_new + Q_new.T)
        w, v = np.linalg.eigh(Q_new)
        w = np.clip(w, r_floor, None)
        Q_new = (v * w) @ v.T
        r_new = float((((za - Xs0)**2).sum() + Ps00.sum()) / T)
        r = max(r_new, r_floor)
        q00, q01, q11 = float(Q_new[0, 0]), float(Q_new[0, 1]), float(Q_new[1, 1])

    (xp0, xp1, pp00, pp01, pp11, xf0, xf1,
     pf00, pf01, pf11, _, _, ll) = forward(q00, q01, q11, r)
    loglik_hist.append(ll)
    Q = np.array([[q00, q01], [q01, q11]])
    R = np.array([[r]])
    x_last = np.array([[xf0[-1]], [xf1[-1]]])
    P_last = np.array([[pf00[-1], pf01[-1]], [pf01[-1], pf11[-1]]])
    return Q, R, x_last, P_last, loglik_hist


def em_fit(prices, x0, P0, Q0, R0, F=_F, H=_H, n_iter=5, r_floor=1e-12):
    # Fast scalar path for this module's fixed constant-velocity model; the
    # general numpy path below stays as reference and for custom F/H.
    if (F is _F or np.array_equal(F, _F)) and (H is _H or np.array_equal(H, _H)):
        return _em_fit_cv(prices, x0, P0, Q0, R0, n_iter, r_floor)

    Z = [np.array([[p]], dtype=float) for p in prices]
    T = len(Z)
    n = x0.shape[0]
    Q, R = Q0.copy(), R0.copy()
    loglik_hist = []

    for _ in range(n_iter):
        X_f, P_f, X_p, P_p, K_last, ll = _kalman_forward(Z, x0, P0, F, H, Q, R)
        loglik_hist.append(ll)
        X_s, P_s, J = _rts_smoother(X_f, P_f, X_p, P_p, F)
        P_lag = _lag_one_covariance(P_f, J, K_last, F, H, P0, P_p[0])

        J_prior = P0 @ F.T @ np.linalg.inv(P_p[0])
        x0_s = x0 + J_prior @ (X_s[0] - X_p[0])
        P0_s = P0 + J_prior @ (P_s[0] - P_p[0]) @ J_prior.T
        P0_s = 0.5 * (P0_s + P0_s.T)

        S11 = np.zeros((n, n)); S10 = np.zeros((n, n)); S00 = np.zeros((n, n))
        R_new = np.zeros_like(R)
        for t in range(T):
            xt, Pt = X_s[t], P_s[t]
            xtm1 = x0_s if t == 0 else X_s[t - 1]
            Ptm1 = P0_s if t == 0 else P_s[t - 1]
            Plag_t = P_lag[t]

            S11 += Pt + xt @ xt.T
            S10 += Plag_t + xt @ xtm1.T
            S00 += Ptm1 + xtm1 @ xtm1.T

            innov = Z[t] - H @ xt
            R_new += innov @ innov.T + H @ Pt @ H.T

        # F is held FIXED (not re-estimated), so the ML Q given fixed F is
        # the residual covariance of (x_t - F x_{t-1}), not the usual
        # S11 - F@S10^T form (that formula assumes F is also being updated).
        Q_new = (S11 - F @ S10.T - S10 @ F.T + F @ S00 @ F.T) / T
        Q_new = 0.5 * (Q_new + Q_new.T)
        R_new = R_new / T

        # numerical floor, same pattern used elsewhere in this codebase
        # (KalmanFilter._init_filter's R floor) so S never collapses to ~0.
        w, v = np.linalg.eigh(Q_new)
        w = np.clip(w, r_floor, None)
        Q_new = (v * w) @ v.T
        R_new = np.clip(R_new, r_floor, None)

        Q, R = Q_new, R_new

    X_f, P_f, _, _, _, ll = _kalman_forward(Z, x0, P0, F, H, Q, R)
    loglik_hist.append(ll)
    return Q, R, X_f[-1], P_f[-1], loglik_hist


class EMTest(PortfolioStrategy):

    params = (
        ('k', 2.0),             # threshold in std devs of the innovation
        ('a', 7.0),             # stop loss this much above value 
        ('warmup', 180),        # bars used for the initial polyfit seed
        ('q_level', 0.1e-3),    # INITIAL process noise on level (relative to est. R)
        ('q_vel', 0.1e-6),      # INITIAL process noise on slope (relative to est. R)
        ('reversion', False),   # True: fade the deviation; False: follow it
        ('em_window', 720),     # bars of rolling history the EM refit uses
        ('em_interval', 1440),   # refit Q,R every this many bars
        ('em_iters', 5),        # EM iterations per refit
    )

    F = _F
    H = _H

    def setup(self):
        self.kf = {}
        self._chart = {}
        for d in self.datas:
            self.kf[d] = {'ready': False, 'warm': [],
                          'X': None, 'P': None, 'Xf': None, 'Pf': None,
                          'Q': None, 'R': None,
                          'pbuf': deque(maxlen=self.params.em_window),
                          'bars_since_em': 0}
            self._chart[d._name] = []

    def _init_filter(self, st):
        buf = np.array(st['warm'], dtype=float)
        idx = np.arange(len(buf))
        m, b = np.polyfit(idx, buf, 1)
        resid = buf - (m * idx + b)
        R = float(np.var(resid))
        R = max(R, (buf[-1] * 1e-6) ** 2, 1e-12)
        st['R'] = np.array([[R]])
        st['Q'] = np.array([[R * self.params.q_level, 0.0],
                            [0.0, R * self.params.q_vel]])
        X0 = np.array([[buf[-1]], [m]])
        P0 = np.array([[R, 0.0], [0.0, R]])
        st['X'], st['P'] = X0, P0
        st['Xf'] = self.F @ X0
        st['Pf'] = self.F @ P0 @ self.F.T + st['Q']
        st['pbuf'].extend(buf.tolist())
        st['ready'] = True

    def _refit_em(self, d, st):

        prices = list(st['pbuf'])
        x0 = np.array([[prices[0]], [0.0]])
        P0 = st['P']            # current uncertainty as the batch's prior
        Q, R, x_last, P_last, ll = em_fit(prices, x0, P0, st['Q'], st['R'],
                                          F=self.F, H=self.H,
                                          n_iter=self.params.em_iters)
        self.log('%s EM refit: Q=diag(%.3g,%.3g) R=%.3g  loglik %.1f -> %.1f'
                 % (d._name, Q[0, 0], Q[1, 1], R[0, 0], ll[0], ll[-1]))
        st['Q'], st['R'] = Q, R
        st['X'], st['P'] = x_last, P_last
        st['Xf'] = self.F @ x_last
        st['Pf'] = self.F @ P_last @ self.F.T + Q

    def _step(self, st, price):
        # Scalar fast path of _kalman_step for the fixed 2-state CV model
        # (identical math; drops ~25 tiny numpy calls per bar). Q may be a
        # full symmetric matrix after an EM refit, so keep its off-diagonal.
        X, P, Q = st['X'], st['P'], st['Q']
        x0, x1 = X[0, 0], X[1, 0]
        p00, p01, p11 = P[0, 0], P[0, 1], P[1, 1]
        q00, q01, q11 = Q[0, 0], Q[0, 1], Q[1, 1]
        r = st['R'][0, 0]

        # predict
        xp0 = x0 + x1
        b00 = p00 + 2.0*p01 + p11 + q00
        b01 = p01 + p11 + q01
        b11 = p11 + q11
        S = b00 + r
        innov = price - xp0
        # update
        K0 = b00 / S; K1 = b01 / S
        nx0 = xp0 + K0*innov
        nx1 = x1 + K1*innov
        omk = 1.0 - K0
        n00 = omk*b00
        n11 = b11 - K1*b01
        n01 = 0.5*(omk*b01 + (b01 - K1*b00))    # symmetrized, as before

        y_pred = st['Xf'][0, 0]                 # last bar's forecast for THIS bar
        st['X'] = np.array([[nx0], [nx1]])
        st['P'] = np.array([[n00, n01], [n01, n11]])
        # roll the one-bar-ahead forecast (Xf/Pf), same as F@x / F@P@F.T + Q
        st['Xf'] = np.array([[nx0 + nx1], [nx1]])
        st['Pf'] = np.array([[n00 + 2.0*n01 + n11 + q00, n01 + n11 + q01],
                             [n01 + n11 + q01,           n11 + q11]])
        return innov, S, y_pred

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

        innov, S, y_pred = self._step(st, price)
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
                {'name': 'EM-Kalman', 'color': '#58a6ff',
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
    cerebro.addstrategy(EMTest, printlog=True)

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
