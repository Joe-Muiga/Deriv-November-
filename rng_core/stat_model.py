"""rng_core.stat_model - Layer 3: stationary Bayesian mixture over the digit alphabet, per horizon h, with an explicit
uniform null and an anytime-valid e-process.

Experts (E = 40), each a predictive distribution for D[t+h] given D[..t]:
    U            uniform (explicit null, prior weight 0.5)
    F            global frequencies, KT/Dirichlet alpha = 0.5
    C1..C4       hierarchical back-off context models (last k digits), enabled once n >= 50 * 10^k
    D            lag-h difference mod 10 (stationary M1)
    S0..S31      single-lag dependence tables P(d_{t+h} | d_{t-i}), i = 0..31 (i = 0 is the newest digit)
    B            bit decomposition: parity bit given the last 6 parities, then the high bits given parity

Mixture weight  ~  prior_m * exp(r_m),  r_m = sum of (ln p_m(y) - ln 0.1) over predictions scored since m was enabled. No decay.
Everything is stationary: counts are never discounted.

E-process: log_e = sum over scored predictions of ln(p_mix(y) / 0.1). Because each p_mix is fixed BEFORE its digit exists, exp(log_e)
is a nonnegative supermartingale under 'digits are iid uniform' (Ville: P(sup E >= 1/alpha) <= alpha).

Ranking vs probability. p_mix (the honest predictive distribution, null included) is what the e-process scores. The top-4 RANKING
comes from p_rank = the same mixture with the uniform expert removed (weights renormalised over the other experts): when the null
dominates, p_mix = 0.1 + (1-w_U)(p_rank - 0.1) has the same ordering as p_rank but the differences fall below float resolution, so
ranking from p_mix would silently tie-break to the lowest digits. Under a fair stream any predictable top-4 rule hits exactly 40%, so
ranking from p_rank costs nothing there and uses every bit of evidence elsewhere; calibration (decide.py) is what keeps it honest.

Two implementations with identical maths: StatModel (online, O(E) numpy per tick and horizon) and batch_horizon() (vectorised,
used to load history and by the evaluation harness). tests assert they agree.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import UNIFORM_FLOOR

LN01 = math.log(0.1)
A_F = 0.5                  # KT alpha for the marginal
BETA_C = 100.0             # back-off strength of the context models (tuned on held-out seeds, see docs/v2/REPORT.md)
BETA_S = 300.0             # shrinkage of the single-lag tables toward F (tuned on held-out seeds)
NLAG = 32
PAR_CTX = 6                # parities in the B expert's context
PRIOR_U = 0.5

NAMES = ["U", "F", "C1", "C2", "C3", "C4", "D"] + [f"S{i}" for i in range(NLAG)] + ["B"]
E = len(NAMES)
I_U, I_F, I_C, I_D, I_S, I_B = 0, 1, 2, 6, 7, 7 + NLAG          # C1..C4 -> 2..5, S -> 7..38, B -> 39
PRIOR = np.full(E, (1.0 - PRIOR_U) / (E - 1))
PRIOR[I_U] = PRIOR_U
CTX_MIN = {1: 50 * 10, 2: 50 * 100, 3: 50 * 1000, 4: 50 * 10000}   # enable C_k only when n >= 50 * 10^k
POW10 = [10 ** i for i in range(5)]
LOGPRIOR = np.log(PRIOR)
_AR10 = np.arange(10)


def default_cfg() -> dict:
    return {"ctx_min": dict(CTX_MIN), "beta_c": BETA_C, "beta_s": BETA_S}


# ── batch (vectorised) ──────────────────────────────────────────────────────────────────────────
def _ctx_counts(ce: np.ndarray, ev_t: np.ndarray, y: np.ndarray, cq: np.ndarray, q_t: np.ndarray, A: int, n_total: int) -> np.ndarray:
    """For every query q: counts (A,) of outcomes y[e] over events e with ce[e] == cq[q] and ev_t[e] <= q_t[q] (stationary)."""
    nq = len(cq)
    if len(ce) == 0:
        return np.zeros((nq, A))
    N = n_total + 2
    key = ce.astype(np.int64) * N + ev_t
    order = np.argsort(key, kind="stable")
    key = key[order]
    oh = np.zeros((len(ce), A))
    oh[np.arange(len(ce)), y[order]] = 1.0
    cum = np.vstack([np.zeros((1, A)), np.cumsum(oh, axis=0)])
    start = np.searchsorted(key, cq.astype(np.int64) * N, side="left")
    end = np.searchsorted(key, cq.astype(np.int64) * N + q_t, side="right")
    return cum[end] - cum[start]


def _ctx_index(D: np.ndarray, k: int) -> np.ndarray:
    """ctx[t] = sum_{i<k} D[t-i] * 10^i for t >= k-1 (earlier entries use clipped indices; such queries are disabled)."""
    n = len(D)
    t = np.arange(n)
    c = np.zeros(n, np.int64)
    for i in range(k):
        c += D[np.maximum(t - i, 0)].astype(np.int64) * POW10[i]
    return c


def expert_dists(D: np.ndarray, h: int, cfg: Optional[dict] = None):
    """Generator over experts: yields (name_index, P (n,10), enabled (n,) bool) for query times t = 0..n-1."""
    cfg = {**default_cfg(), **(cfg or {})}
    D = np.asarray(D, np.int64)
    n = len(D)
    t = np.arange(n)
    oh = np.zeros((n, 10))
    oh[t, D] = 1.0
    cumF = np.cumsum(oh, axis=0)                                           # counts of D[0..t]
    PF = (cumF + A_F) / (t[:, None] + 1.0 + 10 * A_F)
    yield I_U, np.full((n, 10), 0.1), np.ones(n, bool)
    yield I_F, PF, np.ones(n, bool)
    chain = PF
    j = np.arange(n)
    for k in (1, 2, 3, 4):
        cq = _ctx_index(D, k)
        ev = j[j >= h + k - 1]
        ce = cq[ev - h]
        cnt = _ctx_counts(ce, ev, D[ev], cq, t, 10, n)
        chain = (cnt + cfg['beta_c'] * chain) / (cnt.sum(axis=1, keepdims=True) + cfg['beta_c'])
        yield I_C + k - 1, chain, t + 1 >= cfg["ctx_min"][k]
    ev = j[j >= h]
    delta = (D[ev] - D[ev - h]) % 10
    cnt = _ctx_counts(np.zeros(len(ev), np.int64), ev, delta, np.zeros(n, np.int64), t, 10, n)
    pm = (cnt + A_F) / (cnt.sum(axis=1, keepdims=True) + 10 * A_F)
    idx = (np.arange(10)[None, :] - D[:, None]) % 10
    yield I_D, pm[t[:, None], idx], np.ones(n, bool)
    for i in range(NLAG):
        ev = j[j >= h + i]
        cq = D[np.maximum(t - i, 0)]
        cnt = _ctx_counts(D[ev - h - i], ev, D[ev], cq, t, 10, n)
        yield I_S + i, (cnt + cfg['beta_s'] * PF) / (cnt.sum(axis=1, keepdims=True) + cfg['beta_s']), t >= i
    par = (D & 1).astype(np.int64)
    cq = np.zeros(n, np.int64)
    for i in range(PAR_CTX):
        cq += par[np.maximum(t - i, 0)] << i
    ev = j[j >= h + PAR_CTX - 1]
    cnt = _ctx_counts(cq[ev - h], ev, par[ev], cq, t, 2, n)
    p1 = (cnt[:, 1] + 0.5) / (cnt.sum(axis=1) + 1.0)                         # P(parity = 1)
    PB = np.zeros((n, 10))
    for par_v in (0, 1):
        cols = np.arange(par_v, 10, 2)                                       # digits 2*hi + par_v
        cnts = cumF[:, cols]
        ph = (cnts + 0.5) / (cnts.sum(axis=1, keepdims=True) + 2.5)
        PB[:, cols] = (p1 if par_v else 1.0 - p1)[:, None] * ph
    yield I_B, PB, t >= PAR_CTX - 1


def batch_horizon(D: np.ndarray, h: int, cfg: Optional[dict] = None, keep_last: int = 0) -> dict:
    """Prequential mixture for ONE horizon over the whole array. Predictions are made at t = 0..n-1 for target t+h.

    Returns pmix (n,10) (finalised, row t = prediction made at t), r_final (E,), log_e, per-target arrays and, for loading an online
    state, the last `keep_last` per-expert predictions (and enabled masks).
    """
    D = np.asarray(D, np.int64)
    n = len(D)
    t = np.arange(n)
    contrib = np.zeros((E, n))                       # contrib[m, s] for target s = t + h  (added to r_m from time s on)
    enabled = np.zeros((E, n), bool)
    for m, P, en in expert_dists(D, h, cfg):
        enabled[m] = en
        if n > h:
            q = np.arange(0, n - h)                 # query times with a scored target
            lg = np.log(np.maximum(P[q, D[q + h]], 1e-300)) - LN01
            contrib[m, q + h] = np.where(en[q], lg, 0.0)
    r_last = np.cumsum(contrib, axis=1, out=contrib)[:, -1].copy()               # r_m(t): everything scored up to and including target t
    logw = contrib                                                              # in place: from here on `contrib` holds log-weights
    logw += LOGPRIOR[:, None]
    logw[~enabled] = -np.inf
    # normalisers computed row by row (no second (E, n) array): max and sum of exp for all experts (w) and with the null removed (w2)
    mx1 = logw.max(axis=0)
    mx2 = np.full(n, -np.inf)
    for m in range(E):
        if m != I_U:
            np.maximum(mx2, logw[m], out=mx2)
    S1, S2 = np.zeros(n), np.zeros(n)
    for m in range(E):
        S1 += np.exp(logw[m] - mx1)
        if m != I_U:
            S2 += np.exp(logw[m] - mx2)
    num = np.zeros((n, 10))
    num2 = np.zeros((n, 10))
    last = {}
    for m, P, en in expert_dists(D, h, cfg):
        num += (np.exp(logw[m] - mx1) / S1)[:, None] * P
        if m != I_U:
            num2 += (np.exp(logw[m] - mx2) / S2)[:, None] * P
        if keep_last:
            last[m] = P[n - keep_last:].copy()
    pm = np.maximum(num, UNIFORM_FLOOR)
    pm /= pm.sum(axis=1, keepdims=True)
    out = {"pmix": pm, "prank": num2, "r_final": r_last, "enabled": enabled}
    if n > h:
        ll = np.log(pm[np.arange(0, n - h), D[h:]]) - LN01          # e-process increments, target index h..n-1
        out["log_e_incr"] = ll
        out["log_e"] = float(ll.sum())
    else:
        out["log_e_incr"], out["log_e"] = np.zeros(0), 0.0
    if keep_last:
        out["last_P"] = last
    return out


# ── online ──────────────────────────────────────────────────────────────────────────────────────
class _HorizonState:
    """Online state of one horizon."""

    def __init__(self, h: int, cfg: dict):
        self.h = h
        self.cfg = cfg
        self.cF = np.zeros(10)
        self.TC = {k: np.zeros((POW10[k], 10)) for k in (1, 2, 3, 4)}
        self.cD = np.zeros(10)
        self.TS = np.zeros((NLAG, 10, 10))
        self.Tpar = np.zeros((1 << PAR_CTX, 2))
        self.r = np.zeros(E)
        self.pend: Deque[tuple] = deque()            # (target_idx, P (E,10), enabled (E,), pmix (10,))
        self.log_e = 0.0
        self.n_scored = 0
        self.hit4 = 0
        self.ll_gain = 0.0
        self.cur: Optional[np.ndarray] = None        # latest mixed distribution (prediction for target t+h)
        self.cur_alt: Optional[np.ndarray] = None    # latest ranking distribution (null expert removed)
        self.seg_t = 0                               # index of the first digit of the current gap-free segment
        self._lag = np.arange(NLAG)

    # ---- one tick ----
    def step(self, buf: np.ndarray, t: int) -> Optional[tuple]:
        """buf: array of recent digits, buf[-1] == D[t] (length >= 48 once t >= 47). Returns the scored (p4_raw, hit, ll_gain) or None."""
        h = self.h
        y = int(buf[-1])
        scored = None
        while self.pend and self.pend[0][0] <= t:
            tgt, P, en, pmix, prank = self.pend.popleft()
            if tgt == t:
                lg = np.log(np.maximum(P[:, y], 1e-300)) - LN01
                self.r += np.where(en, lg, 0.0)
                self.log_e += math.log(pmix[y]) - LN01
                top4 = np.argsort(-prank, kind="stable")[:4]
                hit = int(y in top4)
                self.n_scored += 1
                self.hit4 += hit
                self.ll_gain += math.log(pmix[y]) - LN01
                scored = (float(prank[top4].sum()), hit, math.log(pmix[y]) - LN01)
        n = t + 1
        self.cF[y] += 1.0
        L = len(buf)
        if t - h >= self.seg_t:
            a = int(buf[L - 1 - h])
            self.cD[(y - a) % 10] += 1.0
            valid = t - h - self._lag >= self.seg_t
            if valid.all():
                self.TS[self._lag, buf[L - 1 - h - self._lag], y] += 1.0
            else:
                li = self._lag[valid]
                self.TS[li, buf[L - 1 - h - li], y] += 1.0
            for k in (1, 2, 3, 4):
                if t - h - k + 1 >= self.seg_t:
                    c = 0
                    for i in range(k):
                        c += int(buf[L - 1 - h - i]) * POW10[i]
                    self.TC[k][c, y] += 1.0
            if t - h - PAR_CTX + 1 >= self.seg_t:
                c = 0
                for i in range(PAR_CTX):
                    c += (int(buf[L - 1 - h - i]) & 1) << i
                self.Tpar[c, y & 1] += 1.0
        P, en = self._dists(buf, t)
        logw = np.log(PRIOR) + self.r
        logw = np.where(en, logw, -np.inf)
        w = np.exp(logw - logw.max())
        w /= w.sum()
        pm = np.maximum(w @ P, UNIFORM_FLOOR)
        pm /= pm.sum()
        lw2 = logw.copy()
        lw2[I_U] = -np.inf
        w2 = np.exp(lw2 - lw2.max())
        w2 /= w2.sum()
        pr = w2 @ P
        self.pend.append((t + h, P, en, pm, pr))
        self.cur = pm
        self.cur_alt = pr
        return scored

    def _dists(self, buf: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray]:
        n = t + 1
        L = len(buf)
        P = np.empty((E, 10))
        en = np.zeros(E, bool)
        P[I_U] = 0.1
        en[I_U] = True
        PF = (self.cF + A_F) / (n + 10 * A_F)
        P[I_F] = PF
        en[I_F] = True
        chain = PF
        for k in (1, 2, 3, 4):
            c = 0                                                # buf[L-1-i] is D[t-i], weight 10^i (same as _ctx_index)
            for i in range(min(k, L)):
                c += int(buf[L - 1 - i]) * POW10[i]
            cnt = self.TC[k][c]
            chain = (cnt + self.cfg['beta_c'] * chain) / (cnt.sum() + self.cfg['beta_c'])
            P[I_C + k - 1] = chain
            en[I_C + k - 1] = (n >= self.cfg["ctx_min"][k]) and (t - k + 1 >= self.seg_t)
        pmD = (self.cD + A_F) / (self.cD.sum() + 10 * A_F)
        a = int(buf[-1])
        P[I_D] = pmD[(np.arange(10) - a) % 10]
        en[I_D] = True
        lag = self._lag
        avail = (t - lag) >= self.seg_t
        idx = np.maximum(L - 1 - lag, 0)
        cnt = self.TS[lag, buf[idx]]                                # (32,10)
        P[I_S:I_S + NLAG] = (cnt + self.cfg['beta_s'] * PF) / (cnt.sum(axis=1, keepdims=True) + self.cfg['beta_s'])
        en[I_S:I_S + NLAG] = avail
        c = 0
        for i in range(PAR_CTX):
            c += (int(buf[max(L - 1 - i, 0)]) & 1) << i
        tp = self.Tpar[c]
        p1 = (tp[1] + 0.5) / (tp.sum() + 1.0)
        pb = np.empty(10)
        for par_v in (0, 1):
            cols = np.arange(par_v, 10, 2)
            cn = self.cF[cols]
            pb[cols] = (p1 if par_v else 1.0 - p1) * (cn + 0.5) / (cn.sum() + 2.5)
        P[I_B] = pb
        en[I_B] = t - PAR_CTX + 1 >= self.seg_t
        return P, en


class StatModel:
    """Online stationary mixture for horizons 1..hmax of ONE stream (one segment)."""

    def __init__(self, hmax: int = 10, cfg: Optional[dict] = None):
        self.hmax = hmax
        self.cfg = {**default_cfg(), **(cfg or {})}
        self.H: Dict[int, _HorizonState] = {h: _HorizonState(h, self.cfg) for h in range(1, hmax + 1)}
        self.buf: Deque[int] = deque(maxlen=64)
        self.n = 0                                     # digits seen; the newest has index n-1
        self.fast = True                               # vectorised-over-horizons update (identical results to _HorizonState.step)
        self._stacked = False

    def reset(self) -> None:
        self.__init__(self.hmax, self.cfg)

    def on_digit(self, d: int) -> Dict[int, tuple]:
        """Feed one digit; returns {h: (p4_raw, hit, ll_gain)} for predictions scored by this digit."""
        self.buf.append(int(d))
        t = self.n
        self.n += 1
        arr = np.fromiter(self.buf, np.int64, len(self.buf))
        if self.fast:
            return self._step_all(arr, t)
        out = {}
        for h, st in self.H.items():
            s = st.step(arr, t)
            if s is not None:
                out[h] = s
        return out

    # ---- vectorised over horizons -------------------------------------------------------------------------------
    def _stack(self) -> None:
        """Move the per-horizon arrays into stacked arrays (H, ...) and rebind each _HorizonState's attributes to views of them."""
        H = self.hmax
        hs = list(range(1, H + 1))
        self._hs = np.array(hs)
        self._cF = np.stack([self.H[h].cF for h in hs])
        self._cD = np.stack([self.H[h].cD for h in hs])
        self._TC = {k: np.stack([self.H[h].TC[k] for h in hs]) for k in (1, 2, 3, 4)}
        self._TS = np.stack([self.H[h].TS for h in hs])
        self._Tpar = np.stack([self.H[h].Tpar for h in hs])
        self._r = np.stack([self.H[h].r for h in hs])
        for i, h in enumerate(hs):
            st = self.H[h]
            st.cF, st.cD, st.TS, st.Tpar, st.r = self._cF[i], self._cD[i], self._TS[i], self._Tpar[i], self._r[i]
            st.TC = {k: self._TC[k][i] for k in (1, 2, 3, 4)}
        self._HH, self._LL = np.meshgrid(np.arange(H), np.arange(NLAG), indexing="ij")          # (H, NLAG)
        self._lag = np.arange(NLAG)
        self._stacked = True

    def _step_all(self, buf: np.ndarray, t: int) -> Dict[int, tuple]:
        if not self._stacked:
            self._stack()
        H, hs = self.hmax, self._hs
        cfg = self.cfg
        L = len(buf)
        tail = buf[::-1]                                              # tail[m] = D[t-m]
        Lc = L - 1
        y = int(buf[-1])
        seg = self.H[1].seg_t
        out: Dict[int, tuple] = {}
        # ---- score the predictions that target tick t ----
        rows, pcols, ens, pmys, pranks = [], [], [], [], []
        for h in range(1, H + 1):
            st = self.H[h]
            while st.pend and st.pend[0][0] <= t:
                tgt, P, en, pmix, prank = st.pend.popleft()
                if tgt == t:
                    rows.append(h - 1)
                    pcols.append(P[:, y])
                    ens.append(en)
                    pmys.append(float(pmix[y]))
                    pranks.append(prank)
        if rows:
            lg = np.log(np.maximum(np.array(pcols), 1e-300)) - LN01                  # (k, E)
            self._r[rows] += np.where(np.array(ens), lg, 0.0)
            pr = np.array(pranks)
            top = np.argsort(-pr, axis=1, kind="stable")[:, :4]
            hit = (top == y).any(axis=1)
            p4 = np.take_along_axis(pr, top, axis=1).sum(axis=1)
            for j, hi in enumerate(rows):
                st = self.H[hi + 1]
                g = math.log(pmys[j]) - LN01
                st.log_e += g
                st.n_scored += 1
                st.hit4 += int(hit[j])
                st.ll_gain += g
                out[hi + 1] = (float(p4[j]), int(hit[j]), g)
        # ---- update the counts ----
        n = t + 1
        self._cF[:, y] += 1.0
        hv = (t - hs) >= seg                                        # horizons whose (t-h)-th digit lies in the current segment
        if hv.any():
            ih = np.nonzero(hv)[0]
            a = tail[np.minimum(hs[ih], Lc)]
            self._cD[ih, (y - a) % 10] += 1.0
            valid = (t - hs[:, None] - self._lag[None, :]) >= seg                      # (H, NLAG)
            A = tail[np.minimum(hs[:, None] + self._lag[None, :], Lc)]
            m = valid
            self._TS[self._HH[m], self._LL[m], A[m], y] += 1.0
            c = np.zeros(H, np.int64)
            for k in (1, 2, 3, 4):
                c = c + tail[np.minimum(hs + (k - 1), Lc)] * POW10[k - 1]
                ok = (t - hs - k + 1) >= seg
                if ok.any():
                    ik = np.nonzero(ok)[0]
                    self._TC[k][ik, c[ik], y] += 1.0
            cp = np.zeros(H, np.int64)
            for i in range(PAR_CTX):
                cp = cp + ((tail[np.minimum(hs + i, Lc)] & 1) << i)
            ok = (t - hs - PAR_CTX + 1) >= seg
            if ok.any():
                ip = np.nonzero(ok)[0]
                self._Tpar[ip, cp[ip], y & 1] += 1.0
        # ---- predict ----
        P = np.empty((H, E, 10))
        en = np.zeros((H, E), bool)
        P[:, I_U] = 0.1
        en[:, I_U] = True
        PF = (self._cF + A_F) / (n + 10 * A_F)
        P[:, I_F] = PF
        en[:, I_F] = True
        chain = PF
        cq = 0
        for k in (1, 2, 3, 4):
            cq += int(tail[min(k - 1, Lc)]) * POW10[k - 1] if (k - 1) <= Lc else 0
            cnt = self._TC[k][:, cq, :]
            chain = (cnt + cfg["beta_c"] * chain) / (cnt.sum(axis=1, keepdims=True) + cfg["beta_c"])
            P[:, I_C + k - 1] = chain
            en[:, I_C + k - 1] = (n >= cfg["ctx_min"][k]) and (t - k + 1 >= seg)
        pmD = (self._cD + A_F) / (self._cD.sum(axis=1, keepdims=True) + 10 * A_F)
        P[:, I_D] = pmD[:, (_AR10 - y) % 10]
        en[:, I_D] = True
        lag = self._lag
        avail = (t - lag) >= seg
        cnt = self._TS[:, lag, tail[np.minimum(lag, Lc)]]                                # (H, NLAG, 10)
        P[:, I_S:I_S + NLAG] = (cnt + cfg["beta_s"] * PF[:, None, :]) / (cnt.sum(axis=2, keepdims=True) + cfg["beta_s"])
        en[:, I_S:I_S + NLAG] = avail[None, :]
        c = 0
        for i in range(PAR_CTX):
            c += (int(tail[min(i, Lc)]) & 1) << i
        tp = self._Tpar[:, c, :]                                                         # (H, 2)
        p1 = (tp[:, 1] + 0.5) / (tp.sum(axis=1) + 1.0)
        pb = np.empty((H, 10))
        for par_v in (0, 1):
            cols = _AR10[par_v::2]
            cn = self._cF[:, cols]
            pb[:, cols] = (p1 if par_v else 1.0 - p1)[:, None] * (cn + 0.5) / (cn.sum(axis=1, keepdims=True) + 2.5)
        P[:, I_B] = pb
        en[:, I_B] = (t - PAR_CTX + 1) >= seg
        logw = np.where(en, LOGPRIOR[None, :] + self._r, -np.inf)
        w = np.exp(logw - logw.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        pm = np.maximum(np.einsum("he,hek->hk", w, P), UNIFORM_FLOOR)
        pm /= pm.sum(axis=1, keepdims=True)
        logw[:, I_U] = -np.inf
        w2 = np.exp(logw - logw.max(axis=1, keepdims=True))
        w2 /= w2.sum(axis=1, keepdims=True)
        prk = np.einsum("he,hek->hk", w2, P)
        for i in range(H):
            st = self.H[i + 1]
            st.pend.append((t + i + 1, P[i], en[i], pm[i], prk[i]))
            st.cur = pm[i]
            st.cur_alt = prk[i]
        return out

    def gap(self) -> None:
        """Ticks were missed: index-based features must not span the gap. Pending predictions are dropped (their target index is
        no longer known), the digit buffer is cleared and every lag/context only uses digits after the gap. Counts are kept."""
        self.buf.clear()
        for st in self.H.values():
            st.pend.clear()
            st.seg_t = self.n
            st.cur = None
            st.cur_alt = None

    def dist(self, h: int) -> Optional[np.ndarray]:
        return self.H[h].cur

    def dist_alt(self, h: int) -> Optional[np.ndarray]:
        """Ranking distribution (null expert removed) used for top-4 and p4_raw."""
        return self.H[h].cur_alt

    def log_e(self, h: int) -> float:
        return self.H[h].log_e

    def load(self, D: np.ndarray, hs: Optional[Sequence[int]] = None) -> None:
        """Replace the state by a batch fit of digits D (equivalent to feeding them one by one). Caller then keeps calling on_digit."""
        D = np.asarray(D, np.int64)
        n = len(D)
        self.reset()
        if n < 100:                                      # too short for the batch path: replay online
            for x in D:
                self.on_digit(int(x))
            return
        self.n = n
        self.buf.extend(int(x) for x in D[-64:])
        for h in (hs or list(self.H)):
            st = self.H[h]
            res = batch_horizon(D, h, self.cfg, keep_last=h)
            j = np.arange(n)
            st.cF = np.bincount(D, minlength=10).astype(np.float64)
            for k in (1, 2, 3, 4):
                ev = j[j >= h + k - 1]
                cq = _ctx_index(D, k)
                np.add.at(st.TC[k], (cq[ev - h], D[ev]), 1.0)
            ev = j[j >= h]
            np.add.at(st.cD, (D[ev] - D[ev - h]) % 10, 1.0)
            for i in range(NLAG):
                ev = j[j >= h + i]
                np.add.at(st.TS[i], (D[ev - h - i], D[ev]), 1.0)
            par = D & 1
            cq = np.zeros(n, np.int64)
            for i in range(PAR_CTX):
                cq += par[np.maximum(j - i, 0)] << i
            ev = j[j >= h + PAR_CTX - 1]
            np.add.at(st.Tpar, (cq[ev - h], par[ev]), 1.0)
            st.r = res["r_final"].copy()
            st.log_e = res["log_e"]
            pm = res["pmix"]
            pk = res["prank"]
            st.n_scored = n - h
            tgt = D[h:]
            q = np.arange(0, n - h)
            p = pk[q]
            top = np.argsort(-p, axis=1, kind="stable")[:, :4]
            st.hit4 = int((top == tgt[:, None]).any(axis=1).sum())
            st.ll_gain = res["log_e"]
            for off in range(h):                              # predictions whose target is still ahead: made at t = n-h .. n-1
                tq = n - h + off
                P = np.empty((E, 10))
                en = np.zeros(E, bool)
                for m in range(E):
                    P[m] = res["last_P"][m][off]
                    en[m] = res["enabled"][m, tq]
                st.pend.append((tq + h, P, en, pm[tq].copy(), pk[tq].copy()))   # copies: views would pin the whole (n, 10) batch arrays
            st.cur = pm[n - 1].copy()
            st.cur_alt = pk[n - 1].copy()
        self._loaded_h = list(hs or list(self.H))
        self._stacked = False
