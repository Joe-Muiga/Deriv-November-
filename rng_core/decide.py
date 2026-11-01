"""rng_core.decide - Layer 4: top-4 selection and calibration of 'the true digit is in top4'.

top4 = the 4 highest entries of dist (deterministic tie-break to the lower digit): the exact Bayes optimum for top-4
accuracy. No contiguity constraint, no lag mixing, no cover logic. Each horizon is predicted independently.

Calibration is a Beta-binomial reliability map on prequential (p4_raw, hit) pairs, POOLED across streams of the same status,
with a prior of strength PRIOR_N (default 200) centred on 0.40: with no evidence the calibrated probability IS 0.40. Bins are
monotone (pool-adjacent-violators). p4_lb is a Wilson lower bound on the pseudo-count posterior.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import mathutil as mu
from .types import (EXACT_STATE, NO_STRUCTURE, STAT_SIGNAL, STATUSES, Prediction, finalize_dist, top4_of)

P_FAIR = 0.40
PRIOR_N = 200.0
# bins on the raw top-4 mass p4_raw (upper edges); a near-uniform mixture lives in the first few bins
BIN_EDGES = np.array([0.4005, 0.401, 0.402, 0.404, 0.407, 0.41, 0.415, 0.42, 0.43, 0.45, 0.5, 0.6, 0.8, 0.95, 0.995, 1.0 + 1e-9])


class Top4Calibrator:
    """Pooled reliability map per status. add() after a prediction is scored; map() before publishing."""

    def __init__(self, prior_n: float = PRIOR_N, center: float = P_FAIR, z_lb: float = 1.64):
        self.prior_n, self.center, self.z = prior_n, center, z_lb
        self.n: Dict[str, np.ndarray] = {s: np.zeros(len(BIN_EDGES)) for s in STATUSES}
        self.k: Dict[str, np.ndarray] = {s: np.zeros(len(BIN_EDGES)) for s in STATUSES}

    @staticmethod
    def _bin(p4: float) -> int:
        return int(min(np.searchsorted(BIN_EDGES, p4, side="left"), len(BIN_EDGES) - 1))

    def add(self, status: str, p4_raw: float, hit: int) -> None:
        b = self._bin(p4_raw)
        self.n[status][b] += 1.0
        self.k[status][b] += float(hit)

    def _post(self, status: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, k = self.n[status], self.k[status]
        pn = n + self.prior_n
        pk = k + self.prior_n * self.center
        mean = pk / pn
        # pool adjacent violators over the bins that HAVE data (n > 0): a bin without evidence keeps the prior mean (the fair 0.40)
        # and is neither lifted by lower bins nor allowed to drag observed bins down (conservative, never optimistic)
        out_m, out_n = list(mean), list(pn)
        obs = [i for i in range(len(n)) if n[i] > 0]
        blocks: List[List] = []                        # [value*weight, weight, [bin indices]]
        for i in obs:
            blocks.append([mean[i] * pn[i], pn[i], [i]])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                a = blocks.pop()
                blocks[-1][0] += a[0]
                blocks[-1][1] += a[1]
                blocks[-1][2] += a[2]
        for bv, bw, idx in blocks:
            for i in idx:
                out_m[i], out_n[i] = bv / bw, bw
        return np.array(out_m), np.array(out_n), pk

    def map(self, status: str, p4_raw: float) -> Tuple[float, float]:
        """(p4_cal, p4_lb) for a raw top-4 mass."""
        mean, nn, _ = self._post(status)
        b = self._bin(p4_raw)
        m, nb = float(mean[b]), float(nn[b])
        return m, mu.wilson(m * nb, nb, self.z)[0]

    def reliability(self, status: str) -> List[Tuple[float, float, int]]:
        """[(bin upper edge, observed hit rate, n)] for bins with data."""
        n, k = self.n[status], self.k[status]
        return [(float(BIN_EDGES[i]), float(k[i] / n[i]), int(n[i])) for i in range(len(n)) if n[i] > 0]


def make_prediction(symbol: str, target_idx: int, h: int, dist, status: str, log_e: float, n_scored: int, source: str,
                    cal: Optional[Top4Calibrator] = None, exact_digit: Optional[int] = None, exact_mass: float = 0.999,
                    dist_alt=None) -> Prediction:
    """Build the contract object. For EXACT_STATE the distribution puts exact_mass on the solver's digit and spreads the rest
    in the order of `dist` (so the other 3 top4 slots are filled by the statistical model's ranking)."""
    p = np.asarray(dist, np.float64)
    if status == EXACT_STATE and exact_digit is not None:
        rest = p.copy()
        rest[exact_digit] = 0.0
        rest = rest / rest.sum() if rest.sum() > 0 else np.full(10, 1.0 / 9)
        rest[exact_digit] = 0.0
        p = (1.0 - exact_mass) * rest
        p[exact_digit] = exact_mass
    p = finalize_dist(p)
    alt = finalize_dist(dist_alt) if (dist_alt is not None and status != EXACT_STATE) else None
    rank = alt if alt is not None else p              # top4 / p4_raw come from the ranking distribution when there is one
    t4 = top4_of(rank)
    raw = float(sum(rank[d] for d in t4))
    if cal is not None:
        c, lb = cal.map(status, raw)
    else:
        c, lb = P_FAIR, mu.wilson(P_FAIR * PRIOR_N, PRIOR_N, 1.64)[0]
    return Prediction(symbol=symbol, target_idx=int(target_idx), h=int(h), dist=[float(x) for x in p], top4=t4, p4_raw=raw,
                      p4_cal=float(c), p4_lb=float(lb), status=status, log_e=float(log_e), n_scored=int(n_scored), source=source,
                      dist_alt=None if alt is None else [float(x) for x in alt])
