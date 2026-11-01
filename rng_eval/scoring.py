"""rng_eval.scoring - prequential scoring: top-1 / top-4 hit rate, log-loss, Brier, reliability, risk-coverage."""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from . import ci

LN10 = math.log(10.0)


def top4_mask(P: np.ndarray) -> np.ndarray:
    """(m,10) bool: the 4 highest entries of each row, ties to the lower digit (stable argsort of -P)."""
    order = np.argsort(-P, axis=1, kind="stable")[:, :4]
    mask = np.zeros(P.shape, bool)
    np.put_along_axis(mask, order, True, axis=1)
    return mask


def score(P: np.ndarray, y: np.ndarray, conf: Optional[np.ndarray] = None, n_bins: int = 10) -> Dict:
    """P: (m,10) predictive distributions made BEFORE each outcome; y: (m,) outcomes. conf: optional confidence per row
    (default: the raw top-4 mass) used for the reliability curve and risk-coverage."""
    P = np.asarray(P, np.float64)
    y = np.asarray(y, np.int64)
    m = len(y)
    if m == 0:
        return {"n": 0}
    mask = top4_mask(P)
    hit4 = mask[np.arange(m), y]
    hit1 = P.argmax(axis=1) == y
    py = np.maximum(P[np.arange(m), y], 1e-12)
    ll = float(-np.log(py).mean())
    oh = np.zeros_like(P)
    oh[np.arange(m), y] = 1.0
    brier = float(((P - oh) ** 2).sum(axis=1).mean())
    p4 = (P * mask).sum(axis=1)
    c = p4 if conf is None else np.asarray(conf, np.float64)
    out = {"n": m, "top1": float(hit1.mean()), "top4": float(hit4.mean()), "logloss": ll, "ll_gain": LN10 - ll, "brier": brier,
           "top4_ci": ci.wilson(hit4.sum(), m), "top4_z": ci.binom_z(hit4.sum(), m, 0.4), "mean_p4": float(p4.mean())}
    out["reliability"] = reliability(c, hit4, n_bins)
    out["risk_coverage"] = risk_coverage(c, hit4)
    return out


def reliability(conf: np.ndarray, hit: np.ndarray, n_bins: int = 10):
    """Equal-count bins of the confidence: [(mean_conf, observed_rate, n)]."""
    m = len(conf)
    if m < n_bins * 5:
        return []
    order = np.argsort(conf, kind="stable")
    res = []
    for idx in np.array_split(order, n_bins):
        res.append((float(conf[idx].mean()), float(hit[idx].mean()), int(len(idx))))
    return res


def risk_coverage(conf: np.ndarray, hit: np.ndarray, covs=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)):
    """Accept the top `coverage` share of predictions by confidence; report the hit rate of the accepted ones."""
    m = len(conf)
    order = np.argsort(-conf, kind="stable")
    h = hit[order]
    cs = np.cumsum(h)
    res = []
    for c in covs:
        k = max(1, int(round(c * m)))
        res.append((c, float(cs[k - 1] / k), k))
    return res


def prequential_dists(digits: np.ndarray, predict_fn, h: int = 1, start: int = 0):
    """Drive any predictor: predict_fn(history_array, h) -> dist(10) made from digits[:t+1], scored on digits[t+h].
    Plain O(n * cost) loop used only for small unit tests and oracle checks."""
    P, Y = [], []
    for t in range(start, len(digits) - h):
        P.append(predict_fn(digits[:t + 1], h))
        Y.append(digits[t + h])
    return np.array(P), np.array(Y)
