"""
stream_diag.py - rolling diagnostics of the digit streams themselves (pure python/numpy, O(1) per tick).

Per symbol:
  * chi-square of the digit frequencies per block of TEST_DIAG_BLOCK digits (df=9) and the share of blocks with p<0.05 (expect 5%)
  * block-to-block persistence: Pearson correlation of consecutive blocks' digit-frequency deviations (expect ~0)
  * serial correlation of the digit sequence at lags 1..3 (z = r*sqrt(n))
  * two-sided CUSUM (in sigma units) of the OVER5-zone indicator (digit>=6) against its fair rate 0.4
"""
import math
from collections import deque
from typing import Dict

import numpy as np

CUSUM_K = 0.5       # slack, sigma units
CUSUM_H = 5.0       # alarm level, sigma units
P_OVER = 0.4
SIGMA = math.sqrt(P_OVER * (1 - P_OVER))


def _gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) (Numerical Recipes series / continued fraction)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(500):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - gln) * h)


def chi2_sf(x: float, df: int) -> float:
    return _gammaincc(df / 2.0, x / 2.0)


class SymDiag:
    def __init__(self, block: int):
        self.block = max(50, int(block))
        self.counts = np.zeros(10)
        self.nb = 0
        self.prev_dev = None
        self.pvals: deque = deque(maxlen=200)
        self.corrs: deque = deque(maxlen=200)
        self.n = 0
        self.hist: deque = deque(maxlen=3)                # last 3 digits
        self.sx = self.sxx = 0.0
        self.sxy = [0.0, 0.0, 0.0]                         # lag-1..3 cross sums
        self.nxy = [0, 0, 0]
        self.cp = self.cn = 0.0                            # CUSUM+ / CUSUM-
        self.cmax = 0.0
        self.alarms = 0

    def add(self, d: int) -> None:
        self.n += 1
        self.sx += d
        self.sxx += d * d
        for lag in range(1, 4):                            # hist[-lag] is the digit `lag` ticks back
            if len(self.hist) >= lag:
                self.sxy[lag - 1] += d * self.hist[-lag]
                self.nxy[lag - 1] += 1
        self.hist.append(d)
        z = ((1.0 if d >= 6 else 0.0) - P_OVER) / SIGMA
        self.cp = max(0.0, self.cp + z - CUSUM_K)
        self.cn = max(0.0, self.cn - z - CUSUM_K)
        self.cmax = max(self.cmax, self.cp, self.cn)
        if self.cp > CUSUM_H or self.cn > CUSUM_H:
            self.alarms += 1
            self.cp = self.cn = 0.0                        # restart after an alarm
        self.counts[d] += 1
        if self.counts.sum() >= self.block:
            exp = self.block / 10.0
            chi = float(((self.counts - exp) ** 2 / exp).sum())
            self.pvals.append(chi2_sf(chi, 9))
            dev = self.counts - exp
            if self.prev_dev is not None and dev.std() > 0 and self.prev_dev.std() > 0:
                self.corrs.append(float(np.corrcoef(dev, self.prev_dev)[0, 1]))
            self.prev_dev = dev
            self.nb += 1
            self.counts = np.zeros(10)

    def serial(self) -> list:
        n = self.n
        if n < 50:
            return []
        mean = self.sx / n
        var = self.sxx / n - mean * mean
        out = []
        for lag in range(3):
            m = self.nxy[lag]
            if var <= 0 or m < 50:
                out.append(None)
                continue
            r = (self.sxy[lag] / m - mean * mean) / var
            out.append({"lag": lag + 1, "r": round(r, 5), "z": round(r * math.sqrt(m), 2)})
        return out

    def snapshot(self) -> dict:
        pv = list(self.pvals)
        cr = list(self.corrs)
        persist = None
        if len(cr) >= 3:
            m, sd = float(np.mean(cr)), float(np.std(cr, ddof=1))
            persist = {"mean_r": round(m, 4), "t": round(m / (sd / math.sqrt(len(cr))), 2) if sd > 0 else None, "blocks": len(cr)}
        return {"n": self.n, "blocks": self.nb,
                "chi2_p_last": round(pv[-1], 4) if pv else None,
                "chi2_frac_p05": round(sum(1 for p in pv if p < 0.05) / len(pv), 3) if pv else None,
                "persistence": persist, "serial": self.serial(),
                "cusum": {"pos": round(self.cp, 2), "neg": round(self.cn, 2), "max": round(self.cmax, 2), "alarms": self.alarms}}


class StreamDiag:
    def __init__(self, block: int = 1000):
        self.block = block
        self.sym: Dict[str, SymDiag] = {}

    def add(self, sym: str, digit: int) -> None:
        d = self.sym.get(sym)
        if d is None:
            d = self.sym[sym] = SymDiag(self.block)
        d.add(int(digit))

    def add_many(self, sym: str, digits) -> None:
        for x in digits:
            self.add(sym, int(x))

    def snapshot(self) -> dict:
        per = {s: d.snapshot() for s, d in self.sym.items()}
        pvals = [p for d in self.sym.values() for p in d.pvals]
        return {"per_symbol": per,
                "pooled": {"blocks": len(pvals),
                           "frac_p05": round(sum(1 for p in pvals if p < 0.05) / len(pvals), 3) if pvals else None,
                           "cusum_alarms": sum(d.alarms for d in self.sym.values())}}
