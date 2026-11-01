"""rng_core.mathutil - the few special functions the live code needs without scipy."""
from __future__ import annotations

import math
from typing import Tuple


def norm_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _gser(a: float, x: float) -> float:
    ap, s, d = a, 1.0 / a, 1.0 / a
    for _ in range(2000):
        ap += 1.0
        d *= x / ap
        s += d
        if abs(d) < abs(s) * 1e-16:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a: float, x: float) -> float:
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gammaq(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x)."""
    if x <= 0:
        return 1.0
    if x < a + 1.0:
        return max(0.0, 1.0 - _gser(a, x))
    return min(1.0, _gcf(a, x))


def chi2_sf(x: float, df: float) -> float:
    """P(chi2_df >= x); accurate far into the tail (continued fraction), so tiny p-values are not rounded to 0 early."""
    if df <= 0:
        return 1.0
    return gammaq(df / 2.0, x / 2.0)


def wilson(k: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    den = 1 + z2 / n
    ctr = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(max(p * (1 - p) / n + z2 / (4 * n * n), 0.0)) / den
    return (max(0.0, ctr - half), min(1.0, ctr + half))


def kolmogorov_sf(x: float) -> float:
    """P(sup |Brownian bridge| >= x)."""
    if x <= 0:
        return 1.0
    if x < 0.2:
        return 1.0
    s = 0.0
    for j in range(1, 101):
        t = 2.0 * (-1) ** (j - 1) * math.exp(-2.0 * j * j * x * x)
        s += t
        if abs(t) < 1e-18:
            break
    return min(1.0, max(0.0, s))
