"""rng_eval.ci - confidence intervals and power helpers (numpy/math only)."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_sf(x: float) -> float:
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Inverse normal cdf (Acklam rational approximation + one Halley step, |err| ~ 1e-15)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    e = norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def wilson(k: float, n: float, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    den = 1 + z2 / n
    ctr = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return (max(0.0, ctr - half), min(1.0, ctr + half))


def wilson_lower(k: float, n: float, z: float = 1.64) -> float:
    return wilson(k, n, z)[0]


def sequential_wilson(k: float, n: float, alpha: float = 0.05) -> Tuple[float, float]:
    """Anytime-valid-ish interval: Wilson at the spending level alpha_n = alpha / (n (n+1)) (union over all n).
    Conservative; used so a curve of intervals over n has joint coverage >= 1 - alpha."""
    if n <= 0:
        return (0.0, 1.0)
    a_n = alpha / (n * (n + 1.0))
    return wilson(k, n, norm_ppf(1 - a_n / 2.0))


def required_n(delta: float, alpha: float = 0.05, power: float = 0.8, p0: float = 0.4, z_alpha: float = None) -> int:
    """Sample size to detect hit rate p0+delta against p0 with a one-sided test.
    z_alpha overrides alpha (e.g. z_alpha=5 for a '5 sigma' bar). power=0.5 gives z_beta=0.
    required_n(0.01, z_alpha=5, power=0.5) = 60,000 (the brief's figure)."""
    za = z_alpha if z_alpha is not None else norm_ppf(1 - alpha)
    zb = norm_ppf(power)
    p1 = p0 + delta
    n = ((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / delta) ** 2
    return int(math.ceil(n))


def binom_z(k: float, n: float, p0: float) -> float:
    return (k - n * p0) / math.sqrt(n * p0 * (1 - p0)) if n > 0 else 0.0
