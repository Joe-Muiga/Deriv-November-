"""rng_core.diag - Layer 1: structure diagnostics.

Each test returns a TestResult and declares its minimum n; below it the result has ok=False and p=NaN ('insufficient',
never 'pass'). run_diagnostics() runs every test on the digit stream (and on observation-map streams when quotes are
given), applies Holm within the run and alpha-spending across runs (alpha_j = alpha / (j (j+1))), and returns a
StructureReport with a verdict and recommended solver families. Diagnostics are meant to run on a geometric schedule
(5k, 10k, 20k, ... digits), not on a timer: see geometric_schedule().

Two p-value conventions worth knowing:
  * 'MC-normal' p-values (serial tests below 10x cells, compressibility): the statistic is recomputed on B shuffled
    copies; p comes from the normal tail of (obs - mean_shuffles) / sd_shuffles. A plain rank p-value could never go below
    1/(B+1), which is useless once Holm and alpha-spending push the threshold to ~1e-4. The normal approximation is
    checked against null streams in the harness (rng_eval/run_diag_null.py), not assumed.
  * Maurer uses sd = sigma/sqrt(K) (Maurer's c(L,K) < 1 correction is not applied), which is conservative.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import mathutil as mu
from .exact import gf2
from .observe import OBS_MAPS
from .types import (StructureReport, TestResult, V_BIAS, V_CHANGEPOINT, V_LINEAR, V_MIXED, V_NONE, V_PERIODIC, V_REPEAT,
                    V_SERIAL)

NAN = float("nan")


def alpha_run(alpha: float, j: int) -> float:
    """Alpha spent on diagnostic run number j (j >= 1); sum over all runs <= alpha."""
    return alpha / (j * (j + 1.0))


def geometric_schedule(first: int = 5000, factor: float = 2.0, limit: int = 10_000_000) -> List[int]:
    out, n = [], first
    while n <= limit:
        out.append(int(n))
        n = int(n * factor)
    return out


def _ok(name, stream, stat, p, effect, n, note="") -> TestResult:
    return TestResult(name, stream, float(stat), float(min(1.0, max(p, 1e-300))), float(effect), int(n), True, note)


def _insufficient(name, stream, n, need) -> TestResult:
    return TestResult(name, stream, NAN, NAN, NAN, int(n), False, f"insufficient: need n>={need}")


def _unavailable(name, stream, n, why) -> TestResult:
    return TestResult(name, stream, NAN, NAN, NAN, int(n), False, f"unavailable: {why}")


# ── bit streams ──────────────────────────────────────────────────────────────────────────────────
def bit_streams(d: np.ndarray) -> Dict[str, np.ndarray]:
    """parity bit, the top-bit proxy (digit >= 5) and the bit planes of the 4-bit digit encoding."""
    d = d.astype(np.int64)
    return {"parity": d & 1, "ge5": (d >= 5).astype(np.int64), "bit1": (d >> 1) & 1, "bit2": (d >> 2) & 1, "bit3": (d >> 3) & 1}


# ── frequency / serial ───────────────────────────────────────────────────────────────────────────
def t_freq(d: np.ndarray, A: int = 10, stream: str = "digit") -> TestResult:
    n = len(d)
    if n < 10 * A:
        return _insufficient("freq_chi2", stream, n, 10 * A)
    c = np.bincount(d, minlength=A)[:A]
    e = n / A
    x = float(((c - e) ** 2 / e).sum())
    return _ok("freq_chi2", stream, x, mu.chi2_sf(x, A - 1), float(np.abs(c / n - 1.0 / A).max()), n)


def _kgram_counts(d: np.ndarray, k: int, A: int) -> np.ndarray:
    m = len(d) - k + 1
    idx = np.zeros(m, np.int64)
    for j in range(k):
        idx = idx * A + d[j:m + j]
    return np.bincount(idx, minlength=A ** k)


def _psi2(d: np.ndarray, k: int, A: int) -> float:
    if k == 0:
        return 0.0
    m = len(d) - k + 1
    c = _kgram_counts(d, k, A).astype(np.float64)
    return (A ** k) / m * float((c * c).sum()) - m


def _good(d: np.ndarray, k: int, A: int) -> float:
    return _psi2(d, k, A) - _psi2(d, k - 1, A)


def t_serial(d: np.ndarray, k: int, A: int = 10, stream: str = "digit", B: int = 99, seed: int = 0) -> TestResult:
    """Good's serial test on overlapping k-grams: delta-psi^2_k ~ chi2(A^k - A^(k-1)). Needs n >= 10 A^k for the asymptotic
    law; below that the null is estimated from B shuffles (MC-normal). Minimum n = 1000."""
    n = len(d)
    name = f"serial{k}"
    if n < 1000:
        return _insufficient(name, stream, n, 1000)
    obs = _good(d, k, A)
    df = A ** k - A ** (k - 1)
    if n >= 10 * A ** k:
        return _ok(name, stream, obs, mu.chi2_sf(obs, df), obs / n, n, "chi2")
    rng = np.random.default_rng(seed + 17 * k)
    sims = np.array([_good(rng.permutation(d), k, A) for _ in range(B)])
    sd = float(sims.std(ddof=1)) or 1.0
    z = (obs - float(sims.mean())) / sd
    return _ok(name, stream, z, mu.norm_sf(z), (obs - float(sims.mean())) / n, n, f"mc-normal B={B}")


def t_indep(d: np.ndarray, A: int, stream: str) -> TestResult:
    """Lag-1 chi-square test of independence on the A x A table of consecutive pairs (expected = row*col/n), so a non-uniform
    marginal does not matter. Used for observation-map streams. Reported under the name 'serial2'."""
    n = len(d)
    if n < 1000:
        return _insufficient("serial2", stream, n, 1000)
    T = np.zeros((A, A))
    np.add.at(T, (d[:-1], d[1:]), 1.0)
    m = T.sum()
    r, c = T.sum(1), T.sum(0)
    keep_r, keep_c = r > 0, c > 0
    E = np.outer(r, c) / m
    mask = np.outer(keep_r, keep_c) & (E > 0)
    x2 = float((((T - E) ** 2)[mask] / E[mask]).sum())
    df = (int(keep_r.sum()) - 1) * (int(keep_c.sum()) - 1)
    if df <= 0:
        return _ok("serial2", stream, 0.0, 1.0, 0.0, n, "degenerate")
    return _ok("serial2", stream, x2, mu.chi2_sf(x2, df), x2 / m, n, f"chi2 indep df={df}")


# ── autocorrelation / Ljung-Box ──────────────────────────────────────────────────────────────────
def _lb(x: np.ndarray, L: int) -> Tuple[float, float, float]:
    n = len(x)
    x = x - x.mean()
    den = float((x * x).sum())
    if den <= 0:
        return 0.0, 1.0, 0.0
    r = np.array([float(np.dot(x[:-k], x[k:])) / den for k in range(1, L + 1)])
    q = n * (n + 2.0) * float((r ** 2 / (n - np.arange(1, L + 1))).sum())
    return q, mu.chi2_sf(q, L), float(np.abs(r).max())


def t_ljungbox(d: np.ndarray, L: int = 64, stream: str = "digit") -> List[TestResult]:
    n = len(d)
    L = min(L, n // 10)
    if n < 500 or L < 5:
        return [_insufficient("ljungbox_digits", stream, n, 500), _insufficient("ljungbox_indicators", stream, n, 500)]
    q, p, eff = _lb(d.astype(np.float64), L)
    res = [_ok("ljungbox_digits", stream, q, p, eff, n, f"lags 1..{L}")]
    ps, effs = [], []
    for c in range(10):
        qc, pc, ec = _lb((d == c).astype(np.float64), L)
        ps.append(pc)
        effs.append(ec)
    pmin = min(ps)
    res.append(_ok("ljungbox_indicators", stream, float(np.argmin(ps)), 1.0 - (1.0 - pmin) ** 10, max(effs), n, f"min over 10 digits, lags 1..{L}"))
    return res


# ── spectral ─────────────────────────────────────────────────────────────────────────────────────
def t_dft(x: np.ndarray, name_stream: str, centered_var: Optional[float] = None) -> List[TestResult]:
    """x: +-1 (or centred) series. NIST peak-count test (with the Kim et al. variance) + max-peak test (|F_k|^2/n ~ Exp(1))."""
    n = len(x)
    if n < 1000:
        return [_insufficient("dft_count", name_stream, n, 1000), _insufficient("dft_peak", name_stream, n, 1000)]
    F = np.abs(np.fft.rfft(x))[1:n // 2 + 1]
    var = 1.0 if centered_var is None else centered_var
    T = math.sqrt(math.log(1 / 0.05) * n * var)
    N0, N1 = 0.95 * n / 2.0, float((F < T).sum())
    dz = (N1 - N0) / math.sqrt(n * 0.95 * 0.05 / 3.8)
    p1 = math.erfc(abs(dz) / math.sqrt(2.0))
    K = len(F)
    m = float((F ** 2).max() / (n * var))
    p2 = -math.expm1(-K * math.exp(-m)) if m < 700 else 0.0
    return [_ok("dft_count", name_stream, dz, p1, abs(dz) / math.sqrt(n), n), _ok("dft_peak", name_stream, m, p2, m, n, f"peak at bin {int(np.argmax(F)) + 1}")]


# ── runs / Maurer ────────────────────────────────────────────────────────────────────────────────
def t_runs(b: np.ndarray, stream: str) -> TestResult:
    n = len(b)
    if n < 100:
        return _insufficient("runs", stream, n, 100)
    n1 = int(b.sum())
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return _ok("runs", stream, 0.0, 1e-300, 1.0, n, "constant stream")
    runs = 1 + int((b[1:] != b[:-1]).sum())
    mean = 2.0 * n1 * n0 / n + 1.0
    var = (mean - 1.0) * (mean - 2.0) / (n - 1.0)
    z = (runs - mean) / math.sqrt(var) if var > 0 else 0.0
    return _ok("runs", stream, z, math.erfc(abs(z) / math.sqrt(2.0)), abs(runs - mean) / n, n, "conditional on observed proportion")


_MAURER: Dict[int, Tuple[float, float]] = {}


def _maurer_moments(M: int) -> Tuple[float, float]:
    if M not in _MAURER:
        p = 1.0 / M
        dd = np.arange(1, 40 * M)
        pr = p * (1 - p) ** (dd - 1)
        lg = np.log2(dd)
        e = float((pr * lg).sum())
        _MAURER[M] = (e, float((pr * lg * lg).sum()) - e * e)
    return _MAURER[M]


def t_maurer(d: np.ndarray, stream: str = "digit") -> TestResult:
    """Maurer universal test on non-overlapping digit pairs (alphabet 100), Q = 1000 initialisation blocks."""
    n = len(d)
    Mx, Q = 100, 1000
    nb = n // 2
    K = nb - Q
    if K < 2000:
        return _insufficient("maurer", stream, n, 2 * (Q + 2000))
    blk = (d[0:2 * nb:2].astype(np.int64) * 10 + d[1:2 * nb:2])
    last = np.full(Mx, -1, np.int64)
    s = 0.0
    cnt = 0
    for j in range(nb):                                         # pure python loop over blocks; nb <= ~50k
        b = blk[j]
        if j >= Q and last[b] >= 0:
            s += math.log2(j - last[b])
            cnt += 1
        last[b] = j
    if cnt < 1000:
        return _insufficient("maurer", stream, n, 2 * (Q + 2000))
    e, v = _maurer_moments(Mx)
    f = s / cnt
    z = (f - e) / math.sqrt(v / cnt)
    return _ok("maurer", stream, z, math.erfc(abs(z) / math.sqrt(2.0)), abs(f - e), n, "sd = sigma/sqrt(K) (conservative)")


# ── linear complexity / matrix rank ──────────────────────────────────────────────────────────────
def t_linear_complexity(b: np.ndarray, stream: str, max_bits: int = 40000) -> TestResult:
    n = min(len(b), max_bits)
    if n < 200:
        return _insufficient("linear_complexity", stream, len(b), 200)
    bits = b[:n].tolist()
    L, _ = gf2.berlekamp_massey(bits)
    p = gf2.lc_pvalue(L, n)
    return _ok("linear_complexity", stream, L, p, (n / 2.0 - L), n, f"L={L} vs n/2={n / 2:.0f}")


def _rank32(rows: List[int]) -> int:
    rank = 0
    rows = rows[:]
    for bit in range(32):
        piv = -1
        for i in range(rank, 32):
            if (rows[i] >> bit) & 1:
                piv = i
                break
        if piv < 0:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(32):
            if i != rank and (rows[i] >> bit) & 1:
                rows[i] ^= rows[rank]
        rank += 1
    return rank


def t_matrix_rank(b: np.ndarray, stream: str) -> TestResult:
    """Binary matrix rank test (32x32 matrices from consecutive 1024-bit blocks). Needs >= 38 matrices."""
    nm = len(b) // 1024
    if nm < 38:
        return _insufficient("matrix_rank", stream, len(b), 38 * 1024)
    cnt = [0, 0, 0]
    for m in range(nm):
        blk = b[m * 1024:(m + 1) * 1024].reshape(32, 32)
        rows = [int("".join("1" if v else "0" for v in r), 2) for r in blk]
        r = _rank32(rows)
        cnt[0 if r == 32 else (1 if r == 31 else 2)] += 1
    pr = (0.2888, 0.5776, 0.1336)
    x = sum((c - nm * q) ** 2 / (nm * q) for c, q in zip(cnt, pr))
    return _ok("matrix_rank", stream, x, math.exp(-x / 2.0), x / nm, len(b))


def t_lattice(n: int, stream: str = "quote") -> TestResult:
    return _unavailable("lattice", stream, n, "needs a high-resolution uniform-like stream; only digits/last-decimal available")


# ── repeats / period ─────────────────────────────────────────────────────────────────────────────
_HB = np.uint64(0x9E3779B97F4A7C15 | 1)
_HBINV = np.uint64(pow(int(_HB), -1, 1 << 64))


def _roll_hash(d: np.ndarray, w: int) -> np.ndarray:
    """Polynomial hash (mod 2^64, odd base) of every w-gram, O(n): window sums of c[k]*B^-k, rescaled by B^(i+w-1)."""
    n = len(d)
    c = d.astype(np.uint64) + np.uint64(1)
    with np.errstate(over="ignore"):
        powb = np.cumprod(np.concatenate([[np.uint64(1)], np.full(n - 1, _HB, np.uint64)]))        # B^k
        powi = np.cumprod(np.concatenate([[np.uint64(1)], np.full(n - 1, _HBINV, np.uint64)]))      # B^-k
        S = np.concatenate([[np.uint64(0)], np.cumsum(c * powi)])
        i = np.arange(n - w + 1)
        return (S[i + w] - S[i]) * powb[i + w - 1]


def _longest_repeat(d: np.ndarray) -> int:
    """Longest repeated substring length (exact; hash hits are verified)."""
    n = len(d)
    lo, w = 0, 1
    while w <= n - 1 and _dup_exists(d, w):
        lo = w
        w *= 2
    hi = min(w, n)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _dup_exists(d, mid):
            lo = mid
        else:
            hi = mid
    return lo


def _dup_exists(d: np.ndarray, w: int) -> bool:
    n = len(d)
    if w < 1 or n - w + 1 < 2:
        return False
    h = _roll_hash(d, w)
    order = np.argsort(h, kind="stable")
    hs = h[order]
    same = np.nonzero(hs[1:] == hs[:-1])[0]
    for k in same[:50]:                                  # verify real equality (hash collisions are astronomically rare)
        i, j = int(order[k]), int(order[k + 1])
        if np.array_equal(d[i:i + w], d[j:j + w]):
            return True
    return False


def t_repeat(d: np.ndarray, stream: str = "digit") -> TestResult:
    n = len(d)
    if n < 1000:
        return _insufficient("repeat", stream, n, 1000)
    L = _longest_repeat(d)
    m = n - L + 1
    lam = m * (m - 1) / 2.0 * 10.0 ** (-L)
    p = -math.expm1(-lam)                                # P(a repeat of length >= L exists) ~ 1 - exp(-C(m,2) 10^-L)  (clumping makes it conservative)
    return _ok("repeat", stream, L, p, L - 2 * math.log10(n / math.sqrt(2.0)), n, f"longest repeat {L}")


def t_period(d: np.ndarray, stream: str = "digit", max_lag: int = 20000) -> TestResult:
    """Match rate at every lag 1..max_lag via FFT autocorrelation of the 10 indicator series; max z, Sidak-corrected."""
    n = len(d)
    if n < 1000:
        return _insufficient("period", stream, n, 1000)
    max_lag = min(max_lag, n // 2)
    m = 1 << (2 * n - 1).bit_length()
    ac = np.zeros(max_lag + 1)
    for c in range(10):
        x = (d == c).astype(np.float64)
        F = np.fft.rfft(x, m)
        ac += np.fft.irfft(F * np.conj(F), m)[:max_lag + 1]
    lags = np.arange(1, max_lag + 1)
    cnt = n - lags
    z = (ac[1:] - cnt / 10.0) / np.sqrt(cnt * 0.09)
    k = int(np.argmax(z))
    zmax = float(z[k])
    p1 = mu.norm_sf(zmax)
    p = 1.0 - (1.0 - p1) ** max_lag if p1 < 1e-8 and max_lag * p1 < 1e-3 else min(1.0, max_lag * p1)
    frac = float(ac[k + 1] / cnt[k])
    return _ok("period", stream, zmax, p, frac, n, f"best lag {k + 1}, match fraction {frac:.3f}")


# ── mutual information ───────────────────────────────────────────────────────────────────────────
def t_mutual_info(d: np.ndarray, stream: str = "digit", max_lag: int = 64, A: int = 10, perm: int = 0, seed: int = 0) -> TestResult:
    """Miller-Madow bias-corrected MI at lags 1..max_lag. Null: G = 2 n I ~ chi2((A-1)^2) per lag (Sidak over lags); with
    perm > 0 the max-over-lags corrected MI is also compared with `perm` shuffles and the larger p is reported."""
    n = len(d)
    L = min(max_lag, n // 20)
    if n < 1000 or L < 2:
        return _insufficient("mutual_info", stream, n, 1000)
    df = (A - 1) ** 2

    def mi_all(x: np.ndarray):
        out = []
        for k in range(1, L + 1):
            joint = np.bincount(x[:-k] * A + x[k:], minlength=A * A).reshape(A, A).astype(np.float64)
            m = joint.sum()
            pxy = joint / m
            px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
            nz = pxy > 0
            out.append(float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum()))
        return np.array(out)

    mi = mi_all(d.astype(np.int64))
    corrected = mi - df / (2.0 * (n - np.arange(1, L + 1)))
    G = 2.0 * (n - np.arange(1, L + 1)) * mi
    pk = np.array([mu.chi2_sf(g, df) for g in G])
    k = int(np.argmin(pk))
    p = 1.0 - (1.0 - pk[k]) ** L if pk[k] > 1e-12 else min(1.0, L * pk[k])
    note = f"best lag {k + 1}, corrected MI {corrected[k]:.5f} nats"
    if perm > 0:
        rng = np.random.default_rng(seed)
        mx = np.array([mi_all(rng.permutation(d).astype(np.int64)).max() for _ in range(perm)])
        pperm = (1 + int((mx >= mi.max()).sum())) / (perm + 1.0)
        p = max(p, pperm)
        note += f"; perm p={pperm:.4f}"
    return _ok("mutual_info", stream, float(G[k]), p, float(corrected[k]), n, note)


# ── compressibility ──────────────────────────────────────────────────────────────────────────────
def _loss_order(d: np.ndarray, k: int) -> float:
    """Adaptive KT code length (nats) of d with a context of the previous k digits (alpha = 0.5)."""
    n = len(d)
    if k == 0:
        ctx = np.zeros(n, np.int64)
    else:
        ctx = np.zeros(n, np.int64)
        for j in range(1, k + 1):
            sh = np.zeros(n, np.int64)
            sh[j:] = d[:-j]
            ctx = ctx * 10 + sh
    key = ctx * 10 + d
    order = np.argsort(key, kind="stable")
    ks = key[order]
    start = np.concatenate([[True], ks[1:] != ks[:-1]])
    gid = np.cumsum(start) - 1
    pos = np.arange(n) - np.nonzero(start)[0][gid]
    cyx = np.empty(n)
    cyx[order] = pos
    order2 = np.argsort(ctx, kind="stable")
    cs = ctx[order2]
    start2 = np.concatenate([[True], cs[1:] != cs[:-1]])
    gid2 = np.cumsum(start2) - 1
    pos2 = np.arange(n) - np.nonzero(start2)[0][gid2]
    cx = np.empty(n)
    cx[order2] = pos2
    return float(-np.log((cyx + 0.5) / (cx + 5.0)).sum())


def bma_codelength(d: np.ndarray, orders: Sequence[int] = (0, 1, 2, 3)) -> float:
    """Code length (nats) of a Bayesian mixture over context orders (a valid prefix code, = CTW-style order averaging)."""
    tot = np.array([_loss_order(d, k) for k in orders])
    m = tot.min()
    return float(m - math.log(np.exp(-(tot - m)).sum() / len(orders)))


def t_compress(d: np.ndarray, stream: str = "digit", B: int = 200, seed: int = 0) -> TestResult:
    """Compressibility vs shuffled surrogates. For each context order k = 1..3 the statistic is the code-length saving of an
    adaptive order-k KT model over the memoryless code (negative under the null: it is the Occam penalty). The null mean/sd
    and skew come from B shuffles; z gets a Cornish-Fisher skew correction, then p = min(1, 2 * 3 * min_k p_k) (3 orders,
    factor 2 for estimating mean/sd from B shuffles). Calibrated against null streams in rng_eval/run_diag_null.py."""
    n = len(d)
    if n < 2000:
        return _insufficient("compress", stream, n, 2000)
    rng = np.random.default_rng(seed + 5)
    l0 = _loss_order(d, 0)                                    # permutation invariant
    obs = np.array([l0 - _loss_order(d, k) for k in (1, 2, 3)])
    sims = np.empty((B, 3))
    for i in range(B):
        sims[i] = [l0 - _loss_order(rng.permutation(d), k) for k in (1, 2, 3)]
        time.sleep(0.0005)                                    # hand the GIL back to the event loop between surrogates
    m, sd = sims.mean(axis=0), sims.std(axis=0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    skew = (((sims - m) ** 3).mean(axis=0)) / sd ** 3
    z = (obs - m) / sd
    zc = np.where(z > 1.0, z - skew / 6.0 * (z * z - 1.0), z)
    pk = np.array([mu.norm_sf(float(x)) for x in zc])
    kbest = int(np.argmin(pk))
    p = min(1.0, 6.0 * float(pk.min()))
    return _ok("compress", stream, float(z[kbest]), p, float((obs - m)[kbest] / n), n,
               f"order {kbest + 1}, extra saving {float((obs - m)[kbest]):.1f} nats over shuffles, B={B}")


# ── change points ────────────────────────────────────────────────────────────────────────────────
def t_changepoint(d: np.ndarray, stream: str = "digit", block: int = 500) -> Tuple[TestResult, Optional[int]]:
    """CUSUM (Brownian-bridge sup) over per-block mean and per-block chi-square; detects reseeds that change statistics."""
    n = len(d)
    m = n // block
    if m < 10:
        return _insufficient("changepoint", stream, n, 10 * block), None
    dd = d[:m * block].reshape(m, block)
    zm = (dd.mean(axis=1) - 4.5) / math.sqrt(8.25 / block)
    cnt = np.bincount((np.arange(m)[:, None] * 10 + dd).ravel(), minlength=m * 10).reshape(m, 10)
    chi = ((cnt - block / 10.0) ** 2 / (block / 10.0)).sum(axis=1)
    zc = (chi - 9.0) / math.sqrt(18.0)
    best_p, best_k, best_x = 1.0, None, 0.0
    for z in (zm, zc):
        sd = float(z.std(ddof=1))                      # empirical scale: a stationary NON-uniform stream has larger block variance
        if sd <= 1e-12:                                # but is not a change point
            continue
        S = np.cumsum(z - z.mean())
        x = float(np.abs(S).max() / (sd * math.sqrt(m)))
        p = mu.kolmogorov_sf(x)
        if p < best_p or best_k is None:
            best_p, best_k, best_x = p, (int(np.argmax(np.abs(S))) + 1) * block, x
    p = min(1.0, 2.0 * best_p)
    return _ok("changepoint", stream, best_x, p, best_x, n, f"largest shift near offset {best_k}"), best_k


# ── window scan (rolling windows) ────────────────────────────────────────────────────────────────
def window_scan(d: np.ndarray, size: int = 2000, step: int = 1000) -> List[dict]:
    """Cheap tests (frequency chi2, Ljung-Box lag 1..16) on rolling windows -> [{start, p_freq, p_lb}]."""
    out = []
    for s in range(0, max(0, len(d) - size + 1), step):
        w = d[s:s + size]
        f = t_freq(w)
        q, p, _ = _lb(w.astype(np.float64), 16)
        out.append({"start": s, "p_freq": f.p, "p_lb": p})
    return out


# ── families, Holm, verdict ──────────────────────────────────────────────────────────────────────
FAMILY = {"freq_chi2": V_BIAS, "serial2": V_SERIAL, "serial3": V_SERIAL, "serial4": V_SERIAL, "ljungbox_digits": V_SERIAL,
          "ljungbox_indicators": V_SERIAL, "runs": V_SERIAL, "maurer": V_SERIAL, "mutual_info": V_SERIAL, "compress": V_SERIAL,
          "dft_count": V_PERIODIC, "dft_peak": V_PERIODIC, "period": V_PERIODIC, "repeat": V_REPEAT,
          "linear_complexity": V_LINEAR, "matrix_rank": V_LINEAR, "changepoint": V_CHANGEPOINT}

SOLVERS = {V_LINEAR: ["bm_recurrence", "gf2_state"], V_PERIODIC: ["cycle", "bm_recurrence"], V_REPEAT: ["cycle"],
           V_SERIAL: [], V_BIAS: [], V_CHANGEPOINT: []}


def holm(results: List[TestResult], alpha: float) -> List[TestResult]:
    """Holm step-down over the tests that actually ran; returns the rejected ones."""
    live = sorted([r for r in results if r.ok and r.p == r.p], key=lambda r: r.p)
    m = len(live)
    rej = []
    for i, r in enumerate(live):
        if r.p <= alpha / (m - i):
            rej.append(r)
        else:
            break
    return rej


def verdict_of(rejected: List[TestResult]) -> Tuple[str, List[str]]:
    fams = {FAMILY.get(r.name, V_SERIAL) for r in rejected}
    if not fams:
        return V_NONE, []
    if V_CHANGEPOINT in fams:
        v = V_CHANGEPOINT
    elif len(fams) == 1:
        v = next(iter(fams))
    else:
        v = V_MIXED
    solvers: List[str] = []
    for f in fams:
        for s in SOLVERS.get(f, []):
            if s not in solvers:
                solvers.append(s)
    return v, solvers


def run_diagnostics(digits: np.ndarray, run_index: int = 1, alpha: float = 0.01, quotes: Optional[np.ndarray] = None,
                    decimals: int = 2, seed: int = 0, B_compress: int = 200, B_mc: int = 99, max_bm_bits: int = 40000,
                    map_streams: Sequence[str] = ("INCR_Q10", "INCR_SIGN", "PIT10"), perm_mi: int = 0) -> StructureReport:
    """Run every applicable test on the digit stream (and observation-map streams if quotes are given)."""
    d = np.asarray(digits, np.int64)
    n = len(d)
    res: List[TestResult] = []
    res.append(t_freq(d))
    for k in (2, 3, 4):
        res.append(t_serial(d, k, B=B_mc, seed=seed))
    res.extend(t_ljungbox(d))
    bits = bit_streams(d)
    for nm in ("parity", "ge5"):
        res.extend(t_dft((2 * bits[nm] - 1).astype(np.float64), nm))
        res.append(t_runs(bits[nm], nm))
    res.extend(t_dft((d - 4.5).astype(np.float64), "digit", centered_var=8.25)[1:])
    res.append(t_maurer(d))
    for nm in ("parity", "ge5", "bit1", "bit2", "bit3"):
        res.append(t_linear_complexity(bits[nm], nm, max_bm_bits))
    for nm in ("parity", "ge5"):
        res.append(t_matrix_rank(bits[nm], nm))
    res.append(t_lattice(n))
    res.append(t_repeat(d))
    res.append(t_period(d))
    res.append(t_mutual_info(d, perm=perm_mi, seed=seed))
    res.append(t_compress(d, B=B_compress, seed=seed))
    cp, cp_off = t_changepoint(d)
    res.append(cp)
    if quotes is not None and len(quotes) == n:
        for mname in map_streams:
            s, A = OBS_MAPS[mname](quotes, d, decimals)
            s = np.asarray(s, np.int64)
            if len(s) < 1000:
                continue
            tag = mname
            # serial dependence only: the marginal of an increment-derived stream is not uniform under the null and says nothing
            # about the predictability of the digit, so no frequency test on map streams
            res.append(t_indep(s, A, tag))
            if A == 10:
                res.extend(t_ljungbox(s, 32, tag))
    a_run = alpha_run(alpha, run_index)
    rej = holm(res, a_run)
    verdict, solvers = verdict_of(rej)
    rep = StructureReport(verdict=verdict, results=res, recommended_solvers=solvers, n=n, run_index=run_index, alpha_run=a_run,
                          rejected=[f"{r.name}@{r.stream}" for r in rej],
                          families=sorted({FAMILY.get(r.name, V_SERIAL) for r in rej}))
    if any(r.name == "changepoint" for r in rej):
        rep.changepoint_idx = cp_off
    return rep
