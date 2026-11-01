"""rng_eval.zoo - generator zoo (brief section 9): null, statistical-signal, weak-generator and hostile streams.

Every stream function is deterministic given its seed. A 'digit stream' is an int8 array; a 'tick stream' adds epochs and
quotes so TickStore / gap handling can be exercised (quote = price whose last decimal is the digit, 2 decimals).
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from rng_core.exact import gens
from rng_core.exact.gens import GENS, MAPS

T0 = 1_760_000_000          # stream start epoch used for the time-seeded generators


# ── null streams ────────────────────────────────────────────────────────────────────────────────
def null_pcg64(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 10, n).astype(np.int8)


def null_drbg(seed: int, n: int) -> np.ndarray:
    """SHA-256 counter-mode DRBG; bytes >= 250 rejected so digits are exactly uniform."""
    out = np.empty(0, np.uint8)
    blocks, ctr = [], 0
    need = int(n * 1.05) + 64
    got = 0
    seedb = int(seed).to_bytes(8, "little")
    while got < need:
        buf = b"".join(hashlib.sha256(seedb + (ctr + i).to_bytes(8, "little")).digest() for i in range(64))
        ctr += 64
        a = np.frombuffer(buf, np.uint8)
        a = a[a < 250]
        blocks.append(a)
        got += len(a)
    a = np.concatenate(blocks)[:n]
    return (a % 10).astype(np.int8)


def null_urandom(seed: int, n: int) -> np.ndarray:
    """os.urandom stream (NOT reproducible; `seed` ignored)."""
    out = []
    got = 0
    while got < n:
        a = np.frombuffer(os.urandom(max(64, int((n - got) * 1.1) + 16)), np.uint8)
        a = a[a < 250]
        out.append(a)
        got += len(a)
    return (np.concatenate(out)[:n] % 10).astype(np.int8)


# ── statistical-signal streams: top-4 hit rate in the known context is 0.40 + lift ──────────────
def _probs(fav: np.ndarray, lift: float) -> np.ndarray:
    p = np.full(10, 0.1 - lift / 6.0)
    p[fav] = 0.1 + lift / 4.0
    return p


def sig_bias(seed: int, n: int, lift: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fav = rng.choice(10, 4, replace=False)
    return rng.choice(10, size=n, p=_probs(fav, lift)).astype(np.int8)


def _ctx_stream(seed: int, n: int, lift: float, ctx_fn, n_ctx: int, warm: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fav = np.array([rng.choice(10, 4, replace=False) for _ in range(n_ctx)])
    cdf = np.cumsum(np.array([_probs(f, lift) for f in fav]), axis=1)
    u = rng.random(n)
    d = np.zeros(n, np.int64)
    d[:warm] = rng.integers(0, 10, warm)
    for t in range(warm, n):
        c = ctx_fn(d, t)
        d[t] = min(int(np.searchsorted(cdf[c], u[t])), 9)
    return d.astype(np.int8)


def sig_markov2(seed: int, n: int, lift: float) -> np.ndarray:
    return _ctx_stream(seed, n, lift, lambda d, t: d[t - 2] * 10 + d[t - 1], 100, 2)


def sig_lag17(seed: int, n: int, lift: float) -> np.ndarray:
    return _ctx_stream(seed, n, lift, lambda d, t: d[t - 17], 10, 17)


SIGNALS = {"bias": sig_bias, "markov2": sig_markov2, "lag17": sig_lag17}


# ── weak generators wrapped by every observation map ────────────────────────────────────────────
def weak_digits(name: str, mp: str, seed: int, n: int) -> np.ndarray:
    spec = GENS[name]
    words = spec.fn(seed, n)
    return gens.digits_from_words(spec, mp, words)


def weak_words(name: str, seed: int, n: int) -> List[int]:
    return GENS[name].fn(seed, n)


def time_seeded(mp: str, n: int, offset: int = 3, t0: int = T0) -> Tuple[np.ndarray, int]:
    """Python random seeded with the epoch second (t0 - offset): returns (digits, seed)."""
    seed = t0 - offset
    return weak_digits("mt19937", mp, seed, n), seed


# ── ticks ───────────────────────────────────────────────────────────────────────────────────────
def make_ticks(digits: np.ndarray, t0: int = T0, step: int = 1, dec: int = 2):
    """(epochs, quotes) with round(quote * 10^dec) % 10 == digit."""
    n = len(digits)
    i = np.arange(n)
    epochs = (t0 + i * step).astype(np.float64)
    base = 10000 + ((i // 9) % 331) * 10
    quotes = (base + digits.astype(np.int64)) / (10.0 ** dec)
    return epochs, quotes


def quotes_for_digits(digits: np.ndarray, seed: int = 0, dec: int = 2, sigma: float = 0.5, start: float = 1000.0) -> np.ndarray:
    """Price-like quotes (random-walk coarse level, independent of the digits) whose last decimal IS the given digit, so a digit
    stream with known truth can drive the quote-derived observation maps without a degenerate (monotone) price path."""
    rng = np.random.default_rng(10_000_019 + seed)
    scale = 10 ** (dec - 1)
    coarse = np.floor((start + np.cumsum(sigma * rng.standard_normal(len(digits)))) * scale) / scale
    return coarse + np.asarray(digits, np.int64) / 10.0 ** dec


def make_price_ticks(u: np.ndarray, t0: int = T0, step: int = 1, dec: int = 2, sigma: float = 0.5, start: float = 1000.0):
    """Random-walk price built from uniform outputs u in (0,1): increment = sigma * Phi^-1(u); digit = last decimal.
    This is how a synthetic index would turn generator output into a PRICE LEVEL (the digit is then not a bit-slice)."""
    from rng_eval.ci import norm_ppf
    z = np.array([norm_ppf(min(max(x, 1e-12), 1 - 1e-12)) for x in u])
    price = np.round((start + np.cumsum(sigma * z)) * 10 ** dec) / 10 ** dec
    digits = (np.round(price * 10 ** dec).astype(np.int64) % 10).astype(np.int8)
    i = np.arange(len(u))
    return (t0 + i * step).astype(np.float64), price, digits


# ── hostile conditions (operate on tick arrays) ─────────────────────────────────────────────────
def hostile_missed(epochs, quotes, digits, rate: float, seed: int):
    rng = np.random.default_rng(seed)
    keep = rng.random(len(epochs)) >= rate
    keep[0] = True
    return epochs[keep], quotes[keep], digits[keep], int((~keep).sum())


def hostile_dups(epochs, quotes, digits, rate: float, seed: int):
    """Duplicated ticks: the same (epoch, quote) record arrives twice in a row."""
    rng = np.random.default_rng(seed)
    rep = np.where(rng.random(len(epochs)) < rate, 2, 1)
    idx = np.repeat(np.arange(len(epochs)), rep)
    return epochs[idx], quotes[idx], digits[idx], int((rep - 1).sum())


def hostile_reseed(name: str, mp: str, seed: int, n: int, every: int) -> Tuple[np.ndarray, List[int]]:
    """Generator reseeded every `every` draws. Returns (digits, reseed_positions)."""
    out, pos = [], []
    i, k = 0, 0
    while i < n:
        m = min(every, n - i)
        out.append(weak_digits(name, mp, seed * 1000 + k, m))
        pos.append(i)
        i += m
        k += 1
    return np.concatenate(out), pos


def hostile_interleaved(name: str, mp: str, seed: int, n: int, which: int = 0, symbols: int = 2) -> np.ndarray:
    """`symbols` symbols sharing ONE generator (consecutive outputs go to consecutive symbols); returns symbol `which`."""
    d = weak_digits(name, mp, seed, n * symbols)
    return d[which::symbols][:n]


def hostile_mixed(name: str, mp: str, seed: int, n: int, p_weak: float = 0.5) -> np.ndarray:
    """Each tick comes from the weak generator with probability p_weak, else from a CSPRNG (SHA-256 DRBG)."""
    rng = np.random.default_rng(seed + 99)
    pick = rng.random(n) < p_weak
    w = weak_digits(name, mp, seed, n)
    c = null_drbg(seed, n)
    return np.where(pick, w, c).astype(np.int8)
