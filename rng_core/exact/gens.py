"""rng_core.exact.gens - reference implementations of the catalogued weak generators + observation maps.

Used by (a) the exact solvers (to roll a recovered state forward) and (b) rng_eval.zoo (to build test streams).
Pure Python ints + numpy; no imports from the bot. A generator emits RAW OUTPUT WORDS; an observation map turns a word
into a digit (and, for solvers, into known-bit constraints on the word).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

M64 = (1 << 64) - 1
M32 = 0xFFFFFFFF


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def seed_words(seed: int, k: int) -> List[int]:
    """k pseudo-random 32-bit words derived from `seed` (used where a generator has no canonical seeding)."""
    out, s = [], int(seed)
    for _ in range(k):
        s = splitmix64(s)
        out.append(s & M32)
    return out


# ── LCG family (state fits uint64 products except java/trunc48) ─────────────────────────────────
@dataclass
class LCG:
    name: str
    a: int
    c: int
    m: int
    w: int                      # bits of the emitted word
    shift: int                  # emitted word = (x >> shift) & (2**w - 1)
    state_bits: int
    seed_fn: Callable[[int], int]
    vec_ok: bool = True         # a*x fits uint64 -> numpy brute force possible

    def step(self, x: int) -> int:
        return (self.a * x + self.c) % self.m

    def out(self, x: int) -> int:
        return (x >> self.shift) & ((1 << self.w) - 1)

    def outputs_from_state(self, x: int, n: int, advance_first: bool = True) -> List[int]:
        """n outputs; the first output is taken AFTER one step (as every generator here emits post-step)."""
        res = []
        for _ in range(n):
            x = self.step(x)
            res.append(self.out(x))
        return res


LCGS: Dict[str, LCG] = {
    "minstd": LCG("minstd", 16807, 0, 2**31 - 1, 31, 0, 31, lambda s: (s % (2**31 - 1)) or 1),
    "randu": LCG("randu", 65539, 0, 2**31, 31, 0, 31, lambda s: s | 1),
    "glibc_lcg": LCG("glibc_lcg", 1103515245, 12345, 2**31, 31, 0, 31, lambda s: s & 0x7FFFFFFF),
    "msvc": LCG("msvc", 214013, 2531011, 2**32, 15, 16, 32, lambda s: s & M32),
    "nr_lcg": LCG("nr_lcg", 1664525, 1013904223, 2**32, 32, 0, 32, lambda s: s & M32),
    "borland": LCG("borland", 22695477, 1, 2**32, 15, 16, 32, lambda s: s & M32),
    "java": LCG("java", 0x5DEECE66D, 0xB, 2**48, 32, 16, 48, lambda s: (s ^ 0x5DEECE66D) & (2**48 - 1), vec_ok=False),
    "trunc_lcg48": LCG("trunc_lcg48", 0x2875A2E7B175, 0x11, 2**48, 16, 32, 48, lambda s: splitmix64(s) & (2**48 - 1), vec_ok=False),
}


# ── xorshift / LFSR / PCG / middle square / MT / glibc TYPE_3 ───────────────────────────────────
def xorshift32_next(x: int) -> int:
    x ^= (x << 13) & M32
    x ^= x >> 17
    x ^= (x << 5) & M32
    return x & M32


def xorshift128_next(s: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x, y, z, w = s
    t = (x ^ (x << 11)) & M32
    return (y, z, w, (w ^ (w >> 19) ^ t ^ (t >> 8)) & M32)


LFSR32_POLY = 0xD0000001          # Galois taps 32,31,29,1 (maximal)


def lfsr32_next(x: int) -> int:
    return ((x >> 1) ^ (-(x & 1) & LFSR32_POLY)) & M32


def pcg32_outputs(seed: int, n: int, trunc: int = 16) -> List[int]:
    state = splitmix64(seed)
    inc = (splitmix64(seed + 1) | 1) & M64
    out = []
    for _ in range(n):
        old = state
        state = (old * 6364136223846793005 + inc) & M64
        xs = (((old >> 18) ^ old) >> 27) & M32
        rot = old >> 59
        o = ((xs >> rot) | (xs << ((-rot) & 31))) & M32
        out.append(o >> trunc if trunc else o)
    return out


def middle_square_outputs(seed: int, n: int) -> List[int]:
    x = 10_000_000 + (splitmix64(seed) % 89_999_999)
    out = []
    for _ in range(n):
        x = (x * x // 10_000) % 100_000_000
        out.append(x)
    return out


def glibc_type3_outputs(seed: int, n: int) -> List[int]:
    """glibc random() TYPE_3 (additive feedback r[i] = r[i-3] + r[i-31]); output r >> 1 (31 bits)."""
    r = [0] * 34
    r[0] = (seed & M32) or 1
    for i in range(1, 31):
        prev = r[i - 1] if r[i - 1] < 2**31 else r[i - 1] - 2**32          # signed 32-bit
        hi, lo = int(prev / 127773), prev - int(prev / 127773) * 127773
        word = 16807 * lo - 2836 * hi
        if word < 0:
            word += 2147483647
        r[i] = word & M32
    for i in range(31, 34):
        r[i] = r[i - 31]
    for i in range(34, 344):
        r.append((r[i - 31] + r[i - 3]) & M32)
    out = []
    for i in range(344, 344 + n):
        r.append((r[i - 31] + r[i - 3]) & M32)
        out.append(r[i] >> 1)
    return out


def mt_outputs(seed: int, n: int) -> List[int]:
    rg = random.Random(seed)
    return [rg.getrandbits(32) for _ in range(n)]


def pyrandom_words(seed: int, n_ticks: int) -> List[int]:
    """random.random() consumes TWO 32-bit words per float: a = w0 >> 5 (27 bits), b = w1 >> 6 (26 bits).
    Returned as the interleaved word list [a0, b0, a1, b1, ...] of RAW 32-bit MT outputs (so solvers see words)."""
    rg = random.Random(seed)
    return [rg.getrandbits(32) for _ in range(2 * n_ticks)]


# ── generator registry: raw output words per tick ───────────────────────────────────────────────
@dataclass
class GenSpec:
    name: str
    w: int                                   # bits per emitted word (None-like: modulus used instead when modulus is set)
    fn: Callable[[int, int], List[int]]      # (seed, n_ticks) -> list of n_ticks words (or 2*n for stride 2)
    words_per_tick: int = 1
    modulus: Optional[int] = None            # non power of two output range (middle square)
    family: str = "other"                    # lcg | linear | mt | other


def _lcg_fn(spec: LCG):
    def fn(seed: int, n: int) -> List[int]:
        return spec.outputs_from_state(spec.seed_fn(seed), n)
    return fn


def _xorshift32_fn(seed: int, n: int) -> List[int]:
    x = seed_words(seed, 1)[0] or 1
    out = []
    for _ in range(n):
        x = xorshift32_next(x)
        out.append(x)
    return out


def _xorshift128_fn(seed: int, n: int) -> List[int]:
    s = tuple(seed_words(seed, 4))
    if not any(s):
        s = (1, 2, 3, 4)
    out = []
    for _ in range(n):
        s = xorshift128_next(s)
        out.append(s[3])
    return out


def _lfsr32_fn(seed: int, n: int) -> List[int]:
    x = seed_words(seed, 1)[0] or 1
    out = []
    for _ in range(n):
        x = lfsr32_next(x)
        out.append(x)
    return out


GENS: Dict[str, GenSpec] = {}
for _n, _l in LCGS.items():
    GENS[_n] = GenSpec(_n, _l.w, _lcg_fn(_l), family="lcg")
GENS["xorshift32"] = GenSpec("xorshift32", 32, _xorshift32_fn, family="linear")
GENS["xorshift128"] = GenSpec("xorshift128", 32, _xorshift128_fn, family="linear")
GENS["lfsr32"] = GenSpec("lfsr32", 32, _lfsr32_fn, family="linear")
GENS["mt19937"] = GenSpec("mt19937", 32, mt_outputs, family="mt")
GENS["pyrandom"] = GenSpec("pyrandom", 32, lambda s, n: pyrandom_words(s, n), words_per_tick=2, family="mt")
GENS["pcg32_trunc"] = GenSpec("pcg32_trunc", 16, lambda s, n: pcg32_outputs(s, n, 16))
GENS["middle_square"] = GenSpec("middle_square", 27, middle_square_outputs, modulus=100_000_000)
GENS["glibc_type3"] = GenSpec("glibc_type3", 31, glibc_type3_outputs)


# ── observation maps: word -> digit, and digit -> known bits of the word ────────────────────────
MAPS = ("top", "mod10", "scaled")


def _window(spec: GenSpec, mp: str) -> Tuple[int, int]:
    """(low bit s, width wb) of the bit window the map reads."""
    w = spec.w
    if mp == "scaled" and w >= 24:
        return (w - 24, 16)
    return (0, min(w, 16)) if mp == "scaled" else (0, w)


def digit_of(spec: GenSpec, mp: str, word: int, word2: int = 0) -> int:
    """Digit emitted for one tick. For stride-2 generators (pyrandom) the tick value is the 53-bit random() float."""
    if spec.words_per_tick == 2:
        v = ((word >> 5) << 26) | (word2 >> 6)             # 53-bit mantissa integer
        if mp == "mod10":
            return v % 10
        return (v * 10) >> 53                              # top: floor(10 * random())
    if mp == "mod10":
        return word % 10
    if spec.modulus is not None:
        return (word * 10) // spec.modulus
    s, wb = _window(spec, mp)
    return (((word >> s) & ((1 << wb) - 1)) * 10) >> wb


def digits_from_words(spec: GenSpec, mp: str, words: List[int]) -> np.ndarray:
    if spec.words_per_tick == 2:
        return np.array([digit_of(spec, mp, words[2 * i], words[2 * i + 1]) for i in range(len(words) // 2)], np.int8)
    return np.array([digit_of(spec, mp, x) for x in words], np.int8)


def known_bits(spec: GenSpec, mp: str, digit: int) -> List[Tuple[int, int, int]]:
    """Constraints implied by observing `digit`: [(word_offset, bit_index, value)] on the RAW word(s) of that tick.
    top/scaled: the common binary prefix of the interval [d/10, (d+1)/10) of the read window (2.2 bits on average).
    mod10: the parity of the digit equals bit 0 of the word (10 is even). Non power-of-two generators give nothing for top."""
    if spec.words_per_tick == 2:
        if mp == "mod10":
            return [(1, 6, digit & 1)]                     # LSB of the 53-bit value = bit 6 of the second word
        return _prefix_bits(digit, 27, 5, 0)               # top bits of v live in bits 31.. of the first word (a >> 5)
    if mp == "mod10":
        return [(0, 0, digit & 1)]
    if spec.modulus is not None:
        return []
    s, wb = _window(spec, mp)
    return _prefix_bits(digit, wb, s, 0)


def _prefix_bits(digit: int, wb: int, s: int, word_off: int) -> List[Tuple[int, int, int]]:
    """Known top bits of a wb-bit window starting at word bit s (window top bit is word bit s + wb - 1 [+ extra shift for pyrandom])."""
    res = []
    for k in range(1, wb + 1):
        lo = (digit << k) // 10
        hi = (((digit + 1) << k) - 1) // 10
        if lo != hi:
            break
        bit = (lo & 1)
        res.append((word_off, s + wb - k, bit))
    return res


# ── MT19937 as a stream of 'virtual block' words (used by the solver; any segment alignment works) ──────────
MT_N, MT_M, MT_MAG = 624, 397, 0x9908B0DF


def mt_temper(y: int) -> int:
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    return y & M32


class MTStream:
    """Emits tempered outputs of an MT19937 whose NEXT 624 untempered output words are `words` (index 0). After 624 outputs the
    block is twisted in place exactly like the reference code. Because the twist recurrence w[k+624] = f(w[k], w[k+1], w[k+397])
    holds at any alignment, this represents ANY mid-stream state of a real MT19937."""

    def __init__(self, words):
        self.mt = [int(w) & M32 for w in words]
        self.i = 0

    def _twist(self) -> None:
        mt = self.mt
        for k in range(MT_N):
            y = (mt[k] & 0x80000000) | (mt[(k + 1) % MT_N] & 0x7FFFFFFF)
            mt[k] = mt[(k + MT_M) % MT_N] ^ (y >> 1) ^ (MT_MAG if (y & 1) else 0)
        self.i = 0

    def next(self) -> int:
        if self.i >= MT_N:
            self._twist()
        y = self.mt[self.i]
        self.i += 1
        return mt_temper(y)
