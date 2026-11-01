"""rng_core.exact.base - Layer 2 interfaces: fitted models, the validation protocol and the live monitor.

A FIT is a plain dict (JSON-serialisable, so a worker process can return it):
    {"kind": "gen" | "mtseed" | "cycle" | "bits", "name": ..., "gen": ..., "map": ..., "params": {...}, "n_fit": int, "n_check": int,
     "solve_s": float}
`build_model(fit)` turns it into an object with .digit(k) (exact digit of segment tick k) or, for bit-level fits, .allowed(k, history).

Validation protocol (brief section 7, phase 4): a candidate is trusted only if it reproduces >= V = 24 ticks that were NOT used by
the solve (chance 1e-24). Only then does the orchestrator publish EXACT_STATE. The monitor then watches live: the first miss
triggers re-verification (suspect), two misses within 50 ticks drop the state. A tick-index gap starts a new segment; the model can
be advanced by the estimated number of missed ticks, and is dropped on the next two misses when that estimate was wrong.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import gens
from .gens import GENS, LCGS, M32, MTStream

V_DEFAULT = 24


class ExactModel:
    """Base: .digit(k) -> int for exact models; bit-level models override allowed()."""
    exact = True

    def digit(self, k: int) -> int:
        raise NotImplementedError


class GenModel(ExactModel):
    """A catalogued generator whose state at segment tick 0 is known. Digits are produced lazily and cached."""

    def __init__(self, gen: str, mp: str, state, skip_words: int = 0, seed: Optional[int] = None):
        self.gen, self.mp, self.state, self.seed, self.skip_words = gen, mp, state, seed, skip_words
        self.spec = GENS[gen]
        self.wpt = self.spec.words_per_tick
        self._digits: List[int] = []
        self._words_done = 0
        self._it = self._make_iter()

    def _make_iter(self):
        g = self.gen
        if g in LCGS:
            lcg = LCGS[g]
            x = int(self.state)

            def it():
                nonlocal x
                while True:
                    x = lcg.step(x)
                    yield lcg.out(x)
            return it()
        if g == "xorshift32":
            x = int(self.state)

            def it():
                nonlocal x
                while True:
                    x = gens.xorshift32_next(x)
                    yield x
            return it()
        if g == "xorshift128":
            s = tuple(int(v) for v in self.state)

            def it():
                nonlocal s
                while True:
                    s = gens.xorshift128_next(s)
                    yield s[3]
            return it()
        if g == "lfsr32":
            x = int(self.state)

            def it():
                nonlocal x
                while True:
                    x = gens.lfsr32_next(x)
                    yield x
            return it()
        if g in ("mt19937", "pyrandom"):
            if self.seed is not None:                      # time-seeded: random.Random(seed), discard skip words
                rg = random.Random(self.seed)
                for _ in range(self.skip_words):
                    rg.getrandbits(32)

                def it():
                    while True:
                        yield rg.getrandbits(32)
                return it()
            st = MTStream(self.state)

            def it():
                while True:
                    yield st.next()
            return it()
        raise NotImplementedError(f"no model for generator {g}")

    def digit(self, k: int) -> int:
        while len(self._digits) <= k:
            if self.wpt == 2:
                a, b = next(self._it), next(self._it)
                self._digits.append(gens.digit_of(self.spec, self.mp, a, b))
            else:
                self._digits.append(gens.digit_of(self.spec, self.mp, next(self._it)))
        return self._digits[k]


class CycleModel(ExactModel):
    """digit(k) = vec[(k - k0) mod p]; vec holds the last period of the segment (vec[0] is segment tick k0)."""

    def __init__(self, vec: Sequence[int], k0: int = 0):
        self.vec = [int(v) for v in vec]
        self.k0 = int(k0)

    def digit(self, k: int) -> int:
        return self.vec[(k - self.k0) % len(self.vec)]


class BitModel:
    """Bit-level model: one bit per tick of a GF(2)-linear recurrence (BM). Predicts the SET of digits consistent with the bit
    (5 of 10 digits), i.e. a top-4 hit rate of 0.8 instead of 0.4. Needs the observed bits to roll forward."""
    exact = False

    def __init__(self, stream: str, L: int, C: int, bits: Sequence[int]):
        self.stream, self.L, self.C = stream, int(L), int(C)
        self.bits: List[int] = [int(b) for b in bits]          # observed bits of the segment so far

    @staticmethod
    def bit_of(stream: str, digit: int) -> int:
        return (digit & 1) if stream == "parity" else (1 if digit >= 5 else 0)

    def push(self, digit: int) -> None:
        self.bits.append(self.bit_of(self.stream, digit))

    def predict_bits(self, steps: int) -> List[int]:
        from .gf2 import bm_predict
        return bm_predict(self.bits, self.L, self.C, steps)

    def allowed_digits(self, bit: int) -> List[int]:
        if self.stream == "parity":
            return [d for d in range(10) if d & 1 == bit]
        return [d for d in range(10) if (d >= 5) == bool(bit)]


def build_model(fit: dict, digits: Optional[Sequence[int]] = None):
    k, p = fit["kind"], fit["params"]
    if k == "cycle":
        return CycleModel(p["vec"], p.get("k0", 0))
    if k == "gen":
        st = p["state"]
        if fit["gen"] == "xorshift128":
            st = tuple(st)
        elif fit["gen"] in ("mt19937", "pyrandom"):
            st = [int(x) for x in st]
        return GenModel(fit["gen"], fit["map"], st)
    if k == "mtseed":
        return GenModel(fit["gen"], fit["map"], None, skip_words=p["skip_words"], seed=p["seed"])
    if k == "bits":
        return BitModel(p["stream"], p["L"], p["C"], p.get("bits", []) if digits is None else
                        [BitModel.bit_of(p["stream"], int(d)) for d in digits])
    raise ValueError(k)


class ExactMonitor:
    """Live use of a validated fit on one segment. `base` is the segment-relative index of the next tick to be observed.

    predict(h) -> ('digit', d) | ('set', [digits]) | None   for the tick h ahead of the newest observed one.
    observe(d)  -> 'hit' | 'miss' | 'drop'
    """

    def __init__(self, fit: dict, digits: Sequence[int], miss_window: int = 50, max_miss: int = 2):
        self.fit = fit
        self.model = build_model(fit, digits if fit["kind"] == "bits" else None)
        self.k = len(digits)                      # next tick (segment-relative) to be observed
        self.shift = 0                            # ticks to skip because ticks were missed
        self.miss: Deque[int] = deque()
        self.window, self.max_miss = miss_window, max_miss
        self.suspect = False
        self.hits = 0
        self.misses = 0
        self.dropped = False
        self.n_obs = 0
        self._pred_bits: Optional[List[int]] = None

    def skip(self, missed: int) -> None:
        """`missed` ticks were not observed: the generator advanced anyway (exact models only)."""
        if self.model.exact:
            self.shift += max(0, int(missed))
        else:
            self.dropped = True                   # a bit-level fit needs the observed history: re-solve

    def predict(self, h: int):
        if self.dropped:
            return None
        if self.model.exact:
            return ("digit", self.model.digit(self.k + self.shift + h - 1))
        bits = self.model.predict_bits(h)
        return ("set", self.model.allowed_digits(bits[-1]))

    def observe(self, d: int) -> str:
        if self.dropped:
            return "drop"
        pred = self.predict(1)
        if self.model.exact:
            ok = pred[1] == d
        else:
            ok = d in pred[1]
            self.model.push(d)
        self.k += 1
        self.n_obs += 1
        while self.miss and self.miss[0] <= self.n_obs - self.window:
            self.miss.popleft()
        if ok:
            self.hits += 1
            if self.suspect and not self.miss:
                self.suspect = False
            return "hit"
        self.misses += 1
        self.miss.append(self.n_obs)
        self.suspect = True                         # first miss -> re-verification state (mass is reduced by the caller)
        if len(self.miss) >= self.max_miss:
            self.dropped = True
            return "drop"
        return "miss"
