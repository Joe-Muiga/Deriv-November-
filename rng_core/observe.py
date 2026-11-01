"""rng_core.observe - Layer 0: TickStore (numpy ring buffers, absolute indices, gap segments) and observation maps.

TickStore keeps raw epoch/quote/digit per symbol. A missed or duplicated tick (judged from epochs) starts a new SEGMENT:
index-based features and deterministic solvers only ever run inside one segment. Epochs from Deriv are integer seconds,
so gap detection works in whole seconds relative to the median gap (see docs/v2/PHASE0_AUDIT.md, finding 2).
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np


class TickStore:
    """O(1) amortised append, absolute tick index, gap segments.

    Storage is a double-size buffer: when it fills, the newest `capacity` ticks are copied to the front (one memmove per
    `capacity` appends), so every view handed out is contiguous.
    """

    def __init__(self, capacity: int = 300_000, nominal_gap: Optional[float] = None, gap_factor: float = 1.5,
                 min_gaps: int = 30, max_segments: int = 10_000):
        self.cap = int(capacity)
        self._ep = np.zeros(2 * self.cap, np.float64)
        self._q = np.zeros(2 * self.cap, np.float64)
        self._d = np.zeros(2 * self.cap, np.int8)
        self._abs0 = 0                       # absolute index of buffer position 0
        self._lo = 0                         # first valid buffer position
        self._hi = 0                         # one past the last valid buffer position
        self.nominal_gap = nominal_gap
        self.gap_factor = float(gap_factor)
        self.min_gaps = int(min_gaps)
        self._gaps: Deque[float] = deque(maxlen=200)
        self._T: Optional[float] = nominal_gap
        self.seg_starts: Deque[int] = deque([0], maxlen=max_segments)   # absolute index where each segment begins
        self.missed = 0                      # ticks inferred missing (sum over gaps)
        self.dups = 0                        # rejected duplicate / out-of-order epochs
        self.gap_events = 0
        self._last_ep: Optional[float] = None

    # ── size / indices ───────────────────────────────────────────────────────
    @property
    def n_total(self) -> int:
        """Absolute index the NEXT tick will get (== number of ticks ever accepted)."""
        return self._abs0 + self._hi

    @property
    def size(self) -> int:
        return self._hi - self._lo

    @property
    def first_idx(self) -> int:
        return self._abs0 + self._lo

    @property
    def last_epoch(self) -> Optional[float]:
        return self._last_ep

    @property
    def gap_estimate(self) -> Optional[float]:
        return self._T

    # ── append ───────────────────────────────────────────────────────────────
    def _roll(self) -> None:
        c = self.cap
        self._ep[:c] = self._ep[c:]
        self._q[:c] = self._q[c:]
        self._d[:c] = self._d[c:]
        self._abs0 += c
        self._hi -= c
        self._lo = max(0, self._lo - c)

    def _note_gap(self, dt: float) -> bool:
        """True when dt (seconds since the previous accepted tick) means a missed tick."""
        T = self._T
        if T is None or T <= 0:
            return False
        return dt > self.gap_factor * T

    def append(self, epoch: float, quote: float, digit: int) -> int:
        """Store one tick; returns its absolute index, or -1 when the epoch is not newer than the last one (duplicate)."""
        if self._last_ep is not None and epoch <= self._last_ep:
            self.dups += 1
            return -1
        if self._hi == 2 * self.cap:
            self._roll()
        i = self._hi
        if self._last_ep is not None:
            dt = epoch - self._last_ep
            if self._note_gap(dt):
                self.missed += max(1, int(round(dt / self._T)) - 1)
                self.gap_events += 1
                self.seg_starts.append(self._abs0 + i)
            else:
                self._gaps.append(dt)
                if self.nominal_gap is None and len(self._gaps) >= self.min_gaps and (self._abs0 + i) % 20 == 0:
                    self._T = float(np.median(self._gaps))
                elif self._T is None and len(self._gaps) >= self.min_gaps:
                    self._T = float(np.median(self._gaps))
        self._ep[i], self._q[i], self._d[i] = epoch, quote, digit
        self._hi = i + 1
        self._last_ep = float(epoch)
        return self._abs0 + i

    def extend(self, epochs, quotes, digits) -> int:
        """Bulk load (history). Gap logic is vectorised: T = median gap unless nominal_gap is set. Returns ticks accepted.
        Gaps are counted over everything offered; only segments inside the retained window keep a start index."""
        ep = np.asarray(epochs, np.float64)
        q = np.asarray(quotes, np.float64)
        d = np.asarray(digits, np.int8)
        if len(ep) == 0:
            return 0
        last = self._last_ep if self._last_ep is not None else -np.inf
        run = np.maximum.accumulate(np.concatenate([[last], ep[:-1]]))
        keep = ep > run
        self.dups += int((~keep).sum())
        ep, q, d = ep[keep], q[keep], d[keep]
        if len(ep) == 0:
            return 0
        first = self._last_ep is None
        dts = np.diff(ep) if first else np.diff(ep, prepend=self._last_ep)       # dts[k] = gap before ep[k+1] (first) / ep[k]
        gap_of = np.concatenate([[0.0], dts]) if first else dts                   # gap_of[k] = gap before ep[k]
        T = self.nominal_gap
        if T is None and len(dts):
            T = float(np.median(dts))
        if T is not None and T > 0:
            self._T = T
        gpos = np.nonzero(gap_of > self.gap_factor * self._T)[0] if self._T else np.zeros(0, np.int64)
        if first and len(gpos) and gpos[0] == 0:
            gpos = gpos[1:]
        for p in gpos:
            self.missed += max(1, int(round(gap_of[p] / self._T)) - 1)
            self.gap_events += 1
        n_all = len(ep)
        skip = max(0, n_all - self.cap)                    # only the newest `cap` ticks fit
        if skip:
            self._abs0 += self._hi + skip
            self._hi = self._lo = 0
            self.seg_starts = deque([self._abs0], maxlen=self.seg_starts.maxlen)
            self._last_ep = None
            ep, q, d = ep[skip:], q[skip:], d[skip:]
            gap_of = gap_of[skip:]
            gpos = gpos[gpos >= skip] - skip
            first_flag = True
        else:
            first_flag = first
        if self._hi + len(ep) > 2 * self.cap:               # make room: keep the newest `cap` already stored
            keep_from = self._hi - min(self._hi, self.cap - len(ep)) if len(ep) < self.cap else self._hi
            n_keep = self._hi - keep_from
            self._ep[:n_keep] = self._ep[keep_from:self._hi]
            self._q[:n_keep] = self._q[keep_from:self._hi]
            self._d[:n_keep] = self._d[keep_from:self._hi]
            self._abs0 += keep_from
            self._hi = n_keep
            self._lo = max(0, self._lo - keep_from)
        i0 = self._hi
        n = len(ep)
        self._ep[i0:i0 + n], self._q[i0:i0 + n], self._d[i0:i0 + n] = ep, q, d
        self._hi = i0 + n
        for p in gpos:
            if not (first_flag and p == 0):
                self.seg_starts.append(self._abs0 + i0 + int(p))
        if self._T:
            ok = gap_of[1 if first_flag else 0:]
            self._gaps.extend(float(x) for x in ok[ok <= self.gap_factor * self._T][-200:])
        self._last_ep = float(ep[-1])
        return n

    # ── views (valid until the next append / extend) ────────────────────────
    def digits(self, last: Optional[int] = None) -> np.ndarray:
        a = self._d[self._lo:self._hi]
        return a[-last:] if last else a

    def quotes(self, last: Optional[int] = None) -> np.ndarray:
        a = self._q[self._lo:self._hi]
        return a[-last:] if last else a

    def epochs(self, last: Optional[int] = None) -> np.ndarray:
        a = self._ep[self._lo:self._hi]
        return a[-last:] if last else a

    def segment(self, back: int = 0) -> Tuple[int, int]:
        """(first absolute idx, one-past-last absolute idx) of the `back`-th newest segment, clipped to what is stored."""
        starts = list(self.seg_starts)
        k = len(starts) - 1 - back
        if k < 0:
            return (self.n_total, self.n_total)
        lo = max(starts[k], self.first_idx)
        hi = starts[k + 1] if k + 1 < len(starts) else self.n_total
        return (lo, hi)

    def segment_digits(self, back: int = 0) -> Tuple[int, np.ndarray]:
        lo, hi = self.segment(back)
        base = self._abs0
        return lo, self._d[lo - base:hi - base]

    def segment_quotes(self, back: int = 0) -> np.ndarray:
        lo, hi = self.segment(back)
        base = self._abs0
        return self._q[lo - base:hi - base]

    def segment_epochs(self, back: int = 0) -> np.ndarray:
        lo, hi = self.segment(back)
        base = self._abs0
        return self._ep[lo - base:hi - base]

    def info(self) -> dict:
        return {"n_total": self.n_total, "stored": self.size, "segments": len(self.seg_starts), "missed": self.missed,
                "dups": self.dups, "gap_events": self.gap_events, "gap_s": self._T,
                "mem_mb": round((self._ep.nbytes + self._q.nbytes + self._d.nbytes) / 1e6, 2)}


# ── observation maps ────────────────────────────────────────────────────────────────────────────
def _phi(z: np.ndarray) -> np.ndarray:
    """Standard normal cdf (Abramowitz-Stegun 7.1.26 erf, |err| < 1.5e-7)."""
    x = np.asarray(z, np.float64) / np.sqrt(2.0)
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return 0.5 * (1.0 + s * y)


def _incr(quotes: np.ndarray) -> np.ndarray:
    q = np.asarray(quotes, np.float64)
    return np.diff(q)


def map_digit_last(quotes, digits, decimals: int = 2):
    """DIGIT_LAST: the digit stream itself (alphabet 10)."""
    return np.asarray(digits, np.int64), 10


def map_incr_q10(quotes, digits, decimals: int = 2):
    """Increments in pip units, quantised to 10 symbols by the window's own deciles (alphabet 10, length n-1).
    Uses the whole window to place the cut points, so it is a diagnostic map only (not causal)."""
    inc = _incr(quotes)
    if len(inc) < 20:
        return np.zeros(len(inc), np.int64), 10
    qs = np.quantile(inc, np.linspace(0.1, 0.9, 9))
    r = np.searchsorted(qs, inc, side="right")
    # ties (many zero increments) make deciles collapse; break ties by position so symbols stay balanced
    return r.astype(np.int64), 10


def map_incr_sign(quotes, digits, decimals: int = 2):
    """sign of the increment, flat increments dropped: 0 down, 1 up (alphabet 2). A flat tick is ~1% of ticks, so keeping it as a
    third symbol would make the stream non-uniform under the null."""
    sg = np.sign(_incr(quotes)).astype(np.int64)
    return (sg[sg != 0] + 1) // 2, 2


def map_pit10(quotes, digits, decimals: int = 2):
    """Probability-integral-transform stream: u = Phi((dx - median)/(1.4826 MAD)), digit = floor(10 u) (alphabet 10, n-1).
    HYPOTHESIS (brief 6.2): keep it only if the harness shows it raises detection power."""
    inc = _incr(quotes)
    if len(inc) < 20:
        return np.zeros(len(inc), np.int64), 10
    med = np.median(inc)
    mad = np.median(np.abs(inc - med)) * 1.4826
    if mad <= 0:
        mad = float(np.std(inc)) or 1.0
    u = _phi((inc - med) / mad)
    return np.minimum((u * 10).astype(np.int64), 9), 10


OBS_MAPS: Dict[str, Callable] = {
    "DIGIT_LAST": map_digit_last,
    "INCR_Q10": map_incr_q10,
    "INCR_SIGN": map_incr_sign,
    "PIT10": map_pit10,
}


def observe(name: str, quotes, digits, decimals: int = 2) -> Tuple[np.ndarray, int]:
    """Apply a registered observation map -> (symbol stream, alphabet size)."""
    return OBS_MAPS[name](quotes, digits, decimals)
