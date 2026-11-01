"""
set_predictor.py - "predict a SET of next digits, then buy the Over/Under contract that covers it".

Per symbol, on every tick:
  1. Estimate P(next last digit = d) for d = 0..9   (hierarchical back-off over the last 1..3 digits).
  2. Take the SET_SIZE (default 3) most probable digits  -> the predicted set.
  3. Find every Over/Under contract that wins on ALL digits of the set:
        Over  k  wins on digits  > k   -> covers the set when k <= min(set) - 1
        Under k  wins on digits  < k   -> covers the set when k >= max(set) + 1
  4. "Clustered" rule: a covering contract that wins on more than SET_MAX_COVER_WIDTH digits is too
     wide (it pays almost nothing), so a spread-out set produces NO signal for that symbol/tick.

No Matches/Differs and no Even/Odd here. No payout floor or ceiling: the engine ranks signals by
(model win probability x live payout/stake).

Timing (the "lag") is identical to predictor.py: HORIZON = duration + entry delay. A pair is
(digits t-k+1..t) -> digit t+HORIZON, never anything that was not known at the decision tick.

The predictor scores ITSELF live (walk-forward): every prediction is kept until the digit it was made
for arrives, then counted as a top-1 hit and/or a top-N hit. Chance level is 10% / 30% for N=3, so the
scoreboard shows whether the model beats a random guess. Nothing is hidden or smoothed in that report.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

import config
import predictor as P

logger = logging.getLogger(__name__)

Candidate = P.Candidate     # ("OVER"|"UNDER", barrier)


def _cfg(name: str, default: Any) -> Any:
    return getattr(config, name, default)


# ─── Pure helpers ────────────────────────────────────────────────────────────

def win_digits(cand: Candidate) -> List[int]:
    return [d for d in range(10) if P.wins_digit(cand, d)]


def covering_contracts(digits: Sequence[int], max_width: int = 5) -> List[Tuple[Candidate, int]]:
    """
    Every Over/Under contract that wins on ALL of `digits` and wins on at most `max_width` digits,
    tightest (= highest payout) first. Returns [(candidate, width)]. Empty list = spread out, no trade.

        {7,8,9} -> OVER 6 (width 3), OVER 5 (4), OVER 4 (5)
        {0,1,2} -> UNDER 3 (3), UNDER 4 (4), UNDER 5 (5)
        {1,5,9} -> []   (would need OVER 0 / UNDER 9, which win on 9 of 10 digits)
    """
    s = sorted({int(d) for d in digits})
    if not s:
        return []
    lo, hi = s[0], s[-1]
    out: List[Tuple[Candidate, int]] = []
    for k in range(0, lo):                 # OVER k wins on digits > k ; barrier 0..8
        if k <= 8 and 9 - k <= max_width:
            out.append((("OVER", k), 9 - k))
    for k in range(hi + 1, 10):            # UNDER k wins on digits < k ; barrier 1..9
        if k >= 1 and k <= max_width:
            out.append((("UNDER", k), k))
    out.sort(key=lambda x: x[1])
    return out


def contract_win_prob(dist: np.ndarray, cand: Candidate) -> float:
    return float(sum(dist[d] for d in win_digits(cand)))


def tag(cand: Candidate) -> str:
    return f"{cand[0]}{'' if cand[1] is None else cand[1]}"


# ─── Parameters / records ────────────────────────────────────────────────────

@dataclass
class SetParams:
    context_sizes: Tuple[int, ...] = field(default_factory=lambda: tuple(_cfg("SET_CONTEXT_SIZES", (1, 2, 3))))
    set_size: int = field(default_factory=lambda: int(_cfg("SET_SIZE", 3)))
    max_cover_width: int = field(default_factory=lambda: int(_cfg("SET_MAX_COVER_WIDTH", 5)))
    # Dirichlet back-off strengths: how many "virtual observations" of the lower-order estimate a
    # context starts with. Bigger = more shrinkage toward the shorter context / uniform.
    prior_strength: float = field(default_factory=lambda: float(_cfg("SET_PRIOR_STRENGTH", 50.0)))
    backoff_strength: float = field(default_factory=lambda: float(_cfg("SET_BACKOFF_STRENGTH", 30.0)))
    min_set_prob: float = field(default_factory=lambda: float(_cfg("SET_MIN_SET_PROB", 0.0)))
    confirm_fraction: float = field(default_factory=lambda: float(_cfg("CONFIRM_FRACTION", 0.2)))
    duration_ticks: int = field(default_factory=lambda: int(_cfg("CONTRACT_DURATION_TICKS", 1)))
    entry_delay_ticks: int = field(default_factory=lambda: int(_cfg("PREDICTOR_ENTRY_DELAY_TICKS", 0)))

    @property
    def horizon(self) -> int:
        return P.horizon_ticks(self.duration_ticks, self.entry_delay_ticks)


@dataclass
class SetSignal:
    symbol: str
    epoch: Optional[float]                       # epoch of the tick the prediction was made on
    wall: float                                  # wall-clock time the signal was created
    ranked: Tuple[int, ...]                      # predicted digits, most likely first
    digits: Tuple[int, ...]                      # the same digits, sorted
    probs: Tuple[float, ...]                     # model probability of each ranked digit
    set_prob: float                              # model P(next digit is in the set)
    contracts: List[Tuple[Candidate, int, float]]  # (cand, width, model P(win)), tightest first


class ScoreBoard:
    """Live walk-forward accuracy of the predicted set."""

    def __init__(self, window: int = 1000):
        self.n = 0
        self.top1 = 0
        self.topn = 0
        self.recent: Deque[Tuple[bool, bool]] = deque(maxlen=window)

    def record(self, hit1: bool, hitn: bool) -> None:
        self.n += 1
        self.top1 += int(hit1)
        self.topn += int(hitn)
        self.recent.append((bool(hit1), bool(hitn)))

    def rates(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.n:
            return None, None
        return self.top1 / self.n, self.topn / self.n

    def recent_rates(self) -> Tuple[Optional[float], Optional[float]]:
        if not self.recent:
            return None, None
        m = len(self.recent)
        return sum(a for a, _ in self.recent) / m, sum(b for _, b in self.recent) / m


# ─── Predictor ───────────────────────────────────────────────────────────────

class SetPredictor:
    def __init__(self, symbol: str = "", params: Optional[SetParams] = None):
        self.symbol = symbol
        self.p = params or SetParams()
        self.ks = tuple(sorted({int(k) for k in self.p.context_sizes if int(k) >= 1}))
        self.counts0 = np.zeros(10, dtype=np.float64)
        self.counts: Dict[int, np.ndarray] = {k: np.zeros((10 ** k, 10)) for k in self.ks}
        self.history: List[int] = []
        self.last_epoch: Optional[float] = None
        self.fitted = False
        self.fit_report: dict = {}
        self.score = ScoreBoard()
        self._pending: Deque[Tuple[int, ...]] = deque()      # predictions waiting for their digit

    # -- helpers
    @staticmethod
    def _ctx_index(ds: Sequence[int]) -> int:
        idx = 0
        for d in ds:
            idx = idx * 10 + int(d)
        return idx

    def context(self, k: int) -> Optional[Tuple[int, ...]]:
        if len(self.history) < k:
            return None
        return tuple(self.history[-k:])

    def _load(self, D: np.ndarray) -> None:
        h = self.p.horizon
        self.counts0 = np.bincount(D, minlength=10).astype(np.float64)
        for k in self.ks:
            self.counts[k] = P.Predictor._pair_counts(D, k, h, 0, len(D)).astype(np.float64)

    def distribution(self) -> np.ndarray:
        """P(next digit = d | what is known NOW). Interpolated back-off: uniform -> overall digit
        frequencies -> 1-digit context -> 2-digit context -> 3-digit context. Each level starts from the
        level below as its prior, so a context seen only a few times stays close to the shorter one."""
        a, b = self.p.prior_strength, self.p.backoff_strength
        p = (self.counts0 + a * 0.1) / (self.counts0.sum() + a)
        for k in self.ks:
            ctx = self.context(k)
            if ctx is None:
                continue
            row = self.counts[k][self._ctx_index(ctx)]
            p = (row + b * p) / (row.sum() + b)
        return p

    def _top(self, dist: np.ndarray) -> Tuple[int, ...]:
        order = np.argsort(-dist, kind="stable")
        return tuple(int(x) for x in order[: self.p.set_size])

    # -- warm-up ----------------------------------------------------------------
    def fit(self, digits: Sequence[int], last_epoch: Optional[float] = None) -> dict:
        D = np.asarray(list(digits), dtype=np.int64)
        self._load(D)
        self.history = [int(x) for x in D[-64:]]
        self.last_epoch = last_epoch
        self._pending.clear()
        self.fitted = True
        rep = {"ticks": int(len(D)), "horizon": self.p.horizon}
        if len(D) >= 500:
            rep.update(self.backtest(D))
        self.fit_report = rep
        logger.info(f"SET FIT {self.symbol}: {rep}")
        return rep

    def backtest(self, D: np.ndarray) -> dict:
        """Static hold-out check: fit on the older (1-CONFIRM_FRACTION) of the ticks, then measure how often
        the top-1 / top-N digits contain the digit that really came HORIZON ticks later in the newer part."""
        n, h = len(D), self.p.horizon
        n_train = int(n * (1.0 - self.p.confirm_fraction))
        tmp = SetPredictor(self.symbol, self.p)
        tmp._load(D[:n_train])
        kmax = max(self.ks)
        m = t1 = tn = 0
        for j in range(max(n_train, h + kmax - 1), n):
            i = j - h
            tmp.history = [int(x) for x in D[max(0, i - kmax + 1): i + 1]]
            top = tmp._top(tmp.distribution())
            t = int(D[j])
            m += 1
            t1 += int(top[0] == t)
            tn += int(t in top)
        return {"holdout_n": m,
                "holdout_top1": round(t1 / m, 4) if m else None,
                "holdout_topN": round(tn / m, 4) if m else None}

    # -- online -----------------------------------------------------------------
    def on_tick(self, digit: int, epoch: Optional[float] = None) -> bool:
        """Feed one settled tick digit. Scores the prediction that was made HORIZON ticks ago, learns the
        pair (context ending HORIZON ticks ago -> this digit), then predicts again. False if duplicate."""
        if epoch is not None and self.last_epoch is not None and epoch <= self.last_epoch:
            return False
        if epoch is not None:
            self.last_epoch = epoch
        digit = int(digit)
        h = self.p.horizon
        if len(self._pending) >= h:
            pend = self._pending.popleft()
            self.score.record(digit == pend[0], digit in pend)
        self.history.append(digit)
        if len(self.history) > 256:
            del self.history[:-128]
        self.counts0[digit] += 1
        for k in self.ks:
            if len(self.history) >= k + h:
                ctx = self._ctx_index(self.history[-(k + h):-h])
                self.counts[k][ctx, digit] += 1
        if self.fitted and len(self.history) >= max(self.ks):
            self._pending.append(self._top(self.distribution()))
        return True

    # -- decision input ---------------------------------------------------------
    def signal(self, epoch: Optional[float] = None, wall: float = 0.0) -> Optional[SetSignal]:
        """The current predicted set and the Over/Under contracts that cover it, or None when the set is
        spread out (no covering contract within SET_MAX_COVER_WIDTH) or too unlikely (SET_MIN_SET_PROB)."""
        if not self.fitted or len(self.history) < max(self.ks):
            return None
        dist = self.distribution()
        ranked = self._top(dist)
        set_prob = float(sum(dist[d] for d in ranked))
        if set_prob < self.p.min_set_prob:
            return None
        covers = covering_contracts(ranked, self.p.max_cover_width)
        if not covers:
            return None
        contracts = [(c, w, contract_win_prob(dist, c)) for c, w in covers]
        return SetSignal(symbol=self.symbol, epoch=epoch, wall=wall, ranked=ranked,
                         digits=tuple(sorted(ranked)), probs=tuple(float(dist[d]) for d in ranked),
                         set_prob=set_prob, contracts=contracts)

    def describe(self) -> str:
        r = self.fit_report
        return (f"set={self.p.set_size} horizon={self.p.horizon} "
                f"holdout top1={r.get('holdout_top1')} top{self.p.set_size}={r.get('holdout_topN')}")
