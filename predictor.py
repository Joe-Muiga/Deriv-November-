"""
predictor.py – walk-forward last-digit predictor for Deriv digit contracts
(Even/Odd and Over/Under). This is the ONLY strategy that places trades.

Model
  Conditional frequency tables. For each context size k in CONTEXT_SIZES the
  previous k digits are the context; counts[k][context, d] is how often digit
  d appeared HORIZON ticks after that context. From the counts:
      P(even | ctx), P(digit > b | ctx), P(digit < b | ctx)

Lag (the important part)
  HORIZON = CONTRACT_DURATION_TICKS + ENTRY_DELAY_TICKS.
  The bot sees tick t, requests a proposal and buys. Deriv's entry tick is the
  FIRST tick after the purchase and a 1-tick contract settles on that same
  tick, so with a round trip shorter than the tick interval the settlement
  tick is t+1 (HORIZON = 1). A slower round trip adds ENTRY_DELAY_TICKS.
  Training pairs are (digits t-k+1..t) -> digit t+HORIZON, never anything else.
  See settlement_index() and the tests.

Anti-overfitting (fit())
  Warm-up ticks are split in time order: older (1-CONFIRM_FRACTION) = train,
  newer CONFIRM_FRACTION = confirm. A (k, context, candidate) is "tradable" only if
    (a) train win-rate > break-even + EDGE_MARGIN AND one-sided z-test vs
        break-even passes a Bonferroni-corrected threshold (alpha / #tested),
    (b) confirm win-rate > break-even (with >= MIN_CONFIRM_SAMPLES),
    (c) train count >= MIN_CONTEXT_SAMPLES.
  Counts keep updating online after fit(); selection is refreshed each boot.

Decision (rank() + accept())
  Wilson lower bound (WILSON_Z) of the context's win-rate must exceed
  break_even + EDGE_MARGIN, break_even = 1/(payout/stake) from the LIVE
  proposal, and payout/stake must be >= MIN_PAYOUT_RATIO. On a fair
  generator it is expected, and correct, that this almost never trades.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)

Candidate = Tuple[str, Optional[int]]   # ("EVEN"|"ODD"|"OVER"|"UNDER", barrier|None)

# Deriv's own contract_type strings
CONTRACT_TYPE = {"EVEN": "DIGITEVEN", "ODD": "DIGITODD",
                 "OVER": "DIGITOVER", "UNDER": "DIGITUNDER"}


# ─── Pure helpers ────────────────────────────────────────────────────────────

def break_even(payout_ratio: float) -> float:
    """Win probability at which a contract with this payout/stake breaks even."""
    return 1.0 / float(payout_ratio)


def wilson_lower(wins: float, n: float, z: float = 2.0) -> float:
    if n <= 0:
        return 0.0
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, (centre - margin) / denom)


def wins_digit(cand: Candidate, d: int) -> bool:
    kind, b = cand
    if kind == "EVEN":
        return d % 2 == 0
    if kind == "ODD":
        return d % 2 == 1
    if kind == "OVER":
        return d > b
    if kind == "UNDER":
        return d < b
    raise ValueError(cand)


def true_probability(cand: Candidate) -> float:
    return sum(1 for d in range(10) if wins_digit(cand, d)) / 10.0


def estimated_payout_ratio(cand: Candidate, factor: float = 0.975) -> float:
    """Fallback only (used before a live proposal exists): ~Deriv's 2.5% margin."""
    return factor / true_probability(cand)


def all_candidates(barriers: Sequence[int] = tuple(range(10)),
                   even_odd: bool = True) -> List[Candidate]:
    out: List[Candidate] = []
    if even_odd:
        out += [("EVEN", None), ("ODD", None)]
    out += [("OVER", b) for b in barriers if 0 <= b <= 8]
    out += [("UNDER", b) for b in barriers if 1 <= b <= 9]
    return out


def contract_type(cand: Candidate) -> str:
    return CONTRACT_TYPE[cand[0]]


def infer_decimals(quotes: Sequence[float], cap: int = 6) -> int:
    """Decimal count of the index's quotes = most decimals seen in the sample."""
    best = 0
    for q in list(quotes)[:2000]:
        s = repr(float(q))
        if "e" in s or "E" in s:
            continue
        frac = s.split(".")[1].rstrip("0") if "." in s else ""
        best = max(best, len(frac))
    return min(best, cap)


def last_digit(quote: float, decimals: int) -> int:
    """Identical to signal_engine._last_digit."""
    return abs(int(round(float(quote) * (10 ** decimals)))) % 10


def digits_from_quotes(quotes: Sequence[float], decimals: int) -> List[int]:
    return [last_digit(q, decimals) for q in quotes]


def horizon_ticks(duration_ticks: int, entry_delay_ticks: int) -> int:
    return int(duration_ticks) + int(entry_delay_ticks)


def settlement_index(tick_epochs: Sequence[float], last_known_idx: int,
                     buy_time: float, duration_ticks: int) -> Optional[int]:
    """
    Index of the tick whose last digit settles a digit contract bought at
    `buy_time`: the entry tick is the first tick strictly after the purchase
    and counts as tick 1 of the contract, so exit = entry + duration - 1.
    Returns None if that tick is not in tick_epochs yet.
    """
    entry = None
    for i in range(last_known_idx, len(tick_epochs)):
        if tick_epochs[i] > buy_time:
            entry = i
            break
    if entry is None:
        return None
    exit_idx = entry + int(duration_ticks) - 1
    return exit_idx if exit_idx < len(tick_epochs) else None


# ─── Parameters ──────────────────────────────────────────────────────────────

def _cfg(name: str, default: Any) -> Any:
    return getattr(config, name, default)


@dataclass
class PredictorParams:
    context_sizes: Tuple[int, ...] = field(default_factory=lambda: tuple(_cfg("CONTEXT_SIZES", (1, 2, 3))))
    min_context_samples: int = field(default_factory=lambda: int(_cfg("MIN_CONTEXT_SAMPLES", 300)))
    min_confirm_samples: int = field(default_factory=lambda: int(_cfg("MIN_CONFIRM_SAMPLES", 30)))
    edge_margin: float = field(default_factory=lambda: float(_cfg("EDGE_MARGIN", 0.01)))
    wilson_z: float = field(default_factory=lambda: float(_cfg("WILSON_Z", 2.0)))
    confirm_fraction: float = field(default_factory=lambda: float(_cfg("CONFIRM_FRACTION", 0.2)))
    duration_ticks: int = field(default_factory=lambda: int(_cfg("CONTRACT_DURATION_TICKS", 1)))
    entry_delay_ticks: int = field(default_factory=lambda: int(_cfg("PREDICTOR_ENTRY_DELAY_TICKS", 0)))
    min_payout_ratio: float = field(default_factory=lambda: float(_cfg("MIN_PAYOUT_RATIO", 1.8)))
    max_payout_ratio: float = field(default_factory=lambda: float(_cfg("MAX_PAYOUT_RATIO", 0.0)))  # 0 = no ceiling
    barriers: Tuple[int, ...] = field(default_factory=lambda: tuple(_cfg("PREDICTOR_BARRIERS", tuple(range(10)))))
    even_odd: bool = True
    alpha: float = field(default_factory=lambda: float(_cfg("PREDICTOR_ALPHA", 0.05)))
    payout_factor: float = field(default_factory=lambda: float(_cfg("PREDICTOR_PAYOUT_FACTOR", 0.975)))

    @property
    def horizon(self) -> int:
        return horizon_ticks(self.duration_ticks, self.entry_delay_ticks)


@dataclass
class Pick:
    cand: Candidate
    k: int
    context: Tuple[int, ...]
    n: int
    p_hat: float
    p_lower: float


@dataclass
class Decision:
    ok: bool
    reason: str
    ratio: float = 0.0
    break_even: float = 0.0
    edge: float = 0.0          # p_lower - break_even


# ─── Predictor ───────────────────────────────────────────────────────────────

class Predictor:
    def __init__(self, symbol: str = "", params: Optional[PredictorParams] = None,
                 payout_ratio_fn: Optional[Callable[[Candidate], Optional[float]]] = None):
        self.symbol = symbol
        self.p = params or PredictorParams()
        self.payout_ratio_fn = payout_ratio_fn
        self.candidates = all_candidates(self.p.barriers, self.p.even_odd)
        self._win = np.array([[1.0 if wins_digit(c, d) else 0.0 for d in range(10)]
                              for c in self.candidates])           # [cand, digit]
        self.counts_all: Dict[int, np.ndarray] = {}
        self.counts_train: Dict[int, np.ndarray] = {}
        self.selected: Dict[Tuple[int, int, int], dict] = {}        # (k, ctx, cand_idx) -> info
        self.history: List[int] = []
        self.last_epoch: Optional[float] = None
        self.fitted = False
        self.fit_report: dict = {}
        self._reset_counts()

    # -- payout ratio used for selection / ranking (live when known)
    def ratio_for(self, cand: Candidate) -> float:
        r = None
        if self.payout_ratio_fn is not None:
            try:
                r = self.payout_ratio_fn(cand)
            except Exception:
                r = None
        return float(r) if r else estimated_payout_ratio(cand, self.p.payout_factor)

    def _reset_counts(self) -> None:
        for k in self.p.context_sizes:
            self.counts_all[k] = np.zeros((10 ** k, 10), dtype=np.int64)
            self.counts_train[k] = np.zeros((10 ** k, 10), dtype=np.int64)

    @staticmethod
    def _ctx_index(ds: Sequence[int]) -> int:
        idx = 0
        for d in ds:
            idx = idx * 10 + int(d)
        return idx

    @staticmethod
    def _pair_counts(D: np.ndarray, k: int, h: int, tgt_lo: int, tgt_hi: int) -> np.ndarray:
        """Counts of (context ending at i) -> D[i+h] for targets in [tgt_lo, tgt_hi)."""
        out = np.zeros((10 ** k, 10), dtype=np.int64)
        n = len(D)
        # target index j = i + h, context = D[j-h-k+1 .. j-h]
        j = np.arange(max(tgt_lo, h + k - 1), min(tgt_hi, n))
        if len(j) == 0:
            return out
        i = j - h
        ctx = np.zeros(len(j), dtype=np.int64)
        for off in range(k):                      # oldest digit first
            ctx = ctx * 10 + D[i - (k - 1) + off]
        np.add.at(out, (ctx, D[j]), 1)
        return out

    # -- warm-up ---------------------------------------------------------------
    def fit(self, digits: Sequence[int], last_epoch: Optional[float] = None) -> dict:
        D = np.asarray(list(digits), dtype=np.int64)
        n, h = len(D), self.p.horizon
        self._reset_counts()
        self.selected = {}
        self.history = [int(x) for x in D[-64:]]
        self.last_epoch = last_epoch
        n_train = int(n * (1.0 - self.p.confirm_fraction))
        conf_counts: Dict[int, np.ndarray] = {}
        for k in self.p.context_sizes:
            self.counts_train[k] = self._pair_counts(D, k, h, 0, n_train)
            conf_counts[k] = self._pair_counts(D, k, h, n_train, n)
            self.counts_all[k] = self.counts_train[k] + conf_counts[k]

        # pass 1: collect hypotheses actually tested (n >= MIN_CONTEXT_SAMPLES, ratio >= floor)
        tests: List[Tuple[int, int, int, int, float, float]] = []   # k, ctx, ci, n, p, be
        ratios = [self.ratio_for(c) for c in self.candidates]
        for k in self.p.context_sizes:
            tr = self.counts_train[k]
            ns = tr.sum(axis=1)
            wins = tr @ self._win.T                                  # [ctx, cand]
            for ctx in np.nonzero(ns >= self.p.min_context_samples)[0]:
                for ci in range(len(self.candidates)):
                    if ratios[ci] < self.p.min_payout_ratio:
                        continue
                    if self.p.max_payout_ratio > 0 and ratios[ci] > self.p.max_payout_ratio:
                        continue
                    tests.append((k, int(ctx), ci, int(ns[ctx]),
                                  float(wins[ctx, ci] / ns[ctx]), break_even(ratios[ci])))
        m = max(len(tests), 1)
        q = 1.0 - self.p.alpha / m
        # alpha/m >= 0.5 means "no multiple-testing correction at all": z_crit 0 instead of an
        # invalid inverse-CDF argument (large PREDICTOR_ALPHA values used for demo experiments).
        z_crit = 0.0 if q <= 0.5 else NormalDist().inv_cdf(min(q, 1.0 - 1e-12))
        passed_train = 0
        for (k, ctx, ci, nt, p_tr, be) in tests:
            if p_tr <= be + self.p.edge_margin:
                continue
            z = (p_tr - be) / math.sqrt(be * (1.0 - be) / nt)
            if z <= z_crit:
                continue
            passed_train += 1
            cc = conf_counts[k][ctx]
            nc = int(cc.sum())
            if nc < self.p.min_confirm_samples:
                continue
            p_cf = float(cc @ self._win[ci]) / nc
            if p_cf <= be:
                continue
            self.selected[(k, ctx, ci)] = {"n_train": nt, "p_train": p_tr, "n_confirm": nc,
                                           "p_confirm": p_cf, "be": be, "z": z}
        self.fitted = True
        self.fit_report = {"ticks": n, "train": n_train, "confirm": n - n_train,
                           "horizon": h, "tested": len(tests), "z_crit": z_crit,
                           "passed_train": passed_train, "tradable": len(self.selected)}
        logger.info(f"PREDICTOR FIT {self.symbol}: {self.fit_report}")
        return self.fit_report

    # -- online ----------------------------------------------------------------
    def on_tick(self, digit: int, epoch: Optional[float] = None) -> bool:
        """Append a settled-in-the-past tick digit; updates counts with the pair
        (context ending HORIZON ticks ago -> this digit). Returns False if duplicate."""
        if epoch is not None and self.last_epoch is not None and epoch <= self.last_epoch:
            return False
        if epoch is not None:
            self.last_epoch = epoch
        self.history.append(int(digit))
        if len(self.history) > 256:
            del self.history[:-128]
        h = self.p.horizon
        for k in self.p.context_sizes:
            if len(self.history) >= k + h:
                ctx = self._ctx_index(self.history[-(k + h):-h])
                self.counts_all[k][ctx, int(digit)] += 1
        return True

    def context(self, k: int) -> Optional[Tuple[int, ...]]:
        """Last k digits KNOWN NOW (uses nothing after the latest tick)."""
        if len(self.history) < k:
            return None
        return tuple(self.history[-k:])

    # -- decision --------------------------------------------------------------
    def rank(self) -> List[Pick]:
        """Candidates from the tradable set for the current context, best first
        by p_lower - estimated break-even. Payout/edge is re-checked live in accept()."""
        picks: List[Pick] = []
        if not self.fitted:
            return picks
        for (k, ctx, ci), _info in self.selected.items():
            cur = self.context(k)
            if cur is None or self._ctx_index(cur) != ctx:
                continue
            row = self.counts_all[k][ctx]
            n = int(row.sum())
            if n == 0:
                continue
            w = float(row @ self._win[ci])
            picks.append(Pick(self.candidates[ci], k, cur, n, w / n,
                              wilson_lower(w, n, self.p.wilson_z)))
        picks.sort(key=lambda x: x.p_lower - break_even(self.ratio_for(x.cand)), reverse=True)
        return picks

    def accept(self, pick: Pick, ratio: float) -> Decision:
        """Payout floor + edge test against the LIVE payout/stake ratio."""
        if not ratio or ratio < self.p.min_payout_ratio:
            return Decision(False, f"payout ratio {ratio:.4f} < floor {self.p.min_payout_ratio}", ratio)
        if self.p.max_payout_ratio > 0 and ratio > self.p.max_payout_ratio:
            return Decision(False, f"payout ratio {ratio:.4f} > ceiling {self.p.max_payout_ratio}", ratio)
        be = break_even(ratio)
        edge = pick.p_lower - be
        if pick.p_lower <= be + self.p.edge_margin:
            return Decision(False, f"p_lower {pick.p_lower:.4f} <= be {be:.4f} + margin", ratio, be, edge)
        return Decision(True, "ok", ratio, be, edge)

    def best_with_ratios(self, ratio_of: Callable[[Candidate], Optional[float]]) -> Optional[Tuple[Pick, Decision]]:
        """Offline helper (tests / signal_engine route): best accepted pick."""
        best = None
        for pk in self.rank():
            r = ratio_of(pk.cand)
            if r is None:
                continue
            d = self.accept(pk, r)
            if d.ok and (best is None or d.edge > best[1].edge):
                best = (pk, d)
        return best


# ─── Registry + signal_engine bridge ─────────────────────────────────────────

_REGISTRY: Dict[str, Predictor] = {}


def get_predictor(symbol: str) -> Predictor:
    if symbol not in _REGISTRY:
        _REGISTRY[symbol] = Predictor(symbol)
    return _REGISTRY[symbol]


def register(symbol: str, predictor: Predictor) -> None:
    _REGISTRY[symbol] = predictor


def evaluate_for_signal_engine(ticks: Optional[List[Any]], symbol: str):
    """signal_engine.SignalEngine.evaluate() routes PREDICTOR_SYMBOLS here.
    Uses the live predictor (kept current by bot_engine) and an ESTIMATED payout
    ratio; bot_engine itself uses rank()/accept() with real proposals."""
    from signal_engine import SignalResult, NONE_RESULT
    pr = _REGISTRY.get(symbol)
    if pr is None or not pr.fitted:
        return NONE_RESULT
    best = pr.best_with_ratios(pr.ratio_for)
    if best is None:
        return NONE_RESULT
    pk, d = best
    kind, b = pk.cand
    return SignalResult(direction=kind, strength=3, score=pk.p_lower, strategy="PREDICTOR",
                        reason=f"k={pk.k} ctx={pk.context} n={pk.n} p={pk.p_hat:.3f} lb={pk.p_lower:.3f} edge={d.edge:.3f}",
                        contract_kind="DIGIT", digit=b, match_type=kind)
