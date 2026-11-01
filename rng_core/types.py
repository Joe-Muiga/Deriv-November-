"""rng_core.types - shared dataclasses and constants (brief section 6, 'Shared contract').

Everything here is plain Python + numpy; nothing imports the rest of the bot.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple

LN10 = math.log(10.0)
UNIFORM_FLOOR = 1e-6          # every dist entry is >= this (contract)

# Prediction.status
NO_STRUCTURE = "NO_STRUCTURE"
STAT_SIGNAL = "STAT_SIGNAL"
EXACT_STATE = "EXACT_STATE"
STATUSES = (NO_STRUCTURE, STAT_SIGNAL, EXACT_STATE)

# StructureReport.verdict
V_NONE, V_BIAS, V_SERIAL, V_LINEAR, V_LATTICE = "NONE", "BIAS", "SERIAL", "LINEAR", "LATTICE"
V_PERIODIC, V_REPEAT, V_CHANGEPOINT, V_MIXED = "PERIODIC", "REPEAT", "CHANGEPOINT", "MIXED"
VERDICTS = (V_NONE, V_BIAS, V_SERIAL, V_LINEAR, V_LATTICE, V_PERIODIC, V_REPEAT, V_CHANGEPOINT, V_MIXED)


class Tick(NamedTuple):
    epoch: float
    quote: float
    digit: int
    idx: int          # absolute tick index (monotone over the life of a TickStore)


@dataclass
class Prediction:
    """One prediction for ONE target tick. dist sums to 1 and every entry >= UNIFORM_FLOOR."""
    symbol: str
    target_idx: int
    h: int
    dist: List[float]
    top4: Tuple[int, int, int, int]
    p4_raw: float
    p4_cal: float
    p4_lb: float
    status: str = NO_STRUCTURE
    log_e: float = 0.0
    n_scored: int = 0
    source: str = "uniform"
    dist_alt: Optional[List[float]] = None   # ranking distribution (null expert removed); top4 and p4_raw come from it when set


@dataclass
class TestResult:
    """name/stream identify the test; p is NaN when ok is False (insufficient data / unavailable)."""
    name: str
    stream: str
    stat: float
    p: float
    effect: float
    n: int
    ok: bool
    note: str = ""

    @property
    def significant_p(self) -> float:
        return self.p if self.ok and self.p == self.p else 1.0


@dataclass
class StructureReport:
    verdict: str = V_NONE
    results: List[TestResult] = field(default_factory=list)
    recommended_solvers: List[str] = field(default_factory=list)
    n: int = 0
    run_index: int = 0
    alpha_run: float = 0.0
    rejected: List[str] = field(default_factory=list)       # names rejected after Holm
    changepoint_idx: Optional[int] = None                     # offset inside the analysed window
    families: List[str] = field(default_factory=list)         # families of the rejected tests (BIAS, SERIAL, LINEAR, ...)


def finalize_dist(p, floor: float = UNIFORM_FLOOR):
    """Clip to the floor and renormalise (contract: sums to 1, each >= floor)."""
    import numpy as np
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, floor)
    return p / p.sum()


def top4_of(p) -> Tuple[int, int, int, int]:
    """Four highest entries, most probable first; ties go to the lower digit (deterministic)."""
    import numpy as np
    o = np.argsort(-np.asarray(p, dtype=np.float64), kind="stable")[:4]          # stable: equal entries keep the lower digit first
    return (int(o[0]), int(o[1]), int(o[2]), int(o[3]))
