"""
exp_mode.py - the pre-registered primary test (pure functions, no I/O).

H0: the hit rate of a calibrated digit model equals the base rate (0.40 for OVER5 / UNDER4), so a flat-stake
balance drifts down at the house edge.  Primary test (frozen; never tune it after the fact):
  * all symbols not on the blocklist, horizon h=1 only, contracts OVER5 (wins 6-9) and UNDER4 (wins 0-3)
  * each tick: P(d) for 10 digits -> zone totals P(6-9), P(0-3), calibrated per cover
  * fire only if one zone's calibrated total >= TEST_ZONE_THRESHOLD; skip ticks whose most likely digit is 4 or 5
  * flat stake TEST_STAKE, demo only, max one contract per (symbol, target tick) per arm
Per-symbol and per-horizon breakdowns are EXPLORATORY.
"""
import hashlib
import json
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

import config

OVER5 = ("OVER", 5)
UNDER4 = ("UNDER", 4)
BASE_RATE = 0.40
OVER_DIGITS = (6, 7, 8, 9)
UNDER_DIGITS = (0, 1, 2, 3)

# every key below is part of the frozen config hash
FROZEN_KEYS = (
    "TEST_STAKE", "TEST_ZONE_THRESHOLD", "TEST_MAX_OPEN", "TEST_MAX_TRADES", "TEST_APPLY_GATE", "TEST_GATE_Z",
    "TEST_HONOUR_MUTE", "TEST_CONTROL", "TEST_CONTROL_MODE", "TEST_CONTROL_SEED", "TEST_VERDICT_MIN",
    "TEST_SPRT", "TEST_SPRT_P1", "TEST_SPRT_ALPHA", "TEST_SPRT_BETA", "MH_DEADLINE_FRAC", "MH_DEADLINE_RTT",
    "MH_DECAY", "MH_HISTORY_TICKS", "MH_GATE", "MH_GATE_Z", "PREDICTOR_SYMBOLS",
)


def frozen_config() -> dict:
    out = {}
    for k in FROZEN_KEYS:
        v = getattr(config, k, None)
        out[k] = list(v) if isinstance(v, (tuple, list)) else v
    out["contracts"] = ["OVER5", "UNDER4"]
    out["h"] = 1
    return out


def config_hash(cfg: Optional[dict] = None) -> str:
    cfg = frozen_config() if cfg is None else cfg
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


class ZoneDecision:
    __slots__ = ("fire", "reason", "cand", "zo", "zu", "co", "cu", "p_cal", "threshold", "top")

    def __init__(self, fire, reason, cand, zo, zu, co, cu, p_cal, threshold, top):
        self.fire, self.reason, self.cand = fire, reason, cand
        self.zo, self.zu, self.co, self.cu = zo, zu, co, cu      # raw zone probs, calibrated zone probs
        self.p_cal, self.threshold, self.top = p_cal, threshold, top


def zone_probs(p: Sequence[float]) -> Tuple[float, float]:
    p = np.asarray(p, dtype=float)
    return float(p[list(OVER_DIGITS)].sum()), float(p[list(UNDER_DIGITS)].sum())


def zone_decision(p: Sequence[float], cal_over: Optional[Callable[[float], float]],
                  cal_under: Optional[Callable[[float], float]], threshold: float) -> ZoneDecision:
    """The frozen fire/skip rule.  `cal_*` map a raw zone total to a calibrated probability (None = no calibration)."""
    pa = np.asarray(p, dtype=float)
    zo, zu = zone_probs(pa)
    top = int(np.argmax(pa))
    if cal_over is None or cal_under is None:
        return ZoneDecision(False, "no_calibration", None, zo, zu, None, None, None, threshold, top)
    co, cu = float(cal_over(zo)), float(cal_under(zu))
    if top in (4, 5):
        return ZoneDecision(False, "skip_45_dominant", None, zo, zu, co, cu, None, threshold, top)
    if max(co, cu) < threshold:
        return ZoneDecision(False, "skip_below_threshold", None, zo, zu, co, cu, None, threshold, top)
    if co == cu:
        return ZoneDecision(False, "skip_tie", None, zo, zu, co, cu, None, threshold, top)
    if co > cu:
        return ZoneDecision(True, "fire_over5", OVER5, zo, zu, co, cu, co, threshold, top)
    return ZoneDecision(True, "fire_under4", UNDER4, zo, zu, co, cu, cu, threshold, top)


def best_zone(zd: ZoneDecision) -> Optional[str]:
    """Zone the model leans to on this tick (for shadow scoring); None when calibration is missing."""
    if zd.co is None:
        return "over" if zd.zo >= zd.zu else "under"
    return "over" if zd.co >= zd.cu else "under"


def zone_hit(zone: str, digit: int) -> bool:
    return digit in (OVER_DIGITS if zone == "over" else UNDER_DIGITS)


def label_alignment(realized_h: Optional[int], h: int) -> str:
    """ALIGNED: settled on the intended tick; LATE: later; UNKNOWN: exit tick not found; ANOMALY: earlier than h."""
    if realized_h is None:
        return "UNKNOWN"
    if realized_h == h:
        return "ALIGNED"
    return "LATE" if realized_h > h else "ANOMALY"


def control_cover(mode: str, rng) -> tuple:
    """Control-arm contract for the same moment (same flat stake): fixed OVER5 / UNDER4, or random."""
    if mode == "over":
        return OVER5
    if mode == "under":
        return UNDER4
    return OVER5 if rng.random() < 0.5 else UNDER4
