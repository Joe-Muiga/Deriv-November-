"""rng_core.selector - which predictions are worth acting on (brief section: selection layer).

Pure function of the calibrated Prediction; NO payout / price term here (the payout model is broker-specific and lives in the engine).

RANK_MODE
    off         : act on every prediction whose status is not NO_STRUCTURE (the engine's own gates then decide).
    calibrated  : additionally require the Wilson lower bound of the calibrated top-4 probability to beat the fair-RNG top-4
                  rate (0.40) by `min_edge`. With no evidence the calibrator returns 0.40 and its lower bound is below it, so
                  the selector abstains until the shadow-scored record supports acting.
NO_STRUCTURE predictions are published (and scored) but never acted on unless V2_ACT_ON_NO_STRUCTURE=1.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .types import NO_STRUCTURE, Prediction

P_FAIR = 0.40


class Selector:
    def __init__(self, mode: str = "off", min_edge: float = 0.0, act_on_no_structure: bool = False):
        if mode not in ("off", "calibrated"):
            raise ValueError(f"V2_RANK_MODE must be off|calibrated, got {mode!r}")
        self.mode, self.min_edge, self.act_on_no_structure = mode, float(min_edge), bool(act_on_no_structure)

    def admit(self, pred: Prediction) -> bool:
        if pred is None:
            return False
        if pred.status == NO_STRUCTURE and not self.act_on_no_structure:
            return False
        if self.mode == "calibrated":
            return pred.p4_lb > P_FAIR + self.min_edge
        return True

    @staticmethod
    def score(pred: Prediction) -> float:
        """Ranking key for choosing between candidates: calibrated lower-bound edge over the fair top-4 rate."""
        return float(pred.p4_lb - P_FAIR)


def risk_coverage(scores: Sequence[float], hits: Sequence[int], points: int = 10) -> List[Tuple[float, float, int]]:
    """Risk-coverage curve: abstain below a score threshold. Returns [(coverage, top-4 hit rate among acted, n_acted)] for the
    `points` most-selective-to-least-selective score quantiles. A calibrated, informative score has hit rate falling as coverage
    rises; a flat curve at 0.40 means the score carries no information."""
    s, h = np.asarray(scores, float), np.asarray(hits, float)
    if len(s) == 0:
        return []
    order = np.argsort(-s, kind="stable")
    hs = h[order]
    out = []
    for k in np.linspace(max(1, len(s) // points), len(s), points).astype(int):
        out.append((k / len(s), float(hs[:k].mean()), int(k)))
    return out
