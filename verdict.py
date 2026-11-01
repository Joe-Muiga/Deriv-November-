"""
verdict.py - honest statistics for the follower test (pure functions).

Only follower trades placed after a snapshot count, ONE observation per collector signal
(4 followers copying one signal = 1 observation).  The statistic is profit per stake
(win = ratio-1, loss = -1) with a 95% normal-approximation confidence interval.

States (never hidden, negatives included):
  not enough data              fewer than VERDICT_MIN_TRADES signals
  indistinguishable from chance  the CI contains zero (equivalently: win rate vs break-even)
  confirmed profitable         CI lower bound above zero
  confirmed unprofitable       CI upper bound below zero
"""
import math
from typing import Dict, Iterable, List, Optional, Tuple

Z95 = 1.96

NOT_ENOUGH = "not enough data"
CHANCE = "indistinguishable from chance"
PROFIT = "confirmed profitable"
LOSS = "confirmed unprofitable"


def state(n: int, ci_low: Optional[float], ci_high: Optional[float], min_signals: int) -> str:
    if n < min_signals or ci_low is None or ci_high is None:
        return NOT_ENOUGH
    if ci_low > 0:
        return PROFIT
    if ci_high < 0:
        return LOSS
    return CHANCE


def summarize(obs: Iterable[Tuple[float, float]], min_signals: int) -> dict:
    """obs = iterable of (profit_in_stakes, payout_ratio), one per signal."""
    data = list(obs)
    n = len(data)
    if n == 0:
        return {"n": 0, "wins": 0, "win_rate": None, "avg_ratio": None, "break_even": None,
                "mean": None, "ci_low": None, "ci_high": None, "state": NOT_ENOUGH}
    wins = sum(1 for u, _ in data if u > 0)
    mean = sum(u for u, _ in data) / n
    avg_ratio = sum(r for _, r in data) / n
    if n > 1:
        var = sum((u - mean) ** 2 for u, _ in data) / (n - 1)
        se = math.sqrt(var / n)
        lo, hi = mean - Z95 * se, mean + Z95 * se
    else:
        lo = hi = None
    return {"n": n, "wins": wins, "win_rate": wins / n, "avg_ratio": avg_ratio,
            "break_even": (1.0 / avg_ratio) if avg_ratio else None,
            "mean": mean, "ci_low": lo, "ci_high": hi, "state": state(n, lo, hi, min_signals)}


def by_source(rows: Iterable[Tuple[str, float, float]], min_signals: int) -> Dict[str, dict]:
    rows = list(rows)
    out = {"all": summarize([(u, r) for _, u, r in rows], min_signals)}
    for src in ("default", "engine"):
        out[src] = summarize([(u, r) for s, u, r in rows if s == src], min_signals)
    return out


def out_of_zone(minute_rows: List[List[float]], zone_minutes: set) -> dict:
    """Collector's own trades outside every traded zone, from minute_stats.
    CI is approximate (binomial variance with the average payout ratio)."""
    n = w = 0
    psu = rs = 0.0
    for m, r in enumerate(minute_rows):
        if m in zone_minutes:
            continue
        n += int(r[0])
        w += int(r[1])
        psu += float(r[4])
        rs += float(r[3])
    if n == 0:
        return {"n": 0}
    p = w / n
    ratio = rs / n if rs else 0.0
    var = p * (1 - p) * ratio * ratio
    se = math.sqrt(var / n)
    mean = psu / n
    return {"n": n, "wins": w, "win_rate": p, "avg_ratio": ratio,
            "break_even": (1.0 / ratio) if ratio else None,
            "mean": mean, "ci_low": mean - Z95 * se, "ci_high": mean + Z95 * se, "var": var}


def compare(in_zone: dict, out_zone: dict) -> Optional[dict]:
    """Difference in profit per stake: in-zone follower signals minus out-of-zone collector trades."""
    if not in_zone.get("n") or not out_zone.get("n") or in_zone.get("mean") is None:
        return None
    if in_zone["ci_low"] is None:
        return None
    se_in = (in_zone["ci_high"] - in_zone["mean"]) / Z95
    se_out = math.sqrt(out_zone["var"] / out_zone["n"]) if out_zone.get("var") is not None else 0.0
    se = math.sqrt(se_in ** 2 + se_out ** 2)
    diff = in_zone["mean"] - out_zone["mean"]
    return {"diff": diff, "ci_low": diff - Z95 * se, "ci_high": diff + Z95 * se}
