"""
zone_engine.py - zone selection from the collector's minute-of-hour data (pure functions).

Minute = UTC minute (identical to Kenya's, which is a whole-hour offset).
A candidate window is 3..8 contiguous minutes (no wrapping). Score:
    shrunk_rate = (wins + k*overall_rate) / (n + k),  k = 200
    expected profit per stake = shrunk_rate * avg_payout_ratio - 1
An ENGINE zone qualifies if pooled trades >= min_trades and expected profit > 0.
Up to max_zones non-overlapping zones are chosen, best first.

The pre-registered DEFAULT zone is a hypothesis to be tested (minutes 53..59 by default),
traded every hour whatever the data says. An engine zone overlapping it is tagged "default".
This module claims nothing about an edge; it only ranks historical windows.
"""
from typing import Dict, List, Optional, Tuple

# rows[m] = [trades, wins, stake_sum, payout_ratio_sum, profit_stake_units]


def shrunk_rate(wins: float, n: float, overall_rate: float, k: float = 200.0) -> float:
    return (wins + k * overall_rate) / (n + k)


def _pool(rows: List[List[float]], s: int, e: int) -> Tuple[int, int, float]:
    n = w = 0
    rs = 0.0
    for i in range(s, e + 1):
        n += int(rows[i][0])
        w += int(rows[i][1])
        rs += float(rows[i][3])
    return n, w, rs


def describe_window(rows: List[List[float]], s: int, e: int, k: float = 200.0) -> dict:
    tot_n = sum(int(r[0]) for r in rows)
    overall = (sum(int(r[1]) for r in rows) / tot_n) if tot_n else 0.0
    n, w, rs = _pool(rows, s, e)
    ratio = (rs / n) if n else 0.0
    sr = shrunk_rate(w, n, overall, k) if tot_n else 0.0
    return {"start": s, "end": e, "trades": n, "wins": w,
            "raw_rate": (w / n) if n else None, "shrunk_rate": sr, "avg_ratio": ratio,
            "ev": (sr * ratio - 1.0) if n and ratio else None}


def select_zones(rows: List[List[float]], min_trades: int = 150, max_zones: int = 3, k: float = 200.0,
                 min_len: int = 3, max_len: int = 8) -> List[dict]:
    tot_n = sum(int(r[0]) for r in rows)
    if tot_n <= 0:
        return []
    cands: List[dict] = []
    for s in range(60):
        for length in range(min_len, max_len + 1):
            e = s + length - 1
            if e > 59:
                break
            n, _, _ = _pool(rows, s, e)
            if n < min_trades:
                continue
            d = describe_window(rows, s, e, k)
            if d["ev"] is not None and d["ev"] > 0:
                d["source"] = "engine"
                cands.append(d)
    cands.sort(key=lambda z: (-z["ev"], -z["trades"], z["start"]))
    chosen: List[dict] = []
    used = set()
    for z in cands:
        mins = set(range(z["start"], z["end"] + 1))
        if mins & used:
            continue
        chosen.append(z)
        used |= mins
        if len(chosen) >= max_zones:
            break
    return chosen


def build_plan(rows: List[List[float]], default_start: int, default_end: int, min_trades: int = 150,
               max_zones: int = 3, k: float = 200.0) -> dict:
    """Zones + a 60-entry tag list (None / 'default' / 'engine') + contiguous sessions."""
    engine = select_zones(rows, min_trades, max_zones, k)
    tags: List[Optional[str]] = [None] * 60
    for m in range(default_start, default_end + 1):
        tags[m] = "default"
    default_desc = describe_window(rows, default_start, default_end, k)
    default_desc["source"] = "default"
    default_desc["pre_registered"] = True
    zones = [default_desc]
    for z in engine:
        overlaps = any(default_start <= m <= default_end for m in range(z["start"], z["end"] + 1))
        if overlaps:
            z["source"] = "default"          # tagged default because it overlaps the hypothesis zone
        for m in range(z["start"], z["end"] + 1):
            if tags[m] is None:
                tags[m] = z["source"]
        zones.append(z)
    return {"zones": zones, "tags": tags, "sessions": sessions_from_tags(tags)}


def sessions_from_tags(tags: List[Optional[str]]) -> List[dict]:
    """Contiguous runs of traded minutes inside one hour: [{'start': s, 'end': e}] (e inclusive)."""
    out: List[dict] = []
    start = None
    for m in range(61):
        active = m < 60 and tags[m] is not None
        if active and start is None:
            start = m
        elif not active and start is not None:
            out.append({"start": start, "end": m - 1})
            start = None
    return out
