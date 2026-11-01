"""
stats_import.py - one-time import of the old PREDICTOR_STATS env var into minute_stats.

Old format (v1):  {"v":1, "m":[[wins,losses,net_pnl] x60], "t":[trades,wins,stake_sum,net_pnl,ratio_sum]}
The old file stores profit in currency; minute_stats stores it in stake units, so profit is
divided by the average stake (t[2]/t[0]).  Per-minute stake and payout sums were not stored,
so they are filled with trades x average stake / average ratio (only the average is used later).
Runs once: skipped if minute_stats already has data or the meta flag is set.
"""
import json
import logging
import os
from typing import List, Optional

logger = logging.getLogger("follower.import")

FLAG = "predictor_stats_imported_v1"


def convert(raw: str) -> Optional[List[List[float]]]:
    try:
        d = json.loads(raw)
        if d.get("v") != 1 or len(d.get("m", [])) != 60 or len(d.get("t", [])) != 5:
            return None
        trades, _wins, stake_sum, _pnl, ratio_sum = d["t"]
        if not trades or stake_sum <= 0:
            return None
        avg_stake = stake_sum / trades
        avg_ratio = ratio_sum / trades
        rows: List[List[float]] = []
        for w, l, net in d["m"]:
            n = int(w) + int(l)
            rows.append([n, int(w), n * avg_stake, n * avg_ratio, float(net) / avg_stake])
        return rows
    except Exception:
        return None


def maybe_import(db_module) -> bool:
    raw = (os.environ.get("PREDICTOR_STATS") or "").strip()
    if not raw:
        return False
    rows = convert(raw)
    if rows is None:
        logger.warning("FOLLOWER import: PREDICTOR_STATS is not valid v1 JSON - skipped")
        return False
    done = db_module.import_rows_once(rows, FLAG)
    if done:
        logger.info("FOLLOWER import: %d old trades imported into minute_stats (one-time)",
                    sum(int(r[0]) for r in rows))
    return done
