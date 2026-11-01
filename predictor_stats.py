"""
predictor_stats.py – compact minute-of-hour trade statistics that survive
Render redeploys.

Stored as ONE small JSON (~1.5 KB) in the PREDICTOR_STATS env var through
env_persist (RENDER_API_KEY); without a key it falls back to a local JSON file
(lost on redeploy – a warning is logged). No per-trade history.

  {"v":1, "m":[[wins,losses,net_pnl] x60], "t":[trades,wins,stake_sum,net_pnl,payout_ratio_sum]}

Minute = UTC minute (0-59) at which the trade was OPENED.
Writes are batched: at most once per FLUSH_SECONDS, plus an explicit forced
flush before a redeploy.
"""
import json
import logging
import math
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import config
import env_persist
from predictor import wilson_lower

logger = logging.getLogger(__name__)

ENV_KEY = "PREDICTOR_STATS"
LOCAL_FILE = os.environ.get("PREDICTOR_STATS_FILE", "predictor_stats.json")


def _blank() -> dict:
    return {"v": 1, "m": [[0, 0, 0.0] for _ in range(60)], "t": [0, 0, 0.0, 0.0, 0.0]}


class MinuteStats:
    def __init__(self, local_file: str = LOCAL_FILE):
        self._lock = threading.Lock()
        self._data = _blank()
        self._dirty = False
        self._last_flush = 0.0
        self._local_file = local_file
        self.flush_seconds = float(getattr(config, "PREDICTOR_STATS_FLUSH_SECONDS", 30))
        self._warned = False

    # -- load / save --------------------------------------------------------
    def load(self) -> None:
        raw = os.environ.get(ENV_KEY, "").strip()
        if not raw and os.path.exists(self._local_file):
            try:
                raw = open(self._local_file).read()
            except Exception:
                raw = ""
        if raw:
            try:
                d = json.loads(raw)
                if d.get("v") == 1 and len(d.get("m", [])) == 60 and len(d.get("t", [])) == 5:
                    self._data = d
                    logger.info(f"PREDICTOR STATS loaded: {d['t'][0]} trades")
            except Exception as exc:
                logger.error(f"PREDICTOR STATS: could not parse stored stats: {exc}")
        if not (getattr(config, "STATS_PERSIST_TO_RENDER", False) and env_persist.configured()) \
                and not self._warned:
            self._warned = True
            logger.info("Predictor stats are kept in memory + a local file only (no Render env-var "
                        "writes, so the service is never restarted by stats saves). They reset if "
                        "Render restarts the instance.")

    def serialize(self) -> str:
        with self._lock:
            return json.dumps(self._data, separators=(",", ":"))

    def _write_local(self, payload: str) -> None:
        try:
            with open(self._local_file, "w") as f:
                f.write(payload)
        except Exception as exc:
            logger.debug(f"local stats write failed: {exc}")

    async def flush(self, force: bool = False) -> bool:
        now = time.time()
        if not self._dirty and not force:
            return True
        if not force and now - self._last_flush < self.flush_seconds:
            return True
        payload = self.serialize()
        self._write_local(payload)
        persist = bool(getattr(config, "STATS_PERSIST_TO_RENDER", False)) and env_persist.configured()
        # Writing a Render env var counts as a config change on Render and can restart the
        # service, so it is only done when the scheduled self-redeploy is in use.
        ok = await env_persist.put_env_vars_async({ENV_KEY: payload}) if persist else True
        if not persist:
            os.environ[ENV_KEY] = payload
        self._last_flush = now
        if ok:
            self._dirty = False
        return ok

    # -- recording ----------------------------------------------------------
    def record(self, opened_at: float, stake: float, payout_ratio: float, profit: float) -> None:
        minute = datetime.fromtimestamp(opened_at, tz=timezone.utc).minute
        won = profit > 0
        with self._lock:
            m = self._data["m"][minute]
            m[0 if won else 1] += 1
            m[2] = round(m[2] + profit, 4)
            t = self._data["t"]
            t[0] += 1
            t[1] += 1 if won else 0
            t[2] = round(t[2] + stake, 4)
            t[3] = round(t[3] + profit, 4)
            t[4] = round(t[4] + payout_ratio, 6)
            self._dirty = True

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._data))


# ─── Analysis (dashboard) ────────────────────────────────────────────────────

def analyze(snap: dict, min_trades_per_minute: Optional[int] = None,
            window_minutes: Optional[int] = None, z: Optional[float] = None) -> dict:
    mtm = int(min_trades_per_minute if min_trades_per_minute is not None
              else getattr(config, "MIN_TRADES_PER_MINUTE", 30))
    wm = int(window_minutes if window_minutes is not None else getattr(config, "WINDOW_MINUTES", 10))
    z = float(z if z is not None else getattr(config, "WILSON_Z", 2.0))
    trades, wins, stake, pnl, ratio_sum = snap["t"]
    avg_ratio = (ratio_sum / trades) if trades else 0.0
    be = (1.0 / avg_ratio) if avg_ratio else None
    overall = {
        "trades": trades, "wins": wins,
        "win_rate": (wins / trades) if trades else None,
        "avg_payout_ratio": avg_ratio or None,
        "break_even": be, "net_profit": pnl, "stake_sum": stake,
    }
    minutes: List[dict] = []
    for i, (w, l, p) in enumerate(snap["m"]):
        n = w + l
        minutes.append({"minute": i, "trades": n, "wins": w,
                        "win_rate": (w / n) if n else None,
                        "wilson_lower": wilson_lower(w, n, z) if n else None,
                        "net_profit": p, "reliable": n >= mtm})
    # best contiguous (wrapping) window by pooled Wilson lower bound
    min_pooled = mtm * wm
    best = None
    for s in range(60):
        idx = [(s + j) % 60 for j in range(wm)]
        n = sum(snap["m"][i][0] + snap["m"][i][1] for i in idx)
        if n < min_pooled:
            continue
        w = sum(snap["m"][i][0] for i in idx)
        lb = wilson_lower(w, n, z)
        if best is None or lb > best["wilson_lower"]:
            best = {"start": s, "end": (s + wm - 1) % 60, "minutes": wm, "trades": n, "wins": w,
                    "win_rate": w / n, "wilson_lower": lb,
                    "net_profit": round(sum(snap["m"][i][2] for i in idx), 4)}
    if best is not None:
        best["beats_break_even"] = (be is not None and best["wilson_lower"] > be)
    return {"overall": overall, "minutes": minutes,
            "window": best if best else {"status": "insufficient data", "needed_trades": min_pooled},
            "params": {"min_trades_per_minute": mtm, "window_minutes": wm, "wilson_z": z}}


_stats: Optional[MinuteStats] = None


def get_stats() -> MinuteStats:
    global _stats
    if _stats is None:
        _stats = MinuteStats()
        _stats.load()
    return _stats
