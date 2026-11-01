"""
donkey_guard.py — loss-control + edge logging for the Donkey digit bot.

What it does (and does NOT do)
  * It LIMITS how much a bad stretch can cost: session stop-loss / take-profit
    (in stake units), a real consecutive-loss pause, and a trades-per-hour cap.
  * It LOGS every settled Donkey trade with the payout Deriv actually quoted,
    so you can compare real win rate vs breakeven (see analyze_edge.py).
  * It does NOT create an edge. If the win rate sits below breakeven, this
    only makes the losses arrive slower.

All thresholds come from config.* with the defaults below, so config edits
are optional. Stakes stay fixed — nothing here touches stake size.
"""

import csv
import json
import logging
import os
import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

try:
    import config  # type: ignore
except Exception:  # allows standalone tests
    config = None

logger = logging.getLogger("donkey_guard")

EDGE_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "donkey_edge_log.csv")
EDGE_LOG_FIELDS = [
    "ts", "symbol", "signal_kind", "contract", "barrier",
    "stake", "payout", "breakeven", "won", "pnl",
]


def _cfg(name: str, default):
    return getattr(config, name, default) if config is not None else default


class DonkeyGuard:
    def __init__(self, edge_log_path: str = EDGE_LOG_PATH):
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._path = edge_log_path
        self._trade_times: Deque[float] = deque()
        self._reset_session()
        self._halt_until = 0.0
        self._halt_reason = ""
        self._pause_until = 0.0
        self._consec_losses = 0
        self.load_from_env()

    # ── survive redeploys ───────────────────────────────────────────────
    # The bot redeploys after every leg (fixed_cycle.py), and each deploy is
    # a fresh container, so in-memory counters would reset every leg and the
    # session limits could never trip. fixed_cycle.enter_cooldown_now() saves
    # export_state() into the DONKEY_GUARD_STATE env var; we reload it here.
    def export_state(self) -> str:
        with self._lock:
            return json.dumps({
                "halt_until": self._halt_until, "halt_reason": self._halt_reason,
                "pause_until": self._pause_until,
                "consec_losses": self._consec_losses,
                "session_pnl": self._session_pnl,
                "session_stake_unit": self._session_stake_unit,
            })

    def load_from_env(self) -> None:
        raw = os.environ.get("DONKEY_GUARD_STATE", "")
        if not raw:
            return
        try:
            d = json.loads(raw)
            self._halt_until = float(d.get("halt_until", 0.0))
            self._halt_reason = str(d.get("halt_reason", ""))
            self._pause_until = float(d.get("pause_until", 0.0))
            self._consec_losses = int(d.get("consec_losses", 0))
            self._session_pnl = float(d.get("session_pnl", 0.0))
            self._session_stake_unit = float(d.get("session_stake_unit", 0.0))
            logger.info(f"DONKEY GUARD: restored state (session pnl "
                        f"{self._session_pnl:+.2f}, streak {self._consec_losses})")
        except Exception as exc:
            logger.warning(f"DONKEY GUARD: could not restore state: {exc}")

    # ── config (read live so edits take effect) ─────────────────────────
    @property
    def enabled(self) -> bool:
        return bool(_cfg("DONKEY_GUARD_ENABLED", True))

    def _reset_session(self) -> None:
        self._session_pnl = 0.0
        self._session_stake_unit = 0.0  # learned from first trade

    # ── gate: call before opening any new entry ─────────────────────────
    def can_enter(self, now: Optional[float] = None) -> Tuple[bool, str]:
        if not self.enabled:
            return True, ""
        now = time.time() if now is None else now
        with self._lock:
            if now < self._halt_until:
                mins = (self._halt_until - now) / 60
                return False, f"{self._halt_reason} — halted {mins:.0f}min more"
            if now < self._pause_until:
                mins = (self._pause_until - now) / 60
                return False, f"loss-streak pause — {mins:.1f}min more"

            cap = int(_cfg("DONKEY_GUARD_MAX_TRADES_PER_HOUR", 60))
            while self._trade_times and now - self._trade_times[0] > 3600:
                self._trade_times.popleft()
            if cap > 0 and len(self._trade_times) >= cap:
                return False, f"trade cap {cap}/hour reached"
            return True, ""

    def note_entry(self, now: Optional[float] = None) -> None:
        """Call when a new contract is actually bought (feeds the hourly cap)."""
        with self._lock:
            self._trade_times.append(time.time() if now is None else now)

    # ── settlement hook ─────────────────────────────────────────────────
    def record(self, *, symbol: str, signal_kind: str, contract: str,
               barrier, stake: float, payout: float, won: bool, pnl: float,
               now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        breakeven = (stake / payout) if payout and payout > stake > 0 else ""
        try:
            row = {
                "ts": int(now), "symbol": symbol, "signal_kind": signal_kind,
                "contract": contract, "barrier": barrier,
                "stake": round(stake, 4), "payout": round(payout, 4),
                "breakeven": round(breakeven, 5) if breakeven != "" else "",
                "won": int(bool(won)), "pnl": round(pnl, 4),
            }
            # Render's disk is wiped on every redeploy, so also emit the row
            # to the log stream; analyze_edge.py --logs rebuilds from it.
            logger.info("EDGE_LOG," + ",".join(str(row[k]) for k in EDGE_LOG_FIELDS))
            self._append_csv(row)
        except Exception as exc:  # logging must never break trading
            logger.warning(f"edge log write failed: {exc}")

        if not self.enabled:
            return
        with self._lock:
            if self._session_stake_unit == 0.0 and stake > 0:
                self._session_stake_unit = stake
            unit = self._session_stake_unit or stake or 1.0
            self._session_pnl += pnl
            self._consec_losses = 0 if won else self._consec_losses + 1

            max_losses = int(_cfg("DONKEY_GUARD_CONSEC_LOSS_LIMIT", 8))
            if max_losses > 0 and self._consec_losses >= max_losses:
                mins = float(_cfg("DONKEY_GUARD_CONSEC_LOSS_PAUSE_MINS", 30))
                self._pause_until = now + mins * 60
                logger.warning(
                    f"DONKEY GUARD: {self._consec_losses} losses in a row — "
                    f"pausing new entries {mins:.0f}min")
                self._consec_losses = 0

            sl = float(_cfg("DONKEY_GUARD_SESSION_STOP_LOSS_STAKES", 15))
            tp = float(_cfg("DONKEY_GUARD_SESSION_TAKE_PROFIT_STAKES", 10))
            halt_mins = float(_cfg("DONKEY_GUARD_HALT_MINS", 120))
            in_stakes = self._session_pnl / unit
            reason = ""
            if sl > 0 and in_stakes <= -sl:
                reason = f"session stop-loss hit ({in_stakes:+.1f} stakes)"
            elif tp > 0 and in_stakes >= tp:
                reason = f"session take-profit hit ({in_stakes:+.1f} stakes)"
            if reason:
                self._halt_until = now + halt_mins * 60
                self._halt_reason = reason
                logger.warning(f"DONKEY GUARD: {reason} — no new entries "
                               f"for {halt_mins:.0f}min, then fresh session")
                self._reset_session()  # next session starts from zero

    # ── internals ───────────────────────────────────────────────────────
    def _append_csv(self, row: dict) -> None:
        new = not os.path.exists(self._path)
        with self._file_lock:
            with open(self._path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=EDGE_LOG_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)


def signal_kind_from_reason(reason: str) -> str:
    """Donkey reasons start 'Frequency:' (signal 1) or 'Trend filter:' (signal 2)."""
    r = (reason or "").lower()
    if r.startswith("frequency"):
        return "freq"
    if r.startswith("trend"):
        return "trend"
    if "combined" in r or "both" in r:
        return "combined"
    return "other"


_singleton: Optional[DonkeyGuard] = None


def get_guard() -> DonkeyGuard:
    """Shared instance so bot_engine and fixed_cycle see the same state."""
    global _singleton
    if _singleton is None:
        _singleton = DonkeyGuard()
    return _singleton
