"""
trade_journal.py – Persistent trade log stored as newline-delimited JSON.

Every trade (open + close) is appended to trades.jsonl so nothing is lost
across restarts.  Also maintains an in-memory summary for the dashboard.

Render's free tier does NOT have persistent disk storage between deploys,
but the journal survives restarts within the same session and provides
rich in-session analytics.
"""

import json
import logging
import os
import time
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
import threading

logger = logging.getLogger(__name__)

JOURNAL_PATH = os.path.join(os.path.dirname(__file__), "trades.jsonl")


@dataclass
class JournalEntry:
    # Identity
    contract_id:  str
    symbol:       str
    asset_class:  str
    direction:    str          # LONG | SHORT

    # Execution
    stake:        float
    entry_price:  float
    entry_time:   str          # ISO 8601 UTC

    # Result (filled on close)
    exit_price:   float = 0.0
    exit_time:    str   = ""
    pnl:          float = 0.0
    payout:       float = 0.0
    won:          bool  = False
    duration_sec: int   = 0

    # Strategy metadata
    htf_bias:     str  = ""
    smc_structure:str  = ""
    m1_signal:    int  = 0
    m2_signal:    int  = 0
    m3_signal:    int  = 0
    modules_fired: int = 0

    # Balances
    balance_before: float = 0.0
    balance_after:  float = 0.0


class TradeJournal:
    """
    Thread-safe trade journal with JSON persistence.
    """

    def __init__(self, path: str = JOURNAL_PATH):
        self._path   = path
        self._lock   = threading.Lock()
        self._trades: Dict[str, JournalEntry] = {}  # contract_id → entry
        self._closed: List[JournalEntry] = []        # completed trades

        # Session statistics
        self.session_start = datetime.datetime.utcnow().isoformat()
        self._load_session()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_session(self):
        """Load today's closed trades from the journal file for continuity."""
        if not os.path.exists(self._path):
            return
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        try:
            with open(self._path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        if d.get("exit_time", "").startswith(today) and d.get("won") is not None:
                            entry = JournalEntry(**{k: d[k] for k in JournalEntry.__dataclass_fields__ if k in d})
                            self._closed.append(entry)
                    except Exception:
                        continue
            logger.info(f"Journal loaded {len(self._closed)} trades from today")
        except Exception as exc:
            logger.warning(f"Could not load journal: {exc}")

    def _append_to_file(self, entry: JournalEntry):
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
        except Exception as exc:
            logger.warning(f"Journal write failed: {exc}")

    # ── Trade lifecycle ───────────────────────────────────────────────────────

    def open_trade(self, contract_id: str, symbol: str, direction: str,
                   stake: float, entry_price: float, balance_before: float,
                   asset_class: str = "", htf_bias: str = "",
                   smc_structure: str = "", m1: int = 0, m2: int = 0,
                   m3: int = 0, modules: int = 0) -> JournalEntry:
        entry = JournalEntry(
            contract_id   = contract_id,
            symbol        = symbol,
            asset_class   = asset_class,
            direction     = direction,
            stake         = stake,
            entry_price   = entry_price,
            entry_time    = datetime.datetime.utcnow().isoformat(),
            htf_bias      = htf_bias,
            smc_structure = smc_structure,
            m1_signal     = m1,
            m2_signal     = m2,
            m3_signal     = m3,
            modules_fired = modules,
            balance_before = balance_before,
        )
        with self._lock:
            self._trades[contract_id] = entry
        logger.info(f"Journal OPEN  | {contract_id} | {direction} {symbol} "
                    f"@ {entry_price:.5f} | ${stake:.2f}")
        return entry

    def close_trade(self, contract_id: str, exit_price: float,
                    pnl: float, payout: float, balance_after: float) -> Optional[JournalEntry]:
        with self._lock:
            entry = self._trades.pop(contract_id, None)
        if not entry:
            logger.warning(f"Journal: unknown contract_id {contract_id}")
            return None

        now = datetime.datetime.utcnow()
        try:
            entry_dt = datetime.datetime.fromisoformat(entry.entry_time)
            dur      = int((now - entry_dt).total_seconds())
        except Exception:
            dur = 0

        entry.exit_price   = exit_price
        entry.exit_time    = now.isoformat()
        entry.pnl          = round(pnl, 6)
        entry.payout       = round(payout, 6)
        entry.won          = pnl > 0
        entry.duration_sec = dur
        entry.balance_after = balance_after

        with self._lock:
            self._closed.append(entry)
        self._append_to_file(entry)

        outcome = "✅ WIN " if entry.won else "❌ LOSS"
        logger.info(f"Journal CLOSE | {contract_id} | {outcome} | "
                    f"pnl=${pnl:+.4f} | balance=${balance_after:.4f}")
        return entry

    # ── Analytics ─────────────────────────────────────────────────────────────

    @property
    def closed_trades(self) -> List[JournalEntry]:
        with self._lock:
            return list(self._closed)

    @property
    def open_trades(self) -> List[JournalEntry]:
        with self._lock:
            return list(self._trades.values())

    def session_summary(self) -> dict:
        closed = self.closed_trades
        if not closed:
            return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
                    "total_pnl": 0, "gross_profit": 0, "gross_loss": 0,
                    "profit_factor": 0, "avg_rr": 0, "best_trade": 0,
                    "worst_trade": 0, "streak": 0}

        wins   = [t for t in closed if t.won]
        losses = [t for t in closed if not t.won]
        pnls   = [t.pnl for t in closed]
        gp     = sum(t.pnl for t in wins)
        gl     = abs(sum(t.pnl for t in losses))
        pf     = round(gp / gl, 3) if gl > 0 else float("inf")

        # Average R:R (win_avg / loss_avg in $ terms relative to stake)
        avg_win_r  = (gp / len(wins)  / (sum(t.stake for t in wins)  / len(wins)))  if wins  else 0
        avg_loss_r = (gl / len(losses)/ (sum(t.stake for t in losses)/ len(losses))) if losses else 1
        avg_rr     = round(avg_win_r / avg_loss_r, 2) if avg_loss_r else 0

        # Current win/loss streak
        streak = 0
        if closed:
            last_won = closed[-1].won
            for t in reversed(closed):
                if t.won == last_won:
                    streak += 1
                else:
                    break
            if not last_won:
                streak = -streak

        # Best symbols
        sym_pnl: Dict[str, float] = {}
        for t in closed:
            sym_pnl[t.symbol] = sym_pnl.get(t.symbol, 0) + t.pnl
        best_sym  = max(sym_pnl, key=sym_pnl.get) if sym_pnl else ""
        worst_sym = min(sym_pnl, key=sym_pnl.get) if sym_pnl else ""

        return {
            "session_start":  self.session_start,
            "trades":         len(closed),
            "open_trades":    len(self.open_trades),
            "wins":           len(wins),
            "losses":         len(losses),
            "win_rate":       round(len(wins) / len(closed) * 100, 1),
            "total_pnl":      round(sum(pnls), 4),
            "gross_profit":   round(gp, 4),
            "gross_loss":     round(gl, 4),
            "profit_factor":  pf,
            "avg_rr":         avg_rr,
            "best_trade":     round(max(pnls), 4),
            "worst_trade":    round(min(pnls), 4),
            "streak":         streak,
            "best_symbol":    best_sym,
            "worst_symbol":   worst_sym,
            "by_asset_class": self._by_asset_class(closed),
        }

    def _by_asset_class(self, trades: List[JournalEntry]) -> dict:
        classes: Dict[str, dict] = {}
        for t in trades:
            ac = t.asset_class or "unknown"
            if ac not in classes:
                classes[ac] = {"trades": 0, "wins": 0, "pnl": 0.0}
            classes[ac]["trades"] += 1
            classes[ac]["wins"]   += int(t.won)
            classes[ac]["pnl"]    += t.pnl
        for ac in classes:
            n = classes[ac]["trades"]
            classes[ac]["win_rate"] = round(classes[ac]["wins"] / n * 100, 1) if n else 0
            classes[ac]["pnl"]      = round(classes[ac]["pnl"], 4)
        return classes

    def recent_trades(self, n: int = 20) -> List[dict]:
        closed = self.closed_trades
        return [asdict(t) for t in closed[-n:]]
