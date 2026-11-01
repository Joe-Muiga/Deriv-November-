"""
strategy_stats.py

Tracks closed-trade outcomes per (strategy, symbol) pair, maintains a
rolling win rate over a configurable window, and flags underperforming
pairs using config.STRATEGY_WIN_RATE_FLOOR / STRATEGY_WIN_RATE_MIN_TRADES.

Persistence: SQLite if the sqlite3 module + a writable DB file are
available, otherwise falls back to a local JSON file. Either way, stats
survive Render restarts as long as the underlying disk/volume persists
(ephemeral filesystems will still lose data on redeploy — mount a disk
if you need cross-deploy durability).

Thread-safety: a single RLock guards all read/write access, since
bot_engine.py may call record_trade() from multiple async tasks and the
dashboard route may read concurrently.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import config

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False


# ── STORAGE LOCATIONS ─────────────────────────────────────────
DATA_DIR = os.environ.get("STRATEGY_STATS_DIR", os.path.dirname(os.path.abspath(__file__)))
SQLITE_PATH = os.path.join(DATA_DIR, "strategy_stats.db")
JSON_PATH = os.path.join(DATA_DIR, "strategy_stats.json")

DEFAULT_WINDOW = 100
WIN_RATE_FLOOR = config.STRATEGY_WIN_RATE_FLOOR
MIN_TRADES_FOR_FLOOR = config.STRATEGY_WIN_RATE_MIN_TRADES


@dataclass
class TradeRecord:
    strategy: str
    symbol: str
    entry_score: float
    won: bool
    stake: float
    payout: float
    timestamp: float  # unix epoch seconds
    features: Optional[str] = None   # JSON-encoded feature dict, or None


def _wilson_interval(wins: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a binomial proportion.
    z=1.96 -> ~95% CI. Returns (low, high), both in [0, 1].
    With n=0 returns (0.0, 1.0) — maximally uncertain.
    """
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + (z ** 2) / n
    center = p + (z ** 2) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) / n) + (z ** 2) / (4 * n ** 2))
    low = (center - margin) / denom
    high = (center + margin) / denom
    return max(0.0, low), min(1.0, high)


class _SqliteBackend:
    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> "sqlite3.Connection":
        # sqlite3 connections aren't safe to share across threads by
        # default; keep one per thread, guarded externally by the RLock.
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            self._local.conn = conn
        return conn

    def _init_db(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                entry_score REAL,
                won INTEGER NOT NULL,
                stake REAL,
                payout REAL,
                timestamp REAL NOT NULL,
                features TEXT
            )
            """
        )
        # Defensive upgrade path for databases created before `features`
        # existed — CREATE TABLE IF NOT EXISTS above is a no-op against an
        # already-existing table, so old schemas need this ALTER to gain
        # the new column without losing history.
        try:
            conn.execute("ALTER TABLE trades ADD COLUMN features TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_strategy_symbol_ts "
            "ON trades (strategy, symbol, timestamp)"
        )
        conn.commit()
        conn.close()

    def insert(self, rec: TradeRecord):
        conn = self._conn()
        conn.execute(
            "INSERT INTO trades (strategy, symbol, entry_score, won, stake, payout, timestamp, features) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (rec.strategy, rec.symbol, rec.entry_score, int(rec.won), rec.stake, rec.payout, rec.timestamp, rec.features),
        )
        conn.commit()

    def recent_results(self, strategy: str, symbol: str, window: int) -> List[bool]:
        conn = self._conn()
        cur = conn.execute(
            "SELECT won FROM trades WHERE strategy=? AND symbol=? "
            "ORDER BY timestamp DESC LIMIT ?",
            (strategy, symbol, window),
        )
        return [bool(row[0]) for row in cur.fetchall()]

    def count(self, strategy: str, symbol: str) -> int:
        conn = self._conn()
        cur = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE strategy=? AND symbol=?",
            (strategy, symbol),
        )
        return cur.fetchone()[0]

    def recent_trades(self, strategy: str, symbol: str, window: int) -> List[Tuple[bool, float, float]]:
        """Most recent `window` trades as (won, stake, payout) tuples, newest first."""
        conn = self._conn()
        cur = conn.execute(
            "SELECT won, stake, payout FROM trades WHERE strategy=? AND symbol=? "
            "ORDER BY timestamp DESC LIMIT ?",
            (strategy, symbol, window),
        )
        return [(bool(row[0]), row[1], row[2]) for row in cur.fetchall()]

    def all_pairs(self) -> List[Tuple[str, str]]:
        conn = self._conn()
        cur = conn.execute("SELECT DISTINCT strategy, symbol FROM trades")
        return [(row[0], row[1]) for row in cur.fetchall()]

    def all_trades_full(self, window: Optional[int] = None) -> List[Tuple[str, str, bool, float, float, float, Optional[str]]]:
        """
        All trades (or the most recent `window`) across every (strategy,
        symbol) pair, as (strategy, symbol, won, stake, payout, timestamp,
        features) tuples, newest first. Used by get_hourly_payout_ratio()
        to bucket realized payout by UTC hour, and by get_feature_rows()
        (Implementation Brief v6, PART 3) to decode logged feature
        vectors — recent_trades()/recent_results() don't expose timestamp
        or features, which these need.
        """
        conn = self._conn()
        if window:
            cur = conn.execute(
                "SELECT strategy, symbol, won, stake, payout, timestamp, features FROM trades "
                "ORDER BY timestamp DESC LIMIT ?",
                (window,),
            )
        else:
            cur = conn.execute(
                "SELECT strategy, symbol, won, stake, payout, timestamp, features FROM trades "
                "ORDER BY timestamp DESC"
            )
        return [(row[0], row[1], bool(row[2]), row[3], row[4], row[5], row[6]) for row in cur.fetchall()]


class _JsonBackend:
    """
    Fallback backend. Stores trades as a flat list per (strategy, symbol)
    key in a single JSON file. Fine for demo/research-scale trade counts;
    not intended for very high volume (whole file is rewritten on each
    insert).
    """

    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(self.path):
            self._write({})

    def _read(self) -> Dict[str, List[dict]]:
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write(self, data: Dict[str, List[dict]]):
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, self.path)  # atomic on POSIX

    @staticmethod
    def _key(strategy: str, symbol: str) -> str:
        return f"{strategy}::{symbol}"

    def insert(self, rec: TradeRecord):
        data = self._read()
        key = self._key(rec.strategy, rec.symbol)
        data.setdefault(key, [])
        data[key].append(asdict(rec))
        self._write(data)

    def recent_results(self, strategy: str, symbol: str, window: int) -> List[bool]:
        data = self._read()
        rows = data.get(self._key(strategy, symbol), [])
        rows_sorted = sorted(rows, key=lambda r: r["timestamp"], reverse=True)
        return [bool(r["won"]) for r in rows_sorted[:window]]

    def count(self, strategy: str, symbol: str) -> int:
        data = self._read()
        return len(data.get(self._key(strategy, symbol), []))

    def recent_trades(self, strategy: str, symbol: str, window: int) -> List[Tuple[bool, float, float]]:
        """Most recent `window` trades as (won, stake, payout) tuples, newest first."""
        data = self._read()
        rows = data.get(self._key(strategy, symbol), [])
        rows_sorted = sorted(rows, key=lambda r: r["timestamp"], reverse=True)[:window]
        return [(bool(r["won"]), r["stake"], r["payout"]) for r in rows_sorted]

    def all_pairs(self) -> List[Tuple[str, str]]:
        data = self._read()
        pairs = []
        for key in data.keys():
            if "::" in key:
                strategy, symbol = key.split("::", 1)
                pairs.append((strategy, symbol))
        return pairs

    def all_trades_full(self, window: Optional[int] = None) -> List[Tuple[str, str, bool, float, float, float, Optional[str]]]:
        """Same contract as _SqliteBackend.all_trades_full() — see there."""
        data = self._read()
        rows: List[Tuple[str, str, bool, float, float, float, Optional[str]]] = []
        for key, entries in data.items():
            if "::" not in key:
                continue
            strategy, symbol = key.split("::", 1)
            for r in entries:
                rows.append((strategy, symbol, bool(r["won"]), r["stake"], r["payout"], r["timestamp"], r.get("features")))
        rows.sort(key=lambda r: r[5], reverse=True)
        if window:
            rows = rows[:window]
        return rows


class StrategyStats:
    def __init__(self):
        self._lock = threading.RLock()
        self._backend = None
        self._backend_name = None
        self._init_backend()

    def _init_backend(self):
        if SQLITE_AVAILABLE:
            try:
                self._backend = _SqliteBackend(SQLITE_PATH)
                self._backend_name = "sqlite"
                return
            except sqlite3.Error:
                pass  # fall through to JSON
        self._backend = _JsonBackend(JSON_PATH)
        self._backend_name = "json"

    # ── PUBLIC API ────────────────────────────────────────────

    def record_trade(
        self,
        strategy: str,
        symbol: str,
        entry_score: float,
        won: bool,
        stake: float,
        payout: float,
        timestamp: Optional[float] = None,
        features: Optional[dict] = None,
    ) -> None:
        """Record a closed trade outcome. Call this whenever a trade closes."""
        rec = TradeRecord(
            strategy=strategy,
            symbol=symbol,
            entry_score=float(entry_score),
            won=bool(won),
            stake=float(stake),
            payout=float(payout),
            timestamp=float(timestamp) if timestamp is not None else time.time(),
            features=json.dumps(features) if features is not None else None,
        )
        with self._lock:
            self._backend.insert(rec)

    def get_win_rate(
        self, strategy: str, symbol: str, window: int = DEFAULT_WINDOW
    ) -> Tuple[float, float, float, int]:
        """
        Rolling win rate over the most recent `window` trades for this
        (strategy, symbol) pair, with a Wilson 95% confidence interval.

        Returns (rate, ci_low, ci_high, n) where n is the number of
        trades the rate was computed over (<= window). With n=0, returns
        (0.0, 0.0, 1.0, 0).
        """
        with self._lock:
            results = self._backend.recent_results(strategy, symbol, window)
        n = len(results)
        if n == 0:
            return 0.0, 0.0, 1.0, 0
        wins = sum(results)
        rate = wins / n
        ci_low, ci_high = _wilson_interval(wins, n)
        return rate, ci_low, ci_high, n

    def get_avg_win_payout_ratio(
        self, strategy: str, symbol: str, window: int = DEFAULT_WINDOW
    ) -> Optional[float]:
        """
        Average payout/stake ratio across WINNING trades only, over the
        most recent `window` trades for this (strategy, symbol) pair.

        This is the raw ratio a winning stake returns (e.g. 1.9 on a
        $1 stake), i.e. the input needed to derive Kelly's `b` (net
        odds) via b = ratio - 1. Losing trades are excluded since they
        don't inform the win-payout side of the formula.

        Returns None if there are no winning trades in the window, or
        no valid (stake > 0) winning trades — callers should treat
        None as "not enough data" rather than 0.
        """
        with self._lock:
            rows = self._backend.recent_trades(strategy, symbol, window)
        ratios = [payout / stake for won, stake, payout in rows if won and stake > 0]
        if not ratios:
            return None
        return sum(ratios) / len(ratios)

    def get_hourly_payout_ratio(
        self, window: int = 5000
    ) -> Dict[int, Tuple[Optional[float], int]]:
        """
        Implementation Brief v5 / B2.

        Realized average win-payout ratio (payout/stake on winning
        trades), bucketed by UTC hour of trade close (0-23), across ALL
        (strategy, symbol) pairs and ALL settled trades this backend has
        (up to the most recent `window` trades overall, to bound cost on
        the JSON fallback backend).

        Independent payout auditing found off-peak/quiet-session payouts
        can run meaningfully below the advertised headline rate — a
        direct hit to the Kelly `b` term (net odds) for every Rise/Fall
        trade placed during quiet hours, independent of signal quality.
        This is the cheap way to act on that: the data (payout, stake,
        timestamp) is already captured on every settled trade, so this
        just aggregates what's already there instead of guessing.

        Returns {hour: (avg_ratio_or_None, n)}. avg_ratio is None for
        hours with no winning trades logged yet — callers should treat
        None (or a low n) as "not enough data, use a neutral weight"
        rather than as 0.
        """
        with self._lock:
            rows = self._backend.all_trades_full(window=window)

        buckets: Dict[int, List[float]] = {h: [] for h in range(24)}
        for _strategy, _symbol, won, stake, payout, ts, _features in rows:
            if not won or stake <= 0:
                continue
            hour = time.gmtime(ts).tm_hour
            buckets[hour].append(payout / stake)

        return {
            h: ((sum(ratios) / len(ratios)) if ratios else None, len(ratios))
            for h, ratios in buckets.items()
        }

    def get_feature_rows(self, strategy: str, symbol: str, window: int = 500) -> List[dict]:
        """
        Rows with a non-null feature vector, newest first, for training a
        calibration model (Implementation Brief v6, PART 5). Each dict:
        {"won": bool, "stake": float, "payout": float, "timestamp": float,
         **decoded feature dict}. Rows recorded before this field existed
        are skipped (features is None), not zero-filled.
        """
        with self._lock:
            rows = self._backend.all_trades_full(window=window)
        out = []
        for strat, sym, won, stake, payout, ts, features_json in rows:
            if strat != strategy or sym != symbol:
                continue
            if not features_json:
                continue
            try:
                decoded = json.loads(features_json)
            except (TypeError, ValueError):
                continue
            row = {"won": won, "stake": stake, "payout": payout, "timestamp": ts}
            row.update(decoded)
            out.append(row)
        return out

    def is_underperforming(self, strategy: str, symbol: str) -> bool:
        """
        True if this (strategy, symbol) pair has >= STRATEGY_WIN_RATE_MIN_TRADES
        logged trades AND its rolling win rate (window=STRATEGY_WIN_RATE_MIN_TRADES)
        falls below config.STRATEGY_WIN_RATE_FLOOR. Small samples are never
        flagged, regardless of how bad the raw rate looks.
        """
        with self._lock:
            total = self._backend.count(strategy, symbol)
        if total < MIN_TRADES_FOR_FLOOR:
            return False
        rate, _ci_low, _ci_high, n = self.get_win_rate(
            strategy, symbol, window=MIN_TRADES_FOR_FLOOR
        )
        if n < MIN_TRADES_FOR_FLOOR:
            return False
        return rate < WIN_RATE_FLOOR

    def all_stats(self, window: int = DEFAULT_WINDOW) -> List[dict]:
        """
        Returns a list of dicts, one per (strategy, symbol) pair seen so
        far, for consumption by a dashboard route. Each dict:
            {
                "strategy": str,
                "symbol": str,
                "win_rate": float,
                "ci_low": float,
                "ci_high": float,
                "n": int,               # trades in the rolling window
                "total_trades": int,    # all-time trades for this pair
                "underperforming": bool,
            }
        """
        with self._lock:
            pairs = self._backend.all_pairs()

        out = []
        for strategy, symbol in pairs:
            rate, ci_low, ci_high, n = self.get_win_rate(strategy, symbol, window=window)
            total = self._backend.count(strategy, symbol)
            out.append(
                {
                    "strategy": strategy,
                    "symbol": symbol,
                    "win_rate": round(rate, 4),
                    "ci_low": round(ci_low, 4),
                    "ci_high": round(ci_high, 4),
                    "n": n,
                    "total_trades": total,
                    "underperforming": self.is_underperforming(strategy, symbol),
                }
            )
        out.sort(key=lambda d: (d["strategy"], d["symbol"]))
        return out

    def get_take_invert_stats(self, symbol: str, window: int = 500) -> dict:
        """
        Aggregates realized win/loss counts for `symbol`, split by whether
        each trade was executed as TAKE (features["inverted"] is False or
        missing) or INVERT (features["inverted"] is True) — across ALL
        strategies that trade this symbol, since
        config.META_LABEL_DEFAULT_ACTION_BY_SYMBOL is a per-symbol (not
        per-strategy) default and the same symbol can trade under more
        than one strategy (e.g. VOL_BREAKOUT vs VOL_REV_MULT for the same
        VOL_MULTIPLIER_SYMBOLS entry, depending on regime).

        Used by meta_labeling.py's Bayesian TAKE-vs-INVERT bandit — this
        is the ONLY data it trains on (no borrowed/generic priors, no
        other symbols' data, just this symbol's own realized outcomes).
        Rows with no features JSON (pre-dating the "inverted" field) are
        skipped rather than guessed at.

        Returns {"wins_take", "losses_take", "wins_invert", "losses_invert"}.
        """
        with self._lock:
            rows = self._backend.all_trades_full(window=window)
        wins_take = losses_take = wins_invert = losses_invert = 0
        for (_strategy, sym, won, _stake, _payout, _ts, features_json) in rows:
            if sym != symbol or not features_json:
                continue
            try:
                decoded = json.loads(features_json)
            except (TypeError, ValueError):
                continue
            if "inverted" not in decoded:
                continue
            if bool(decoded["inverted"]):
                if won:
                    wins_invert += 1
                else:
                    losses_invert += 1
            else:
                if won:
                    wins_take += 1
                else:
                    losses_take += 1
        return {
            "wins_take": wins_take, "losses_take": losses_take,
            "wins_invert": wins_invert, "losses_invert": losses_invert,
        }

    def get_dashboard_analytics(
        self,
        scaled_account_balance: float = 20.0,
        scaled_stake: float = 1.0,
    ) -> dict:
        """
        Dashboard analytics requested by the user (Aug 2026): a %P&L-of-
        staked figure/graph, a profit-factor graph, and a "what if I were
        trading a $20 account staking $1/trade" projection. Computed fresh
        from full trade history each call — fine at demo-account trade
        volumes, not optimized for high-frequency re-polling at scale.

        Returns:
          total_trades, total_staked, total_pnl
          pct_pnl_of_staked: (total_pnl / total_staked) * 100, exactly as
            requested. NOT the same as return-on-balance — this is P&L
            relative to money actually put at risk across all trades,
            which differs from balance growth % whenever stake size
            varies trade to trade (it does here — stake scales with
            balance/Kelly).
          profit_factor_series: [{"timestamp", "profit_factor"}, ...] —
            the CUMULATIVE profit factor (gross profit / gross loss) as
            of each trade's close, in chronological order. A single
            point-in-time number isn't a graph; this is the running value
            so the dashboard can chart its trend over time.
          scaled_account: a replay of the REAL trade sequence — same wins
            and losses, in the same order — at a fixed $`scaled_stake`
            per trade against a starting balance of
            $`scaled_account_balance`. Each trade's real payout RATIO
            (payout / stake, e.g. 1.9x) is applied to the fixed scaled
            stake, so a trade that returned 1.9x on its real ~$40 stake
            also returns 1.9x on the $1 scaled stake — a faithful replay
            of what actually happened, not a resimulated/hypothetical
            market. Compounds: each trade's scaled P&L is added to the
            running scaled balance before the next trade, matching "if
            I'm not withdrawing profits".
          growth_projection: LINEAR extrapolation of the scaled account's
            OBSERVED average $ P&L per trade x observed trades/day
            (computed from real trade timestamps) out to 1 day/week/month.
            This is explicitly not a forecast or guarantee — it assumes
            future trades behave statistically like past ones, which
            isn't certain, and win rate/edge can and does change over
            time, especially on this little history. See its "basis"
            field, which restates this every time it's read.
        """
        with self._lock:
            rows = self._backend.all_trades_full()  # newest-first

        if not rows:
            return {
                "total_trades": 0, "total_staked": 0.0, "total_pnl": 0.0,
                "pct_pnl_of_staked": 0.0, "pct_pnl_series": [], "profit_factor_series": [],
                "scaled_account": None, "growth_projection": None,
                "note": "No trade history yet.",
            }

        rows_chrono = list(reversed(rows))  # oldest-first for time-series work

        total_staked = sum(r[3] for r in rows_chrono)
        total_pnl = sum((r[4] - r[3]) for r in rows_chrono)
        pct_pnl_of_staked = (total_pnl / total_staked * 100.0) if total_staked > 0 else 0.0

        # ── Cumulative profit-factor AND %P&L-of-staked series ────────
        # Both computed in the same pass since they walk the same
        # chronological trade list.
        profit_factor_series = []
        pct_pnl_series = []
        cum_gp, cum_gl = 0.0, 0.0
        cum_staked, cum_pnl = 0.0, 0.0
        for (_strategy, _symbol, _won, stake, payout, ts, _features) in rows_chrono:
            pnl = payout - stake
            if pnl > 0:
                cum_gp += pnl
            else:
                cum_gl += -pnl
            pf = (cum_gp / cum_gl) if cum_gl > 0 else (float("inf") if cum_gp > 0 else 0.0)
            profit_factor_series.append({"timestamp": ts, "profit_factor": pf})

            cum_staked += stake
            cum_pnl += pnl
            pct = (cum_pnl / cum_staked * 100.0) if cum_staked > 0 else 0.0
            pct_pnl_series.append({"timestamp": ts, "pct_pnl_of_staked": round(pct, 4)})

        # ── Scaled $20 / $1-per-trade replay ─────────────────────────
        scaled_balance = scaled_account_balance
        scaled_series = [{"timestamp": rows_chrono[0][5], "balance": round(scaled_balance, 4)}]
        scaled_pnl_total = 0.0
        for (_strategy, _symbol, _won, stake, payout, ts, _features) in rows_chrono:
            ratio = (payout / stake) if stake > 0 else 0.0
            scaled_pnl = (scaled_stake * ratio) - scaled_stake
            scaled_balance += scaled_pnl
            scaled_pnl_total += scaled_pnl
            scaled_series.append({"timestamp": ts, "balance": round(scaled_balance, 4)})

        scaled_pct_return = (
            (scaled_pnl_total / scaled_account_balance * 100.0)
            if scaled_account_balance > 0 else 0.0
        )

        # ── Growth projection — linear extrapolation of the observed rate
        first_ts = rows_chrono[0][5]
        last_ts = rows_chrono[-1][5]
        span_days = max((last_ts - first_ts) / 86400.0, 1.0 / 1440.0)  # floor ~1min, avoid /0
        trades_per_day = len(rows_chrono) / span_days
        avg_pnl_per_trade_scaled = scaled_pnl_total / len(rows_chrono)
        avg_daily_pnl_scaled = avg_pnl_per_trade_scaled * trades_per_day

        growth_projection = {
            "basis": (
                "Linear extrapolation of this account's OBSERVED average $ "
                "P&L per trade x observed trades/day over the logged history. "
                "Not a forecast or guarantee — assumes future trades behave "
                "statistically like past ones."
            ),
            "observed_trades_per_day": round(trades_per_day, 2),
            "observed_avg_pnl_per_trade_usd": round(avg_pnl_per_trade_scaled, 4),
            "projected_pnl_1_day":   round(avg_daily_pnl_scaled, 4),
            "projected_pnl_1_week":  round(avg_daily_pnl_scaled * 7, 4),
            "projected_pnl_1_month": round(avg_daily_pnl_scaled * 30, 4),
            # Floored at 0 — a real account can't go negative; at the
            # current observed rate this projection can imply a wipeout
            # well before 1 month, which is itself an important thing for
            # the dashboard to show plainly rather than a misleading
            # negative number.
            "projected_balance_1_day":   round(max(0.0, scaled_balance + avg_daily_pnl_scaled), 4),
            "projected_balance_1_week":  round(max(0.0, scaled_balance + avg_daily_pnl_scaled * 7), 4),
            "projected_balance_1_month": round(max(0.0, scaled_balance + avg_daily_pnl_scaled * 30), 4),
        }

        return {
            "total_trades": len(rows_chrono),
            "total_staked": round(total_staked, 4),
            "total_pnl": round(total_pnl, 4),
            "pct_pnl_of_staked": round(pct_pnl_of_staked, 4),
            "pct_pnl_series": pct_pnl_series,
            "profit_factor_series": profit_factor_series,
            "scaled_account": {
                "starting_balance": scaled_account_balance,
                "stake_per_trade": scaled_stake,
                "final_balance": round(scaled_balance, 4),
                "total_pnl": round(scaled_pnl_total, 4),
                "pct_return": round(scaled_pct_return, 4),
                "balance_series": scaled_series,
            },
            "growth_projection": growth_projection,
        }

    @property
    def backend_name(self) -> str:
        return self._backend_name


# Module-level singleton — import and use directly:
#   from strategy_stats import stats
#   stats.record_trade(...)
stats = StrategyStats()


if __name__ == "__main__":
    # Quick smoke test
    import random

    random.seed(0)
    for i in range(120):
        stats.record_trade(
            strategy="mean_reversion",
            symbol="R_75",
            entry_score=round(random.uniform(0.6, 0.9), 3),
            won=random.random() < 0.52,
            stake=1.0,
            payout=1.9 if random.random() < 0.52 else 0.0,
            timestamp=time.time() - (120 - i),
        )
    print("Backend:", stats.backend_name)
    print("Win rate:", stats.get_win_rate("mean_reversion", "R_75"))
    print("Underperforming:", stats.is_underperforming("mean_reversion", "R_75"))
    print("All stats:", stats.all_stats())
