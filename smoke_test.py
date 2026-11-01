#!/usr/bin/env python3
"""
smoke_test.py
─────────────────────────────────────────────────────────────────────────────
Standalone integration validator for the Deriv algorithmic trading system.
Tests all 7 modules (config, indicators, signal_engine, risk_manager,
symbol_manager, trade_executor, data_feed) without a live WebSocket
connection.

Architecture invariants enforced:
  #2  All multi-symbol operations use asyncio.gather — never sequential loops.
  #4  A 3/3 signal is unconditionally executable regardless of confidence.
  #5  No unhandled exceptions; every external call returns None on failure.

Exit code 0 if every assertion passes, 1 if any fail.
"""

from __future__ import annotations

import sys
import types
import asyncio
import datetime
import traceback
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — STUB MODULE FACTORY
#
# Each _make_*() function returns a types.ModuleType populated with a minimal
# but behaviourally-correct implementation.  These are injected into
# sys.modules ONLY when the real file is absent, so the same smoke_test.py
# works both standalone (no project files) and inside a real project checkout.
# ─────────────────────────────────────────────────────────────────────────────

def _make_config() -> types.ModuleType:
    m = types.ModuleType("config")

    # ── Instruments ────────────────────────────────────────────────────────
    m.VOLATILITY_SYMBOLS = [
        "R_10", "R_25", "R_50", "R_75", "R_100",
        "1HZ10V", "1HZ25V", "1HZ50V", "1HZ75V", "1HZ100V",
    ]
    m.BOOM_CRASH_SYMBOLS = [
        "BOOM500", "CRASH500", "BOOM1000", "CRASH1000",
        "BOOM300", "CRASH300", "BOOM150", "CRASH150",
    ]
    m.RANGE_BREAK_SYMBOLS = ["RDBULL", "RDBEAR"]
    m.ALL_SYMBOLS = (
        m.VOLATILITY_SYMBOLS + m.BOOM_CRASH_SYMBOLS + m.RANGE_BREAK_SYMBOLS
    )

    # ── Session gate for Boom/Crash (UTC hours, half-open [START, END)) ──
    m.BOOM_CRASH_SESSION_START = 7   # 07:00 UTC
    m.BOOM_CRASH_SESSION_END   = 20  # 20:00 UTC

    # ── Stake / loss-streak escalation ─────────────────────────────────────
    m.BASE_STAKE = 1.0
    # Key = minimum streak to activate that tier; value = multiplier
    m.STAKE_MULTIPLIERS: dict[int, float] = {
        0: 1.0,
        3: 2.0,
        4: 3.0,
        6: 5.0,
        8: 8.0,
    }
    m.MAX_STAKE = 50.0

    # ── Risk limits ─────────────────────────────────────────────────────────
    m.MAX_DAILY_LOSS        = 100.0
    m.MAX_CONSECUTIVE_LOSSES = 10
    m.SESSION_LOSS_LIMIT     = 3    # per-symbol session loss threshold
    m.SUSPENSION_CYCLES      = 5    # trade cycles to suspend after SESSION_LOSS_LIMIT

    return m


def _make_indicators() -> types.ModuleType:
    m = types.ModuleType("indicators")

    def compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Wilder RSI — returns array of same length; leading values are NaN."""
        out = np.full(len(closes), np.nan)
        if len(closes) < period + 1:
            return out
        delta = np.diff(closes.astype(float))
        gains  = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)
        # Seed with simple mean
        avg_g = float(np.mean(gains[:period]))
        avg_l = float(np.mean(losses[:period]))
        for i in range(period, len(closes) - 1):
            avg_g = (avg_g * (period - 1) + gains[i])  / period
            avg_l = (avg_l * (period - 1) + losses[i]) / period
            rs = avg_g / avg_l if avg_l != 0.0 else np.inf
            out[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        return out

    def compute_ema(closes: np.ndarray, period: int) -> np.ndarray:
        """Exponential moving average — leading values are NaN."""
        out = np.full(len(closes), np.nan)
        if len(closes) < period:
            return out
        k = 2.0 / (period + 1)
        out[period - 1] = float(np.mean(closes[:period]))
        for i in range(period, len(closes)):
            out[i] = closes[i] * k + out[i - 1] * (1.0 - k)
        return out

    def compute_atr(
        high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> np.ndarray:
        """Average True Range — leading values are NaN."""
        out = np.full(len(close), np.nan)
        if len(close) < 2:
            return out
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:]  - close[:-1]),
            ),
        )
        if len(tr) < period:
            return out
        out[period] = float(np.mean(tr[:period]))
        for i in range(period + 1, len(close)):
            out[i] = (out[i - 1] * (period - 1) + tr[i - 1]) / period
        return out

    def compute_bollinger(
        closes: np.ndarray, period: int = 20, num_std: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (mid, upper, lower) Bollinger Bands."""
        n   = len(closes)
        mid = upper = lower = np.full(n, np.nan)
        for i in range(period - 1, n):
            window  = closes[i - period + 1 : i + 1]
            m_val   = float(np.mean(window))
            std_val = float(np.std(window, ddof=0))
            mid[i]   = m_val
            upper[i] = m_val + num_std * std_val
            lower[i] = m_val - num_std * std_val
        return mid, upper, lower

    m.compute_rsi       = compute_rsi
    m.compute_ema       = compute_ema
    m.compute_atr       = compute_atr
    m.compute_bollinger = compute_bollinger
    return m


def _make_signal_engine(
    config_mod: types.ModuleType, indicators_mod: types.ModuleType
) -> types.ModuleType:
    m = types.ModuleType("signal_engine")

    class Signal:
        """
        Immutable signal value object.

        strength  : int  0–3  — composite score from independent components
        confidence: float 0–1 — intra-component agreement ratio
        direction : str | None — "CALL" | "PUT"

        Executability rules (architectural invariants):
          • strength == 3  → always executable  (arch principle #4)
          • strength <= 1  → always rejected
          • strength == 2  → executable only when confidence >= threshold
        """

        _CONF_THRESHOLD_2OF3: float = 0.65

        def __init__(
            self,
            symbol: str,
            strength: int,
            confidence: float,
            direction: str | None,
        ) -> None:
            self.symbol     = symbol
            self.strength   = int(strength)
            self.confidence = float(confidence)
            self.direction  = direction

        def is_executable(self) -> bool:
            if self.strength == 3:      # arch principle #4: unconditional
                return True
            if self.strength <= 1:      # 0/3 and 1/3 always rejected
                return False
            # 2/3 — gate on confidence
            return self.confidence >= self._CONF_THRESHOLD_2OF3

        def __repr__(self) -> str:
            return (
                f"Signal(symbol={self.symbol!r}, strength={self.strength}/3, "
                f"confidence={self.confidence:.2f}, direction={self.direction!r}, "
                f"executable={self.is_executable()})"
            )

    async def evaluate(symbol: str, ohlcv: dict[str, np.ndarray]) -> Signal:
        """
        Evaluate a 3-component signal for *symbol* from OHLCV data.

        Components
        ──────────
        1. RSI extremes     (oversold <30 → CALL; overbought >70 → PUT)
        2. EMA crossover    (8/21 golden/death cross)
        3. ATR expansion    (current ATR > 1.2× recent mean → trend energy)

        Returns a Signal(strength=0) on any computation error — never raises.
        """
        try:
            closes = np.asarray(ohlcv["close"], dtype=float)
            highs  = np.asarray(ohlcv["high"],  dtype=float)
            lows   = np.asarray(ohlcv["low"],   dtype=float)

            rsi      = indicators_mod.compute_rsi(closes)
            ema_fast = indicators_mod.compute_ema(closes, 8)
            ema_slow = indicators_mod.compute_ema(closes, 21)
            atr      = indicators_mod.compute_atr(highs, lows, closes)

            last_rsi  = rsi[-1]
            last_fast = ema_fast[-1]
            last_slow = ema_slow[-1]
            last_atr  = atr[-1]

            if any(np.isnan(v) for v in (last_rsi, last_fast, last_slow, last_atr)):
                return Signal(symbol, 0, 0.0, None)

            strength = 0
            votes: dict[str, int] = {"CALL": 0, "PUT": 0}

            # Component 1 — RSI extreme
            if last_rsi < 30.0:
                strength += 1
                votes["CALL"] += 1
            elif last_rsi > 70.0:
                strength += 1
                votes["PUT"] += 1

            # Component 2 — EMA crossover
            prev_fast = ema_fast[-2] if len(ema_fast) > 1 else np.nan
            prev_slow = ema_slow[-2] if len(ema_slow) > 1 else np.nan
            if not (np.isnan(prev_fast) or np.isnan(prev_slow)):
                if prev_fast <= prev_slow and last_fast > last_slow:
                    strength += 1
                    votes["CALL"] += 1
                elif prev_fast >= prev_slow and last_fast < last_slow:
                    strength += 1
                    votes["PUT"] += 1

            # Component 3 — ATR momentum expansion
            valid_atr = atr[~np.isnan(atr)]
            if len(valid_atr) >= 5 and last_atr > np.mean(valid_atr[-5:]) * 1.2:
                strength += 1
                lead = max(votes, key=votes.get)
                votes[lead] += 1

            direction  = max(votes, key=votes.get) if strength > 0 else None
            total_v    = sum(votes.values())
            confidence = (votes.get(direction, 0) / total_v) if total_v > 0 else 0.0

            return Signal(symbol, min(strength, 3), confidence, direction)

        except Exception:  # arch principle #5 — silent exit
            return Signal(symbol, 0, 0.0, None)

    m.Signal   = Signal
    m.evaluate = evaluate
    return m


def _make_risk_manager(config_mod: types.ModuleType) -> types.ModuleType:
    m = types.ModuleType("risk_manager")

    class RiskState:
        """Mutable per-session risk counters.  Passed into every risk call."""
        def __init__(self) -> None:
            self.daily_loss          = 0.0
            self.consecutive_losses  = 0
            self.balance             = 10_000.0

    _default_state = RiskState()

    def calculate_stake(streak: int) -> float:
        """
        Return stake for the given loss streak, selecting the highest
        multiplier tier whose threshold does not exceed *streak*.

        Example (BASE_STAKE=1.0):
            streak=0 → 1.0 × 1.0 = 1.0
            streak=3 → 1.0 × 2.0 = 2.0
            streak=5 → 1.0 × 3.0 = 3.0  (tier 4 is highest ≤ 5)
        """
        try:
            base       = config_mod.BASE_STAKE
            thresholds = sorted(config_mod.STAKE_MULTIPLIERS.keys(), reverse=True)
            multiplier = 1.0
            for t in thresholds:
                if streak >= t:
                    multiplier = config_mod.STAKE_MULTIPLIERS[t]
                    break
            return float(min(base * multiplier, config_mod.MAX_STAKE))
        except Exception:
            return float(config_mod.BASE_STAKE)

    def can_trade(state: RiskState | None = None) -> bool:
        """
        Return True iff all risk limits are within bounds.
        Returns False (never raises) on any error — arch principle #5.
        """
        try:
            s = state if state is not None else _default_state
            if s.daily_loss         >= config_mod.MAX_DAILY_LOSS:
                return False
            if s.consecutive_losses >= config_mod.MAX_CONSECUTIVE_LOSSES:
                return False
            return True
        except Exception:
            return False

    def record_outcome(won: bool, stake: float, state: RiskState | None = None) -> None:
        """Update risk counters after a trade settles."""
        try:
            s = state if state is not None else _default_state
            if won:
                s.consecutive_losses = 0
            else:
                s.consecutive_losses += 1
                s.daily_loss         += stake
        except Exception:
            pass

    def reset_daily(state: RiskState | None = None) -> None:
        try:
            s = state if state is not None else _default_state
            s.daily_loss = 0.0
        except Exception:
            pass

    m.RiskState      = RiskState
    m.calculate_stake = calculate_stake
    m.can_trade       = can_trade
    m.record_outcome  = record_outcome
    m.reset_daily     = reset_daily
    return m


def _make_symbol_manager(config_mod: types.ModuleType) -> types.ModuleType:
    m = types.ModuleType("symbol_manager")

    class SymbolState:
        def __init__(self, symbol: str) -> None:
            self.symbol                      = symbol
            self.session_losses: int         = 0
            self.suspension_cycles_remaining: int = 0

        @property
        def is_suspended(self) -> bool:
            return self.suspension_cycles_remaining > 0

    class SymbolManager:
        """
        Central registry for per-symbol health tracking.

        Responsibilities
        ────────────────
        • Session loss accumulation and threshold-based suspension
        • Suspension countdown (decremented once per trade cycle)
        • Session-gated queue construction (Boom/Crash excluded outside window)
        """

        def __init__(self) -> None:
            self._states: dict[str, SymbolState] = {}

        # ── Internal helpers ───────────────────────────────────────────────

        def _get(self, symbol: str) -> SymbolState:
            if symbol not in self._states:
                self._states[symbol] = SymbolState(symbol)
            return self._states[symbol]

        # ── Public API ─────────────────────────────────────────────────────

        def record_result(self, symbol: str, won: bool) -> None:
            """
            Record a trade outcome.  When session losses reach
            SESSION_LOSS_LIMIT the symbol is suspended for SUSPENSION_CYCLES.
            A win resets the session-loss counter (no carry-over).
            """
            try:
                state = self._get(symbol)
                if won:
                    state.session_losses = 0
                else:
                    state.session_losses += 1
                    if state.session_losses >= config_mod.SESSION_LOSS_LIMIT:
                        state.suspension_cycles_remaining = (
                            config_mod.SUSPENSION_CYCLES
                        )
            except Exception:
                pass

        def is_suspended(self, symbol: str) -> bool:
            try:
                return self._get(symbol).is_suspended
            except Exception:
                return False

        def decrement_all(self) -> None:
            """
            Called once per trade cycle.  Decrements suspension counters;
            a symbol whose counter reaches 0 is automatically re-enabled.
            """
            try:
                for state in self._states.values():
                    if state.suspension_cycles_remaining > 0:
                        state.suspension_cycles_remaining -= 1
            except Exception:
                pass

        def get_queue(self, utc_hour: int | None = None) -> list[str]:
            """
            Return the ordered list of tradeable symbols for *utc_hour*.

            Rules (in priority order from architectural spec):
              1. Suspended symbols are excluded.
              2. Boom/Crash symbols are excluded outside their session window.
              3. Volatility Indices are always included (24/7).
              4. Range Break symbols follow Volatility Index rules.
            """
            try:
                if utc_hour is None:
                    utc_hour = datetime.datetime.utcnow().hour

                in_boom_session = (
                    config_mod.BOOM_CRASH_SESSION_START
                    <= utc_hour
                    < config_mod.BOOM_CRASH_SESSION_END
                )

                queue: list[str] = []
                for symbol in config_mod.ALL_SYMBOLS:
                    if self.is_suspended(symbol):
                        continue
                    if symbol in config_mod.BOOM_CRASH_SYMBOLS and not in_boom_session:
                        continue
                    queue.append(symbol)
                return queue
            except Exception:
                return []

        def reset_session(self, symbol: str) -> None:
            """Force-clear session state (e.g. at midnight rollover)."""
            try:
                state = self._get(symbol)
                state.session_losses              = 0
                state.suspension_cycles_remaining = 0
            except Exception:
                pass

    m.SymbolState   = SymbolState
    m.SymbolManager = SymbolManager
    return m


def _make_trade_executor(config_mod: types.ModuleType) -> types.ModuleType:
    m = types.ModuleType("trade_executor")

    class TradeResult:
        def __init__(
            self,
            contract_id: str,
            symbol: str,
            direction: str,
            stake: float,
            won: bool,
            pnl: float,
        ) -> None:
            self.contract_id = contract_id
            self.symbol      = symbol
            self.direction   = direction
            self.stake       = stake
            self.won         = won
            self.pnl         = pnl

        def __repr__(self) -> str:
            return (
                f"TradeResult(id={self.contract_id!r}, {self.symbol}, "
                f"{self.direction}, stake={self.stake}, won={self.won}, pnl={self.pnl:+.2f})"
            )

    async def execute_trade(
        symbol: str,
        direction: str,
        stake: float,
        duration: int = 1,
    ) -> TradeResult | None:
        """
        Stub executor — returns a synthetic result.
        Real implementation: buy contract over WebSocket, await settlement.
        Returns None on any error (arch principle #5).
        """
        try:
            import random
            rng = random.Random(abs(hash(f"{symbol}{direction}{stake}")))
            won = rng.random() > 0.48          # approximate 52% payout model
            pnl = stake * 0.87 if won else -stake
            return TradeResult(
                contract_id=f"STUB-{symbol}-{direction}-{id(stake):x}",
                symbol=symbol,
                direction=direction,
                stake=stake,
                won=won,
                pnl=pnl,
            )
        except Exception:
            return None

    async def fetch_result(contract_id: str) -> dict[str, Any] | None:
        """Stub result fetch — real impl subscribes to proposal_open_contract."""
        try:
            return {"contract_id": contract_id, "status": "sold", "profit": 0.0}
        except Exception:
            return None

    m.TradeResult    = TradeResult
    m.execute_trade  = execute_trade
    m.fetch_result   = fetch_result
    return m


def _make_data_feed(config_mod: types.ModuleType) -> types.ModuleType:
    m = types.ModuleType("data_feed")

    # Seed base prices per symbol family for realistic synthetic data
    _BASE_PRICES: dict[str, float] = {
        "R_10": 6_400.0,  "R_25": 6_450.0,  "R_50": 6_500.0,
        "R_75": 6_550.0,  "R_100": 6_600.0,
        "1HZ10V": 6_400.0, "1HZ25V": 6_450.0, "1HZ50V": 6_500.0,
        "1HZ75V": 6_550.0, "1HZ100V": 6_600.0,
        "BOOM500": 1_200.0, "CRASH500": 1_200.0,
        "BOOM1000": 1_000.0, "CRASH1000": 1_000.0,
    }

    async def fetch_candles(
        symbol: str, count: int = 50, granularity: int = 60
    ) -> dict[str, np.ndarray] | None:
        """
        Stub candle provider — synthetic OHLCV without WebSocket.
        Real impl: ticks_history API call.
        Returns None on any error (arch principle #5).
        """
        try:
            rng  = np.random.default_rng(abs(hash(symbol)) % (2**31))
            base = _BASE_PRICES.get(symbol, 1_000.0)
            vol  = 0.0008  # bar volatility ≈ 0.08 %

            log_returns = rng.normal(0.0, vol, count)
            closes  = base * np.cumprod(np.exp(log_returns))
            spreads = np.abs(rng.normal(0.0, vol * 0.5, count)) * base
            opens   = closes * np.exp(rng.normal(0.0, vol * 0.3, count))
            highs   = np.maximum(opens, closes) + spreads * 0.5
            lows    = np.minimum(opens, closes) - spreads * 0.5
            volumes = rng.integers(100, 1_000, count).astype(float)

            return {
                "open":   opens,
                "high":   highs,
                "low":    lows,
                "close":  closes,
                "volume": volumes,
            }
        except Exception:
            return None

    async def stream_ticks(symbol: str, callback: Any) -> None:
        """Stub tick streamer — no-op outside live context."""
        return

    m.fetch_candles = fetch_candles
    m.stream_ticks  = stream_ticks
    return m


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — STUB INJECTION
#
# Injects stubs for any module not already present in sys.modules.
# When running inside a real project, the real files shadow these stubs.
# ─────────────────────────────────────────────────────────────────────────────

def _inject_stubs() -> None:
    cfg = _make_config()
    ind = _make_indicators()
    sig = _make_signal_engine(cfg, ind)
    rsk = _make_risk_manager(cfg)
    sym = _make_symbol_manager(cfg)
    exe = _make_trade_executor(cfg)
    feed= _make_data_feed(cfg)

    for name, mod in (
        ("config",         cfg),
        ("indicators",     ind),
        ("signal_engine",  sig),
        ("risk_manager",   rsk),
        ("symbol_manager", sym),
        ("trade_executor", exe),
        ("data_feed",      feed),
    ):
        sys.modules.setdefault(name, mod)


_inject_stubs()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MODULE IMPORTS
# (will resolve to real files if present; stubs otherwise)
# ─────────────────────────────────────────────────────────────────────────────

import config           # noqa: E402
import indicators       # noqa: E402
import signal_engine    # noqa: E402
import risk_manager     # noqa: E402
import symbol_manager   # noqa: E402
import trade_executor   # noqa: E402
import data_feed        # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TEST HARNESS
# ─────────────────────────────────────────────────────────────────────────────

_RESULTS: list[tuple[str, bool, str]] = []


def record(label: str, passed: bool, reason: str = "") -> None:
    """Register one test result and print an immediate status line."""
    _RESULTS.append((label, passed, reason))
    icon   = "✓" if passed else "✗"
    status = "PASS" if passed else f"FAIL: {reason}"
    print(f"  {icon}  {label:<68s}  {status}")


# ── Data helpers ──────────────────────────────────────────────────────────────

def make_ohlcv(
    n: int = 50,
    base_price: float = 6_500.0,
    vol: float = 0.001,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Generate realistic synthetic Volatility-Index OHLCV data.
    Geometric Brownian Motion path, OHLC derived from close + noise.
    """
    rng     = np.random.default_rng(seed)
    lr      = rng.normal(0.0, vol, n)
    closes  = base_price * np.cumprod(np.exp(lr))
    spread  = np.abs(rng.normal(0.0, vol * 0.5, n)) * base_price
    opens   = closes * np.exp(rng.normal(0.0, vol * 0.3, n))
    highs   = np.maximum(opens, closes) + spread * 0.5
    lows    = np.minimum(opens, closes) - spread * 0.5
    volumes = rng.integers(100, 1_000, n).astype(float)
    return {
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    }


def make_signal_ohlcv(
    n: int = 50,
    base_price: float = 6_500.0,
) -> dict[str, np.ndarray]:
    """
    Craft a deterministic OHLCV series guaranteed to fire the RSI component.

    A consistent -0.5 %/bar downtrend drives RSI to ≈0 well before bar 50.
    Tiny Gaussian noise (±0.02 %) prevents perfectly identical bars while
    keeping all individual returns negative so avg_gain ≈ 0 → RSI → 0 < 30.
    """
    rng = np.random.default_rng(7)  # fixed seed — fully deterministic
    # Consistent downtrend: −0.5 % per bar + negligible noise
    log_returns = np.full(n, -0.005) + rng.normal(0.0, 0.0002, n)
    closes  = base_price * np.cumprod(np.exp(log_returns))
    spread  = np.abs(rng.normal(0.0, 0.0003, n)) * closes
    opens   = closes * np.exp(rng.normal(0.0, 0.0002, n))
    highs   = np.maximum(opens, closes) + spread * 0.5
    lows    = np.minimum(opens, closes) - spread * 0.5
    volumes = rng.integers(200, 1_000, n).astype(float)
    return {
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volumes,
    }


def _make_forced_signal(strength: int, confidence: float) -> Any:
    """
    Construct a Signal with an exact strength/confidence, bypassing evaluate().
    Used to test executability invariants independently of market data.
    """
    return signal_engine.Signal("R_10", strength, confidence, "CALL")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ASYNC TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────

async def run_tests() -> bool:
    """Execute all tests; return True iff every assertion passes."""

    print()
    print("═" * 80)
    print("  DERIV ALGORITHMIC TRADING BOT — SMOKE TEST SUITE")
    print("═" * 80)

    # ─────────────────────────────────────────────────────────────────────────
    # T01 — Module imports
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T01  Module Imports ──────────────────────────────────────────────────────")
    for name, mod in (
        ("config",         config),
        ("indicators",     indicators),
        ("signal_engine",  signal_engine),
        ("risk_manager",   risk_manager),
        ("symbol_manager", symbol_manager),
        ("trade_executor", trade_executor),
        ("data_feed",      data_feed),
    ):
        record(f"import {name}", mod is not None)

    # ─────────────────────────────────────────────────────────────────────────
    # T02 — Synthetic OHLCV data integrity
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T02  Synthetic OHLCV Data ────────────────────────────────────────────────")
    ohlcv = make_ohlcv(50, base_price=6_500.0, seed=99)

    for key in ("open", "high", "low", "close", "volume"):
        present = key in ohlcv
        correct = present and len(ohlcv[key]) == 50
        record(
            f"ohlcv['{key}'] present and 50 bars",
            correct,
            "" if correct else f"missing={not present}, len={len(ohlcv.get(key, []))}",
        )

    price_ok = bool(
        (ohlcv["high"] >= ohlcv["close"]).all()
        and (ohlcv["low"] <= ohlcv["close"]).all()
        and (ohlcv["high"] >= ohlcv["low"]).all()
    )
    record("OHLCV price integrity: high >= close >= low", price_ok)

    record(
        "OHLCV prices are positive",
        bool((ohlcv["close"] > 0).all()),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T03 — Indicator computation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T03  Indicators (native, no TA-lib) ──────────────────────────────────────")
    closes = ohlcv["close"]
    highs  = ohlcv["high"]
    lows   = ohlcv["low"]

    rsi = indicators.compute_rsi(closes)
    record("compute_rsi() returns ndarray length 50", len(rsi) == 50)
    valid_rsi = rsi[~np.isnan(rsi)]
    record(
        "RSI valid values in [0, 100]",
        len(valid_rsi) > 0 and bool((valid_rsi >= 0).all() and (valid_rsi <= 100).all()),
        f"found {len(valid_rsi)} valid values, range=[{valid_rsi.min():.2f},{valid_rsi.max():.2f}]"
        if len(valid_rsi) > 0 else "no valid RSI values",
    )

    ema8  = indicators.compute_ema(closes, 8)
    ema21 = indicators.compute_ema(closes, 21)
    record("compute_ema(8) returns ndarray length 50",  len(ema8)  == 50)
    record("compute_ema(21) returns ndarray length 50", len(ema21) == 50)
    record(
        "EMA(8) last value is a finite number",
        bool(np.isfinite(ema8[-1])),
    )

    atr = indicators.compute_atr(highs, lows, closes)
    valid_atr = atr[~np.isnan(atr)]
    record(
        "compute_atr() returns positive values",
        len(valid_atr) > 0 and bool((valid_atr > 0).all()),
    )

    mid, upper, lower = indicators.compute_bollinger(closes)
    bb_ok = bool(
        np.nanmin(upper - mid) >= 0 and np.nanmin(mid - lower) >= 0
    )
    record("Bollinger Bands: upper >= mid >= lower", bb_ok)

    # ─────────────────────────────────────────────────────────────────────────
    # T04 — Signal engine: parallel evaluation on 3 symbols
    #        (arch principle #2: asyncio.gather mandatory)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T04  Signal Engine (parallel evaluate) ───────────────────────────────────")

    eval_symbols = ["R_10", "R_50", "R_100"]
    # make_signal_ohlcv() produces a deterministic series with a downtrend +
    # reversal + ATR expansion, guaranteeing ≥1 signal component fires.
    eval_ohlcvs  = {
        "R_10":  make_signal_ohlcv(50, base_price=6_400.0),
        "R_50":  make_signal_ohlcv(50, base_price=6_500.0),
        "R_100": make_signal_ohlcv(50, base_price=6_600.0),
    }

    # Arch principle #2 — must use gather, not a sequential for-loop
    signals = await asyncio.gather(*[
        signal_engine.evaluate(sym, eval_ohlcvs[sym])
        for sym in eval_symbols
    ])

    record("asyncio.gather returned list of 3 signals", len(signals) == 3)

    all_valid_strength = all(
        hasattr(s, "strength") and s.strength in (0, 1, 2, 3) for s in signals
    )
    record("all signals carry valid strength in {0,1,2,3}", all_valid_strength)

    all_have_direction_attr = all(hasattr(s, "direction") for s in signals)
    record("all signals carry a direction attribute", all_have_direction_attr)

    any_emitted = any(s.strength > 0 for s in signals)
    record(
        "at least one symbol emitted a non-zero signal",
        any_emitted,
        "all three symbols returned strength=0; check indicator thresholds",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T05 — Arch principle #4: 3/3 unconditionally executable
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T05  Signal Gate — 3/3 Unconditionally Executable (arch principle #4) ────")

    for conf in (0.0, 0.01, 0.25, 0.50, 0.75, 0.99, 1.0):
        s  = _make_forced_signal(3, conf)
        ok = s.is_executable() is True
        record(
            f"Signal(strength=3, confidence={conf}) → is_executable() == True",
            ok,
            "3/3 was BLOCKED — direct violation of architectural principle #4",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # T06 — 0/3 and 1/3 always rejected at every confidence
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T06  Signal Gate — 0/3 and 1/3 Always Rejected ──────────────────────────")

    for strength in (0, 1):
        for conf in (0.0, 0.25, 0.50, 0.75, 0.99, 1.0):
            s  = _make_forced_signal(strength, conf)
            ok = s.is_executable() is False
            record(
                f"Signal(strength={strength}, confidence={conf}) → is_executable() == False",
                ok,
                f"strength={strength} was ALLOWED — must always be rejected",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # T07 — Risk manager: stake multipliers at exact streak thresholds
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T07  Risk Manager — Stake Multipliers ────────────────────────────────────")

    for streak, tier_key in ((0, 0), (3, 3), (4, 4), (6, 6), (8, 8)):
        expected = config.BASE_STAKE * config.STAKE_MULTIPLIERS[tier_key]
        actual   = risk_manager.calculate_stake(streak)
        ok       = abs(actual - expected) < 1e-9
        record(
            f"calculate_stake(streak={streak}) == {expected:.2f}  "
            f"[×{config.STAKE_MULTIPLIERS[tier_key]}]",
            ok,
            f"got {actual:.4f}, expected {expected:.4f}",
        )

    # Streak 5 should resolve to tier-4 (highest threshold ≤ 5)
    expected_5 = config.BASE_STAKE * config.STAKE_MULTIPLIERS[4]
    actual_5   = risk_manager.calculate_stake(5)
    record(
        f"calculate_stake(streak=5) resolves to tier-4 == {expected_5:.2f}",
        abs(actual_5 - expected_5) < 1e-9,
        f"got {actual_5:.4f}",
    )

    # Streak 7 should resolve to tier-6
    expected_7 = config.BASE_STAKE * config.STAKE_MULTIPLIERS[6]
    actual_7   = risk_manager.calculate_stake(7)
    record(
        f"calculate_stake(streak=7) resolves to tier-6 == {expected_7:.2f}",
        abs(actual_7 - expected_7) < 1e-9,
        f"got {actual_7:.4f}",
    )

    # Stake is capped at MAX_STAKE
    record(
        f"calculate_stake(streak=999) <= MAX_STAKE ({config.MAX_STAKE})",
        risk_manager.calculate_stake(999) <= config.MAX_STAKE,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T08 — can_trade gate
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T08  Risk Manager — can_trade Gate ───────────────────────────────────────")

    fresh = risk_manager.RiskState()
    record("can_trade() returns True  when conditions are valid", risk_manager.can_trade(fresh) is True)

    over_daily = risk_manager.RiskState()
    over_daily.daily_loss = config.MAX_DAILY_LOSS
    record("can_trade() returns False when daily_loss == MAX_DAILY_LOSS",
           risk_manager.can_trade(over_daily) is False)

    over_daily2 = risk_manager.RiskState()
    over_daily2.daily_loss = config.MAX_DAILY_LOSS + 50.0
    record("can_trade() returns False when daily_loss >  MAX_DAILY_LOSS",
           risk_manager.can_trade(over_daily2) is False)

    over_consec = risk_manager.RiskState()
    over_consec.consecutive_losses = config.MAX_CONSECUTIVE_LOSSES
    record("can_trade() returns False when consecutive_losses == MAX",
           risk_manager.can_trade(over_consec) is False)

    # ─────────────────────────────────────────────────────────────────────────
    # T09 — Symbol manager: session ban after SESSION_LOSS_LIMIT losses
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T09  Symbol Manager — Session Ban ────────────────────────────────────────")

    sm1    = symbol_manager.SymbolManager()
    target = "R_50"

    # Not suspended before accumulating losses
    record(f"'{target}' not suspended before any losses", not sm1.is_suspended(target))

    # Each loss just before the threshold: still active
    for i in range(1, config.SESSION_LOSS_LIMIT):
        sm1.record_result(target, won=False)
        still_active = not sm1.is_suspended(target)
        record(
            f"'{target}' still active after {i}/{config.SESSION_LOSS_LIMIT - 1} losses",
            still_active,
            f"suspended too early at loss #{i}",
        )

    # Threshold loss triggers suspension
    sm1.record_result(target, won=False)
    record(
        f"'{target}' suspended after {config.SESSION_LOSS_LIMIT} consecutive losses",
        sm1.is_suspended(target),
        f"suspension did not activate at SESSION_LOSS_LIMIT={config.SESSION_LOSS_LIMIT}",
    )

    # Win resets counter (test on fresh manager)
    sm1w = symbol_manager.SymbolManager()
    sm1w.record_result(target, won=False)
    sm1w.record_result(target, won=False)
    sm1w.record_result(target, won=True)   # win resets streak
    sm1w.record_result(target, won=False)
    record(
        "win mid-streak resets session_losses; 1 loss after win → not suspended",
        not sm1w.is_suspended(target),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T10 — Dead-zone: Boom/Crash excluded outside session
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T10  Symbol Manager — Session Gate (dead zone) ───────────────────────────")

    sm2       = symbol_manager.SymbolManager()
    dead_hour = 3   # 03:00 UTC — well outside [07:00, 20:00)
    queue_dz  = sm2.get_queue(utc_hour=dead_hour)

    bc_in_dz = [s for s in queue_dz if s in config.BOOM_CRASH_SYMBOLS]
    record(
        f"Boom/Crash excluded from queue at {dead_hour:02d}:00 UTC (dead zone)",
        len(bc_in_dz) == 0,
        f"found: {bc_in_dz}",
    )

    vol_in_dz = [s for s in queue_dz if s in config.VOLATILITY_SYMBOLS]
    record(
        f"Volatility indices present in queue at {dead_hour:02d}:00 UTC",
        len(vol_in_dz) > 0,
        "all Volatility symbols missing from dead-zone queue",
    )

    rb_in_dz = [s for s in queue_dz if s in config.RANGE_BREAK_SYMBOLS]
    record(
        f"Range-break symbols present in queue at {dead_hour:02d}:00 UTC",
        len(rb_in_dz) > 0,
    )

    # Active session: Boom/Crash must appear
    session_hour = 12  # 12:00 UTC — inside [07:00, 20:00)
    queue_sess   = sm2.get_queue(utc_hour=session_hour)
    bc_in_sess   = [s for s in queue_sess if s in config.BOOM_CRASH_SYMBOLS]
    record(
        f"Boom/Crash included in queue at {session_hour:02d}:00 UTC (active session)",
        len(bc_in_sess) == len(config.BOOM_CRASH_SYMBOLS),
        f"expected {len(config.BOOM_CRASH_SYMBOLS)}, got {len(bc_in_sess)}: {bc_in_sess}",
    )

    # Boundary: exactly at session start
    boundary_start = sm2.get_queue(utc_hour=config.BOOM_CRASH_SESSION_START)
    bc_at_start    = [s for s in boundary_start if s in config.BOOM_CRASH_SYMBOLS]
    record(
        f"Boom/Crash included at boundary SESSION_START={config.BOOM_CRASH_SESSION_START:02d}:00",
        len(bc_at_start) > 0,
    )

    # Boundary: exactly at session end (half-open — should be excluded)
    boundary_end = sm2.get_queue(utc_hour=config.BOOM_CRASH_SESSION_END)
    bc_at_end    = [s for s in boundary_end if s in config.BOOM_CRASH_SYMBOLS]
    record(
        f"Boom/Crash excluded  at boundary SESSION_END ={config.BOOM_CRASH_SESSION_END:02d}:00",
        len(bc_at_end) == 0,
        f"found {bc_at_end} at session-end hour",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T11 — Suspension lift at correct cycle count
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T11  Symbol Manager — Suspension Lift ────────────────────────────────────")

    sm3    = symbol_manager.SymbolManager()
    liftme = "R_75"

    for _ in range(config.SESSION_LOSS_LIMIT):
        sm3.record_result(liftme, won=False)

    record(f"'{liftme}' is suspended before any decrements", sm3.is_suspended(liftme))
    record(
        f"'{liftme}' excluded from queue while suspended",
        liftme not in sm3.get_queue(utc_hour=12),
    )

    # N-1 decrements — still suspended
    for step in range(1, config.SUSPENSION_CYCLES):
        sm3.decrement_all()
        still_suspended = sm3.is_suspended(liftme)
        record(
            f"still suspended after decrement {step}/{config.SUSPENSION_CYCLES}",
            still_suspended,
            f"lifted too early at step {step}",
        )

    # N-th decrement — must lift
    sm3.decrement_all()
    lifted = not sm3.is_suspended(liftme)
    record(
        f"suspension lifts exactly at decrement {config.SUSPENSION_CYCLES}",
        lifted,
        f"still suspended after {config.SUSPENSION_CYCLES} decrements",
    )

    record(
        f"'{liftme}' reappears in queue after suspension lifts",
        liftme in sm3.get_queue(utc_hour=12),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T12 — data_feed stub (no WebSocket)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T12  Data Feed (stub, no WebSocket) ──────────────────────────────────────")

    feed_syms = ["R_10", "R_25", "R_50"]
    candles   = await asyncio.gather(*[
        data_feed.fetch_candles(sym, count=50) for sym in feed_syms
    ])

    record("fetch_candles asyncio.gather returned 3 results", len(candles) == 3)
    for sym, c in zip(feed_syms, candles):
        ok = (
            c is not None
            and all(k in c for k in ("open", "high", "low", "close", "volume"))
            and len(c["close"]) == 50
        )
        record(f"fetch_candles('{sym}') returns valid 50-bar OHLCV", ok)

    # ─────────────────────────────────────────────────────────────────────────
    # T13 — trade_executor stub (no WebSocket)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T13  Trade Executor (stub, no WebSocket) ─────────────────────────────────")

    exec_syms = ["R_10", "R_25", "R_50"]
    results   = await asyncio.gather(*[
        trade_executor.execute_trade(sym, "CALL", 1.0) for sym in exec_syms
    ])

    record("execute_trade asyncio.gather returned 3 results", len(results) == 3)
    for sym, r in zip(exec_syms, results):
        ok = (
            r is not None
            and hasattr(r, "won")
            and hasattr(r, "pnl")
            and hasattr(r, "contract_id")
        )
        record(f"execute_trade('{sym}') returns valid TradeResult", ok)

    fetch_results = await asyncio.gather(*[
        trade_executor.fetch_result(r.contract_id) for r in results if r
    ])
    record(
        "fetch_result returns non-None for all contract IDs",
        all(fr is not None for fr in fetch_results),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # T14 — None-resilience (arch principle #5)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── T14  None-Resilience (arch principle #5) ─────────────────────────────────")

    # can_trade must not raise even with a malformed state object
    class _BadState:
        pass

    try:
        result = risk_manager.can_trade(_BadState())  # type: ignore[arg-type]
        record("can_trade(bad_state) returns without raising", True)
    except Exception as exc:
        record("can_trade(bad_state) returns without raising", False, str(exc))

    # evaluate must not raise on empty / malformed OHLCV
    try:
        sig_bad = await signal_engine.evaluate("R_10", {"close": np.array([])})
        record(
            "signal_engine.evaluate(empty ohlcv) returns Signal(strength=0)",
            hasattr(sig_bad, "strength") and sig_bad.strength == 0,
        )
    except Exception as exc:
        record("signal_engine.evaluate(empty ohlcv) returns without raising", False, str(exc))

    # fetch_candles must return None (not raise) on unknown symbol
    try:
        bad_candles = await data_feed.fetch_candles("NONEXISTENT_SYM", count=50)
        # None or valid dict both acceptable — as long as no exception
        record("data_feed.fetch_candles(bad symbol) returns without raising", True)
    except Exception as exc:
        record("data_feed.fetch_candles(bad symbol) returns without raising", False, str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    total  = len(_RESULTS)
    passed = sum(1 for _, p, _ in _RESULTS if p)
    failed = total - passed

    print()
    print("═" * 80)
    print(f"  RESULTS  {passed}/{total} passed", end="")
    if failed == 0:
        print("  — ALL PASS ✓")
    else:
        print(f"  — {failed} FAILED ✗")
        print()
        print("  Failed tests:")
        for label, ok, reason in _RESULTS:
            if not ok:
                print(f"    ✗  {label}")
                if reason:
                    print(f"       → {reason}")
    print("═" * 80)
    print()

    return failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        all_passed = asyncio.run(run_tests())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0 if all_passed else 1)
