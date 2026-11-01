"""
risk_manager.py – Phase C risk overlay for the SIFM bot.

v17 — PLS (Progressive Loss Scaling) rewrite.

KEY CHANGES vs v16
──────────────────
• Replaced all aggressive/Martingale-adjacent scaling with the
  research-documented safest recovery method: Progressive Loss Scaling.
  PLS scales stake and concurrency UP only on confirmed win streaks,
  and collapses INSTANTLY to baseline on any loss. Martingale-style
  doubling-after-loss is never used anywhere in this module.

BASE STAKE
──────────
  base_stake = config.BASE_STAKE_PCT × current_live_balance
  Recalculated from live balance before EVERY single trade.
  As balance grows, base stake grows — pure compounding.
  Fixed dollar amounts are never used as base.

PLS WIN SCALING
───────────────
  Streak 0–2   →  1.0×  stake
  Streak ≥ 3   →  1.5×  stake
  Streak ≥ 5   →  2.0×  stake
  Streak ≥ 8   →  3.0×  stake
  Streak ≥ 12  →  4.0×  stake

  Final stake = base × multiplier, hard-capped at config.MAX_STAKE.

ON ANY LOSS — IMMEDIATE RESET
──────────────────────────────
  _win_streak  → 0
  _multiplier  → 1.0
  _extra_slots → 0
  Next base recalculated immediately from current (reduced) balance.
  Stake is NEVER increased to recover a loss.

CONCURRENT SLOT SCALING
────────────────────────
  Default: config.MAX_CONCURRENT_TRADES
  Streak ≥ 3   →  +2 extra slots
  Streak ≥ 5   →  +4 extra slots
  Streak ≥ 8   →  +6 extra slots
  Streak ≥ 12  →  +8 extra slots
  Any loss     →  reset to default immediately

can_trade() GATES
──────────────────
  True iff: open_contracts < current_concurrent_limit
        AND current_balance > MIN_STAKE
  Nothing else blocks trading.

POST-CLOSE LOG
───────────────
  PLS | Streak: +{N} | Multiplier: {M}x | Balance: ${B:.4f} | Next stake: ${S:.4f}

KELLY OVERLAY (v18 addition)
──────────────────────────────
  Sits ON TOP of PLS. Does not touch PLS's own math (base stake,
  multiplier tiers, streak logic are all untouched from v17).

  Per (strategy, symbol) pair, using strategy_stats.py's rolling data:
    p = rolling win rate
    b = avg_win_payout_ratio - 1   (net odds: profit per $1 staked on a win)
    q = 1 - p
    f* = (b*p - q) / b
    adjusted_f* = f* × config.KELLY_FRACTION_MULTIPLIER

  Requires config.KELLY_MIN_TRADES (falls back to 20 if absent — this
  constant does not exist in config.py yet, see reply) logged trades
  for that pair before it activates. Below that, or if payout data is
  missing, the overlay is a no-op and PLS's stake passes through
  unchanged.

  Once active:
    adjusted_f* <= 0   → stake forced to $0.00 for that pair, no matter
                         what PLS's multiplier says.
    adjusted_f* > 0    → kelly_stake = adjusted_f* × balance
                         final_stake = min(pls_stake, kelly_stake)
                         (Kelly caps/scales PLS down, never scales it up)

  Every recalculation is logged with its p/b/q/f* inputs regardless of
  outcome.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import config
import strategy_stats

logger = logging.getLogger(__name__)


# ── PLS tier table ──────────────────────────────────────────────────────────
#
# Disabled per user request: stake no longer scales with win streak.
# _pls_tier() always returns (1.0, 0) — kept as a function (rather than
# deleted) so _update_multiplier()/callers don't need to change.
_PLS_TIERS: list[tuple[int, float, int]] = [
    (0, 1.0, 0),   # baseline — always matches, streak has no effect
]


def _pls_tier(win_streak: int) -> tuple[float, int]:
    """
    Win-streak stake scaling is disabled — always returns (1.0, 0)
    regardless of win_streak. Stake sizing instead comes purely from
    base_stake = BASE_STAKE_PCT × current_balance in _compute_stake().
    """
    return 1.0, 0


# ── Bot state enum ────────────────────────────────────────────────────────────

class BotState(Enum):
    RUNNING  = auto()
    DRAINING = auto()
    STOPPED  = auto()


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    symbol:      str
    direction:   str
    stake:       float
    entry_price: float
    exit_price:  Optional[float] = None
    pnl:         float = 0.0
    won:         bool  = False
    timestamp:   float = field(default_factory=time.time)


# ── RiskManager ───────────────────────────────────────────────────────────────

class RiskManager:
    """
    Stateful risk and position-sizing layer using Progressive Loss Scaling.

    PLS Design
    ──────────
    base_stake   = BASE_STAKE_PCT × live_balance  (recalculated every trade)
    actual_stake = base_stake × pls_multiplier    (streak-driven, win-only)
    actual_stake clamped to [MIN_STAKE, MAX_STAKE]

    Any loss instantly collapses: multiplier → 1.0×, extra_slots → 0,
    win_streak → 0. Stake NEVER increases to recover a loss. This is
    explicitly not Martingale — there is no doubling after a loss, and
    scaling only ever happens off confirmed, already-banked wins.
    """

    def __init__(
        self,
        risk_per_trade: float = None,
        min_stake:      float = None,
        max_stake:      float = None,
        max_concurrent: int   = None,
        deriv_client=None,
    ):
        # Config resolution
        # STAKE_PCT_MULTIPLIER, when set, is a deliberate user override of
        # the conservative BASE_STAKE_PCT default — e.g. 0.30 for 30% of
        # balance per trade. PLS scaling and the Kelly overlay below still
        # apply on top of whichever value ends up here; neither is bypassed.
        override_pct = getattr(config, "STAKE_PCT_MULTIPLIER", None)
        self.base_stake_pct = (
            override_pct if override_pct is not None
            else getattr(config, "BASE_STAKE_PCT", 0.01)
        )
        self.risk_per_trade = risk_per_trade or self.base_stake_pct
        self.min_stake      = min_stake      or getattr(config, "MIN_STAKE",             0.35)
        self.max_stake      = max_stake      or getattr(config, "MAX_STAKE",             50.0)
        self.max_concurrent = max_concurrent or getattr(config, "MAX_CONCURRENT_TRADES", 15)
        self._deriv_client  = deriv_client

        # ── Kelly overlay config ──────────────────────────────────────
        self.kelly_fraction_multiplier = getattr(config, "KELLY_FRACTION_MULTIPLIER", 0.25)
        # Implementation Brief v5 / A3 — config.KELLY_MIN_TRADES is now an
        # explicit, documented constant in config.py; the getattr default
        # of 20 here is just a defensive fallback if it's ever removed.
        self.kelly_min_trades = getattr(config, "KELLY_MIN_TRADES", 20)

        # Live balance — updated by set_balance() and _fetch_live_balance()
        self._current_balance: float = 0.0
        self._balance_cycle:   int   = -1
        self._current_cycle:   int   = 0

        # Day-reset bookkeeping
        self._day_tag:           str   = ""
        self._day_start_balance: float = 0.0

        # PLS win-streak counter (never goes negative; loss resets to 0)
        self._win_streak: int = 0

        # ── Equity curve stabilization state (win-rate/drawdown pass, Aug
        # 2026) — see config.py's EQUITY CURVE STABILIZATION section.
        # Separate from PLS's _win_streak: PLS already resets to 0 on any
        # loss but that only tells you "not currently winning", not "how
        # many losses in a row" — this counter tracks the latter.
        self._peak_balance: float = 0.0
        self._loss_streak:  int   = 0

        # Explicit PLS state mirrors (used for immediate reset on loss)
        self._multiplier:  float = 1.0
        self._extra_slots: int   = 0

        # Concurrent trade tracking
        self._open_trade_count: int = 0

        # Bot state
        self._bot_state: BotState = BotState.RUNNING

        # Pause-cycle countdown (kept for bot_engine compatibility)
        self._pause_cycles_remaining: int = 0

        # Session stats
        self.total_trades: int   = 0
        self.wins:         int   = 0
        self.losses:       int   = 0
        self.total_pnl:    float = 0.0
        self._trades: List[TradeRecord] = []

    # ── Bot-state management ──────────────────────────────────────────────────

    def set_bot_state(self, state: BotState) -> None:
        self._bot_state = state
        logger.info(f"Bot state → {state.name}")

    @property
    def bot_state(self) -> BotState:
        return self._bot_state

    # ── Cycle clock ───────────────────────────────────────────────────────────

    def tick_cycle(self) -> None:
        self._current_cycle += 1

    # ── Pause-cycle management (bot_engine compatibility) ─────────────────────

    @property
    def pause_cycles_remaining(self) -> int:
        return self._pause_cycles_remaining

    def decrement_pause(self) -> None:
        if self._pause_cycles_remaining > 0:
            self._pause_cycles_remaining -= 1

    def consume_pause_cycle(self) -> None:
        self.decrement_pause()

    # ── Balance management ────────────────────────────────────────────────────

    def set_balance(self, balance: float) -> None:
        self._current_balance = balance
        if self._day_start_balance == 0:
            self._day_start_balance = balance
        if balance > self._peak_balance:
            self._peak_balance = balance
        logger.info(f"BALANCE UPDATED: ${balance:.4f}")
        self._balance_cycle   = self._current_cycle
        self._handle_day_rollover(balance)

    async def _fetch_live_balance(self) -> float:
        """
        Return the most current balance available.
        Hits the Deriv client when stale; falls back to cached value if
        the client is absent or the call fails.
        """
        if self._deriv_client is None:
            return self._current_balance

        if self._balance_cycle == self._current_cycle and self._current_balance > 0:
            return self._current_balance

        try:
            balance = await self._deriv_client.get_balance()
            if balance is not None and balance >= 0:
                self._current_balance = float(balance)
                if self._current_balance > self._peak_balance:
                    self._peak_balance = self._current_balance
                self._balance_cycle   = self._current_cycle
                self._handle_day_rollover(self._current_balance)
        except Exception:
            pass

        return self._current_balance

    def _handle_day_rollover(self, balance: float) -> None:
        today = _today_tag()
        if today != self._day_tag:
            self._day_tag        = today
            self._win_streak     = 0
            self._multiplier     = 1.0
            self._extra_slots    = 0
            self._day_start_balance = balance if balance > 0 else self._day_start_balance
            logger.info(
                f"New trading day {today} | "
                f"Starting balance: ${self._day_start_balance:.4f}")
        if self._day_start_balance == 0.0 and balance > 0:
            self._day_start_balance = balance
            logger.info(f"Day-start balance initialised: ${balance:.4f}")

    # ── PLS multiplier update ──────────────────────────────────────────────────

    def _update_multiplier(self) -> None:
        """Sync _multiplier and _extra_slots to the current win streak tier."""
        self._multiplier, self._extra_slots = _pls_tier(self._win_streak)

    # ── PLS properties (spec-required public interface) ────────────────────────

    @property
    def current_streak(self) -> int:
        """Current consecutive win count (0 after any loss)."""
        return self._win_streak

    @property
    def current_multiplier(self) -> float:
        """Active PLS stake multiplier derived from win streak."""
        return self._multiplier

    @property
    def current_concurrent_limit(self) -> int:
        """
        Effective concurrent-trade ceiling.
        = MAX_CONCURRENT_TRADES + win-streak slot bonus.
        Collapses to MAX_CONCURRENT_TRADES on any loss.
        """
        return self.max_concurrent + self._extra_slots

    @property
    def next_stake(self) -> float:
        """
        Preview of the stake that would be used for the very next trade,
        computed from the current cached balance (synchronous; no await).
        """
        return self._compute_stake(self._current_balance)

    # ── Legacy aliases (bot_engine compatibility) ──────────────────────────────

    @property
    def win_streak(self) -> int:
        return self._win_streak

    @property
    def loss_streak(self) -> int:
        # PLS tracks only win streaks; loss streak is always 0 or 1
        return 0

    # ── PLS stake computation ───────────────────────────────────────────────────

    def _compute_stake(self, balance: float) -> float:
        """
        Core PLS formula (synchronous, no I/O).

          base_stake   = max(BASE_STAKE_PCT × balance, MIN_STAKE)
          actual_stake = base_stake × current_multiplier
          actual_stake = clamp(actual_stake, MIN_STAKE, MAX_STAKE)
        """
        if balance <= 0:
            return self.min_stake

        base_stake   = max(balance * self.base_stake_pct, self.min_stake)
        actual_stake = base_stake * self._multiplier
        actual_stake = min(actual_stake, self.max_stake)
        actual_stake = max(actual_stake, self.min_stake)
        return round(actual_stake, 2)

    # ── Dynamic, ATR-normalized stop-loss (Implementation Brief v4, Fix H) ────

    def compute_dynamic_stop_loss_pct(self, atr_pct: float, multiplier: int) -> float:
        """
        Converts a target stop distance of STOP_ATR_MULT * ATR (real price
        terms) into the stop_loss_pct (% of stake) that buy_multiplier()
        expects, so the dollar stop stays calibrated to actual volatility
        regardless of which multiplier a symbol is forced into.

        Only used for config.VOL_MULTIPLIER_SYMBOLS (see bot_engine.py's
        _execute()) — STOP_LOSS_MAP / DEFAULT_STOP_LOSS_PCT keep governing
        Boom/Crash and everything else exactly as before.
        """
        stop_atr_mult = getattr(config, "STOP_ATR_MULT", 2.0)
        raw_pct = stop_atr_mult * atr_pct * multiplier * 100.0
        lo = getattr(config, "DYNAMIC_STOP_LOSS_PCT_MIN", 15.0)
        hi = getattr(config, "DYNAMIC_STOP_LOSS_PCT_MAX", 90.0)
        return float(max(lo, min(hi, raw_pct)))

    # ── Kelly overlay ────────────────────────────────────────────────────────

    def compute_kelly_fraction(self, strategy: str, symbol: str) -> Optional[float]:
        """
        Compute the config.KELLY_FRACTION_MULTIPLIER-adjusted Kelly
        fraction for a (strategy, symbol) pair using strategy_stats.py's
        rolling win rate and average win-payout ratio.

        Returns:
          None  → not enough data yet (fewer than self.kelly_min_trades
                  logged trades for this pair, or no winning-trade
                  payout data available). Caller should treat this as
                  "overlay inactive" and let PLS's stake pass through.
          0.0   → data is sufficient but there is no computable edge
                  (b <= 0) or the Kelly formula itself is <= 0. Caller
                  must force stake to zero for this pair.
          >0.0  → the adjusted Kelly-optimal fraction of balance to risk.

        Every call is logged with its inputs, regardless of outcome.
        """
        rate, ci_low, ci_high, n = strategy_stats.stats.get_win_rate(
            strategy, symbol, window=strategy_stats.DEFAULT_WINDOW
        )
        if n < self.kelly_min_trades:
            logger.info(
                f"KELLY | {strategy}/{symbol} | n={n} < "
                f"kelly_min_trades={self.kelly_min_trades} | overlay inactive, "
                f"PLS stake passes through unchanged")
            return None

        avg_win_ratio = strategy_stats.stats.get_avg_win_payout_ratio(
            strategy, symbol, window=strategy_stats.DEFAULT_WINDOW
        )
        if avg_win_ratio is None:
            logger.info(
                f"KELLY | {strategy}/{symbol} | n={n} but no winning-trade "
                f"payout data available | overlay inactive, PLS stake passes "
                f"through unchanged")
            return None

        p = rate
        q = 1.0 - p
        b = avg_win_ratio - 1.0  # net odds: profit per $1 staked on a win

        if b <= 0:
            logger.info(
                f"KELLY | {strategy}/{symbol} | p={p:.4f} q={q:.4f} "
                f"avg_win_payout_ratio={avg_win_ratio:.4f} b={b:.4f} n={n} | "
                f"b<=0, no computable edge -> fraction forced to 0.0")
            return 0.0

        raw_fraction = (b * p - q) / b
        adjusted_fraction = raw_fraction * self.kelly_fraction_multiplier

        logger.info(
            f"KELLY | {strategy}/{symbol} | p={p:.4f} q={q:.4f} b={b:.4f} n={n} | "
            f"raw_f*={raw_fraction:.4f} × multiplier={self.kelly_fraction_multiplier} "
            f"-> adjusted_f*={adjusted_fraction:.4f}")

        return adjusted_fraction

    def _apply_kelly_overlay(
        self,
        pls_stake: float,
        balance:   float,
        strategy:  Optional[str],
        symbol:    Optional[str],
    ) -> float:
        """
        Gate/scale a PLS-computed stake through the Kelly overlay.
        PLS's own math is never touched — this only ever caps or zeroes
        what PLS already produced.
        """
        if strategy is None or symbol is None:
            # No pair context supplied — overlay can't run, PLS stake stands.
            return pls_stake

        kelly_fraction = self.compute_kelly_fraction(strategy, symbol)

        if kelly_fraction is None:
            return pls_stake  # insufficient data — PLS passes through unchanged

        if kelly_fraction <= 0:
            logger.info(
                f"KELLY | {strategy}/{symbol} | fraction<=0 -> forcing stake to "
                f"$0.00 (overriding PLS stake of ${pls_stake:.4f})")
            return 0.0

        kelly_stake = balance * kelly_fraction
        final_stake = min(pls_stake, kelly_stake)
        final_stake = round(final_stake, 2)

        if final_stake < pls_stake:
            logger.info(
                f"KELLY | {strategy}/{symbol} | kelly_stake=${kelly_stake:.4f} < "
                f"pls_stake=${pls_stake:.4f} -> capped to ${final_stake:.4f}")

        return final_stake

    # ── Equity curve stabilization dampener ─────────────────────────────────

    def _stability_dampener_mult(self, balance: float) -> float:
        """
        Continuous stake multiplier in (0, 1] combining two signals — takes
        whichever is more conservative rather than multiplying them (see
        config.py's EQUITY CURVE STABILIZATION comment for why). Returns
        1.0 (no-op) when both features are disabled or the required
        peak-balance history isn't available yet.
        """
        mult = 1.0

        if getattr(config, "DRAWDOWN_DAMPENER_ENABLED", False) and self._peak_balance > 0:
            drawdown_pct = max(0.0, (self._peak_balance - balance) / self._peak_balance)
            start_pct = getattr(config, "DRAWDOWN_DAMPENER_START_PCT", 0.015)
            full_pct  = getattr(config, "DRAWDOWN_DAMPENER_FULL_PCT", 0.06)
            floor     = getattr(config, "DRAWDOWN_DAMPENER_FLOOR", 0.40)
            if drawdown_pct <= start_pct:
                dd_mult = 1.0
            elif drawdown_pct >= full_pct or full_pct <= start_pct:
                dd_mult = floor
            else:
                # Linear ramp from 1.0 at start_pct down to floor at full_pct.
                span = (drawdown_pct - start_pct) / (full_pct - start_pct)
                dd_mult = 1.0 - span * (1.0 - floor)
            mult = min(mult, dd_mult)

        if getattr(config, "LOSS_STREAK_DAMPENER_ENABLED", False):
            ls_mult = 1.0
            for streak_count, tier_mult in getattr(config, "LOSS_STREAK_DAMPENER_TABLE", []):
                if self._loss_streak >= streak_count:
                    ls_mult = tier_mult
            mult = min(mult, ls_mult)

        return mult

    def next_stake_for(self, strategy: str, symbol: str) -> float:
        """
        Synchronous preview of the stake that would be used for the very
        next trade on this (strategy, symbol) pair, PLS + Kelly overlay
        combined, using the current cached balance.
        """
        if getattr(config, "MANUAL_STAKE_MODE", False):
            return float(getattr(config, "MANUAL_STAKE_AMOUNT", self.min_stake))
        pls_stake = self._compute_stake(self._current_balance)
        return self._apply_kelly_overlay(
            pls_stake, self._current_balance, strategy, symbol
        )

    # ── Stake calculation (PLS + Kelly overlay) ─────────────────────────────

    async def calculate_stake(
        self,
        strategy: Optional[str] = None,
        symbol:   Optional[str] = None,
    ) -> float:
        """
        Fetch live balance, apply the PLS formula (unchanged from v17),
        then gate/scale the result through the Kelly overlay for this
        (strategy, symbol) pair, log, and return the final stake.

        strategy/symbol are optional for backward compatibility with
        existing callers — if omitted, the Kelly overlay is skipped
        entirely and behavior is identical to v17 (pure PLS).

        Log format:
          STAKE: $X (pls=$Y base=$Z ×M streak=+N [strategy=... symbol=...])
        """
        if getattr(config, "MANUAL_STAKE_MODE", False):
            manual_stake = float(getattr(config, "MANUAL_STAKE_AMOUNT", self.min_stake))
            logger.info(
                f"STAKE: ${manual_stake:.2f} — MANUAL_STAKE_MODE active, "
                f"BASE_STAKE_PCT/Kelly/dampener/exposure-ceiling all bypassed"
                + (f" strategy={strategy} symbol={symbol}" if strategy and symbol else "")
            )
            return manual_stake

        balance = await self._fetch_live_balance()

        base      = max(self._current_balance * self.base_stake_pct, self.min_stake)
        pls_stake = round(base * self._multiplier, 2)
        pls_stake = min(pls_stake, self.max_stake)
        pls_stake = max(pls_stake, self.min_stake)

        stake = self._apply_kelly_overlay(pls_stake, self._current_balance, strategy, symbol)

        # Equity curve stabilization — see config.py's EQUITY CURVE
        # STABILIZATION section and _stability_dampener_mult() above.
        # Applied after Kelly (which is per-pair edge sizing) and before
        # the exposure ceiling (which is a hard portfolio-wide cap) — this
        # sits between them as a soft, account-wide "how rough has it
        # been lately" throttle.
        dampener = self._stability_dampener_mult(self._current_balance)
        if dampener < 1.0:
            damped_stake = round(stake * dampener, 2)
            logger.info(
                f"STABILITY DAMPENER: ${stake:.4f} -> ${damped_stake:.4f} "
                f"(×{dampener:.2f} | drawdown_from_peak=${max(0.0, self._peak_balance - self._current_balance):.4f} "
                f"| loss_streak={self._loss_streak})"
            )
            stake = damped_stake

        # Exposure ceiling: never let this trade push total committed stake
        # across all open positions past EXPOSURE_CEILING_PCT of balance.
        ceiling_pct = getattr(config, "EXPOSURE_CEILING_PCT", 0.90)
        exposure_ceiling = self._current_balance * ceiling_pct
        already_committed = self._committed_exposure()
        room = exposure_ceiling - already_committed

        if room <= 0:
            logger.info(
                f"STAKE: $0.00 — exposure ceiling reached "
                f"(committed=${already_committed:.4f} >= "
                f"ceiling=${exposure_ceiling:.4f})"
                + (f" strategy={strategy} symbol={symbol}" if strategy and symbol else "")
            )
            return 0.0

        if stake > room:
            logger.info(
                f"STAKE: clamped ${stake:.4f} -> ${room:.4f} — exposure "
                f"ceiling (committed=${already_committed:.4f}, "
                f"ceiling=${exposure_ceiling:.4f})"
            )
            stake = round(room, 2)

        if stake < self.min_stake:
            logger.info(
                f"STAKE: $0.00 — post-clamp stake ${stake:.4f} below "
                f"MIN_STAKE ${self.min_stake:.2f}"
            )
            return 0.0

        pair_suffix = f" strategy={strategy} symbol={symbol}" if strategy and symbol else ""
        logger.info(
            f"STAKE: ${stake:.4f} "
            f"(pls=${pls_stake:.4f} "
            f"base=${base:.4f} "
            f"×{self._multiplier:.1f} "
            f"streak=+{self._win_streak}"
            f"{pair_suffix})")

        return stake

    # ── PLS can_trade gate ──────────────────────────────────────────────────────

    def can_trade(self) -> bool:
        """
        Return True iff ALL of:
          1. open_contracts < current_concurrent_limit
          2. current_balance > MIN_STAKE

        Nothing else blocks trading.
        Symbol suspension is handled by symbol_manager, not here.
        """
        if self._current_balance <= self.min_stake:
            logger.debug(
                f"can_trade: balance ${self._current_balance:.4f} "
                f"<= MIN_STAKE ${self.min_stake:.2f}")
            return False

        if self._open_trade_count >= self.current_concurrent_limit:
            logger.debug(
                f"can_trade: concurrent limit reached "
                f"({self._open_trade_count}/{self.current_concurrent_limit})")
            return False

        return True

    # ── PLS result recording ────────────────────────────────────────────────────

    def record_result(self, won: bool) -> None:
        """
        Update the PLS win streak and emit the mandated post-trade log line.

        WIN  → increment streak → _update_multiplier() raises tier.
        LOSS → immediate reset: win_streak → 0, multiplier → 1.0,
               extra_slots → 0. Stake is NEVER raised to recover a loss —
               this is the core distinction from Martingale.

        Log format (win):
          PLS | Streak: +{N} | Multiplier: {M}x | Balance: ${B:.4f} | Next stake: ${S:.4f}
        Log format (loss):
          PLS | Streak: +0 (reset) | Multiplier: 1.0x | Balance: ${B:.4f} | Next stake: ${S:.4f}
        """
        if won:
            self.wins        += 1
            self._win_streak += 1
            self._loss_streak = 0
            self._update_multiplier()
        else:
            self.losses          += 1
            self._win_streak      = 0
            self._multiplier      = 1.0
            self._extra_slots     = 0
            self._loss_streak    += 1

        balance = self._current_balance if self._current_balance > 0 else self.min_stake
        ns      = self._compute_stake(balance)

        streak_label = (
            f"+{self._win_streak}" if won
            else "+0 (reset)"
        )

        logger.info(
            f"PLS | Streak: {streak_label} | "
            f"Multiplier: {self._multiplier:.1f}x | "
            f"Balance: ${balance:.4f} | "
            f"Next stake: ${ns:.4f}")

    # ── Trade lifecycle ─────────────────────────────────────────────────────────

    def _committed_exposure(self) -> float:
        """
        Sum of stake for every currently-open trade (exit_price is None).
        Used to clamp new stakes so total exposure across concurrent
        positions can never exceed EXPOSURE_CEILING_PCT of balance —
        necessary once STAKE_PCT_MULTIPLIER allows large per-trade stakes,
        where even 2-3 concurrent trades could otherwise over-commit past
        100% of the account.
        """
        return sum(
            rec.stake for rec in self._trades if rec.exit_price is None
        )

    def register_open(
        self,
        symbol:      str,
        direction:   str,
        stake:       float,
        entry_price: float,
    ) -> TradeRecord:
        rec = TradeRecord(
            symbol=symbol,
            direction=direction,
            stake=stake,
            entry_price=entry_price,
        )
        self._trades.append(rec)
        self._open_trade_count += 1
        self.total_trades      += 1
        logger.info(
            f"Trade OPEN  | {symbol} {direction} | "
            f"stake=${stake:.2f} | price={entry_price} | "
            f"streak=+{self._win_streak} | "
            f"multiplier={self._multiplier:.1f}x | "
            f"concurrent={self._open_trade_count}/{self.current_concurrent_limit}")
        return rec

    def register_close(
        self,
        rec:        TradeRecord,
        exit_price: float,
        pnl:        float,
    ) -> None:
        rec.exit_price         = exit_price
        rec.pnl                = pnl
        rec.won                = pnl > 0
        self._open_trade_count = max(0, self._open_trade_count - 1)
        self.total_pnl        += pnl

        self.record_result(rec.won)

        logger.info(
            f"Trade CLOSE | {rec.symbol} {rec.direction} | "
            f"pnl=${pnl:+.4f} | {'WIN' if rec.won else 'LOSS'} | "
            f"streak=+{self._win_streak} | "
            f"balance=${self._current_balance:.4f} | "
            f"open={self._open_trade_count}")

    # ── Session helpers ─────────────────────────────────────────────────────────

    @property
    def current_balance(self) -> float:
        return self._current_balance

    @property
    def day_start_balance(self) -> float:
        return self._day_start_balance

    @property
    def daily_pnl(self) -> float:
        return self._current_balance - self._day_start_balance

    @property
    def daily_pnl_pct(self) -> float:
        if self._day_start_balance == 0:
            return 0.0
        return self.daily_pnl / self._day_start_balance

    @staticmethod
    def minutes_until_midnight() -> float:
        now      = datetime.datetime.utcnow()
        midnight = (now + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        return (midnight - now).total_seconds() / 60

    def summary(self) -> dict:
        total = self.wins + self.losses
        return {
            "current_balance":    round(self._current_balance, 4),
            "day_start_balance":  round(self._day_start_balance, 4),
            "daily_pnl":          round(self.daily_pnl, 4),
            "daily_pnl_pct":      round(self.daily_pnl_pct * 100, 2),
            "total_trades":       self.total_trades,
            "wins":               self.wins,
            "losses":             self.losses,
            "win_rate":           round(self.wins / total * 100, 1) if total else 0.0,
            "total_pnl":          round(self.total_pnl, 4),
            "open_trades":        self._open_trade_count,
            # PLS state
            "current_streak":     self.current_streak,
            "current_multiplier": self.current_multiplier,
            "next_stake":         self.next_stake,
            "concurrent_limit":   self.current_concurrent_limit,
            "bot_state":          self._bot_state.name,
            # Equity curve stabilization (see config.py / _stability_dampener_mult)
            "peak_balance":            round(self._peak_balance, 4),
            "drawdown_from_peak_pct":  round(
                ((self._peak_balance - self._current_balance) / self._peak_balance * 100)
                if self._peak_balance > 0 else 0.0, 2),
            "loss_streak":             self._loss_streak,
            "stability_dampener_mult": round(self._stability_dampener_mult(self._current_balance), 3),
        }


# ── Module-level helpers ──────────────────────────────────────────────────────

def _today_tag() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")
