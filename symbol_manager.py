import time
import logging
from datetime import datetime, timezone
import config

logger = logging.getLogger(__name__)

# ── Symbol category groupings (fallback to config if defined there) ──
_BOOM_CRASH_500_300 = getattr(config, 'BOOM_CRASH_500_300', ["BOOM500", "BOOM300", "CRASH300"])
_CRASH500_ONLY = getattr(config, 'CRASH500_ONLY', ["CRASH500"])
_BOOM_CRASH_1000 = getattr(config, 'BOOM_CRASH_1000', ["BOOM1000", "CRASH1000"])
_RANGE_BREAK_SYMBOLS = getattr(config, 'RANGE_BREAK_SYMBOLS', [])
_DIGIT_SYMBOLS = getattr(config, 'DIGIT_SYMBOLS', [])
_MEAN_REVERSION_SYMBOLS = getattr(config, 'MEAN_REVERSION_SYMBOLS', getattr(config, 'MEAN_REVERSION', []))
_STEP_SYMBOLS = getattr(config, 'STEP_SYMBOLS', [])
_JUMP_SYMBOLS = getattr(config, 'JUMP_SYMBOLS', [])
_BEAR_BULL_SYMBOLS = getattr(config, 'BEAR_BULL_SYMBOLS', [])

_BOOM_CRASH_ALL = set(_BOOM_CRASH_500_300) | set(_CRASH500_ONLY) | set(_BOOM_CRASH_1000)
_BEAR_BULL_SET = set(_BEAR_BULL_SYMBOLS)

# Bear/Bull ("Daily Reset") indices reset to a baseline at 00:00 GMT and then
# trend (up for Bull, down for Bear) with constant volatility until the next
# reset. The window immediately following the reset is flagged — not
# blocked — since behavior right after reset may differ from mid-cycle
# trending. Duration is configurable via config.BEAR_BULL_TREND_SHIFT_MINS
# (defaults to 20 if unset).
_BEAR_BULL_POST_RESET_MINS = getattr(config, 'BEAR_BULL_TREND_SHIFT_MINS', 20)


class SymbolManager:
    def __init__(self):
        self._suspension_until = {}    # symbol -> unix expiry timestamp
        self._last_traded = {}         # symbol -> unix timestamp of last placement
        self._session_losses = {}      # symbol -> loss count this UTC session
        self._symbol_wins = {}         # symbol -> win count this session
        self._symbol_trades = {}       # symbol -> total trade count this session
        self._active_symbols = set()   # symbols with currently open contracts
        self._jump_last_seen = {}      # symbol -> unix timestamp of last detected jump
        self._all_active = []
        self.current_session = "Active"

    def suspend(self, symbol: str, minutes: float) -> None:
        until = time.time() + (minutes * 60)
        self._suspension_until[symbol] = until
        logger.info(
            f"SUSPENDED: {symbol} for {minutes}min "
            f"expires at {datetime.fromtimestamp(until, tz=timezone.utc).strftime('%H:%M:%S')} UTC"
        )

    def is_suspended(self, symbol: str) -> bool:
        until = self._suspension_until.get(symbol, 0)
        now = time.time()
        if now < until:
            remaining = until - now
            logger.debug(f"SUSPENDED: {symbol} {remaining:.0f}s remaining")
            return True
        return False

    def can_trade_now(self, symbol: str) -> bool:
        if self.is_suspended(symbol):
            remaining = self._suspension_until.get(symbol, 0) - time.time()
            logger.info(f"BLOCKED: {symbol} suspended ({remaining:.0f}s remaining)")
            return False

        if symbol in self._active_symbols:
            logger.info(f"BLOCKED: {symbol} already has an active contract")
            return False

        gap_required = config.SYMBOL_MIN_GAP_MINS * 60
        elapsed = time.time() - self._last_traded.get(symbol, 0)
        if elapsed < gap_required:
            logger.info(
                f"BLOCKED: {symbol} min-gap not met "
                f"({elapsed:.0f}s elapsed, {gap_required:.0f}s required)"
            )
            return False

        return True

    def record_trade_placed(self, symbol: str) -> None:
        self._last_traded[symbol] = time.time()
        self._active_symbols.add(symbol)
        logger.info(f"TRADE PLACED: {symbol} | Active: {self._active_symbols}")

    def record_contract_opened(self, symbol: str) -> None:
        self._active_symbols.add(symbol)
        logger.info(f"CONTRACT OPENED: {symbol} | Active: {self._active_symbols}")

    def record_contract_closed(self, symbol: str) -> None:
        self._active_symbols.discard(symbol)
        logger.info(f"CONTRACT CLOSED: {symbol} | Active: {self._active_symbols}")

    def record_result(self, symbol: str, won: bool) -> None:
        self._symbol_trades[symbol] = self._symbol_trades.get(symbol, 0) + 1

        if won:
            self._symbol_wins[symbol] = self._symbol_wins.get(symbol, 0) + 1
            # BUG FIX (drawdown pass, Aug 2026): a win must clear this
            # symbol's consecutive-loss count. Previously _session_losses
            # only ever incremented and was never reset on a win (despite
            # being named/commented as a "consecutive" counter) — it was
            # actually a lifetime count since the last redeploy. On a
            # multi-day unattended run, once a symbol crossed 4 total
            # losses EVER it got the ladder's max 240min suspension on
            # every single subsequent loss, permanently, for the rest of
            # that run — which is what was producing multi-hour dead
            # stretches on the dashboard despite no crash/hang anywhere.
            self._session_losses[symbol] = 0
            # NOTE (popular-indicator pipeline, spec point 8, Aug 2026): the
            # blanket per-symbol win-suspend call that used to sit here
            # (self.suspend(symbol, config.SYMBOL_WIN_SUSPEND_MINS)) has
            # been removed. It blocked EVERY indicator on this symbol for
            # every single trade result, which directly contradicts "the
            # symbol keeps trading normally on whichever other indicator is
            # currently signalling" — performance-based throttling for this
            # pipeline is now scoped to the specific (indicator, symbol)
            # pair that actually underperformed, via pair_suspension.py,
            # not the whole symbol. Win/loss bookkeeping above (counts,
            # consecutive-loss reset) is left in place since other code may
            # still read it.
            logger.info(f"RESULT: {symbol} WON | consecutive-loss count reset")
        else:
            self._session_losses[symbol] = self._session_losses.get(symbol, 0) + 1
            loss_count = self._session_losses[symbol]

            # Escalating ladder (Implementation Brief v2, Requirement 2 /
            # Fix F): 1st CONSECUTIVE loss on a symbol (since its last win,
            # or since process start) -> ladder[0] minutes, 2nd -> ladder[1],
            # ..., loss_count beyond the ladder's length holds at the
            # ladder's last (highest) value. A win resets this to 0 (see
            # above) — that reset is what makes it consecutive rather than
            # lifetime. The suspension *timestamps* themselves still persist
            # across calendar-day boundaries (reset_session() is manual-only,
            # unchanged) — only what counts as "still on a losing streak"
            # was the bug.
            # Escalating ladder (Implementation Brief v2, Requirement 2 /
            # Fix F): kept as bookkeeping (loss_count is still tracked and
            # logged) but no longer calls self.suspend() — per spec point 8
            # (Aug 2026), a losing trade must not block every other
            # indicator on this symbol. Only the specific (indicator,
            # symbol) pair that underperforms gets suspended, and only for
            # a flat 1 hour, via pair_suspension.py (wired into
            # evaluate_popular_indicator()'s picking loop in
            # signal_engine.py). The suspension *timestamps* themselves
            # still persist across calendar-day boundaries
            # (reset_session() is manual-only, unchanged).
            ladder = getattr(config, "SESSION_LOSS_SUSPEND_LADDER_MINS", [60, 120, 180, 240])
            idx = min(loss_count, len(ladder)) - 1
            suspend_mins = ladder[idx]
            logger.info(
                f"RESULT: {symbol} LOST ({loss_count} consecutive) | "
                f"ladder would-be={suspend_mins}min (not applied — symbol-wide "
                f"suspension disabled, see pair_suspension.py)"
            )

    def get_symbol_score(self, symbol: str) -> float:
        return self._symbol_wins.get(symbol, 0) / max(self._symbol_trades.get(symbol, 1), 1)

    def best_symbols(self, n: int) -> list:
        scored = [
            {
                "symbol": s,
                "win_rate": round(self.get_symbol_score(s) * 100, 1),
                "trades": self._symbol_trades.get(s, 0),
            }
            for s in self._symbol_trades
        ]
        return sorted(scored, key=lambda x: x["win_rate"], reverse=True)[:n]

    def is_post_reset(self, symbol: str) -> bool:
        """
        True if `symbol` is a Bear/Bull ("Daily Reset") index currently within
        the configurable post-reset window (config.BEAR_BULL_TREND_SHIFT_MINS
        minutes since 00:00 GMT). This is an informational flag only — it
        does NOT block trading. Callers (e.g. signal_engine.py / risk_manager.py)
        can use it to apply different logic post-reset vs. mid-cycle.
        Returns False for non-Bear/Bull symbols or outside the window.
        """
        if symbol not in _BEAR_BULL_SET:
            return False
        now = datetime.now(timezone.utc)
        minutes_since_reset = now.hour * 60 + now.minute
        return minutes_since_reset < _BEAR_BULL_POST_RESET_MINS

    def get_bear_bull_state(self, symbol: str):
        """
        Returns "post_reset", "mid_cycle", or None (symbol is not a
        configured Bear/Bull index). Convenience wrapper around
        is_post_reset() for callers that want a labeled state rather
        than a bool.
        """
        if symbol not in _BEAR_BULL_SET:
            return None
        return "post_reset" if self.is_post_reset(symbol) else "mid_cycle"

    def is_in_session(self, symbol: str) -> bool:
        now_hour = datetime.now(timezone.utc).hour

        # Explicit dead zone: 00:00–05:00 UTC blocks all Boom/Crash symbols
        if symbol in _BOOM_CRASH_ALL and 0 <= now_hour < 5:
            logger.debug(f"SESSION: {symbol} blocked — dead zone (00:00-05:00 UTC)")
            return False

        if symbol in _BOOM_CRASH_500_300:
            ok = 7 <= now_hour < 12
            if not ok:
                logger.debug(f"SESSION: {symbol} outside window 07:00-12:00 UTC")
            return ok

        if symbol in _CRASH500_ONLY:
            ok = 7 <= now_hour < 16
            if not ok:
                logger.debug(f"SESSION: {symbol} outside window 07:00-16:00 UTC")
            return ok

        if symbol in _BOOM_CRASH_1000:
            ok = 5 <= now_hour < 20
            if not ok:
                logger.debug(f"SESSION: {symbol} outside window 05:00-20:00 UTC")
            return ok

        if symbol in _RANGE_BREAK_SYMBOLS:
            return True

        if symbol in _DIGIT_SYMBOLS or symbol in _MEAN_REVERSION_SYMBOLS:
            return True

        if symbol in _STEP_SYMBOLS:
            return True

        if symbol in _JUMP_SYMBOLS:
            # Preferred window 07:00-20:00 UTC, but always allowed
            if not (7 <= now_hour < 20):
                logger.debug(f"SESSION: {symbol} outside preferred window but still allowed")
            return True

        if symbol in _BEAR_BULL_SET:
            # Daily Reset indices: never blocked by session, but the window
            # right after 00:00 GMT reset is flagged as a distinct state —
            # mirrors the Boom/Crash dead-zone pattern except it flags
            # instead of blocking. See is_post_reset() / get_bear_bull_state().
            if self.is_post_reset(symbol):
                logger.debug(
                    f"SESSION: {symbol} in post-reset window "
                    f"(within {_BEAR_BULL_POST_RESET_MINS}min of 00:00 GMT reset) — "
                    f"flagged, not blocked"
                )
            return True

        # Default: no restriction defined, allow
        return True

    def get_queue(self, active_list: list = None) -> list:
        # Use passed list, or fall back to _all_active, or fall back to config
        source = active_list or self._all_active
        if not source:
            source = list(getattr(config, 'ALL_TRADE_SYMBOLS',
                          getattr(config, 'TRADE_SYMBOLS',
                          getattr(config, 'ALL_SYMBOLS', []))))
            self._all_active = source
            logger.warning(
                f"get_queue: _all_active was empty — fell back to "
                f"config symbol list ({len(source)} symbols)"
            )

        tradeable = []
        suspended_list = []
        session_blocked = []

        now = time.time()
        for symbol in source:
            if symbol in self._active_symbols:
                continue

            if now < self._suspension_until.get(symbol, 0):
                remaining = (self._suspension_until[symbol] - now) / 60
                suspended_list.append(f"{symbol}({remaining:.1f}m)")
                continue

            if not self.is_in_session(symbol):
                session_blocked.append(symbol)
                continue

            gap_required = getattr(config, 'SYMBOL_MIN_GAP_MINS', 1) * 60
            elapsed = now - self._last_traded.get(symbol, 0)
            if elapsed < gap_required:
                continue

            tradeable.append(symbol)

        logger.info(
            f"Queue: {len(tradeable)} tradeable | "
            f"Suspended: {suspended_list} | "
            f"Session-blocked: {session_blocked}"
        )
        return tradeable

    def update_active(self, symbol_list):
        import config
        self._all_active = [
            s for s in symbol_list
            if s in getattr(config, 'ALL_TRADE_SYMBOLS',
                getattr(config, 'TRADE_SYMBOLS',
                getattr(config, 'ALL_SYMBOLS', symbol_list)))
        ]
        logger.info(f"Active pool: {len(self._all_active)} symbols")

    def reset_session(self) -> None:
        """
        NOTE: no longer called automatically on a UTC-midnight (or any
        other calendar) boundary — see Implementation Brief v2,
        Requirement 2 / Fix F. The escalating loss-suspension ladder in
        record_result() must persist across calendar-day changes and be
        reset ONLY by a redeploy (a real process restart naturally
        reconstructs a fresh SymbolManager, wiping this in-memory state
        for free). Kept here only for manual/administrative use.
        """
        self._session_losses.clear()
        self._symbol_wins.clear()
        self._symbol_trades.clear()
        logger.info("Session counters reset (manual reset_session() call — suspensions preserved)")

    def get_suspended_list(self) -> list:
        now = time.time()
        return [
            {"symbol": s, "minutes": round((until - now) / 60, 1)}
            for s, until in self._suspension_until.items()
            if now < until
        ]
