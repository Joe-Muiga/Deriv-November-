"""
pair_suspension.py – per-(indicator, symbol) suspension window.

Popular-indicator pipeline, spec point 8 (Aug 2026): when a specific
(indicator, symbol) pair underperforms, ONLY that pair sits out — for a flat
config.PAIR_SUSPEND_MINUTES window — after which it's automatically eligible
again. The symbol keeps trading normally on whichever other indicator is
currently signalling, and that indicator keeps trading normally on every
other symbol.

This is deliberately separate from strategy_stats.is_underperforming(),
which is a live rolling-window win-rate check with no timer of its own: it
recomputes on every call and stays True for as long as the rate is bad,
however long that takes to recover. Wiring gating directly off
is_underperforming() would make the "suspension" length depend on when the
rolling win rate happens to recover, not a fixed hour. Instead:

  - is_underperforming() is checked only to decide whether to START the
    clock (maybe_suspend()).
  - Once started, is_suspended() alone gates eligibility until it expires,
    regardless of whether the rolling win rate recovers sooner or stays bad
    longer than the hour.
  - maybe_suspend() is idempotent: calling it again while already suspended
    does not push the expiry back out, so this is a flat, non-renewing
    window rather than one that resets on every subsequent bad-looking
    check.

Also deliberately separate from symbol_manager.py's suspend(), which
operates per-symbol (blocking every indicator on that symbol) and is used
for connection/API-error protection (the buy-failure circuit breaker) and,
previously, for blanket win/loss suspension — see symbol_manager.py's
record_result() for why that blanket suspension was removed for this
pipeline.
"""

import time
import logging
from typing import Dict, Tuple

import config

logger = logging.getLogger(__name__)

# (strategy/indicator label, symbol) -> unix timestamp the suspension lifts
_suspended_until: Dict[Tuple[str, str], float] = {}


def is_suspended(strategy: str, symbol: str) -> bool:
    """True if this exact (indicator, symbol) pair is currently sitting out
    its suspension window. Does not affect any other pair."""
    until = _suspended_until.get((strategy, symbol), 0.0)
    return time.time() < until


def maybe_suspend(strategy: str, symbol: str) -> None:
    """
    Call this once, right after strategy_stats.is_underperforming() returns
    True for this (indicator, symbol) pair. Idempotent: if the pair is
    already suspended, this does nothing — that's what makes the window a
    flat PAIR_SUSPEND_MINUTES rather than one that keeps getting pushed back
    out for as long as the rolling win rate stays bad.
    """
    if is_suspended(strategy, symbol):
        return
    minutes = getattr(config, "PAIR_SUSPEND_MINUTES", 60)
    _suspended_until[(strategy, symbol)] = time.time() + minutes * 60
    logger.info(
        f"PAIR SUSPENDED: {strategy}/{symbol} for {minutes}min "
        f"(underperforming) — other indicators on {symbol} and {strategy} "
        f"on other symbols are unaffected"
    )


def seconds_remaining(strategy: str, symbol: str) -> float:
    """How much longer this pair stays suspended, 0 if not suspended."""
    until = _suspended_until.get((strategy, symbol), 0.0)
    return max(0.0, until - time.time())


def clear(strategy: str, symbol: str) -> None:
    """Manually lift a suspension early (ops/debug use only)."""
    _suspended_until.pop((strategy, symbol), None)


def snapshot() -> Dict[str, float]:
    """Currently-suspended pairs -> seconds remaining, for dashboard/debug."""
    now = time.time()
    return {
        f"{strategy}/{symbol}": until - now
        for (strategy, symbol), until in _suspended_until.items()
        if until > now
    }
