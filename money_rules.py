"""
money_rules.py - the follower money rules as pure functions (no I/O, no clock).

Driven by the follower's LIVE Deriv balance (Deriv deducts the stake when a contract opens,
so the balance already equals free money).

  P      = balance at the last reset.  Reset = midnight Kenya time = 21:00 UTC.
  Growth tier (no floor set): tradable money = whole balance.  Losing it all is acceptable.
  Lock tier (floor set):      tradable money = balance - floor.
  floor  = max(previous_floor, 0.7*P, P + 0.5*(peak - P)); only ever moves up; sticky until
           the next reset even if the balance later falls back under L.  The third term applies
           only once peak > P (a gain exists); at peak == P it would equal P and leave nothing
           tradable right after a reset.
  stake  = max(MIN_STAKE, 17.5% of tradable money)   [MIN_STAKE = PREDICTOR_STAKE env var], rounded DOWN to cents.
  buy rule (hard): only buy if balance - stake >= floor.
"""
import math
import os
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Optional



def _min_stake_from_env(default: float = 0.35) -> float:
    """Minimum stake = the PREDICTOR_STAKE Render environment variable (not hard-coded)."""
    try:
        v = float((os.environ.get("PREDICTOR_STAKE") or "").strip())
        return v if v > 0 else default
    except ValueError:
        return default


MIN_STAKE = _min_stake_from_env()
STAKE_FRACTION = 0.175
FLOOR_FRACTION = 0.7
PEAK_SHARE = 0.5
MILESTONE_MULT = 1.325
EPS = 1e-9


@dataclass(frozen=True)
class FState:
    p: Optional[float] = None               # balance at last reset (None = never reset yet)
    peak: Optional[float] = None
    floor: Optional[float] = None           # None = growth tier
    last_reset_date: Optional[str] = None   # Kenya calendar day of the last reset
    milestone_hit: bool = False

    @property
    def tier(self) -> str:
        return "lock" if self.floor is not None else "growth"


# ── reset ────────────────────────────────────────────────────────────────────
def kenya_day(now_utc: datetime) -> str:
    """Kenya calendar day (UTC+3). Midnight Kenya = 21:00 UTC, so 20:59:59Z and 21:00:00Z
    fall on different days."""
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    return (now_utc.astimezone(timezone.utc) + timedelta(hours=3)).date().isoformat()


def kenya_day_ts(ts: float) -> str:
    return kenya_day(datetime.fromtimestamp(ts, tz=timezone.utc))


def needs_reset(state: FState, today: str) -> bool:
    return state.p is None or state.last_reset_date != today


def apply_reset(state: FState, balance: float, lock_threshold: float, today: str) -> FState:
    """P = peak = live balance. Floor starts at 0.7*P only if balance >= L (else growth tier)."""
    floor = FLOOR_FRACTION * balance if balance >= lock_threshold - EPS else None
    return FState(p=balance, peak=balance, floor=floor, last_reset_date=today, milestone_hit=False)


# ── balance updates ──────────────────────────────────────────────────────────
def update_on_balance(state: FState, balance: float, lock_threshold: float) -> FState:
    if state.p is None:
        return state
    peak = max(state.peak if state.peak is not None else balance, balance)
    floor = state.floor
    if balance >= lock_threshold - EPS:
        candidates = [floor or 0.0, FLOOR_FRACTION * state.p]
        if peak > state.p:      # profit lock applies only once there IS a gain above P; otherwise the
            candidates.append(state.p + PEAK_SHARE * (peak - state.p))   # floor would jump to P at reset
        floor = max(candidates)
    return replace(state, peak=peak, floor=floor)


# ── stake / buy rule ─────────────────────────────────────────────────────────
def tradable(state: FState, balance: float) -> float:
    return balance - (state.floor or 0.0)


def _cents_down(x: float) -> float:
    return math.floor(x * 100 + 1e-7) / 100.0


def stake_for(state: FState, balance: float, min_stake: float = MIN_STAKE,
              fraction: float = STAKE_FRACTION) -> Optional[float]:
    """Stake for the next buy, or None if the follower must not trade (stopped)."""
    t = tradable(state, balance)
    if t < min_stake - EPS:
        return None
    stake = max(min_stake, _cents_down(fraction * t))
    stake = min(stake, _cents_down(t))
    if stake < min_stake - EPS:
        return None
    return stake


def can_buy(state: FState, balance: float, stake: float, min_stake: float = MIN_STAKE) -> bool:
    return stake >= min_stake - EPS and balance - stake >= (state.floor or 0.0) - EPS


def is_stopped(state: FState, balance: float, min_stake: float = MIN_STAKE) -> bool:
    return tradable(state, balance) < min_stake - EPS


def milestone(state: FState) -> Optional[float]:
    """Bot-decided target, shown and logged when first hit; it never stops trading."""
    return state.p * MILESTONE_MULT if state.p else None


def effective_balance(balance: float, pending_stakes: float) -> float:
    """Balance minus stakes already sent in this burst but not yet acknowledged."""
    return balance - pending_stakes
