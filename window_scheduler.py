"""
window_scheduler.py - the two-sessions-per-hour clock (pure functions: no network, no Render calls, no sleeping).

  * Two sessions every hour, every day, UTC, judged on Deriv's SERVER clock (now() = local time + measured offset):
        A  opens HH:05:00.000  closes HH:12:01.000        B  opens HH:24:00.000  closes HH:29:01.000
    Open is inclusive, close is exclusive: a buy is allowed when  open <= t < close.  The "+1 s" is
    config.SESSION_CLOSE_EXTRA_SECONDS.  Every session is its own session: no weekday logic, no JSON, no tiers,
    no cluster / keep-open-through-the-gap logic.
  * The bot is connected only from config.WINDOW_PRECONNECT_SECONDS (the connect lead) before a session opens until
    config.SESSION_DRAIN_SECONDS after it closes, and is fully disconnected otherwise.
  * Exactly one redeploy may fire before EACH session (two per hour): in the idle gap after the previous session's
    drain, timed to FINISH just before the next session's connect lead (before A: after B; before B: after A).
    This module only holds the (pure) math; the decision to fire lives in redeploy_guard.
  * Fail closed: if the constants are inconsistent, schedule_ok() is False, should_be_connected() and
    buy_allowed_at() are False and no redeploy is allowed.  (Callers must check schedule_ok() before calling
    upcoming_session(), which raises RuntimeError on an inconsistent schedule.)
  * With config.SESSION_SCHEDULE_ENABLED off (dev override): phase() is always "OPEN", buy_allowed_at() is True,
    should_be_connected() is True, redeploy_allowed() is False.

All schedule constants are read from `config` on every call (nothing is cached), so tests can patch them.
"""
import logging
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import config

logger = logging.getLogger("window")

_HOUR = 3600.0
# Stand-in for "no end" when SESSION_SCHEDULE_ENABLED is off (2100-01-01 UTC; finite so it survives JSON).
_FAR_FUTURE = 4102444800.0


class SessionClosed(RuntimeError):
    """Raised by deriv_client when a buy would be sent at/after the session close. Means 'definitely not sent'."""


@dataclass(frozen=True)
class Session:
    name: str
    open_ts: float
    close_ts: float
    connect_ts: float          # open_ts - WINDOW_PRECONNECT_SECONDS

    @property
    def drain_end_ts(self) -> float:
        return self.close_ts + float(config.SESSION_DRAIN_SECONDS)


@dataclass(frozen=True)
class RedeployPlan:
    session: str               # the session this redeploy prepares: "A" or "B"
    connect_ts: float          # that session's connect lead
    earliest_ts: float         # previous session's drain_end_ts + PRE_A_REDEPLOY_AFTER_DRAIN_BUFFER_SECONDS
    fire_ts: float             # connect_ts - (expected_s + margin_s)
    fits: bool                 # fire_ts >= earliest_ts
    available_s: float         # connect_ts - earliest_ts


# ── clock (server-time offset, NTP-style) ────────────────────────────────────────────────────────────
# One tuple, replaced atomically, so a reader on another thread never sees a half-updated state.
_clock = (0.0, 0.0, False)     # (offset, error, measured_this_session)


def set_clock_offset(offset: float, error: float = 0.0) -> None:
    """Record the measured Deriv-minus-local offset (seconds) and its +/- error. Marks 'measured this session'.
    A non-finite offset is refused (logged), the previous offset is kept and the clock stays 'not measured'."""
    global _clock
    try:
        off, err = float(offset), abs(float(error))
    except (TypeError, ValueError):
        logger.error("CLOCK: offset %r / error %r are not numbers - ignored", offset, error)
        return
    if not (math.isfinite(off) and math.isfinite(err)):
        logger.error("CLOCK: non-finite offset %r / error %r - ignored", offset, error)
        return
    _clock = (off, err, True)


def get_clock_offset() -> float:
    return _clock[0]


def get_clock_error() -> float:
    """Estimated +/- seconds of the current offset."""
    return _clock[1]


def reset_clock_measured() -> None:
    """Called at the start of each lead: the offset is kept, but clock_measured() is False until the next sample."""
    global _clock
    off, err, _ = _clock
    _clock = (off, err, False)


def clock_measured() -> bool:
    return _clock[2]


def now() -> float:
    """time.time() + offset. ALL schedule decisions use this."""
    return time.time() + _clock[0]


# ── constants and consistency ────────────────────────────────────────────────────────────────────────
def _enabled() -> bool:
    return bool(config.SESSION_SCHEDULE_ENABLED)


def _ts(ts: Optional[float]) -> float:
    return now() if ts is None else float(ts)


def _read() -> Tuple[float, float, float, float, float, float, float]:
    """(a_open, a_close, b_open, b_close, lead, drain, cutoff) in seconds-of-hour / seconds."""
    extra = float(config.SESSION_CLOSE_EXTRA_SECONDS)
    return (float(config.SESSION_A_OPEN_SECOND), float(config.SESSION_A_END_SECOND) + extra,
            float(config.SESSION_B_OPEN_SECOND), float(config.SESSION_B_END_SECOND) + extra,
            float(config.WINDOW_PRECONNECT_SECONDS), float(config.SESSION_DRAIN_SECONDS),
            float(config.SESSION_BUY_CUTOFF_SECONDS))


def _problem() -> str:
    """'' if the constants are consistent, else why not."""
    try:
        a_open, a_close, b_open, b_close, lead, drain, cutoff = _read()
    except (AttributeError, TypeError, ValueError) as exc:
        return "constants unreadable: %s" % exc
    if not all(math.isfinite(v) for v in (a_open, a_close, b_open, b_close, lead, drain, cutoff)):
        return "a constant is not a finite number"
    if lead < 0 or drain < 0 or cutoff < 0:
        return "lead, drain and buy cutoff must be >= 0"
    if not (0 <= a_open < a_close):
        return "session A: need 0 <= open < close (open=%g close=%g)" % (a_open, a_close)
    if not (b_open < b_close <= _HOUR):
        return "session B: need open < close <= 3600 (open=%g close=%g)" % (b_open, b_close)
    if cutoff >= min(a_close - a_open, b_close - b_open):
        return "buy cutoff %g leaves no buyable time in a session" % cutoff
    # consecutive sessions, cyclic: previous drain must end strictly before the next connect lead starts
    if not (a_close + drain < b_open - lead):
        return "A drain end %g is not before B connect %g (sessions overlap / lead too long)" % (
            a_close + drain, b_open - lead)
    if not (b_close + drain < _HOUR + a_open - lead):
        return "B drain end %g is not before next A connect %g (sessions overlap / lead too long)" % (
            b_close + drain, _HOUR + a_open - lead)
    return ""


_last_problem = ""


def schedule_ok() -> bool:
    """False (fail closed) if the schedule constants are inconsistent. Logs CRITICAL once per distinct problem."""
    global _last_problem
    problem = _problem()
    if problem != _last_problem:
        _last_problem = problem
        if problem:
            logger.critical("SESSION SCHEDULE INCONSISTENT - failing closed (no connect, no buys, no redeploy): %s", problem)
        else:
            logger.warning("SESSION SCHEDULE consistent again")
    return not problem


def _sessions_around(ts: float) -> List[Session]:
    """Sessions of the previous, current and next hour, in time order. Only valid when schedule_ok()."""
    if not math.isfinite(ts):
        return []
    a_open, a_close, b_open, b_close, lead, _drain, _cut = _read()
    h = math.floor(ts / _HOUR)
    out: List[Session] = []
    for k in (h - 1, h, h + 1):
        base = k * _HOUR
        out.append(Session("A", base + a_open, base + a_close, base + a_open - lead))
        out.append(Session("B", base + b_open, base + b_close, base + b_open - lead))
    return out


def _all_session() -> Session:
    return Session("ALL", 0.0, _FAR_FUTURE, 0.0)


# ── schedule ─────────────────────────────────────────────────────────────────────────────────────────
def phase(ts: float = None) -> str:
    """"IDLE" | "LEAD" | "OPEN" | "DRAIN". Always "OPEN" with the schedule disabled; "IDLE" if not schedule_ok()."""
    if not _enabled():
        return "OPEN"
    ts = _ts(ts)
    if not schedule_ok():
        return "IDLE"
    for s in _sessions_around(ts):
        if s.open_ts <= ts < s.close_ts:
            return "OPEN"
        if s.close_ts <= ts < s.drain_end_ts:
            return "DRAIN"
        if s.connect_ts <= ts < s.open_ts:
            return "LEAD"
    return "IDLE"


def current_session(ts: float = None) -> Optional[Session]:
    """The session with open_ts <= ts < close_ts, else None (also None if not schedule_ok())."""
    if not _enabled():
        return _all_session()
    ts = _ts(ts)
    if not schedule_ok():
        return None
    for s in _sessions_around(ts):
        if s.open_ts <= ts < s.close_ts:
            return s
    return None


def upcoming_session(ts: float = None) -> Session:
    """The earliest session with drain_end_ts > ts (so during a drain it is the draining session).
    Raises RuntimeError if the schedule is inconsistent - check schedule_ok() first."""
    if not _enabled():
        return _all_session()
    ts = _ts(ts)
    if not schedule_ok():
        raise RuntimeError("session schedule is inconsistent (see the CRITICAL log); no upcoming session")
    for s in _sessions_around(ts):
        if s.drain_end_ts > ts:
            return s
    raise RuntimeError("no upcoming session for ts=%r" % ts)       # non-finite ts


def should_be_connected(ts: float = None) -> bool:
    """True in LEAD / OPEN / DRAIN. False if not schedule_ok()."""
    return phase(ts) != "IDLE"


def buy_allowed_at(send_ts: float = None) -> bool:
    """In a session AND open <= send_ts < close - SESSION_BUY_CUTOFF_SECONDS. False if not schedule_ok()."""
    if not _enabled():
        return True
    t = _ts(send_ts)
    if not schedule_ok():
        return False
    cutoff = float(config.SESSION_BUY_CUTOFF_SECONDS)
    for s in _sessions_around(t):
        if s.open_ts <= t < s.close_ts - cutoff:
            return True
    return False


def in_window(ts: float = None) -> bool:
    return current_session(ts) is not None


def seconds_until_connect(ts: float = None) -> float:
    """0 while should_be_connected, else seconds until the next connect lead starts. inf if not schedule_ok()."""
    if not _enabled():
        return 0.0
    ts = _ts(ts)
    if not schedule_ok():
        return math.inf
    if should_be_connected(ts):
        return 0.0
    return max(0.0, upcoming_session(ts).connect_ts - ts)


def seconds_until_open(ts: float = None) -> float:
    """0 while inside a session, else seconds until the next session opens. inf if not schedule_ok()."""
    if not _enabled():
        return 0.0
    ts = _ts(ts)
    if not schedule_ok():
        return math.inf
    if current_session(ts) is not None:
        return 0.0
    nxt = [s.open_ts for s in _sessions_around(ts) if s.open_ts > ts]
    return (min(nxt) - ts) if nxt else math.inf


def seconds_until_close(ts: float = None) -> float:
    """Seconds until the current session's close; 0 when not inside a session (or not schedule_ok())."""
    ts = _ts(ts)
    s = current_session(ts)
    return max(0.0, s.close_ts - ts) if s is not None else 0.0


def session_end(ts: float = None) -> Optional[float]:
    """close_ts of the current session, or None when not inside one."""
    s = current_session(ts)
    return s.close_ts if s is not None else None


def next_connect_in(ts: float = None) -> float:
    """Same as seconds_until_connect (API name kept from the old module)."""
    return seconds_until_connect(ts)


# ── the pre-session redeploy windows (pure math; the decision to fire lives in redeploy_guard) ──────
def _gap_pair(ts: float) -> Optional[Tuple[Session, Session]]:
    """(previous session, next session) when prev.drain_end_ts <= ts < next.connect_ts, else None.
    The gap before A is after B; the gap before B is after A. None if disabled / not schedule_ok()."""
    if not _enabled() or not schedule_ok():
        return None
    ss = _sessions_around(ts)
    for prev, nxt in zip(ss, ss[1:]):
        if prev.drain_end_ts <= ts < nxt.connect_ts:
            return prev, nxt
    return None


def in_pre_session_gap(ts: float = None) -> bool:
    """prev.drain_end_ts <= ts < next connect_ts (the only gaps where a redeploy may fire: before A and before B)."""
    return _gap_pair(_ts(ts)) is not None


def redeploy_plan(ts: float, expected_s: float, margin_s: float) -> Optional[RedeployPlan]:
    """None unless in_pre_session_gap(ts). fire_ts = connect_ts - (expected_s + margin_s); fits = fire_ts >= earliest_ts."""
    ts = _ts(ts)
    pair = _gap_pair(ts)
    if pair is None:
        return None
    prev, nxt = pair
    earliest = prev.drain_end_ts + float(config.PRE_A_REDEPLOY_AFTER_DRAIN_BUFFER_SECONDS)
    fire = nxt.connect_ts - (float(expected_s) + float(margin_s))
    return RedeployPlan(session=nxt.name, connect_ts=nxt.connect_ts, earliest_ts=earliest, fire_ts=fire,
                        fits=bool(fire >= earliest), available_s=nxt.connect_ts - earliest)


def redeploy_allowed(ts: float, expected_s: float, margin_s: float) -> bool:
    """True ONLY if the plan exists (idle gap before A or before B) and fits and plan.fire_ts <= ts < plan.connect_ts.
    Structurally impossible inside a session, in a lead or in a drain."""
    ts = _ts(ts)
    plan = redeploy_plan(ts, expected_s, margin_s)
    return bool(plan is not None and plan.fits and plan.fire_ts <= ts < plan.connect_ts)
