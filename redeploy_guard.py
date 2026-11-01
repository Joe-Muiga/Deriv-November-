"""
redeploy_guard.py - the pre-session redeploy: fired by the running instance while it is idle in the gap before
EACH session (after B for Session A, after A for Session B), timed to FINISH just before that session's connect
lead.  Two redeploys per hour, each with the build cache CLEARED.

  * maybe_fire(): called about once a second by a watcher thread in main.py. It fires AT MOST once per upcoming
    session (key = that session's connect_ts), only while keep_alive session_state == "IDLE" and
    window_scheduler.redeploy_allowed(now, expected, margin) is True, i.e. only inside an idle gap before a session,
    never inside a session, lead or drain. fire_ts = connect lead - (expected deploy duration + safety margin).
  * expected_duration_seconds(): rolling P95 of Render's own deploy durations (createdAt -> finishedAt), read from
    GET /v1/services/{id}/deploys (cached in memory only). Nothing is persisted in a Render env var:
    changing an env var can restart the service.
  * deploy_request(): POST /v1/services/{id}/deploys {"clearCache": "clear"} (no commitId = LATEST commit).
    A deploy hook cannot clear the cache, so with config.PRE_SESSION_REDEPLOY_CLEAR_CACHE on (default) the API key
    (RENDER_API_KEY + RENDER_SERVICE_ID) is REQUIRED and a hook is never used as a silent no-clear fallback.
    With it off the old behaviour applies: {"clearCache": "do_not_clear"}, hook allowed.
  * preflight(): refuse (and back off) when the kill switch is on, the balance target / test is finished, this
    instance does not answer /health, or the latest Render deploys failed / are still building.
  * hold_preconnect(): True after a deploy was fired, until the old instance is replaced; False again if the deploy
    failed or stalled past its deadline, so the old instance then runs the session normally.
  * A skipped session (does not fit, preflight refused, kill switch, target reached) logs a WARNING and shows on the
    dashboard; nothing is retried. Crash-loop memory comes from Render itself (status of the latest deploys).
"""
import logging
import math
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import requests

import config
import render_api
import window_scheduler as ws

logger = logging.getLogger("redeploy")

NO_CLEAR = {"clearCache": "do_not_clear"}
CLEAR = {"clearCache": "clear"}
FAILED = {"build_failed", "update_failed", "pre_deploy_failed", "canceled"}
IN_PROGRESS = {"created", "queued", "build_in_progress", "update_in_progress", "pre_deploy_in_progress"}
FINISHED_OK = {"live", "deactivated"}              # deploys that ran to completion (used for duration samples)

_EXPECTED_REFRESH_SECONDS = 600.0                  # how long a P95 read from Render is reused
_EXPECTED_RETRY_SECONDS = 60.0                     # retry delay after the deploy history could not be read
_POLL_SECONDS = 15.0                               # status poll of a fired deploy
_MIN_SAMPLES = 3                                   # deploy durations needed before the default is replaced

_lock = threading.Lock()
_state = {
    "fails": 0, "next_ok": 0.0, "last_reason": "", "last_deploy_id": "", "last_trigger": 0.0,
    "armed_key": None,        # connect_ts (int) of the session this process saw coming BEFORE its fire time
    "fired_key": None,        # connect_ts (int) of the session a deploy was fired for (never fired twice per key)
    "skipped_key": None,
    "fired_at": 0.0, "fired_expected": 0.0, "fired_connect": 0.0, "resolved": False, "last_poll": 0.0,
    "pub": None, "redeploy_state": "", "redeploy_fire_ts": 0.0, "redeploy_note": "",
}
_exp = {"value": None, "n": 0, "next_refresh": 0.0}


# ── small helpers ────────────────────────────────────────────────────────────
def _now() -> float:
    """Deriv-synced time (window_scheduler.now); the local clock if that is unavailable."""
    try:
        return float(ws.now())
    except Exception:
        return time.time()


def _cfg(name: str, default: float) -> float:
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return float(default)


def _session_state() -> Optional[str]:
    """keep_alive session_state, or None if it cannot be read (the guard then fails closed: no fire)."""
    try:
        import keep_alive
        getter = getattr(keep_alive, "get_state", None)
        if getter is not None:
            return getter("session_state", None)
        return keep_alive._state.get("session_state")
    except Exception:
        return None


def _publish(**kw) -> None:
    try:
        import keep_alive
        keep_alive.update_status(**kw)
    except Exception:
        pass


def _set_dash(state: str, fire_ts: float, note: str, **extra) -> None:
    """Remember + publish the dashboard fields (only when something changed, so a 1 s tick does not spam)."""
    sig = (state, round(fire_ts), note, tuple(sorted(extra.items())))
    _state.update(redeploy_state=state, redeploy_fire_ts=fire_ts, redeploy_note=note)
    if _state["pub"] == sig:
        return
    _state["pub"] = sig
    _publish(redeploy_state=state, redeploy_fire_ts=fire_ts, redeploy_note=note, **extra)


def _hhmmss(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(ts))


def _parse_iso(s) -> Optional[float]:
    """Render timestamps look like 2026-10-21T10:57:03.123456Z; returns epoch seconds or None."""
    if not isinstance(s, str) or not s:
        return None
    m = re.match(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$", s.strip())
    if not m:
        return None
    base, frac, tz = m.group(1).replace(" ", "T"), m.group(2) or "", m.group(3) or "Z"
    frac = frac[:7] if frac else ""
    if len(frac) == 2:
        frac += "0"
    if tz == "Z":
        tz = "+00:00"
    elif ":" not in tz:
        tz = tz[:3] + ":" + tz[3:]
    try:
        return datetime.fromisoformat(base + frac + tz).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None


# ── backoff after failures ───────────────────────────────────────────────────
def backoff_seconds(fails: int) -> float:
    base = _cfg("PRE_A_REDEPLOY_BACKOFF_BASE_SECONDS", 60.0)
    cap = _cfg("PRE_A_REDEPLOY_BACKOFF_MAX_SECONDS", 1800.0)
    return min(cap, base * (2 ** max(0, fails - 1)))


def note_skip(reason: str) -> float:
    """Register a refused/failed redeploy; returns how long the backoff lasts."""
    _state["fails"] += 1
    _state["last_reason"] = reason
    wait = backoff_seconds(_state["fails"])
    _state["next_ok"] = time.time() + wait
    logger.error("PRE-SESSION REDEPLOY FAILURE (%s) - backing off %.0fs (consecutive failures: %d)", reason, wait, _state["fails"])
    return wait


def note_ok() -> None:
    _state["fails"] = 0
    _state["last_reason"] = ""


# ── Render access ────────────────────────────────────────────────────────────
def _headers() -> dict:
    return {"Authorization": f"Bearer {render_api.api_key()}", "Accept": "application/json", "Content-Type": "application/json"}


def recent_deploys(limit: int = 5) -> Optional[list]:
    """Newest-first list of {id, status, createdAt, finishedAt} from Render; None when the API is unavailable.
    finishedAt is whatever Render returns (None if the field is absent: nothing is invented)."""
    if not render_api.api_configured():
        return None
    r = requests.get(f"{render_api.API_BASE}/services/{render_api.service_id()}/deploys",
                     headers=_headers(), params={"limit": limit}, timeout=15)
    r.raise_for_status()
    out = []
    for it in r.json():
        d = it.get("deploy", it)
        out.append({"id": d.get("id"), "status": d.get("status"), "createdAt": d.get("createdAt"),
                    "finishedAt": d.get("finishedAt")})
    return out


def local_health(timeout: float = 5.0) -> bool:
    try:
        r = requests.get(f"http://127.0.0.1:{int(getattr(config, 'PORT', 10000))}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _durations(deps: Optional[list]) -> List[float]:
    """Deploy durations (seconds) from createdAt -> finishedAt of deploys that ran to completion."""
    out: List[float] = []
    for d in deps or []:
        if d.get("status") not in FINISHED_OK:
            continue
        a, b = _parse_iso(d.get("createdAt")), _parse_iso(d.get("finishedAt"))
        if a is not None and b is not None and b > a:
            out.append(b - a)
    return out


def _p95(values: List[float]) -> float:
    s = sorted(values)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]


def expected_duration_seconds() -> float:
    """Rolling P95 of Render deploy durations (createdAt -> finishedAt). PRE_A_REDEPLOY_DEFAULT_DURATION_SECONDS
    until at least 3 usable samples exist (also when the API is unreachable or finishedAt is missing).
    Blocking when the cache is stale: call it from a thread. Cached in memory only."""
    default = _cfg("PRE_A_REDEPLOY_DEFAULT_DURATION_SECONDS", 180.0)
    now = time.time()
    if _exp["value"] is not None and now < _exp["next_refresh"]:
        return _exp["value"]
    try:
        durs = _durations(recent_deploys(20))
    except Exception as exc:
        logger.warning("PRE-SESSION REDEPLOY: could not read deploy history (%s) - using %s", exc,
                       "the last value" if _exp["value"] is not None else "the default duration")
        _exp["next_refresh"] = now + _EXPECTED_RETRY_SECONDS
        if _exp["value"] is None:
            _exp["value"] = default
        return _exp["value"]
    _exp["n"] = len(durs)
    _exp["value"] = _p95(durs) if len(durs) >= _MIN_SAMPLES else default
    _exp["next_refresh"] = now + _EXPECTED_REFRESH_SECONDS
    return _exp["value"]


# ── preflight / request ──────────────────────────────────────────────────────
# Policy refusals do not count as failures (no backoff bump); the others do.
_POLICY_PREFIXES = ("kill switch", "balance target", "experiment finished", "backing off")


def preflight() -> Tuple[bool, str]:
    """Blocking (run in a thread).  (True, '') = go; otherwise (False, reason)."""
    try:
        import kill_switch
        if kill_switch.is_paused():
            return False, "kill switch engaged"
    except Exception:
        pass
    try:
        import target_guard
        if target_guard.is_reached():
            return False, "balance target reached"
    except Exception:
        pass
    try:
        import exp_stats
        if config.TEST_MODE and exp_stats.EXP.is_done():
            return False, "experiment finished (final verdict printed) - no more redeploys"
    except Exception:
        pass
    if time.time() < _state["next_ok"]:
        return False, f"backing off for another {_state['next_ok'] - time.time():.0f}s"
    if not local_health(min(10.0, _cfg("PRE_A_REDEPLOY_HEALTH_TIMEOUT_SECONDS", 180.0))):
        return False, "local /health did not answer"
    try:
        deps = recent_deploys(5)
    except Exception as exc:
        return False, f"could not read Render deploy status: {exc}"
    if deps:
        st = deps[0].get("status")
        if st in FAILED:
            fails = 0
            for d in deps:
                if d.get("status") in FAILED:
                    fails += 1
                else:
                    break
            _state["fails"] = max(_state["fails"], fails - 1)
            return False, f"latest deploy {deps[0].get('id')} is {st}"
        if st in IN_PROGRESS:
            return False, f"latest deploy {deps[0].get('id')} still {st} (must be live before another one)"
    return True, ""


def clear_cache_wanted() -> bool:
    return bool(getattr(config, "PRE_SESSION_REDEPLOY_CLEAR_CACHE", True))


def deploy_request() -> Tuple[bool, str]:
    """Blocking. Ask Render for a deploy of the latest commit, clearing the build cache (default)."""
    clear = clear_cache_wanted()
    body = CLEAR if clear else NO_CLEAR
    hook = (os.environ.get("RENDER_DEPLOY_HOOK_URL") or getattr(config, "RENDER_DEPLOY_HOOK_URL", "") or "").strip()
    try:
        if render_api.api_configured():
            r = requests.post(f"{render_api.API_BASE}/services/{render_api.service_id()}/deploys",
                              headers=_headers(), json=body, timeout=20)
            ok = r.status_code in (200, 201, 202)
            if ok:
                try:
                    _state["last_deploy_id"] = (r.json() or {}).get("id", "") or ""
                except Exception:
                    _state["last_deploy_id"] = ""
            logger.warning("PRE-SESSION REDEPLOY via API (clearCache=%s, latest commit): HTTP %s %s", body["clearCache"],
                           r.status_code, "" if ok else r.text[:200])
        elif clear:
            return False, "clearing the build cache needs RENDER_API_KEY + RENDER_SERVICE_ID (a deploy hook cannot clear it)"
        elif hook:
            hook = re.sub(r"([?&])clear[^&]*&?", r"\1", hook).rstrip("?&")
            r = requests.post(hook, timeout=20)
            ok = r.status_code in (200, 201, 202)
            _state["last_deploy_id"] = ""
            logger.warning("PRE-SESSION REDEPLOY via deploy hook (hooks never clear the cache): HTTP %s", r.status_code)
        else:
            return False, "no RENDER_API_KEY/RENDER_SERVICE_ID or RENDER_DEPLOY_HOOK_URL"
        if ok:
            _state["last_trigger"] = time.time()
            return True, _state["last_deploy_id"]
        return False, f"HTTP {r.status_code}"
    except Exception as exc:
        return False, str(exc)


# ── the pre-session redeploys (before A and before B) ────────────────────────
def _skip(key: int, fire_ts: float, reason: str, failure: bool = False) -> str:
    _state["skipped_key"] = key
    logger.warning("PRE-SESSION REDEPLOY SKIPPED for the next session (%s); no retry - its connect proceeds normally", reason)
    if failure:
        note_skip(reason)
    _set_dash("skipped", fire_ts, reason)
    return f"skipped: {reason}"


def _resolve_failed(why: str) -> None:
    with _lock:
        _state["resolved"] = True
    note_skip(why)
    logger.error("PRE-SESSION REDEPLOY %s - the old instance runs the session normally", why)
    _set_dash("failed", _state["fired_at"], why)
    if _session_state() == "REDEPLOYING":
        _publish(session_state="IDLE")


def _poll_fired(ts: float) -> str:
    """A deploy was fired and the old instance is still running: watch for failure / stall."""
    deadline = _state["fired_at"] + _state["fired_expected"] + _cfg("PRE_A_REDEPLOY_HEALTH_TIMEOUT_SECONDS", 180.0)
    deadline = min(deadline, _state["fired_connect"])
    if ts >= deadline:
        _resolve_failed(f"deploy stalled (not replaced {ts - _state['fired_at']:.0f}s after firing, "
                        f"expected {_state['fired_expected']:.0f}s)")
        return "stalled"
    if ts - _state["last_poll"] >= _POLL_SECONDS:
        _state["last_poll"] = ts
        try:
            deps = recent_deploys(5) or []
            did = _state["last_deploy_id"]
            mine = next((d for d in deps if d.get("id") == did), None) if did else (deps[0] if deps else None)
            if mine and mine.get("status") in FAILED:
                _resolve_failed(f"deploy {mine.get('id')} is {mine.get('status')}")
                return "failed"
            if mine and mine.get("status") == "live":
                note_ok()
        except Exception as exc:
            logger.debug("PRE-SESSION REDEPLOY: status poll failed: %s", exc)
    if _session_state() == "IDLE":                    # the engine's idle loop may have rewritten the badge
        _publish(session_state="REDEPLOYING")
    return "awaiting replacement"


def _maybe_fire(ts: Optional[float]) -> str:
    ts = _now() if ts is None else ts
    if _state["fired_key"] is not None and not _state["resolved"]:
        return _poll_fired(ts)
    if not getattr(config, "PRE_A_REDEPLOY_ENABLED", True):
        return "disabled"
    if not getattr(config, "SESSION_SCHEDULE_ENABLED", True):
        return "schedule disabled"
    if not ws.in_pre_session_gap(ts):
        return "idle: outside the gaps before a session"
    state = _session_state()
    if state != "IDLE":
        return f"not idle ({state})"
    expected = expected_duration_seconds()
    margin = _cfg("PRE_A_REDEPLOY_SAFETY_MARGIN_SECONDS", 240.0)
    plan = ws.redeploy_plan(ts, expected, margin)
    if plan is None:
        return "no plan"
    key = int(plan.connect_ts)
    if key in (_state["fired_key"], _state["skipped_key"]):
        return "done for this session"
    note = f"before session {plan.session}: expected {expected:.0f}s + margin {margin:.0f}s"
    if not plan.fits:
        return _skip(key, plan.fire_ts,
                     f"does not fit: gap {plan.available_s:.0f}s < expected {expected:.0f}s + margin {margin:.0f}s")
    if ts < plan.fire_ts:
        _state["armed_key"] = key                      # this process was idle and watching BEFORE the fire time
        _set_dash("scheduled", plan.fire_ts, note)
        return f"scheduled for {_hhmmss(plan.fire_ts)} UTC"
    if _state["armed_key"] != key:
        # Booted after the fire time (e.g. the instance a deploy just created): never redeploy again at once.
        return "not armed (process started after this session's fire time)"
    if ts - plan.fire_ts > margin / 2.0:
        return _skip(key, plan.fire_ts, f"too late: {ts - plan.fire_ts:.0f}s after the planned fire time")
    if not ws.redeploy_allowed(ts, expected, margin):
        return "not allowed"
    ok, why = preflight()
    if not ok:
        return _skip(key, plan.fire_ts, why, failure=not why.startswith(_POLICY_PREFIXES))
    ok, info = deploy_request()
    if not ok:
        return _skip(key, plan.fire_ts, f"deploy request failed: {info}", failure=True)
    with _lock:
        _state.update(fired_key=key, fired_at=ts, fired_expected=expected, fired_connect=plan.connect_ts,
                      resolved=False, last_poll=ts)
    logger.warning("PRE-SESSION REDEPLOY FIRED at %s UTC (planned %s, cache %s): expected %.0fs, margin %.0fs, "
                   "session %s connect lead %s UTC, deploy %s",
                   _hhmmss(ts), _hhmmss(plan.fire_ts), "CLEARED" if clear_cache_wanted() else "kept", expected, margin,
                   plan.session, _hhmmss(plan.connect_ts), info or "?")
    _set_dash("fired", ts, f"{note}; deploy {info or '?'}", session_state="REDEPLOYING")
    return "fired"


def maybe_fire(ts: float = None) -> str:
    """Called about once a second. Fires AT MOST once per upcoming session (A and B), only while session_state is IDLE and
    ws.redeploy_allowed(...). Returns a short status string. Blocking when it talks to Render: call from a thread."""
    try:
        return _maybe_fire(ts)
    except Exception as exc:
        logger.error("PRE-SESSION REDEPLOY check error: %s", exc, exc_info=True)
        return f"error: {exc}"


def hold_preconnect() -> bool:
    """True after a deploy was fired and until the old instance is replaced; False when the deploy failed, or is
    stalled past fired_at + expected + PRE_A_REDEPLOY_HEALTH_TIMEOUT_SECONDS, or the next session's connect lead has
    arrived (the hold never outlives the lead). No network: safe to call from the engine's wait loop."""
    with _lock:
        if _state["fired_key"] is None or _state["resolved"]:
            return False
        deadline = min(_state["fired_at"] + _state["fired_expected"] + _cfg("PRE_A_REDEPLOY_HEALTH_TIMEOUT_SECONDS", 180.0),
                       _state["fired_connect"])
    return _now() < deadline


def redeploy_state() -> dict:
    """For the dashboard: keys redeploy_state / redeploy_fire_ts / redeploy_note (contract section 8) plus extras."""
    return {"redeploy_state": _state["redeploy_state"], "redeploy_fire_ts": _state["redeploy_fire_ts"],
            "redeploy_note": _state["redeploy_note"], "hold_preconnect": hold_preconnect(),
            "expected_s": _exp["value"], "samples": _exp["n"], "deploy_id": _state["last_deploy_id"],
            "fails": _state["fails"], "backoff_until": _state["next_ok"]}
