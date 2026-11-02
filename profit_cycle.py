"""
profit_cycle.py – the persistent, never-ending profit cycle: state, persistence, redeploy and dashboard fields.
(The decisions are pure functions in target_guard.py; the orchestration - connect, drain, stop - is in bot_engine.py.)

One cycle:   record the live balance as the STARTING BALANCE (flat portfolio) -> trade -> net profit
(= live balance - starting balance) reaches TARGET_PROFIT -> drain -> record the cycle -> CLEAR the starting balance
(verified) -> redeploy -> the fresh process boots, sees an empty starting balance and starts the next cycle.
A cycle that runs the balance below the minimum stake ends differently: clear (verified) -> suspend (no redeploy).

PERSISTENCE - Render's disk is ephemeral, so the durable state lives in THIS service's own Render env vars, written
through the Render API (RENDER_API_KEY, RENDER_SERVICE_ID) and mirrored into os.environ at once:

  CYCLE_STARTING_BALANCE    the cycle's starting balance; "" / "0" = "start a new cycle". Managed by the bot - the only
                            value you may set by hand is "0" (force a new cycle).
  CYCLE_COUNT               completed cycles (profit target reached)
  CYCLE_LIFETIME_PROFIT     sum of the completed cycles' net profit

Everything else - TARGET_PROFIT above all - belongs to the user and is READ-ONLY to the bot: _write() accepts only the
three keys above and env_persist refuses the protected ones even if some other module asks.

The critical transitions (record the start, clear the start) use env_persist.put_env_var_verified_sync: HTTP 200/201 AND
a read-back that equals the value. A redeploy is structurally impossible until the clear of THIS process was verified
(trigger_redeploy() checks) - a redeploy with a stale starting balance would make the next boot resume the finished cycle,
find the target already met and loop. Without RENDER_API_KEY / RENDER_SERVICE_ID nothing can be persisted or redeployed:
the engine then does not start a cycle, logs CRITICAL and shows it on the dashboard.

Render's env-var write API is documented not to deploy automatically ("Changes will not be deployed automatically" on the
bulk endpoint; the single-key endpoint is silent). If a write DID trigger a deploy, trigger_redeploy() notices a deploy that
is already in flight since the clear and does not start a second one, and a loop brake refuses a new deploy while the
latest one is younger than MIN_REDEPLOY_GAP_SECONDS.
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import requests

import env_persist
import render_api
import target_guard as tg

logger = logging.getLogger(__name__)

KEY_START = "CYCLE_STARTING_BALANCE"
KEY_COUNT = "CYCLE_COUNT"
KEY_LIFETIME = "CYCLE_LIFETIME_PROFIT"
BOT_KEYS = frozenset({KEY_START, KEY_COUNT, KEY_LIFETIME})      # the ONLY variables the bot ever writes

# dashboard phases
RECORDING, TRADING, DRAINING, CLEARING, REDEPLOYING, SUSPENDED, IDLE = (
    "recording", "trading", "draining", "clearing", "redeploying", "suspended", "idle")

MIN_REDEPLOY_GAP_SECONDS = 300.0         # loop brake: never start a deploy while the latest one is younger than this
BACKOFF_BASE_SECONDS = 15.0
BACKOFF_MAX_SECONDS = 600.0
_FAILED = {"build_failed", "update_failed", "pre_deploy_failed", "canceled"}
_IN_PROGRESS = {"created", "queued", "build_in_progress", "update_in_progress", "pre_deploy_in_progress"}

_lock = threading.Lock()
_state = {"phase": RECORDING, "message": "", "persistence_missing": False, "clear_unverified": False,
          "redeploy_retrying": False, "drain_timeout": False, "idle_not_suspended": False}
_clear_verified: bool = False            # this process cleared CYCLE_STARTING_BALANCE and read it back
_clear_verified_at: float = 0.0
_last_critical = {"persistence": 0.0}


# ── saved state (os.environ is the in-process source of truth, mirrored to Render on every write) ──
def _float_env(key: str, default: float = 0.0) -> float:
    try:
        v = float((os.environ.get(key) or "").strip().replace(",", "").lstrip("$"))
        return v if v == v and abs(v) != float("inf") else default
    except ValueError:
        return default


def starting_balance() -> float:
    """The saved cycle starting balance, 0.0 when none (empty / "0" / invalid = start a new cycle)."""
    v = _float_env(KEY_START)
    return v if v > 0 else 0.0


def cycle_count() -> int:
    return max(0, int(_float_env(KEY_COUNT)))


def lifetime_profit() -> float:
    return tg.money(_float_env(KEY_LIFETIME))


def persistence_configured() -> bool:
    return render_api.api_configured()


def _write(key: str, value, env_first: bool = False) -> bool:
    """Verified write of a BOT-managed key. Anything else (TARGET_PROFIT, credentials, ...) is refused."""
    if key not in BOT_KEYS:
        raise ValueError(f"profit_cycle may only write {sorted(BOT_KEYS)}, not {key!r}")
    return env_persist.put_env_var_verified_sync(key, value, env_first=env_first)


def warn_persistence_not_configured() -> None:
    now = time.time()
    set_alert("persistence_missing", True)
    if now - _last_critical["persistence"] < 900:        # at most once every 15 minutes
        return
    _last_critical["persistence"] = now
    logger.critical(
        "\n" + "=" * 78 + "\n"
        "CYCLE PERSISTENCE NOT CONFIGURED\n"
        "RENDER_API_KEY and/or RENDER_SERVICE_ID is not set, so the cycle's starting balance cannot be saved, cleared or "
        "redeployed (Render's disk is ephemeral). No cycle is started and the bot does not trade. Set RENDER_API_KEY "
        "(Render -> Account Settings -> API Keys); RENDER_SERVICE_ID is injected by Render itself.\n" + "=" * 78)


# ── transitions ──────────────────────────────────────────────────────────────
def record_start(balance: float) -> bool:
    """Persist `balance` as the new cycle's starting balance (verified). Blocking - run in a thread. Only the caller that
    verified a flat portfolio and a live balance may call this."""
    global _clear_verified
    if not persistence_configured():
        warn_persistence_not_configured()
        return False
    if not tg.valid_balance(balance) or tg.money(balance) <= 0:
        logger.error("CYCLE: refusing to record an unusable starting balance %r", balance)
        return False
    value = f"{tg.money(balance):.2f}"
    if not _write(KEY_START, value):
        set_alert("persistence_missing", True)
        set_phase(RECORDING, "could not save the starting balance on Render - retrying next session")
        return False
    with _lock:
        _clear_verified = False
        _state.update(persistence_missing=False, clear_unverified=False, redeploy_retrying=False, drain_timeout=False,
                      idle_not_suspended=False)
    logger.warning("CYCLE START recorded: starting balance %s (TARGET_PROFIT %s)", value, tg.target_profit() or "OFF")
    set_phase(TRADING)
    return True


def record_completed_cycle(net: Optional[float]) -> bool:
    """Once per completed (profit target reached) cycle, BEFORE the clear: CYCLE_COUNT + 1 and CYCLE_LIFETIME_PROFIT += net.
    Blocking. A failure is logged but never blocks the clear/redeploy (the statistics are not critical)."""
    net = 0.0 if net is None else float(net)
    count, life = cycle_count() + 1, tg.money(lifetime_profit() + net)
    ok = _write(KEY_COUNT, str(count))
    ok = _write(KEY_LIFETIME, f"{life:.2f}") and ok
    if ok:
        logger.warning("CYCLE COMPLETE: net profit %+.2f | cycles %d | lifetime profit %+.2f", net, count, life)
    else:
        logger.error("CYCLE COMPLETE: net profit %+.2f - the lifetime statistics could NOT be persisted", net)
    return ok


def clear_start() -> bool:
    """Clear CYCLE_STARTING_BALANCE ("0") and VERIFY it (HTTP 200/201 + read-back). Blocking. os.environ is cleared at
    once, so this process never resumes the finished cycle even while the write is retried. Sets the flag that unlocks
    trigger_redeploy() / the suspend only on verified success."""
    global _clear_verified, _clear_verified_at
    os.environ[KEY_START] = "0"          # in-process first, whatever Render says: this process never resumes the cycle
    ok = persistence_configured() and _write(KEY_START, "0", env_first=True)
    with _lock:
        _clear_verified = bool(ok)
        _clear_verified_at = time.time() if ok else 0.0
        _state["clear_unverified"] = not ok
    if not persistence_configured():
        warn_persistence_not_configured()
    if ok:
        logger.warning("CYCLE: starting balance CLEARED and verified on Render")
    else:
        logger.critical("CYCLE: starting balance clear NOT VERIFIED - no redeploy / suspend until it is (retrying)")
    return bool(ok)


def clear_verified() -> bool:
    return _clear_verified


def backoff_seconds(attempt: int) -> float:
    return min(BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** max(0, attempt - 1)))


# ── redeploy ─────────────────────────────────────────────────────────────────
def _parse_iso(s) -> Optional[float]:
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def recent_deploys(limit: int = 5) -> Optional[list]:
    """Newest-first [{id, status, createdAt}] from GET /v1/services/{id}/deploys; None without the API."""
    if not render_api.api_configured():
        return None
    r = requests.get(f"{render_api.API_BASE}/services/{render_api.service_id()}/deploys",
                     headers={"Authorization": f"Bearer {render_api.api_key()}", "Accept": "application/json"},
                     params={"limit": limit}, timeout=15)
    r.raise_for_status()
    out = []
    for it in r.json():
        d = it.get("deploy", it)
        out.append({"id": d.get("id"), "status": d.get("status"), "createdAt": d.get("createdAt")})
    return out


def deploy_status(deploy_id: str) -> Optional[str]:
    """Render's status of one deploy (None when unknown / not readable)."""
    try:
        for d in recent_deploys(10) or []:
            if d.get("id") == deploy_id:
                return d.get("status")
    except Exception as exc:
        logger.debug("CYCLE: deploy status read failed: %s", exc)
    return None


def trigger_redeploy() -> Tuple[bool, str]:
    """Redeploy THIS service (render_api.deploy_target(): the deploy hook if one is set, else POST /deploys).
    Blocking - run in a thread; the caller retries with backoff on (False, reason). Returns (True, deploy_id).
    Refused unless this process verified the clear of CYCLE_STARTING_BALANCE."""
    if not _clear_verified:
        logger.critical("CYCLE: redeploy REFUSED - the starting balance clear is not verified (a redeploy now would "
                        "resume the finished cycle and loop)")
        return False, "starting balance clear not verified"
    target = render_api.deploy_target()
    if target is None:
        return False, "no RENDER_API_KEY / RENDER_SERVICE_ID / deploy hook"
    try:
        deps = recent_deploys(5)
    except Exception as exc:                                  # the loop brake fails open: the clear IS verified
        logger.warning("CYCLE: could not read Render's deploy list (%s) - continuing", exc)
        deps = None
    if deps:
        newest = deps[0]
        created = _parse_iso(newest.get("createdAt"))
        if created is not None:
            if created >= _clear_verified_at - 5 and newest.get("status") in _IN_PROGRESS | {"live"}:
                logger.warning("CYCLE: deploy %s (%s) already started since the clear - not deploying twice",
                               newest.get("id"), newest.get("status"))
                return True, str(newest.get("id") or "")
            age = time.time() - created
            if age < MIN_REDEPLOY_GAP_SECONDS:
                return False, f"latest deploy is only {age:.0f}s old (loop brake {MIN_REDEPLOY_GAP_SECONDS:.0f}s)"
    url, headers = target
    try:
        if headers:       # Render API: keep the build cache, new process only
            r = requests.post(url, headers=headers, json={"clearCache": "do_not_clear"}, timeout=20)
        else:             # deploy hook
            r = requests.post(url, timeout=20)
    except Exception as exc:
        return False, f"deploy request failed: {exc}"
    if r.status_code not in (200, 201, 202):
        return False, f"HTTP {r.status_code} {r.text[:150]}"
    try:
        did = str((r.json() or {}).get("id", "") or "")
    except Exception:
        did = ""
    logger.warning("CYCLE: REDEPLOY triggered (deploy %s) - the new process starts the next cycle", did or "?")
    return True, did


# ── dashboard ────────────────────────────────────────────────────────────────
def set_phase(phase: str, message: str = "") -> None:
    with _lock:
        _state["phase"], _state["message"] = phase, message
    _push(cycle_phase=phase, cycle_message=message)


def set_alert(name: str, on: bool = True) -> None:
    with _lock:
        if name not in _state or name in ("phase", "message"):
            return
        _state[name] = bool(on)
    _push(**{f"cycle_alert_{name}": bool(on)})


def _push(**kw) -> None:
    try:
        import keep_alive
        keep_alive.update_status(**kw)
    except Exception:
        pass


def dashboard_fields(balance=None, min_stake=None) -> dict:
    """Everything the dashboard shows about the cycle (pure read; also what the engine publishes every few seconds)."""
    start, tp = starting_balance(), tg.target_profit()
    with _lock:
        st = dict(_state)
    return {
        "cycle_phase": st["phase"], "cycle_message": st["message"],
        "cycle_starting_balance": start if start > 0 else None,
        "target_profit": tp,
        "cycle_target_balance": tg.target_balance_for(start, tp),
        "cycle_net_profit": tg.net_profit(balance, start) if balance is not None else None,
        "min_stake_used": min_stake,
        "cycle_count": cycle_count(), "lifetime_profit": lifetime_profit(),
        "cycle_alert_persistence_missing": st["persistence_missing"],
        "cycle_alert_clear_unverified": st["clear_unverified"],
        "cycle_alert_redeploy_retrying": st["redeploy_retrying"],
        "cycle_alert_drain_timeout": st["drain_timeout"],
        "cycle_alert_idle_not_suspended": st["idle_not_suspended"],
    }


def publish(balance=None, min_stake=None) -> None:
    _push(**dashboard_fields(balance, min_stake))
