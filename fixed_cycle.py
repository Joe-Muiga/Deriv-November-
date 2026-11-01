"""
fixed_cycle.py – fixed, never-ending single-leg-then-cooldown trading cycle.

Spec (Sep 2026, chat-requested — restructured from the earlier two-leg
version): leg 2 is removed entirely. Leg 1 now gets exactly ONE deploy —
no auto-redeploy while it's running — and every time it ends, the bot
goes straight into a cooldown of RANDOM length before redeploying back
into a fresh leg 1:

  1. Trade for config.FIXED_CYCLE_LEG_MINUTES ("leg 1"), on a single
     deploy with no mid-leg redeploy. config.REDEPLOY_INTERVAL_HOURS is
     already set to this same 5 minutes, so restart_scheduler.py's
     existing rolling-redeploy timer IS leg 1's clock — this module
     doesn't need a separate one for it. Unlike the old two-leg version,
     that timer firing is now ALWAYS treated as leg 1 being over; there
     is no "ordinary redeploy that keeps trading" path left to take.
  2. Drain open contracts, then disconnect from Deriv entirely for a
     cooldown drawn fresh, uniformly at random, between config.
     FIXED_CYCLE_COOLDOWN_MIN_MINUTES and config.
     FIXED_CYCLE_COOLDOWN_MAX_MINUTES (health checks only during this
     window), then redeploy back into a fresh leg 1.
  3. Repeat forever.

PERSISTENCE — same reasoning as the two-leg version before it: Render's
disk on this plan is ephemeral, every redeploy is a brand-new container,
so the cooldown deadline (once chosen) has to survive that redeploy
somewhere durable — including any redeploy that happens to land mid-
cooldown for an unrelated reason (a manual deploy, a platform restart,
etc.), which must NOT re-roll the random duration or shorten/lengthen
it. State is written into THIS SERVICE'S OWN Render environment
variables via Render's public API (PUT /v1/services/{serviceId}/
env-vars/{envVarKey}) every time it changes, and read back out of
os.environ at boot — so cooldown_until, once persisted as an absolute
epoch timestamp, survives any number of redeploys unchanged until it
naturally elapses. Needs the same two env vars the old version needed:

  RENDER_API_KEY     – Render Dashboard -> Account Settings -> API Keys
  RENDER_SERVICE_ID  – the "srv-xxxxxxxxxxxxxxxxxxxx" id in this
                        service's Settings-page URL

Without both set, persistence is impossible on Render's architecture no
matter what this module does — see _warn_persistence_not_configured().

Persisted keys (plain env vars on the service):
  FIXED_PHASE            "trading" | "cooldown"
  FIXED_COOLDOWN_UNTIL    epoch seconds the current cooldown ends at
                          (the randomly chosen duration, already baked
                          in as an absolute deadline — nothing needs to
                          re-roll it on a later redeploy)

Public interface:
  load_state()                          -> {"phase", "cooldown_until"}
  in_cooldown_at_boot()                 -> cooldown_until epoch, or None
  clear_cooldown_and_resume_trading()   -> flip phase back to "trading"
                                            for a fresh leg 1
  log_trading_start()                   -> bootstrap/logging only, no
                                            leg counter to initialise
                                            anymore
  on_redeploy_due()                     -> called every settle tick
                                            while restart_scheduler.
                                            is_redeploy_pending() is
                                            True; leg 1 is the only leg,
                                            so this always requests a
                                            cooldown instead of letting
                                            an ordinary redeploy go
                                            through
  request_cooldown() / is_cooldown_requested()
                                         -> pending-flag, mirrors
                                            restart_scheduler.py's
                                            is_redeploy_pending() pattern
  enter_cooldown_now()                  -> draws a fresh random cooldown
                                            length, persists the window,
                                            starts the supervisor thread,
                                            returns cooldown_until
  start_cooldown_supervisor(until)      -> the thread that waits out the
                                            cooldown then fires the
                                            Render deploy hook
"""

import asyncio
import logging
import os
import random
import threading
import time
from typing import Optional

import aiohttp
import requests

import config

logger = logging.getLogger(__name__)

# ── Module-level state ──────────────────────────────────────────────────────
_cooldown_requested:    bool  = False
_persistence_warned_at: float = 0.0


def _push_dashboard_flag(**kwargs) -> None:
    """Soft/optional dashboard visibility — same guarded-lazy-import
    pattern the rest of this module's predecessors used. Never let a
    missing/broken keep_alive break the cycle logic itself."""
    try:
        import keep_alive
        keep_alive.update_status(**kwargs)
    except Exception:
        pass


def _read_float_env(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _warn_persistence_not_configured() -> None:
    global _persistence_warned_at
    now = time.time()
    if now - _persistence_warned_at < 900:   # at most once every 15 min
        return
    _persistence_warned_at = now
    logger.critical(
        "\n" + "=" * 78 + "\n"
        "FIXED-CYCLE PERSISTENCE NOT CONFIGURED\n"
        "RENDER_API_KEY and/or RENDER_SERVICE_ID are not set. The cooldown "
        "deadline will only live in this process's own environment — it "
        "WILL be lost on the next redeploy (Render's disk here is "
        "ephemeral), so a redeploy landing mid-cooldown would wrongly "
        "resume trading immediately instead of waiting out the rest of "
        "the randomly chosen cooldown window. Set both in the Render "
        "dashboard to fix this:\n"
        "  RENDER_API_KEY    -> Account Settings -> API Keys\n"
        "  RENDER_SERVICE_ID -> the srv-xxxxxxxx id in this service's own "
        "Settings-page URL\n" + "=" * 78
    )
    _push_dashboard_flag(fixed_cycle_persistence_missing=True)


def _persist_env_vars(mapping: dict) -> None:
    """Reflects `mapping` in this process's own os.environ immediately,
    and best-effort pushes each key to this service's Render env vars so
    a FUTURE deploy starts from them too. Same async-task-with-sync-
    fallback shape as the previous version's helper of the same name."""
    for key, value in mapping.items():
        os.environ[key] = str(value)

    api_key    = getattr(config, "RENDER_API_KEY", "")
    service_id = getattr(config, "RENDER_SERVICE_ID", "")
    if not api_key or not service_id:
        _warn_persistence_not_configured()
        return

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }

    async def _push():
        async with aiohttp.ClientSession(headers=headers) as session:
            for key, value in mapping.items():
                url = f"https://api.render.com/v1/services/{service_id}/env-vars/{key}"
                try:
                    async with session.put(
                        url, json={"value": str(value)},
                        timeout=aiohttp.ClientTimeout(total=20),
                    ) as resp:
                        if resp.status in (200, 201):
                            logger.info(f"FIXED-CYCLE: persisted {key}={value} ✓")
                        else:
                            body = await resp.text()
                            logger.error(
                                f"FIXED-CYCLE: failed to persist {key} — "
                                f"HTTP {resp.status}: {body[:300]}")
                except Exception as exc:
                    logger.error(f"FIXED-CYCLE: error persisting {key}: {exc}")

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_push())
    except RuntimeError:
        # No running loop (e.g. called from main.py's synchronous boot
        # path) — fall back to plain blocking requests, one call per key.
        for key, value in mapping.items():
            url = f"https://api.render.com/v1/services/{service_id}/env-vars/{key}"
            try:
                resp = requests.put(
                    url, json={"value": str(value)}, headers=headers, timeout=20)
                if resp.status_code in (200, 201):
                    logger.info(f"FIXED-CYCLE: persisted {key}={value} ✓ (sync)")
                else:
                    logger.error(
                        f"FIXED-CYCLE: failed to persist {key} (sync) — "
                        f"HTTP {resp.status_code}: {resp.text[:300]}")
            except Exception as exc:
                logger.error(f"FIXED-CYCLE: error persisting {key} (sync): {exc}")


# ── State loading ────────────────────────────────────────────────────────────

def load_state() -> dict:
    phase = os.environ.get("FIXED_PHASE", "trading").strip().lower()
    if phase not in ("trading", "cooldown"):
        phase = "trading"
    return {
        "phase":          phase,
        "cooldown_until": _read_float_env("FIXED_COOLDOWN_UNTIL", 0.0),
    }


def in_cooldown_at_boot() -> Optional[float]:
    """Returns the cooldown_until epoch if this process booted mid-
    cooldown (main.py must NOT start the bot / connect to Deriv in that
    case), or None if it's clear to trade — either genuinely trading
    already, or a cooldown that has already elapsed (this process IS the
    post-cooldown redeploy). Because cooldown_until is an absolute epoch
    persisted at the moment the random duration was chosen, an unrelated
    redeploy landing mid-cooldown still returns the SAME deadline here —
    the random length is never re-rolled or reset by a stray redeploy."""
    state = load_state()
    if state["phase"] == "cooldown" and state["cooldown_until"] > time.time():
        return state["cooldown_until"]
    return None


def clear_cooldown_and_resume_trading() -> None:
    """Called by main.py at boot when the persisted phase is 'cooldown'
    but the deadline has already passed — i.e. this process is the
    redeploy the cooldown supervisor triggered. Flips the phase back to
    'trading' for a fresh leg 1."""
    logger.info("FIXED-CYCLE: cooldown window elapsed — resuming trading on a fresh leg 1")
    _persist_env_vars({
        "FIXED_PHASE":          "trading",
        "FIXED_COOLDOWN_UNTIL": "0",
    })


# ── Boot logging (no leg counter anymore — leg 1 is the only leg) ──────────

def log_trading_start() -> None:
    """Called once by bot_engine.run() right after connecting, purely
    for logging — leg 1 is the only leg now, so there's nothing to
    bootstrap or persist here, unlike the old two-leg get_or_init_leg()."""
    logger.info("FIXED-CYCLE: starting a fresh leg 1 trading window")


# ── Pending-cooldown flag (mirrors restart_scheduler.py's
#    is_redeploy_pending() pattern) ─────────────────────────────────────────

def is_cooldown_requested() -> bool:
    return _cooldown_requested


def request_cooldown(reason: str) -> None:
    global _cooldown_requested
    if not _cooldown_requested:
        _cooldown_requested = True
        logger.warning(
            f"FIXED-CYCLE: COOLDOWN REQUESTED ({reason}) — no new trades "
            f"will open while open contracts drain")
        _push_dashboard_flag(fixed_cycle_cooldown_reason=reason)


def on_redeploy_due() -> None:
    """
    Called by bot_engine._settle_loop() on every settle tick while
    restart_scheduler.is_redeploy_pending() is True — mirrors exactly
    where the old on_ordinary_redeploy_due() used to be checked. Leg 1
    is now the ONLY leg, so unlike the old two-leg version there's no
    leg number to check: the rolling-redeploy timer firing always means
    leg 1 just finished, so this always requests a cooldown instead of
    letting an ordinary "redeploy and keep trading" path run. No-ops
    once a cooldown is already requested.
    """
    if is_cooldown_requested():
        return
    request_cooldown("leg 1 complete (single-leg cycle)")


def enter_cooldown_now() -> float:
    """
    Called by bot_engine._settle_loop() ONLY once every open contract has
    been actively, confirmably closed. Draws a fresh cooldown length,
    uniformly at random, between config.FIXED_CYCLE_COOLDOWN_MIN_MINUTES
    and config.FIXED_CYCLE_COOLDOWN_MAX_MINUTES, persists the resulting
    absolute deadline (so it survives any redeploy, including one that
    lands mid-cooldown for an unrelated reason — see in_cooldown_at_boot
    above), then starts the supervisor thread that waits it out and
    fires the Render deploy hook. Returns the cooldown_until epoch.
    """
    global _cooldown_requested

    min_mins = getattr(config, "FIXED_CYCLE_COOLDOWN_MIN_MINUTES", 75)
    max_mins = getattr(config, "FIXED_CYCLE_COOLDOWN_MAX_MINUTES", 150)
    if max_mins < min_mins:
        min_mins, max_mins = max_mins, min_mins
    cooldown_mins  = random.uniform(min_mins, max_mins)
    cooldown_until = time.time() + cooldown_mins * 60

    _state = {
        "FIXED_PHASE":          "cooldown",
        "FIXED_COOLDOWN_UNTIL": f"{cooldown_until:.0f}",
    }
    try:  # carry the donkey guard's session state across the redeploy
        from donkey_guard import get_guard
        _state["DONKEY_GUARD_STATE"] = get_guard().export_state()
    except Exception as exc:
        logger.warning(f"FIXED-CYCLE: could not export donkey guard state: {exc}")
    _persist_env_vars(_state)
    _cooldown_requested = False
    _push_dashboard_flag(
        fixed_cycle_phase="cooldown",
        fixed_cycle_cooldown_until=cooldown_until,
    )
    logger.warning(
        f"FIXED-CYCLE: leg 1 complete — {cooldown_mins:.1f} min cooldown "
        f"(randomly chosen between {min_mins:.0f} and {max_mins:.0f} min), "
        f"then back to a fresh leg 1")
    start_cooldown_supervisor(cooldown_until)
    return cooldown_until


# ── Cooldown supervisor — dedicated thread, deliberately NOT tied to the
#    bot's asyncio event loop, same reasoning as the previous version's
#    supervisor of the same shape (that loop is torn down the moment
#    run() returns and asyncio.run() in main.py's _run_bot() exits). ───────

def _fire_deploy_hook() -> None:
    import render_api
    target = render_api.deploy_target()

    while not target:
        logger.critical(
            "\n" + "=" * 78 + "\n"
            "COOLDOWN OVER BUT NO WAY TO REDEPLOY\n"
            "Set RENDER_API_KEY (preferred) or RENDER_DEPLOY_HOOK_URL in the "
            "Render dashboard so this service can redeploy itself into a "
            "fresh leg 1. Rechecking every 5 min.\n" + "=" * 78
        )
        _push_dashboard_flag(fixed_cycle_redeploy_hook_missing=True)
        time.sleep(300)
        target = render_api.deploy_target()

    hook_url, hdrs = target
    try:
        resp = requests.post(hook_url, headers=hdrs, timeout=20)
        if resp.status_code in (200, 201, 202):
            logger.info(f"FIXED-CYCLE: redeploy fired ✓ HTTP {resp.status_code}")
            _push_dashboard_flag(fixed_cycle_redeploy_hook_missing=False)
        else:
            logger.error(
                f"FIXED-CYCLE: redeploy FAILED — HTTP "
                f"{resp.status_code}: {resp.text[:300]}")
    except Exception as exc:
        logger.error(f"FIXED-CYCLE: redeploy ERROR: {exc}")


def _cooldown_supervisor_thread(cooldown_until: float) -> None:
    remaining = cooldown_until - time.time()
    logger.warning(
        f"FIXED-CYCLE COOLDOWN: NOT connecting to Deriv for the next "
        f"{max(0.0, remaining) / 60:.1f} min — health checks only, "
        f"staying alive on Render via the self-ping thread")

    next_log_at = 0.0
    while True:
        now       = time.time()
        remaining = cooldown_until - now
        _push_dashboard_flag(
            fixed_cycle_phase="cooldown",
            fixed_cycle_cooldown_remaining_secs=max(0, int(remaining)),
        )
        if remaining <= 0:
            break
        if now >= next_log_at:
            logger.info(f"FIXED-CYCLE COOLDOWN: {remaining / 60:.1f} min remaining")
            next_log_at = now + 60
        time.sleep(min(5.0, max(0.1, remaining)))

    logger.warning(
        "FIXED-CYCLE COOLDOWN COMPLETE — firing Render deploy hook to "
        "resume trading on a fresh leg 1")
    _fire_deploy_hook()


def start_cooldown_supervisor(cooldown_until: float) -> None:
    t = threading.Thread(
        target=_cooldown_supervisor_thread,
        args=(cooldown_until,),
        name="fixed-cycle-cooldown",
        daemon=True,
    )
    t.start()
