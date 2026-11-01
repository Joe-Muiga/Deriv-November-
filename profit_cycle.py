"""
profit_cycle.py – persistent profit-target trading cycle.

Spec (Sep 2026): the bot captures a "cycle starting balance" the first
time it connects, then trades until the live balance has grown
config.PROFIT_CYCLE_TARGET_PCT (default 50%) above that value. Once the
target is hit it drains any open contracts, then goes into a cooldown of
config.PROFIT_CYCLE_COOLDOWN_MINUTES (default 17) during which it is NOT
connected to Deriv at all — only the Flask health endpoint and the
self-ping keep-alive thread keep running, so Render's health checks (and
the bot's own 24/7-uptime self-ping) keep passing. When the cooldown
elapses this module fires the Render deploy hook, which brings up a
fresh deploy that starts a brand-new cycle from whatever the balance is
at that point — forever.

Starting-balance drawdown override (Sep 2026): independent of the
target-hit path above, checked only at the moment bot_engine sees the
ordinary rolling REDEPLOY_INTERVAL_HOURS redeploy come due
(restart_scheduler.is_redeploy_pending()). If the live balance is
config.PROFIT_CYCLE_STARTING_DRAWDOWN_PCT (default 25%) or more BELOW
this cycle's own starting balance at that moment, the ordinary redeploy
is skipped and the bot goes straight into the same cooldown+redeploy
path a target-hit takes instead — it does NOT redeploy plainly and keep
trading the same losing cycle. Reaching the profit target, at any other
time the bot is live, still immediately blocks any further new trade
from firing (see request_cooldown()) while open contracts drain, exactly
as before — this override doesn't change that.

PERSISTENCE — why this can't just be a local file or an in-memory dict:
Render's disk on this plan is ephemeral (see bot_engine.py's
_recover_open_contracts_from_portfolio docstring for the exact same
point made about open contracts) — every redeploy is a brand-new
container, so anything written to local disk or kept only in this
process's memory is gone the instant the rolling REDEPLOY_INTERVAL_HOURS
redeploy (every 5 min while trading, by default) or the post-cooldown
redeploy happens. To survive that, the cycle's state is written back
into THIS SERVICE'S OWN Render environment variables via Render's public
API (PUT /v1/services/{serviceId}/env-vars/{envVarKey}) every time it
changes, and read back out of os.environ at boot on the next deploy —
env vars are the one piece of a Render web service's configuration that
a fresh deploy of the same service always starts from. This needs two
extra env vars that don't exist yet elsewhere in this project:

  RENDER_API_KEY     – Render Dashboard → Account Settings → API Keys
  RENDER_SERVICE_ID  – the "srv-xxxxxxxxxxxxxxxxxxxx" id in this
                        service's Settings-page URL

Without both set, persistence is impossible on Render's architecture no
matter what this module does — see _warn_persistence_not_configured().

Persisted keys (plain env vars on the service, not a linked env group):
  CYCLE_PHASE              "trading" | "cooldown"
  CYCLE_STARTING_BALANCE   the balance this cycle started from
  CYCLE_COOLDOWN_UNTIL     epoch seconds the current cooldown ends at

Public interface:
  load_state()                          -> {"phase", "starting_balance",
                                              "cooldown_until"}
  in_cooldown_at_boot()                 -> cooldown_until epoch, or None
  clear_cooldown_and_start_new_cycle()  -> reset persisted state so the
                                            next connect captures a fresh
                                            starting balance
  get_or_init_starting_balance(balance) -> this cycle's starting balance,
                                            capturing+persisting a new one
                                            if none survived a redeploy
  check_target(balance, starting)       -> True once balance is
                                            PROFIT_CYCLE_TARGET_PCT above
                                            starting
  request_cooldown() / is_cooldown_requested()
                                         -> target-hit pending flag,
                                            mirrors restart_scheduler.py's
                                            is_redeploy_pending() pattern
  is_starting_drawdown_breached(balance, starting)
                                         -> True once balance has fallen
                                            PROFIT_CYCLE_STARTING_
                                            DRAWDOWN_PCT below this
                                            cycle's own starting balance.
                                            Checked only when an ordinary
                                            rolling redeploy is due; when
                                            True the caller requests a
                                            cooldown instead of letting
                                            that redeploy go through
  enter_cooldown_now()                  -> persists the cooldown window,
                                            starts the supervisor thread,
                                            returns cooldown_until
  start_cooldown_supervisor(until)      -> the thread that waits out the
                                            cooldown then fires the
                                            Render deploy hook
"""

import asyncio
import logging
import os
import threading
import time
from typing import Optional

import aiohttp
import requests

import config

logger = logging.getLogger(__name__)

# ── Module-level state ──────────────────────────────────────────────────────
_cooldown_requested:   bool  = False
_persistence_warned_at: float = 0.0


def _push_dashboard_flag(**kwargs) -> None:
    """Soft/optional dashboard visibility, same guarded-lazy-import pattern
    as restart_scheduler.py's helper of the same name — never let a
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
        "PROFIT-CYCLE PERSISTENCE NOT CONFIGURED\n"
        "RENDER_API_KEY and/or RENDER_SERVICE_ID are not set. The cycle's "
        "starting balance / phase / cooldown deadline will only live in "
        "this process's own environment — they WILL be lost on the next "
        "redeploy (Render's disk here is ephemeral), so the 50%-target "
        "cycle will silently restart from scratch every redeploy instead "
        "of tracking growth across the whole cycle. Set both in the "
        "Render dashboard to fix this:\n"
        "  RENDER_API_KEY    -> Account Settings -> API Keys\n"
        "  RENDER_SERVICE_ID -> the srv-xxxxxxxx id in this service's own "
        "Settings-page URL\n" + "=" * 78
    )
    _push_dashboard_flag(cycle_persistence_missing=True)


def _persist_env_vars(mapping: dict) -> None:
    """
    Reflects `mapping` in this process's own os.environ immediately (so
    anything reading it later in the same process — e.g.
    get_or_init_starting_balance() right after
    clear_cooldown_and_start_new_cycle() — sees the new values without
    waiting on the network call below), and best-effort pushes each key
    to this service's Render env vars so a FUTURE deploy starts from
    them too. Mirrors restart_scheduler.trigger_redeploy()'s
    async-task-with-sync-fallback shape.
    """
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
                            logger.info(f"PROFIT-CYCLE: persisted {key}={value} ✓")
                        else:
                            body = await resp.text()
                            logger.error(
                                f"PROFIT-CYCLE: failed to persist {key} — "
                                f"HTTP {resp.status}: {body[:300]}")
                except Exception as exc:
                    logger.error(f"PROFIT-CYCLE: error persisting {key}: {exc}")

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_push())
    except RuntimeError:
        # No running loop — e.g. called from main.py's synchronous boot
        # path. Fall back to plain blocking requests, one call per key.
        for key, value in mapping.items():
            url = f"https://api.render.com/v1/services/{service_id}/env-vars/{key}"
            try:
                resp = requests.put(
                    url, json={"value": str(value)}, headers=headers, timeout=20)
                if resp.status_code in (200, 201):
                    logger.info(f"PROFIT-CYCLE: persisted {key}={value} ✓ (sync)")
                else:
                    logger.error(
                        f"PROFIT-CYCLE: failed to persist {key} (sync) — "
                        f"HTTP {resp.status_code}: {resp.text[:300]}")
            except Exception as exc:
                logger.error(f"PROFIT-CYCLE: error persisting {key} (sync): {exc}")


# ── State loading ────────────────────────────────────────────────────────────

def load_state() -> dict:
    phase = os.environ.get("CYCLE_PHASE", "trading").strip().lower()
    if phase not in ("trading", "cooldown"):
        phase = "trading"
    return {
        "phase":            phase,
        "starting_balance": _read_float_env("CYCLE_STARTING_BALANCE", 0.0),
        "cooldown_until":   _read_float_env("CYCLE_COOLDOWN_UNTIL", 0.0),
    }


def in_cooldown_at_boot() -> Optional[float]:
    """Returns the cooldown_until epoch if this process booted mid-cooldown
    (main.py must NOT start the bot / connect to Deriv in that case), or
    None if it's clear to trade — either genuinely trading already, or a
    cooldown that has already elapsed (this process IS the post-cooldown
    redeploy)."""
    state = load_state()
    if state["phase"] == "cooldown" and state["cooldown_until"] > time.time():
        return state["cooldown_until"]
    return None


def clear_cooldown_and_start_new_cycle() -> None:
    """Called by main.py at boot when the persisted phase is 'cooldown'
    but the deadline has already passed — i.e. this process is the
    redeploy the cooldown supervisor thread triggered. Resets the
    persisted state so get_or_init_starting_balance() captures a brand
    new starting balance once bot_engine connects."""
    logger.info("PROFIT-CYCLE: cooldown window elapsed — starting a fresh cycle")
    _persist_env_vars({
        "CYCLE_PHASE":            "trading",
        "CYCLE_COOLDOWN_UNTIL":   "0",
        "CYCLE_STARTING_BALANCE": "0",
    })


# ── Starting-balance capture ────────────────────────────────────────────────

def get_or_init_starting_balance(current_balance: float) -> float:
    """
    Called once by bot_engine.run() right after connecting. If a starting
    balance survived from before this redeploy (the normal case for the
    rolling REDEPLOY_INTERVAL_HOURS redeploy while trading), resume it
    unchanged so growth is tracked across the whole cycle, not reset
    every redeploy. Otherwise (first run ever, or first run after
    clear_cooldown_and_start_new_cycle()) capture the current balance as
    the new cycle's baseline and persist it.
    """
    persisted = _read_float_env("CYCLE_STARTING_BALANCE", 0.0)
    if persisted > 0:
        logger.info(
            f"PROFIT-CYCLE: resuming existing cycle — "
            f"starting_balance=${persisted:.4f}")
        return persisted

    target_pct = getattr(config, "PROFIT_CYCLE_TARGET_PCT", 50)
    target = current_balance * (1 + target_pct / 100.0)
    logger.warning(
        f"PROFIT-CYCLE: NEW cycle starting — "
        f"starting_balance=${current_balance:.4f}, "
        f"target=${target:.4f} (+{target_pct:.0f}%)")
    _persist_env_vars({
        "CYCLE_PHASE":            "trading",
        "CYCLE_STARTING_BALANCE": f"{current_balance:.8f}",
        "CYCLE_COOLDOWN_UNTIL":   "0",
    })
    _push_dashboard_flag(
        cycle_phase="trading",
        cycle_starting_balance=round(current_balance, 4),
        cycle_target_balance=round(target, 4),
    )
    return current_balance


# ── Target check / cooldown request (mirrors restart_scheduler.py's
#    is_redeploy_pending() / trigger_redeploy() pending-flag pattern) ────────

def check_target(current_balance: float, starting_balance: float) -> bool:
    if starting_balance <= 0:
        return False
    target_pct = getattr(config, "PROFIT_CYCLE_TARGET_PCT", 50)
    return current_balance >= starting_balance * (1 + target_pct / 100.0)


def request_cooldown() -> None:
    global _cooldown_requested
    if not _cooldown_requested:
        _cooldown_requested = True
        logger.warning(
            "PROFIT-CYCLE: TARGET REACHED — requesting cooldown; no new "
            "trades will open while open contracts drain")
        _push_dashboard_flag(cycle_target_reached=True)


def is_cooldown_requested() -> bool:
    return _cooldown_requested


def is_starting_drawdown_breached(current_balance: float, starting_balance: float) -> bool:
    """
    Checked by bot_engine._settle_loop() ONLY at the moment the ordinary
    rolling REDEPLOY_INTERVAL_HOURS redeploy comes due
    (restart_scheduler.is_redeploy_pending() is True). True once
    current_balance has fallen PROFIT_CYCLE_STARTING_DRAWDOWN_PCT (or
    more) below this cycle's own starting_balance. When this is True the
    rolling redeploy must NOT be allowed to go through as an ordinary
    redeploy: the caller should request a cooldown instead (same as
    hitting the profit target), so the bot drains and starts a brand-new
    cycle from whatever the balance is once the cooldown elapses, rather
    than redeploying straight back into the same losing cycle.
    """
    if starting_balance <= 0:
        return False   # cycle not initialized yet — nothing to compare against

    drawdown_pct = (starting_balance - current_balance) / starting_balance * 100.0
    trigger_pct = getattr(config, "PROFIT_CYCLE_STARTING_DRAWDOWN_PCT", 25)
    return drawdown_pct >= trigger_pct


def enter_cooldown_now() -> float:
    """
    Called by bot_engine._settle_loop() ONLY once every open contract has
    been actively, confirmably closed. Persists the cooldown window (so
    it survives the redeploy that ends it) and starts the supervisor
    thread that waits it out and then fires the Render deploy hook.
    Returns the cooldown_until epoch.
    """
    global _cooldown_requested

    cooldown_mins  = getattr(config, "PROFIT_CYCLE_COOLDOWN_MINUTES", 17)
    cooldown_until = time.time() + cooldown_mins * 60

    _persist_env_vars({
        "CYCLE_PHASE":            "cooldown",
        "CYCLE_COOLDOWN_UNTIL":   f"{cooldown_until:.0f}",
        # Cleared now so that if this exact process somehow reconnects
        # before the redeploy fires, get_or_init_starting_balance() still
        # treats the next connect as a fresh cycle rather than resuming
        # the just-completed one.
        "CYCLE_STARTING_BALANCE": "0",
    })
    _cooldown_requested = False
    _push_dashboard_flag(
        cycle_phase="cooldown",
        cycle_cooldown_until=cooldown_until,
        cycle_target_reached=False,
    )
    start_cooldown_supervisor(cooldown_until)
    return cooldown_until


# ── Cooldown supervisor — dedicated thread, deliberately NOT tied to the
#    bot's asyncio event loop (that loop is torn down the moment run()
#    returns and asyncio.run() in main.py's _run_bot() exits, which would
#    kill a task scheduled on it). Plain blocking sleep + one HTTP POST,
#    same idea as restart_scheduler.start_restart_scheduler()'s
#    dedicated-thread fallback. ───────────────────────────────────────────

def _fire_deploy_hook() -> None:
    hook_url = os.environ.get("RENDER_DEPLOY_HOOK_URL", "") or \
        getattr(config, "RENDER_DEPLOY_HOOK_URL", "")

    while not hook_url:
        logger.critical(
            "\n" + "=" * 78 + "\n"
            "COOLDOWN OVER BUT RENDER_DEPLOY_HOOK_URL IS NOT SET\n"
            "Cannot redeploy to resume trading. Add it in the Render "
            "dashboard — doing so restarts this service itself, which "
            "will pick the new cycle up from there. Rechecking every "
            "5 min.\n" + "=" * 78
        )
        _push_dashboard_flag(cycle_redeploy_hook_missing=True)
        time.sleep(300)
        hook_url = os.environ.get("RENDER_DEPLOY_HOOK_URL", "") or \
            getattr(config, "RENDER_DEPLOY_HOOK_URL", "")

    try:
        resp = requests.post(hook_url, timeout=20)
        if resp.status_code in (200, 201, 202):
            logger.info(f"PROFIT-CYCLE: redeploy hook fired ✓ HTTP {resp.status_code}")
            _push_dashboard_flag(cycle_redeploy_hook_missing=False)
        else:
            logger.error(
                f"PROFIT-CYCLE: redeploy hook FAILED — HTTP "
                f"{resp.status_code}: {resp.text[:300]}")
    except Exception as exc:
        logger.error(f"PROFIT-CYCLE: redeploy hook ERROR: {exc}")


def _cooldown_supervisor_thread(cooldown_until: float) -> None:
    remaining = cooldown_until - time.time()
    logger.warning(
        f"PROFIT-CYCLE COOLDOWN: NOT connecting to Deriv for the next "
        f"{max(0.0, remaining) / 60:.1f} min — health checks only, "
        f"staying alive on Render via the self-ping thread")

    next_log_at = 0.0
    while True:
        now       = time.time()
        remaining = cooldown_until - now
        _push_dashboard_flag(
            cycle_phase="cooldown",
            cycle_cooldown_remaining_secs=max(0, int(remaining)),
        )
        if remaining <= 0:
            break
        if now >= next_log_at:
            logger.info(f"PROFIT-CYCLE COOLDOWN: {remaining / 60:.1f} min remaining")
            next_log_at = now + 60
        time.sleep(min(5.0, max(0.1, remaining)))

    logger.warning(
        "PROFIT-CYCLE COOLDOWN COMPLETE — firing Render deploy hook to "
        "resume trading")
    _fire_deploy_hook()


def start_cooldown_supervisor(cooldown_until: float) -> None:
    t = threading.Thread(
        target=_cooldown_supervisor_thread,
        args=(cooldown_until,),
        name="profit-cycle-cooldown",
        daemon=True,
    )
    t.start()
