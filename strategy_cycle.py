"""
strategy_cycle.py – persistent, balance-trend-driven switching between the
two Donkey strategy variants (ORIGINAL <-> RAW, see signal_engine.py's
"Donkey Strategy" block).

Spec (Sep 2026): only one variant trades at a time, for roughly
STRATEGY_SWITCH_MIN_MINUTES to STRATEGY_SWITCH_MAX_MINUTES (an hour to
3 hours). The switch is triggered by a balance-trend reversal — the
balance rises for a while, then suddenly starts falling — not by a fixed
timer. When that reversal is detected: drain open contracts, disconnect
from Deriv entirely for STRATEGY_SWITCH_COOLDOWN_MINUTES (1 hour,
health-checks only), then redeploy and resume trading on the OTHER
variant. This repeats forever, alternating variants.

This replaces the wall-clock-only A/B cycle signal_engine.py shipped
with (config.DONKEY_CYCLE_START / DONKEY_CYCLE_PHASE_MINUTES, derived
fresh from time.time() every call, no state needed) — that scheme
couldn't know whether a variant was actually working, it just alternated
on a fixed clock. This one needs to remember, across the switch's own
disconnect-and-redeploy, which variant is active and how the current
run's balance has been trending — so unlike the old scheme it DOES need
persistence, for exactly the reason bot_engine.py's redeploy-drain
comments and profit_cycle.py's module docstring both already spell out:
Render's disk on this plan is ephemeral, a redeploy is a brand-new
container. State is therefore written into this service's OWN Render env
vars via Render's API (PUT /v1/services/{serviceId}/env-vars/{envVarKey})
every time it changes, and read back out of os.environ at boot — see
profit_cycle.py's module docstring for the exact same mechanism (this
module deliberately doesn't import that one — same pattern, independent
concern, independent persisted keys, so the two cycles can't step on
each other's env vars). Needs the same two env vars profit_cycle.py
does, and they're shared/reused if that module is also present:

  RENDER_API_KEY     – Render Dashboard -> Account Settings -> API Keys
  RENDER_SERVICE_ID  – the "srv-xxxxxxxxxxxxxxxxxxxx" id in this
                        service's Settings-page URL

Without both set, persistence is impossible on Render's architecture no
matter what this module does — see _warn_persistence_not_configured().

Persisted keys (plain env vars on the service):
  STRAT_ACTIVE_VARIANT       "ORIGINAL" | "RAW" — read by
                              signal_engine._donkey_active_variant()
  STRAT_PHASE                "trading" | "cooldown"
  STRAT_COOLDOWN_UNTIL        epoch seconds the current cooldown ends at
  STRAT_PHASE_STARTED_AT      epoch seconds the active variant's current
                              run began
  STRAT_PHASE_START_BALANCE   balance when that run began
  STRAT_PEAK_BALANCE          highest balance seen so far during that run

Reversal-detection thresholds (config.py, all tunable):
  STRATEGY_SWITCH_MIN_GAIN_PCT   — the run must have risen at least this
                                    % above its starting balance before a
                                    pullback counts as a "reversal" at
                                    all (filters out pure noise on a flat
                                    balance).
  STRATEGY_SWITCH_DRAWDOWN_PCT   — once that's true, a pullback of at
                                    least this % off the run's peak
                                    balance is "suddenly starts to
                                    decrease" and triggers the switch.
  STRATEGY_SWITCH_MIN_MINUTES    — floor: never switch before a variant
                                    has run at least this long, even if
                                    the drawdown condition is met early
                                    (avoids whipsawing on early noise).
  STRATEGY_SWITCH_MAX_MINUTES    — ceiling: force a switch at this point
                                    regardless of trend, so a variant
                                    that's merely flat (never triggers a
                                    reversal either way) doesn't run
                                    forever. Together with MIN_MINUTES
                                    this is the "an hour to 3 hours" the
                                    variant actually runs for.
  STRATEGY_SWITCH_COOLDOWN_MINUTES — how long to stay disconnected once a
                                      switch triggers (1 hour).

Public interface (mirrors profit_cycle.py's shape):
  load_state()                            -> dict of the six keys above
  in_cooldown_at_boot()                   -> cooldown_until epoch, or None
  clear_cooldown_and_resume_trading()     -> flip phase back to "trading"
                                              (the variant itself was
                                              already swapped when the
                                              cooldown began)
  get_or_init_phase(balance)              -> ensures phase_started_at /
                                              phase_start_balance / peak
                                              are captured for a fresh run
  active_variant()                        -> "ORIGINAL" | "RAW", what
                                              signal_engine.py reads
  update_and_check(balance)               -> updates the peak, checks
                                              both the reversal and the
                                              max-duration conditions,
                                              requests a switch if either
                                              fires
  is_switch_requested() / request_switch()
                                           -> pending-flag, mirrors
                                              restart_scheduler.py's
                                              is_redeploy_pending() /
                                              profit_cycle.py's
                                              is_cooldown_requested()
  is_starting_drawdown_breached(balance)  -> True once balance has
                                              fallen
                                              STRATEGY_SWITCH_STARTING_
                                              DRAWDOWN_PCT below the
                                              run's starting balance.
                                              Checked only when an
                                              ordinary rolling redeploy
                                              is due; when True the
                                              caller requests a switch
                                              instead of letting that
                                              redeploy go through as-is
  enter_cooldown_now()                    -> persists the cooldown
                                              window + the swapped
                                              variant, starts the
                                              supervisor thread, returns
                                              cooldown_until
  start_cooldown_supervisor(until)        -> the thread that waits out
                                              the cooldown then fires the
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
_switch_requested:      bool  = False
_persistence_warned_at: float = 0.0
_last_peak_persist_at:  float = 0.0


def _push_dashboard_flag(**kwargs) -> None:
    """Soft/optional dashboard visibility — same guarded-lazy-import
    pattern as profit_cycle.py's / restart_scheduler.py's helper of the
    same name. Never let a missing/broken keep_alive break the switcher."""
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
        "STRATEGY-SWITCH PERSISTENCE NOT CONFIGURED\n"
        "RENDER_API_KEY and/or RENDER_SERVICE_ID are not set. Which Donkey "
        "variant is active, and how its balance has been trending, will "
        "only live in this process's own environment — they WILL be lost "
        "on the next redeploy (Render's disk here is ephemeral), so the "
        "bot will forget which variant it was running and restart the "
        "trend detection from scratch every redeploy instead of tracking "
        "it across the whole run. Set both in the Render dashboard to "
        "fix this:\n"
        "  RENDER_API_KEY    -> Account Settings -> API Keys\n"
        "  RENDER_SERVICE_ID -> the srv-xxxxxxxx id in this service's own "
        "Settings-page URL\n" + "=" * 78
    )
    _push_dashboard_flag(strategy_persistence_missing=True)


def _persist_env_vars(mapping: dict) -> None:
    """Reflects `mapping` in this process's own os.environ immediately,
    and best-effort pushes each key to this service's Render env vars so
    a FUTURE deploy starts from them too. Same async-task-with-sync-
    fallback shape as profit_cycle.py's helper of the same name."""
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
                            logger.info(f"STRATEGY-CYCLE: persisted {key}={value} ✓")
                        else:
                            body = await resp.text()
                            logger.error(
                                f"STRATEGY-CYCLE: failed to persist {key} — "
                                f"HTTP {resp.status}: {body[:300]}")
                except Exception as exc:
                    logger.error(f"STRATEGY-CYCLE: error persisting {key}: {exc}")

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
                    logger.info(f"STRATEGY-CYCLE: persisted {key}={value} ✓ (sync)")
                else:
                    logger.error(
                        f"STRATEGY-CYCLE: failed to persist {key} (sync) — "
                        f"HTTP {resp.status_code}: {resp.text[:300]}")
            except Exception as exc:
                logger.error(f"STRATEGY-CYCLE: error persisting {key} (sync): {exc}")


# ── State loading ────────────────────────────────────────────────────────────

def _default_start_variant() -> str:
    start = str(getattr(config, "DONKEY_CYCLE_START", "ORIGINAL")).strip().upper()
    return start if start in ("ORIGINAL", "RAW") else "ORIGINAL"


def load_state() -> dict:
    phase = os.environ.get("STRAT_PHASE", "trading").strip().lower()
    if phase not in ("trading", "cooldown"):
        phase = "trading"
    variant = os.environ.get("STRAT_ACTIVE_VARIANT", "").strip().upper()
    if variant not in ("ORIGINAL", "RAW"):
        variant = _default_start_variant()
    return {
        "phase":              phase,
        "active_variant":     variant,
        "cooldown_until":     _read_float_env("STRAT_COOLDOWN_UNTIL", 0.0),
        "phase_started_at":   _read_float_env("STRAT_PHASE_STARTED_AT", 0.0),
        "phase_start_balance": _read_float_env("STRAT_PHASE_START_BALANCE", 0.0),
        "peak_balance":       _read_float_env("STRAT_PEAK_BALANCE", 0.0),
    }


def in_cooldown_at_boot() -> Optional[float]:
    """Returns the cooldown_until epoch if this process booted mid-
    cooldown (main.py must NOT start the bot / connect to Deriv in that
    case), or None if it's clear to trade — either genuinely trading
    already, or a cooldown that has already elapsed (this process IS the
    post-cooldown redeploy)."""
    state = load_state()
    if state["phase"] == "cooldown" and state["cooldown_until"] > time.time():
        return state["cooldown_until"]
    return None


def clear_cooldown_and_resume_trading() -> None:
    """Called by main.py at boot when the persisted phase is 'cooldown'
    but the deadline has already passed — i.e. this process is the
    redeploy the cooldown supervisor triggered. The variant swap already
    happened when the cooldown began (see enter_cooldown_now()); this
    just flips the phase back so it's not stuck reading "cooldown"
    forever, and clears the run-tracking fields so
    get_or_init_phase() captures a fresh run on the new variant."""
    logger.info("STRATEGY-CYCLE: cooldown window elapsed — resuming trading on the new variant")
    _persist_env_vars({
        "STRAT_PHASE":              "trading",
        "STRAT_COOLDOWN_UNTIL":     "0",
        "STRAT_PHASE_STARTED_AT":   "0",
        "STRAT_PHASE_START_BALANCE": "0",
        "STRAT_PEAK_BALANCE":       "0",
    })


# ── Run tracking ─────────────────────────────────────────────────────────────

def active_variant() -> str:
    """What signal_engine.py's _donkey_active_variant() calls — cheap
    os.environ read, safe to call on every signal evaluation."""
    return load_state()["active_variant"]


def get_or_init_phase(current_balance: float) -> None:
    """
    Called once by bot_engine.run() right after connecting. If a run is
    already in progress (survived from before this redeploy — the normal
    case for the rolling REDEPLOY_INTERVAL_HOURS redeploy while trading),
    leaves it alone so trend tracking continues across the whole run, not
    reset every redeploy. Otherwise (first run ever, or first run after
    clear_cooldown_and_resume_trading()) captures phase_started_at /
    phase_start_balance / peak_balance fresh, and makes sure
    STRAT_ACTIVE_VARIANT is actually set (bootstrap default on the very
    first run).
    """
    state = load_state()
    if state["phase_start_balance"] > 0:
        logger.info(
            f"STRATEGY-CYCLE: resuming in-progress {state['active_variant']} run — "
            f"started=${state['phase_start_balance']:.4f}, "
            f"peak=${state['peak_balance']:.4f}")
        return

    now = time.time()
    logger.warning(
        f"STRATEGY-CYCLE: NEW run starting on {state['active_variant']} — "
        f"start_balance=${current_balance:.4f}")
    _persist_env_vars({
        "STRAT_PHASE":               "trading",
        "STRAT_ACTIVE_VARIANT":      state["active_variant"],
        "STRAT_PHASE_STARTED_AT":    f"{now:.0f}",
        "STRAT_PHASE_START_BALANCE": f"{current_balance:.8f}",
        "STRAT_PEAK_BALANCE":        f"{current_balance:.8f}",
        "STRAT_COOLDOWN_UNTIL":      "0",
    })
    _push_dashboard_flag(
        strategy_phase="trading",
        strategy_active_variant=state["active_variant"],
        strategy_phase_start_balance=round(current_balance, 4),
    )


# ── Trend tracking / reversal detection (mirrors restart_scheduler.py's
#    is_redeploy_pending() / profit_cycle.py's is_cooldown_requested()
#    pending-flag pattern) ───────────────────────────────────────────────────

def is_switch_requested() -> bool:
    return _switch_requested


def request_switch(reason: str) -> None:
    global _switch_requested
    if not _switch_requested:
        _switch_requested = True
        logger.warning(
            f"STRATEGY-CYCLE: SWITCH REQUESTED ({reason}) — no new trades "
            f"will open while open contracts drain")
        _push_dashboard_flag(strategy_switch_reason=reason)


def update_and_check(current_balance: float) -> None:
    """
    Called on every live balance push (bot_engine._on_balance). Updates
    the run's peak balance (persisted, throttled to at most once every
    10s so a fast stream of ticks doesn't spam the Render API), then
    checks both the reversal condition and the max-duration safety net,
    requesting a switch if either fires. No-ops once a switch is already
    requested or if the run hasn't been initialized yet.
    """
    global _last_peak_persist_at

    if is_switch_requested():
        return

    state = load_state()
    start_balance = state["phase_start_balance"]
    if start_balance <= 0:
        return   # get_or_init_phase() hasn't run yet this connection

    peak = max(state["peak_balance"], current_balance)
    if peak > state["peak_balance"]:
        now = time.time()
        if now - _last_peak_persist_at >= 10:
            _last_peak_persist_at = now
            _persist_env_vars({"STRAT_PEAK_BALANCE": f"{peak:.8f}"})
        else:
            os.environ["STRAT_PEAK_BALANCE"] = f"{peak:.8f}"   # reflect locally, persist later

    elapsed_secs = time.time() - state["phase_started_at"]
    min_secs = getattr(config, "STRATEGY_SWITCH_MIN_MINUTES", 60) * 60
    max_secs = getattr(config, "STRATEGY_SWITCH_MAX_MINUTES", 180) * 60

    if elapsed_secs >= max_secs:
        request_switch(
            f"max duration {max_secs/60:.0f}min reached with no clear reversal")
        return

    if elapsed_secs < min_secs:
        return   # floor — too early to switch even on a real reversal

    gain_pct = (peak - start_balance) / start_balance * 100.0
    min_gain_pct = getattr(config, "STRATEGY_SWITCH_MIN_GAIN_PCT", 2.0)
    if gain_pct < min_gain_pct:
        return   # never really trended up — a pullback here is just noise

    drawdown_pct = (peak - current_balance) / peak * 100.0 if peak > 0 else 0.0
    drawdown_trigger_pct = getattr(config, "STRATEGY_SWITCH_DRAWDOWN_PCT", 1.5)
    if drawdown_pct >= drawdown_trigger_pct:
        request_switch(
            f"reversal: +{gain_pct:.2f}% peak then -{drawdown_pct:.2f}% off peak")


def is_starting_drawdown_breached(current_balance: float) -> bool:
    """
    Checked by bot_engine._settle_loop() ONLY at the moment the ordinary
    rolling REDEPLOY_INTERVAL_HOURS redeploy comes due
    (restart_scheduler.is_redeploy_pending() is True). True once
    current_balance has fallen STRATEGY_SWITCH_STARTING_DRAWDOWN_PCT (or
    more) BELOW the active variant's phase_start_balance — deliberately
    the run's starting balance, not its peak (that's the separate,
    smaller STRATEGY_SWITCH_DRAWDOWN_PCT reversal check in
    update_and_check() above). When this is True the rolling redeploy
    must NOT be allowed to go through as an ordinary redeploy: the caller
    should request a switch instead, so the bot drains and heads into a
    cooldown (with a variant swap) rather than redeploying straight back
    onto the same losing variant.
    """
    state = load_state()
    start_balance = state["phase_start_balance"]
    if start_balance <= 0:
        return False   # run not initialized yet — nothing to compare against

    drawdown_pct = (start_balance - current_balance) / start_balance * 100.0
    trigger_pct = getattr(config, "STRATEGY_SWITCH_STARTING_DRAWDOWN_PCT", 25)
    return drawdown_pct >= trigger_pct


def enter_cooldown_now() -> float:
    """
    Called by bot_engine._settle_loop() ONLY once every open contract has
    been actively, confirmably closed. Swaps the active variant and
    persists the cooldown window (so both survive the redeploy that ends
    it), then starts the supervisor thread that waits it out and fires
    the Render deploy hook. Returns the cooldown_until epoch.
    """
    global _switch_requested

    state          = load_state()
    next_variant   = "RAW" if state["active_variant"] == "ORIGINAL" else "ORIGINAL"
    cooldown_mins  = getattr(config, "STRATEGY_SWITCH_COOLDOWN_MINUTES", 60)
    cooldown_until = time.time() + cooldown_mins * 60

    _persist_env_vars({
        "STRAT_PHASE":                "cooldown",
        "STRAT_COOLDOWN_UNTIL":       f"{cooldown_until:.0f}",
        "STRAT_ACTIVE_VARIANT":       next_variant,
        # Cleared now so the next connect (post-redeploy) captures a
        # fresh phase_started_at / phase_start_balance / peak for the
        # new variant rather than resuming the just-finished run's.
        "STRAT_PHASE_STARTED_AT":     "0",
        "STRAT_PHASE_START_BALANCE":  "0",
        "STRAT_PEAK_BALANCE":         "0",
    })
    _switch_requested = False
    _push_dashboard_flag(
        strategy_phase="cooldown",
        strategy_cooldown_until=cooldown_until,
        strategy_next_variant=next_variant,
    )
    logger.warning(
        f"STRATEGY-CYCLE: switching {state['active_variant']} -> "
        f"{next_variant} after cooldown")
    start_cooldown_supervisor(cooldown_until)
    return cooldown_until


# ── Cooldown supervisor — dedicated thread, deliberately NOT tied to the
#    bot's asyncio event loop, same reasoning as profit_cycle.py's
#    supervisor of the same shape (that loop is torn down the moment
#    run() returns and asyncio.run() in main.py's _run_bot() exits). ──────

def _fire_deploy_hook() -> None:
    hook_url = os.environ.get("RENDER_DEPLOY_HOOK_URL", "") or \
        getattr(config, "RENDER_DEPLOY_HOOK_URL", "")

    while not hook_url:
        logger.critical(
            "\n" + "=" * 78 + "\n"
            "COOLDOWN OVER BUT RENDER_DEPLOY_HOOK_URL IS NOT SET\n"
            "Cannot redeploy to resume trading on the new variant. Add it "
            "in the Render dashboard — doing so restarts this service "
            "itself, which will pick the swapped variant up from there. "
            "Rechecking every 5 min.\n" + "=" * 78
        )
        _push_dashboard_flag(strategy_redeploy_hook_missing=True)
        time.sleep(300)
        hook_url = os.environ.get("RENDER_DEPLOY_HOOK_URL", "") or \
            getattr(config, "RENDER_DEPLOY_HOOK_URL", "")

    try:
        resp = requests.post(hook_url, timeout=20)
        if resp.status_code in (200, 201, 202):
            logger.info(f"STRATEGY-CYCLE: redeploy hook fired ✓ HTTP {resp.status_code}")
            _push_dashboard_flag(strategy_redeploy_hook_missing=False)
        else:
            logger.error(
                f"STRATEGY-CYCLE: redeploy hook FAILED — HTTP "
                f"{resp.status_code}: {resp.text[:300]}")
    except Exception as exc:
        logger.error(f"STRATEGY-CYCLE: redeploy hook ERROR: {exc}")


def _cooldown_supervisor_thread(cooldown_until: float) -> None:
    remaining = cooldown_until - time.time()
    logger.warning(
        f"STRATEGY-CYCLE COOLDOWN: NOT connecting to Deriv for the next "
        f"{max(0.0, remaining) / 60:.1f} min — health checks only, "
        f"staying alive on Render via the self-ping thread")

    next_log_at = 0.0
    while True:
        now       = time.time()
        remaining = cooldown_until - now
        _push_dashboard_flag(
            strategy_phase="cooldown",
            strategy_cooldown_remaining_secs=max(0, int(remaining)),
        )
        if remaining <= 0:
            break
        if now >= next_log_at:
            logger.info(f"STRATEGY-CYCLE COOLDOWN: {remaining / 60:.1f} min remaining")
            next_log_at = now + 60
        time.sleep(min(5.0, max(0.1, remaining)))

    logger.warning(
        "STRATEGY-CYCLE COOLDOWN COMPLETE — firing Render deploy hook to "
        "resume trading on the new variant")
    _fire_deploy_hook()


def start_cooldown_supervisor(cooldown_until: float) -> None:
    t = threading.Thread(
        target=_cooldown_supervisor_thread,
        args=(cooldown_until,),
        name="strategy-cycle-cooldown",
        daemon=True,
    )
    t.start()
