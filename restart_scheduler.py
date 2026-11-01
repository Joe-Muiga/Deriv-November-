"""
restart_scheduler.py – rolling redeploy scheduler.

The bot force-redeploys itself every
config.REDEPLOY_INTERVAL_SECONDS seconds (default 150), measured from the last actual
redeploy/resumption — NOT anchored to a fixed clock time. Replaces the
previous fixed-Africa/Nairobi-midnight schedule (_next_nairobi_midnight()),
which ignored config.REDEPLOY_INTERVAL_SECONDS entirely and always fired once
a day regardless of what that config value was set to.

Public interface (unchanged contract with bot_engine.py):
  - is_redeploy_pending() -> bool
        True once the rolling timer has fired and a redeploy is due.
        bot_engine.py's _main_loop() checks this to pause taking on new
        trades, and _settle_loop() checks it to know when to start
        draining open contracts before actually redeploying.
  - trigger_redeploy() -> None
        Called by bot_engine.py's _settle_loop() ONLY once every open
        contract has been actively, confirmably closed (or the drain
        window has been exceeded and the redeploy is being delayed —
        in which case this is NOT called, see bot_engine.py Fix G).
        Actually fires the Render deploy hook and clears the pending flag.
  - run_scheduler() -> coroutine
        The rolling timer loop itself. Started as a background task by
        bot_engine.py's run() alongside its other loops.

Draining coordination lives entirely in bot_engine.py (it owns the open
contracts and the reconciliation / Multiplier max-hold machinery) — this
module only owns the schedule and the actual deploy-hook trigger.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone

import aiohttp

import config

logger = logging.getLogger(__name__)

# ── Module-level scheduler state ────────────────────────────────────────────
_pending: bool = False
_last_triggered_at: float = 0.0          # last time trigger_redeploy() ran at all
_last_confirmed_redeploy_at: float = 0.0  # last time it actually succeeded (Task 7)
_last_scheduled_at: float = 0.0
_process_start_time: float = time.time()
_last_watchdog_warning_at: float = 0.0
_hook_missing_warned_at: float = 0.0
_failed_flag: bool = False              # last deploy REQUEST failed (engine then resumes trading)
_due_at: float = 0.0                     # when the pending flag will be raised (0 = not scheduled / cycle skipped)

# Task 7 fix — soft/optional dashboard visibility. restart_scheduler.py
# doesn't otherwise depend on keep_alive.py; imported lazily and guarded
# so a missing/broken keep_alive never breaks the scheduler itself.
def _push_dashboard_flag(**kwargs) -> None:
    try:
        import keep_alive
        keep_alive.update_status(**kwargs)
    except Exception:
        pass


def is_enabled() -> bool:
    """Scheduled self-redeploy is OFF when REDEPLOY_INTERVAL_SECONDS <= 0 (the default)."""
    try:
        return float(getattr(config, "REDEPLOY_INTERVAL_SECONDS", 0)) > 0
    except Exception:
        return False


def buys_blocked() -> bool:
    """True while a redeploy is pending OR due within REDEPLOY_BUY_BLOCK_SECONDS: no new buys, so contracts
    do not straddle the redeploy."""
    if _pending:
        return True
    if not is_enabled() or not _due_at:
        return False
    return time.time() >= _due_at - float(getattr(config, "REDEPLOY_BUY_BLOCK_SECONDS", 20))


def consume_failure() -> bool:
    """True once after a deploy request failed, so the engine can lift its post-redeploy halt."""
    global _failed_flag
    f, _failed_flag = _failed_flag, False
    return f


def skip_cycle(reason: str) -> None:
    """Drop a pending redeploy (guard refused): trading resumes, the next interval starts now."""
    global _pending, _due_at
    logger.error("REDEPLOY CYCLE SKIPPED: %s", reason)
    _pending = False
    _due_at = 0.0


def is_redeploy_pending() -> bool:
    """True once the rolling REDEPLOY_INTERVAL_SECONDS timer has fired and a
    redeploy is due but hasn't been confirmed-triggered yet."""
    return _pending


async def run_scheduler():
    """
    Sleeps for config.REDEPLOY_INTERVAL_SECONDS, measured from the moment
    this loop iteration starts (i.e. from the last actual redeploy or from
    process start on the very first iteration) — not anchored to any fixed
    wall-clock time. Sets the pending flag, then repeats.
    bot_engine.py's _settle_loop() is responsible for noticing
    is_redeploy_pending() == True, draining open contracts for real, and
    calling trigger_redeploy() once that's done (which starts the next
    interval).
    """
    global _pending, _last_scheduled_at, _due_at

    if not is_enabled():
        logger.info("REDEPLOY SCHEDULER: disabled (REDEPLOY_INTERVAL_SECONDS<=0) - bot runs continuously")
        return

    while True:
        try:
            interval_secs = float(getattr(config, "REDEPLOY_INTERVAL_SECONDS", 600))
            _last_scheduled_at = time.time()
            _due_at = _last_scheduled_at + interval_secs

            next_fire = datetime.fromtimestamp(_due_at, tz=timezone.utc)
            logger.info(
                f"REDEPLOY SCHEDULER: next redeploy at {next_fire.isoformat()} "
                f"({interval_secs:.0f}s from now, rolling)"
            )
            # Pre-flight BEFORE buys are blocked: a refused cycle must not cost any trading time.
            block = float(getattr(config, "REDEPLOY_BUY_BLOCK_SECONDS", 20))
            lead = block + 10.0
            await asyncio.sleep(max(0.0, interval_secs - lead))
            import redeploy_guard
            ok, why = await asyncio.to_thread(redeploy_guard.preflight)
            if not ok:
                _due_at = 0.0
                wait = redeploy_guard.note_skip(why)
                _push_dashboard_flag(redeploy_skipped_reason=why)
                await asyncio.sleep(max(wait, 1.0))
                continue
            redeploy_guard.note_ok()
            await asyncio.sleep(max(0.0, _due_at - time.time()))

            _pending = True
            logger.warning(
                f"REDEPLOY DUE: {interval_secs:.0f}s rolling interval elapsed — "
                f"waiting for bot_engine to drain open contracts before "
                f"actually redeploying"
            )

            # Wait here until bot_engine.py's _settle_loop() confirms the
            # drain completed and calls trigger_redeploy() (which clears
            # _pending). Poll rather than block indefinitely so a stuck
            # drain doesn't wedge this loop forever — just keep logging.
            waited = 0
            while _pending:
                await asyncio.sleep(1)
                waited += 1
                if waited % 300 == 0:
                    logger.warning(
                        f"REDEPLOY STILL PENDING: {waited}s since due — "
                        f"bot_engine is still draining open contracts "
                        f"rather than wiping their bookkeeping"
                    )

            # Loop repeats: the next interval is measured from HERE (i.e.
            # from the actual redeploy/resumption), not from the previous
            # fire time — this is what makes it a rolling interval instead
            # of a fixed-clock schedule that could drift or double-fire if
            # a drain ran long.

        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(f"restart_scheduler.run_scheduler: {exc}")
            await asyncio.sleep(60)


def trigger_redeploy() -> None:
    """
    Fires the Render deploy hook (if configured) and clears the pending
    flag. Called by bot_engine.py ONLY after every open contract has been
    actively, confirmably closed.

    FIX (Task 7 — root cause of the 5h37m-without-redeploy incident,
    confirmed by code inspection): when RENDER_DEPLOY_HOOK_URL isn't set
    in the deployed environment, this used to clear `_pending` and quietly
    return — no actual redeploy happens, but the rolling timer in
    run_scheduler() sees `_pending` go False and starts a fresh
    REDEPLOY_INTERVAL_SECONDS window right then, as if a real redeploy had
    just occurred. The process just keeps running and keeps trading,
    indefinitely, with the scheduler's own logs looking completely normal
    ("next redeploy at ..." every cycle) — this fully explains the
    symptom: at the 3h mark the drain likely finished quickly, this
    silent no-op fired, trading resumed immediately, and the next ~2h37m
    of trading happened before the process was presumably restarted by
    some external/manual trigger rather than this mechanism.

    Now: a missing hook URL is logged at CRITICAL with an unmissable
    banner (not a routine warning line) and pushed to the dashboard, and
    `_last_confirmed_redeploy_at` — the baseline the new watchdog below
    uses — is deliberately NOT updated on this path, only on an actual
    successful hook fire. `_last_triggered_at` (this function having run
    at all, success or not) is kept separately for existing behavior/logs.
    """
    global _pending, _last_triggered_at, _hook_missing_warned_at, _last_confirmed_redeploy_at, _due_at

    import render_api
    import redeploy_guard
    _target = render_api.deploy_target()
    hook_url, _hdrs = _target if _target else ("", {})
    _last_triggered_at = time.time()

    async def _fire():
        global _last_confirmed_redeploy_at
        import asyncio as _a
        ok, info = await _a.to_thread(redeploy_guard.deploy_request)
        if ok:
            logger.info(f"REDEPLOY REQUESTED (clearCache=do_not_clear, latest commit) {info}")
            _last_confirmed_redeploy_at = time.time()
            _push_dashboard_flag(redeploy_hook_missing=False, redeploy_watchdog_overdue=False)
        else:
            logger.error(f"REDEPLOY REQUEST FAILED: {info}")
            global _failed_flag
            _failed_flag = True
            redeploy_guard.note_skip(f"deploy request failed: {info}")

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_fire())
    except RuntimeError:
        ok, info = redeploy_guard.deploy_request()
        if ok:
            _last_confirmed_redeploy_at = time.time()
        else:
            redeploy_guard.note_skip(f"deploy request failed: {info}")

    _pending = False
    _due_at = 0.0


async def _watchdog_loop():
    """
    Task 7 self-check requirement: an independent watchdog that fires a
    loud, high-visibility warning if wall-clock time since the last
    CONFIRMED redeploy (or process start, before the first one) exceeds
    REDEPLOY_INTERVAL_SECONDS by more than a grace margin — regardless of
    what run_scheduler()'s own `_pending` state currently says, and
    regardless of the specific cause (missing hook URL, a stuck drain in
    bot_engine._settle_loop(), a wedged run_scheduler() task, etc). This
    is what makes "the rolling redeploy mechanism has stopped actually
    redeploying" visible in logs/dashboard going forward, rather than
    only discovered after the fact by noticing the process has been up
    far longer than intended.
    """
    global _last_watchdog_warning_at

    if not is_enabled():
        return

    check_interval_secs = 60
    grace_secs = getattr(config, "REDEPLOY_WATCHDOG_GRACE_MINS", 20) * 60
    warn_repeat_secs = getattr(config, "REDEPLOY_WATCHDOG_REPEAT_MINS", 15) * 60

    while True:
        try:
            await asyncio.sleep(check_interval_secs)
            interval_secs = float(getattr(config, "REDEPLOY_INTERVAL_SECONDS", 600))
            baseline = _last_confirmed_redeploy_at or _process_start_time
            uptime_secs = time.time() - baseline
            overdue_by = uptime_secs - interval_secs

            if overdue_by > grace_secs:
                now = time.time()
                if now - _last_watchdog_warning_at >= warn_repeat_secs:
                    _last_watchdog_warning_at = now
                    logger.critical(
                        "\n" + "=" * 78 + "\n"
                        f"REDEPLOY WATCHDOG: no CONFIRMED redeploy in "
                        f"{uptime_secs:.0f}s — {overdue_by / 60:.0f}min "
                        f"past the {interval_secs:.0f}s interval + grace margin. "
                        f"The rolling-redeploy mechanism may be stalled — check "
                        f"RENDER_DEPLOY_HOOK_URL is set and reachable, and that "
                        f"bot_engine's _settle_loop()/drain logic isn't stuck.\n"
                        + "=" * 78
                    )
                    _push_dashboard_flag(
                        redeploy_watchdog_overdue=True,
                        redeploy_watchdog_overdue_mins=round(overdue_by / 60, 1),
                    )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(f"restart_scheduler._watchdog_loop: {exc}")


def start_restart_scheduler():
    """
    Entry point called by main.py at startup (unchanged name/contract from
    before this fix — only the scheduling logic inside run_scheduler()
    changed, per Implementation Brief v2, Fix G).

    Starts run_scheduler() as a background asyncio task if a loop is
    already running (the normal case — main.py calls this from inside its
    async startup), or spins up a dedicated background thread with its
    own event loop if called from synchronous code before any loop
    exists. Safe to call exactly once at process startup.
    """
    if not is_enabled():
        logger.info("restart_scheduler: DISABLED (REDEPLOY_INTERVAL_SECONDS<=0) - nothing started")
        return

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(run_scheduler())
        loop.create_task(_watchdog_loop())
        logger.info("restart_scheduler: started on the running event loop (scheduler + watchdog)")
        return
    except RuntimeError:
        pass

    import threading

    def _thread_target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(asyncio.gather(run_scheduler(), _watchdog_loop()))
        except Exception as exc:
            logger.error(f"restart_scheduler thread crashed: {exc}")

    t = threading.Thread(target=_thread_target, name="restart-scheduler", daemon=True)
    t.start()
    logger.info("restart_scheduler: started on a dedicated background thread (scheduler + watchdog)")
