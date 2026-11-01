"""
main.py – entry point (Render web service).

  • Main thread  : Flask (dashboard + /health) – always running.
  • bot-engine   : thread running the engine (asyncio). The engine itself owns the hourly session clock
                   (Session A / Session B, connect leads, disconnect, drain); this file only builds it and
                   rebuilds it if it ever crashes or returns. There is no cooldown / cycle / restart timer here.
  • keep-alive   : self-ping of /health.
  • redeploy     : one redeploy before EACH session (two per hour), fired by the redeploy-watcher thread
                   (redeploy_guard.maybe_fire, polled once a second) while the instance is idle after the previous
                   session, so the fresh instance is live before the next connect lead. clearCache is "clear".
  • SIGTERM      : disconnects from Deriv, flushes stats / ledger and follower DB buffers, then exits.

Env vars: DERIV_API_TOKEN (required), RENDER_API_KEY (redeploy + stats persistence),
optional DERIV_APP_ID, DERIV_ACCOUNT_MODE ("demo" default / "real" picks which
account the token's OTP is minted for), DASHBOARD_PIN, PREDICTOR_SYMBOLS, ...

Follower accounts (optional): DATABASE_URL, ADMIN_PASSWORD, ENCRYPTION_KEY - see render.yaml.
"""
import asyncio
import logging
import os
import signal
import sys
import threading
import time

import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("main")

if not config.DERIV_API_TOKEN:
    logger.critical("DERIV_API_TOKEN is not set! Add it in the Render service environment.")

_live = {"engine": None, "loop": None}      # the running engine + its event loop (read by the SIGTERM handler)
_followers_started = False


async def _run_engine(eng):
    _live["loop"] = asyncio.get_running_loop()
    await eng.run()


def _run_bot():
    """Run the engine; if it ever exits or crashes, build a NEW one and start it again (never sit idle)."""
    import target_guard
    if config.MH_MODE:
        from mh_engine import MHEngine as Engine
        from set_engine import account_allowed
        allowed, why = account_allowed()
        if not allowed:
            logger.critical("MH MODE REFUSED TO START: %s", why)
            return                                   # demo-only guard, same as set mode
    elif config.SET_MODE:
        from set_engine import SetEngine as Engine, account_allowed
        allowed, why = account_allowed()
        if not allowed:
            logger.critical("SET MODE REFUSED TO START: %s", why)
            return                                   # no restart loop: this needs a config change
    else:
        from bot_engine import BotEngine as Engine
    while True:
        if target_guard.is_reached():
            logger.warning("Balance target reached - bot stays stopped (no restart)")
            return
        eng = Engine()
        _live["engine"] = eng
        try:
            asyncio.run(_run_engine(eng))            # the engine owns the session clock: it idles / connects / disconnects itself
        except Exception as exc:
            logger.error(f"Bot crashed: {exc}", exc_info=True)
        finally:
            _live["engine"], _live["loop"] = None, None
        if target_guard.is_reached():
            logger.warning("Balance target reached - bot stays stopped (no restart)")
            return
        logger.warning("Bot engine returned or crashed; building a new engine in 5 s")
        time.sleep(5)


def _redeploy_watcher():
    """Polls redeploy_guard.maybe_fire() once a second (it is blocking when it talks to Render, hence its own thread)."""
    import redeploy_guard
    while True:
        try:
            redeploy_guard.maybe_fire()
        except Exception as exc:                     # maybe_fire never raises; belt and braces
            logger.error(f"redeploy watcher error: {exc}", exc_info=True)
        time.sleep(1.0)


async def _graceful_engine_stop(eng):
    """SIGTERM path: no new buys, no reconnect, Deriv socket closed, stats + ledger flushed. Sends nothing to Deriv."""
    eng._session_over = True                         # BotEngine._blocked() -> no new buys
    client = getattr(eng, "client", None)
    if client is not None:
        stop = getattr(client, "set_reconnect_enabled", None)
        if stop is not None:
            try:
                stop(False)
            except Exception:
                pass
        try:
            await asyncio.wait_for(client.shutdown("SIGTERM"), timeout=4)
        except Exception as exc:
            logger.warning("SIGTERM: Deriv shutdown skipped: %s", exc)
    stats = getattr(eng, "stats", None)
    if stats is not None:
        try:
            await asyncio.wait_for(stats.flush(force=True), timeout=3)
        except Exception as exc:
            logger.warning("SIGTERM: stats flush skipped: %s", exc)
    ledger = getattr(eng, "ledger", None)
    if ledger is not None:
        try:
            await asyncio.wait_for(asyncio.to_thread(ledger.flush, 3.0), timeout=4)
        except Exception as exc:
            logger.warning("SIGTERM: ledger flush skipped: %s", exc)


def _shutdown_engine_blocking(timeout: float = 8.0) -> None:
    eng, loop = _live.get("engine"), _live.get("loop")
    if eng is None or loop is None or loop.is_closed():
        return
    try:
        asyncio.run_coroutine_threadsafe(_graceful_engine_stop(eng), loop).result(timeout)
    except Exception as exc:
        logger.warning("SIGTERM: engine did not stop cleanly: %s", exc)


def _on_sigterm(signum, frame):
    logger.warning("SIGTERM - disconnecting from Deriv, flushing stats / ledger and follower DB buffers")
    _shutdown_engine_blocking(8.0)
    if _followers_started:
        try:
            import follower_manager
            follower_manager.shutdown_blocking(8.0)
        except Exception as exc:
            logger.warning("SIGTERM: follower flush skipped: %s", exc)
    raise SystemExit(0)


def _mmss(sec: float) -> str:
    m = int(sec // 60)
    return f"{m:02d}:{sec - 60 * m:06.3f}"


def _log_startup() -> None:
    logger.info("Starting digit predictor bot … (%s)",
                "MH MODE: multi-horizon digit model, h<=%d, all 18 Over/Under covers, admitted on calibrated edge" % config.MH_HMAX_TRADE
                if config.MH_MODE else "SET MODE: top-%d digit set -> covering Over/Under, all symbols in parallel" % config.SET_SIZE
                if config.SET_MODE else "original single-contract mode")
    logger.info(f"  Deriv App ID : {config.DERIV_APP_ID}")
    logger.info(f"  API token    : {'SET ✓' if config.DERIV_API_TOKEN else 'MISSING ✗'}")
    logger.info(f"  Symbols      : {config.PREDICTOR_SYMBOLS}  "
                f"min stake (PREDICTOR_STAKE)={config.PREDICTOR_STAKE}")
    logger.info(f"  Account mode : {config.DERIV_ACCOUNT_MODE}")
    if config.TEST_MODE:
        import exp_mode
        logger.warning("  TEST MODE ON : frozen config hash %s | demo only | flat stake %s | followers OFF", exp_mode.config_hash(), config.TEST_STAKE)
    legacy = list(getattr(config, "LEGACY_REDEPLOY_ENV_SET", []) or [])
    if legacy:
        logger.warning("  Legacy redeploy env vars are set on Render and IGNORED: %s - delete them", ", ".join(legacy))
    if not config.SESSION_SCHEDULE_ENABLED:
        logger.critical("  SESSION SCHEDULE IS OFF (SESSION_SCHEDULE_ENABLED=0): the session clock and the pre-A redeploy are disabled")
    else:
        close_x = config.SESSION_CLOSE_EXTRA_SECONDS
        logger.info("  Sessions     : A %s-%s | B %s-%s (mm:ss.mmm past each hour, UTC, drain/close incl. +%.0fs); "
                    "connect lead %.0fs, disconnect after drain",
                    _mmss(config.SESSION_A_OPEN_SECOND), _mmss(config.SESSION_A_END_SECOND + close_x),
                    _mmss(config.SESSION_B_OPEN_SECOND), _mmss(config.SESSION_B_END_SECOND + close_x),
                    close_x, config.WINDOW_PRECONNECT_SECONDS)
    if config.PRE_A_REDEPLOY_ENABLED and config.SESSION_SCHEDULE_ENABLED:
        import render_api
        has_api = render_api.api_configured()
        has_hook = bool((os.environ.get("RENDER_DEPLOY_HOOK_URL") or getattr(config, "RENDER_DEPLOY_HOOK_URL", "") or "").strip())
        clear = bool(getattr(config, "PRE_SESSION_REDEPLOY_CLEAR_CACHE", True))
        logger.info("  Redeploy     : ON - before EACH session (A and B), idle gap after the previous one, finishing %.0fs before the "
                    "connect lead (expected duration = P95 of Render deploys, default %.0fs), clearCache=%s",
                    config.PRE_A_REDEPLOY_SAFETY_MARGIN_SECONDS, config.PRE_A_REDEPLOY_DEFAULT_DURATION_SECONDS,
                    "clear" if clear else "do_not_clear")
        if clear and not has_api:
            logger.critical("  Redeploy     : ENABLED with cache clearing but RENDER_API_KEY / RENDER_SERVICE_ID is not set - NO redeploy will happen "
                            "(a deploy hook cannot clear the build cache)")
        elif not (has_api or has_hook):
            logger.critical("  Redeploy     : ENABLED but neither RENDER_API_KEY(+RENDER_SERVICE_ID) nor RENDER_DEPLOY_HOOK_URL is set - NO redeploy will happen")
    else:
        logger.info("  Redeploy     : OFF (PRE_A_REDEPLOY_ENABLED=0 or schedule off)")
    import target_guard as _tg
    logger.info(f"  Balance target: {_tg.target() if _tg.target() else 'OFF (TARGET_BALANCE not set)'}")
    logger.info(f"  Port         : {config.PORT}")


if __name__ == "__main__":
    from keep_alive import app, start_keep_alive

    _log_startup()
    start_keep_alive()
    signal.signal(signal.SIGTERM, _on_sigterm)       # always: disconnect from Deriv + flush, even without followers
    threading.Thread(target=_redeploy_watcher, name="redeploy-watcher", daemon=True).start()

    # Follower accounts: own thread + own event loop + Flask Blueprint. Fully optional:
    # any failure here is logged and the main account below starts regardless.
    try:
        if config.TEST_MODE:
            raise RuntimeError("TEST_MODE: follower mirroring is OFF")
        import follower_manager
        import follower_web
        follower_web.register(app)
        _followers_started = bool(follower_manager.start())
    except Exception as exc:
        logger.error(f"Follower system not started: {exc}", exc_info=not config.TEST_MODE)
    threading.Thread(target=_run_bot, name="bot-engine", daemon=True).start()
    logger.info("Bot engine thread started ✓")
    app.run(host="0.0.0.0", port=config.PORT, use_reloader=False, threaded=True)
