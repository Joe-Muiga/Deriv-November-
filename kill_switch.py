"""
kill_switch.py – one-tap trading pause, controlled from the dashboard.

When the switch is ON ("paused"):
  * no NEW contract is ever bought (hard-blocked inside every buy function
    in deriv_client.py, plus a clean early-exit in bot_engine.py);
  * contracts that are already open are left alone and settle normally;
  * the bot stays connected and the dashboard keeps updating.

The state survives redeploys (this bot redeploys itself once an hour, before
Session A): it is mirrored into this service's own KILL_SWITCH env var through
env_persist.put_env_vars_sync (Render API), and read back at boot.
Without RENDER_API_KEY the switch still works, but only until the next
redeploy — the dashboard says so. Note: Render may restart the service when
an env var changes, so toggling the switch can itself restart the instance.

Security: the dashboard URL is public, so toggling requires a PIN. With no
extra setup the PIN is the LAST 6 CHARACTERS OF YOUR DERIV_API_TOKEN (you
type it once per browser; it is remembered on that device). Set
DASHBOARD_PIN if you would rather use your own.
"""

import hmac
import logging
import os
import threading
import time

import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_paused: bool = os.environ.get("KILL_SWITCH", "0").strip().lower() in ("1", "true", "yes", "on")
_changed_at: float = 0.0
_changed_by: str = "boot" if _paused else ""


def is_paused() -> bool:
    return _paused


def snapshot() -> dict:
    try:
        import render_api
        persists = render_api.api_configured()
    except Exception:
        persists = False
    return {
        "paused":     _paused,
        "changed_at": _changed_at,
        "changed_by": _changed_by,
        "persists":   persists,
    }


def expected_pin() -> str:
    pin = (os.environ.get("DASHBOARD_PIN") or "").strip()
    if pin:
        return pin
    tok = (getattr(config, "DERIV_API_TOKEN", "") or "").strip()
    return tok[-6:] if tok else ""


def check_pin(supplied: str) -> bool:
    want = expected_pin()
    if not want:            # no token and no PIN configured -> nothing to check against
        return True
    return hmac.compare_digest(str(supplied or "").strip().encode(), want.encode())


def set_paused(paused: bool, source: str = "dashboard") -> bool:
    global _paused, _changed_at, _changed_by
    with _lock:
        _paused = bool(paused)
        _changed_at = time.time()
        _changed_by = source
    os.environ["KILL_SWITCH"] = "1" if _paused else "0"
    logger.warning(
        f"KILL SWITCH {'ENGAGED — all new trading paused' if _paused else 'RELEASED — trading may resume'} "
        f"(via {source})")
    # Mirror into the dashboard state + persist across redeploys.
    try:
        import keep_alive
        keep_alive.update_status(kill_switch=_paused)
    except Exception:
        pass
    try:
        import env_persist
        if not env_persist.put_env_vars_sync({"KILL_SWITCH": "1" if _paused else "0"}):
            logger.error("KILL SWITCH: state NOT persisted to Render (RENDER_API_KEY/RENDER_SERVICE_ID missing or the "
                         "API refused) - it holds until the next redeploy only")
    except Exception as exc:
        logger.error(f"KILL SWITCH: could not persist state: {exc}")
    return _paused
