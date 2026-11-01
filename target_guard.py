"""
target_guard.py – stop trading and shut the service down once the LIVE Deriv
account balance reaches a target you set by hand.

Set on Render (Environment tab):

    TARGET_BALANCE = 5000        # the account balance you want, in account currency
                                 # empty / 0 / not set  ->  feature OFF

What happens the moment the live balance is >= TARGET_BALANCE:

  1. a process-wide "target reached" flag is set (synchronously, inside the
     balance callback) -> no new proposal or buy can be sent after that point;
  2. open contracts are NOT waited for: they stay on Deriv's side and settle there;
  3. the Deriv WebSocket is closed and its reconnect loop is told to stay down;
  4. this Render service is SUSPENDED through the Render API (needs
     RENDER_API_KEY; RENDER_SERVICE_ID is injected by Render itself).
     Without an API key the bot still stops trading and disconnects from Deriv,
     but the Render service keeps running idle - the dashboard says so.

To trade again: raise TARGET_BALANCE above the current balance on Render, then
Resume the service. If you resume with the target still at/below the balance the
bot will (correctly) stop again within seconds of connecting.

The balance watched is the one Deriv reports for the account the API token's
session is on (DERIV_ACCOUNT_MODE demo/real) - the subscribed live stream, plus
a direct poll every TARGET_CHECK_SECONDS as a safety net.
"""

import logging
import os
import threading
import time
from typing import Optional

import requests

import render_api

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_reached: bool = False
_reached_at: float = 0.0
_reached_balance: float = 0.0
_reached_target: float = 0.0
_suspend_result: str = "not attempted"


def target() -> Optional[float]:
    """Current TARGET_BALANCE, or None when unset / 0 / invalid (feature off).
    Read from the environment every call, so it is never stale."""
    raw = (os.environ.get("TARGET_BALANCE") or "").strip().replace(",", "").lstrip("$")
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        logger.error(f"TARGET_BALANCE={raw!r} is not a number - target feature OFF")
        return None
    return val if val > 0 else None


def check_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get("TARGET_CHECK_SECONDS", "3")))
    except ValueError:
        return 3.0


def is_reached() -> bool:
    return _reached


def should_stop(balance: float) -> bool:
    """True if the balance is at/above the target (and the feature is on)."""
    t = target()
    return t is not None and balance is not None and float(balance) >= t


def mark_reached(balance: float) -> bool:
    """Set the stop flag. Returns True only for the FIRST caller (so the shutdown
    sequence runs exactly once). Safe to call from sync code in the event loop."""
    global _reached, _reached_at, _reached_balance, _reached_target
    with _lock:
        if _reached:
            return False
        _reached = True
        _reached_at = time.time()
        _reached_balance = float(balance)
        _reached_target = float(target() or 0.0)
    logger.warning(
        f"TARGET REACHED: live balance {float(balance):.2f} >= target {_reached_target:.2f} "
        f"- trading stopped, disconnecting")
    try:
        import keep_alive
        keep_alive.update_status(target_reached=True, target_balance=_reached_target,
                                 target_reached_balance=float(balance))
    except Exception:
        pass
    return True


def snapshot() -> dict:
    return {
        "target":          target(),
        "reached":         _reached,
        "reached_at":      _reached_at,
        "reached_balance": _reached_balance,
        "suspend_result":  _suspend_result,
    }


def suspend_service() -> bool:
    """Suspend THIS Render service (POST /v1/services/{id}/suspend). Blocking -
    call from a thread (asyncio.to_thread). Returns True if Render accepted it."""
    global _suspend_result
    if not render_api.api_configured():
        _suspend_result = "no RENDER_API_KEY / service id - service left running (idle)"
        logger.error(f"TARGET: cannot suspend Render service: {_suspend_result}")
        return False
    url = f"{render_api.API_BASE}/services/{render_api.service_id()}/suspend"
    headers = {"Authorization": f"Bearer {render_api.api_key()}", "Accept": "application/json"}
    try:
        r = requests.post(url, headers=headers, timeout=20)
        if r.status_code in (200, 201, 202, 204):
            _suspend_result = "suspended"
            logger.warning("TARGET: Render service SUSPEND accepted - process will be stopped by Render")
            return True
        _suspend_result = f"Render refused: HTTP {r.status_code} {r.text[:150]}"
        logger.error(f"TARGET: {_suspend_result}")
    except Exception as exc:
        _suspend_result = f"suspend request failed: {exc}"
        logger.error(f"TARGET: {_suspend_result}")
    return False
