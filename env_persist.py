"""
env_persist.py – push small values into this service's own Render env vars so
they survive a redeploy (Render's disk is ephemeral). Same mechanism the old
cycle-state persistence used: PUT /v1/services/{id}/env-vars/{key}
with RENDER_API_KEY (+ RENDER_SERVICE_ID, injected by Render).

User-owned variables are READ-ONLY to the bot. PROTECTED_KEYS can never be written through any helper in this module
(TARGET_PROFIT is set by hand on Render; the bot only reads it). Writes to them are refused and logged CRITICAL.

put_env_vars_async / put_env_vars_sync are best-effort. put_env_var_verified_sync is for critical transitions: it
returns True only when Render answered 200/201 AND the value read back from Render equals what was written.
"""
import asyncio
import logging
import os
from typing import Dict, Optional

import aiohttp
import requests

import render_api

logger = logging.getLogger(__name__)

# Never written by the bot, whoever asks. TARGET_PROFIT is the user's manual profit target; the rest are secrets /
# credentials / settings the user owns.
PROTECTED_KEYS = frozenset({
    "TARGET_PROFIT", "TARGET_BALANCE", "RENDER_API_KEY", "RENDER_SERVICE_ID", "RENDER_DEPLOY_HOOK_URL",
    "DERIV_API_TOKEN", "DATABASE_URL", "ADMIN_PASSWORD", "ENCRYPTION_KEY", "DASHBOARD_PIN", "PREDICTOR_STAKE",
})


def is_protected(key: str) -> bool:
    return str(key).strip().upper() in PROTECTED_KEYS


def _refuse_protected(mapping: Dict[str, str]) -> bool:
    """True (and a CRITICAL log) if the mapping contains a protected key - the whole write is then refused."""
    bad = [k for k in mapping if is_protected(k)]
    if bad:
        logger.critical("ENV PERSIST REFUSED: %s is user-owned and read-only to the bot - nothing was written",
                        ", ".join(bad))
    return bool(bad)


def configured() -> bool:
    return render_api.api_configured()


def _url(key: str) -> str:
    return f"{render_api.API_BASE}/services/{render_api.service_id()}/env-vars/{key}"


def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {render_api.api_key()}",
            "Content-Type": "application/json", "Accept": "application/json"}


async def put_env_vars_async(mapping: Dict[str, str]) -> bool:
    """Returns True if every key was stored. Never raises."""
    if _refuse_protected(mapping):
        return False
    for k, v in mapping.items():
        os.environ[k] = str(v)
    if not configured():
        return False
    ok = True
    try:
        async with aiohttp.ClientSession(headers=_headers()) as s:
            for k, v in mapping.items():
                async with s.put(_url(k), json={"value": str(v)},
                                 timeout=aiohttp.ClientTimeout(total=20)) as r:
                    if r.status in (200, 201):
                        logger.info(f"ENV PERSIST: {k} stored ({len(str(v))} bytes)")
                    else:
                        ok = False
                        logger.error(f"ENV PERSIST: {k} failed HTTP {r.status}: {(await r.text())[:200]}")
    except Exception as exc:
        logger.error(f"ENV PERSIST error: {exc}")
        return False
    return ok


def put_env_vars_sync(mapping: Dict[str, str]) -> bool:
    if _refuse_protected(mapping):
        return False
    for k, v in mapping.items():
        os.environ[k] = str(v)
    if not configured():
        return False
    ok = True
    for k, v in mapping.items():
        try:
            r = requests.put(_url(k), json={"value": str(v)}, headers=_headers(), timeout=20)
            if r.status_code not in (200, 201):
                ok = False
                logger.error(f"ENV PERSIST (sync): {k} failed HTTP {r.status_code}: {r.text[:200]}")
        except Exception as exc:
            ok = False
            logger.error(f"ENV PERSIST (sync) error: {exc}")
    return ok


def read_env_var_sync(key: str) -> Optional[str]:
    """The value Render currently stores for `key` (GET /v1/services/{id}/env-vars/{key}), or None when it cannot be
    read. Accepts both response shapes Render uses: {"key", "value"} and {"envVar": {"key", "value"}}."""
    if not configured():
        return None
    try:
        r = requests.get(_url(key), headers=_headers(), timeout=20)
        if r.status_code != 200:
            logger.error(f"ENV PERSIST: read-back of {key} failed HTTP {r.status_code}: {r.text[:200]}")
            return None
        body = r.json() or {}
        if isinstance(body, dict) and isinstance(body.get("envVar"), dict):
            body = body["envVar"]
        val = body.get("value") if isinstance(body, dict) else None
        return None if val is None else str(val)
    except Exception as exc:
        logger.error(f"ENV PERSIST: read-back of {key} error: {exc}")
        return None


def put_env_var_verified_sync(key: str, value, env_first: bool = False) -> bool:
    """CRITICAL write: True only if Render answered 200/201 AND reading the variable back returns exactly `value`.
    Blocking (call from asyncio.to_thread). Never raises. Without RENDER_API_KEY / RENDER_SERVICE_ID it returns False
    (nothing can be persisted).

    os.environ is updated in-process: after a verified success (default), or BEFORE the request with env_first=True
    (used for the clear, so this process never resumes a finished cycle even while the write is being retried)."""
    if _refuse_protected({key: value}):
        return False
    value = str(value)
    if env_first:
        os.environ[key] = value
    if not configured():
        logger.error(f"ENV PERSIST (verified): {key} NOT stored - RENDER_API_KEY / RENDER_SERVICE_ID missing")
        return False
    try:
        r = requests.put(_url(key), json={"value": value}, headers=_headers(), timeout=20)
    except Exception as exc:
        logger.error(f"ENV PERSIST (verified): {key} PUT error: {exc}")
        return False
    if r.status_code not in (200, 201):
        logger.error(f"ENV PERSIST (verified): {key} PUT failed HTTP {r.status_code}: {r.text[:200]}")
        return False
    back = read_env_var_sync(key)
    if back is None or back != value:
        logger.error(f"ENV PERSIST (verified): {key} read-back mismatch (wrote {value!r}, Render has {back!r})")
        return False
    os.environ[key] = value
    logger.info(f"ENV PERSIST (verified): {key}={value} stored and read back")
    return True


async def put_env_var_verified(key: str, value, env_first: bool = False) -> bool:
    """Awaitable form of put_env_var_verified_sync (runs in a worker thread: never blocks the event loop)."""
    return await asyncio.to_thread(put_env_var_verified_sync, key, value, env_first)
