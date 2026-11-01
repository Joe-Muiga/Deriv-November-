"""
env_persist.py – push small values into this service's own Render env vars so
they survive a redeploy (Render's disk is ephemeral). Same mechanism the old
cycle-state persistence used: PUT /v1/services/{id}/env-vars/{key}
with RENDER_API_KEY (+ RENDER_SERVICE_ID, injected by Render).
"""
import logging
import os
from typing import Dict

import aiohttp
import requests

import render_api

logger = logging.getLogger(__name__)


def configured() -> bool:
    return render_api.api_configured()


def _url(key: str) -> str:
    return f"{render_api.API_BASE}/services/{render_api.service_id()}/env-vars/{key}"


def _headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {render_api.api_key()}",
            "Content-Type": "application/json", "Accept": "application/json"}


async def put_env_vars_async(mapping: Dict[str, str]) -> bool:
    """Returns True if every key was stored. Never raises."""
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
