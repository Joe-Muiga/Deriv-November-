"""
render_api.py – tiny shared helper so the bot needs as few manual Render
environment variables as possible.

Render injects RENDER_SERVICE_ID and RENDER_EXTERNAL_URL into every web
service automatically, so neither ever has to be typed in. The only secret
the bot genuinely cannot get on its own is a Render API key, and with that
single key it can do BOTH things the old RENDER_DEPLOY_HOOK_URL and the old
manually-typed RENDER_SERVICE_ID were used for:

  * trigger a redeploy      -> POST /v1/services/{id}/deploys
  * persist its own state   -> PUT  /v1/services/{id}/env-vars/{key}

An explicit RENDER_DEPLOY_HOOK_URL is still honoured if someone sets one
(it takes priority), but it is no longer required.
"""

import os
from typing import Optional, Tuple, Dict

import config

API_BASE = "https://api.render.com/v1"


def api_key() -> str:
    return (os.environ.get("RENDER_API_KEY") or getattr(config, "RENDER_API_KEY", "") or "").strip()


def service_id() -> str:
    return (os.environ.get("RENDER_SERVICE_ID") or getattr(config, "RENDER_SERVICE_ID", "") or "").strip()


def api_configured() -> bool:
    return bool(api_key() and service_id())


def deploy_target() -> Optional[Tuple[str, Dict[str, str]]]:
    """(url, headers) to POST to in order to redeploy this service, or
    None if neither a deploy hook nor an API key is available."""
    hook = (os.environ.get("RENDER_DEPLOY_HOOK_URL")
            or getattr(config, "RENDER_DEPLOY_HOOK_URL", "") or "").strip()
    if hook:
        return hook, {}
    if api_configured():
        return (
            f"{API_BASE}/services/{service_id()}/deploys",
            {"Authorization": f"Bearer {api_key()}",
             "Accept": "application/json",
             "Content-Type": "application/json"},
        )
    return None
