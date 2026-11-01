"""
follower_client.py - one Deriv connection per follower account.

FollowerClient(DerivClient) overrides ONLY the two places where deriv_client reads the
token / app id / account mode from config: _fetch_otp_ws_url() and _authorize().
Everything else (connect loop, _send, balance stream, contract tracking) is inherited, and
every instance owns its own state (socket, pending futures, balance, semaphore, ...), so
nothing is shared with the collector's client.  deriv_client.py itself is not modified.

The token and the OTP URL are never logged.
"""
import json
import logging
import time

import aiohttp

import follower_config as fc
from deriv_client import DerivClient, _OPTIONS_API_BASE

logger = logging.getLogger("follower.client")


class FollowerClient(DerivClient):
    def __init__(self, label: str, token: str, app_id: str, mode: str):
        super().__init__()
        self._f_label = label
        self._f_token = token
        self._f_app_id = app_id
        self._f_mode = (mode or "demo").strip().lower()

    async def _fetch_otp_ws_url(self) -> str:
        """Same two-step REST flow as the collector (GET /accounts, POST /accounts/{id}/otp)
        but with THIS follower's token, app id and the global demo/real mode."""
        app_id, token, mode = self._f_app_id, self._f_token, self._f_mode
        if not app_id:
            raise ValueError("follower app id is empty")
        if not token:
            raise ValueError("follower token is empty")
        if mode not in ("real", "demo"):
            raise ValueError(f"mode must be 'real' or 'demo', got {mode!r}")
        headers = {"Deriv-App-ID": app_id, "Authorization": f"Bearer {token}"}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            async with session.get(f"{_OPTIONS_API_BASE}/accounts", headers=headers) as resp:
                body = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(f"accounts lookup failed: HTTP {resp.status}")
            accounts = body.get("data", body) if isinstance(body, dict) else body
            if not accounts:
                raise RuntimeError("no Options accounts returned")
            chosen = None
            for acct in accounts:
                acct_type = (acct.get("type") or acct.get("account_type")
                             or ("demo" if acct.get("is_virtual") else "real") or "")
                if str(acct_type).lower() == mode:
                    chosen = acct
                    break
            if chosen is None:
                # Never silently trade on the wrong account type for a follower.
                raise RuntimeError(f"no {mode} account found for follower {self._f_label!r}")
            account_id = chosen.get("accountId") or chosen.get("account_id") or chosen.get("id")
            if not account_id:
                raise RuntimeError("account id missing in accounts response")
            logger.info("FOLLOWER %s: using %s account %s", self._f_label, mode, account_id)
            async with session.post(f"{_OPTIONS_API_BASE}/accounts/{account_id}/otp", headers=headers) as resp:
                body = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(f"OTP request failed: HTTP {resp.status}")
            ws_url = (body.get("data") or {}).get("url") or body.get("url")
            if not ws_url:
                raise RuntimeError("OTP response had no url")
            return ws_url

    async def _authorize(self):
        """Legacy authorize message (connect() normally relies on the OTP URL instead)."""
        if not self._f_token:
            raise ValueError("follower token is empty")
        await self._ws.send(json.dumps({"authorize": self._f_token, "req_id": 1}))
        import asyncio
        msg = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=30))
        if msg.get("error"):
            raise RuntimeError(f"Auth failed: {msg['error'].get('message')}")
        acct = msg.get("authorize", {})
        self._balance = float(acct.get("balance", 0))
        self._balance_ts = time.monotonic()
        self._authorized = True
        self._req_id_counter = 10
        self._ready.set()
        for cb in self._balance_callbacks:
            try:
                cb(self._balance)
            except Exception:
                pass

    # Follower balance freshness helper (monotonic seconds since the last balance update).
    def balance_age(self) -> float:
        return time.monotonic() - self._balance_ts if self._balance_ts else 1e9

    def note_balance_after_buy(self, balance_after: float) -> None:
        """A buy can only LOWER the balance; keep the lower value (conservative) until the
        balance stream confirms."""
        if balance_after < self._balance:
            self._balance = balance_after
            self._balance_ts = time.monotonic()
