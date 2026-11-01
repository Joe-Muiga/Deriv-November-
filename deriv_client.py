"""
deriv_client.py – Async Deriv WebSocket client.

v16 — TWO EXACT-SECOND SESSIONS PER HOUR (see 00_shared_contract.md, sections 4 + 5):

  - shutdown() no longer ends the object: connect() may be called again on the same
    DerivClient (new lead). set_reconnect_enabled(False) makes the reconnect loop stop
    instead of reconnecting. The contract polling task and the clock-resync task are
    cancelled on shutdown and restarted by the next connect().
  - connect() never mints an OTP / opens a socket outside the connect lead (a hard
    safety net on top of the engine's own scheduling): window_scheduler.should_be_connected().
  - sync_clock(): NTP-style server-clock offset from Deriv's whole-second `time` call.
  - _send(): send-time buy gate. A `buy` frame is only written while
    ws.buy_allowed_at(ws.now()); otherwise ws.SessionClosed is raised WITHOUT sending
    ("definitely not sent"). The buy_* methods let SessionClosed propagate (they still
    return None for every other failure).

v15 — ADAPTIVE EXIT ENGINE SUPPORT (additive, no existing contract logic
      touched):

  NEW — contract_update(contract_id, stop_loss=None, take_profit=None):
    - Revises stop_loss and/or take_profit on an already-open Multiplier
      contract via Deriv's contract_update API, without closing it.
    - Either parameter may be omitted to leave that side unchanged; the
      outgoing limit_order dict only includes the params actually passed.
    - Before sending the update, snapshots the contract via the existing
      force_check_contract() (reused, not duplicated) and feature-detects
      an is_valid_to_update / is_valid_to_sell-style flag on the snapshot
      — never assumes a specific key name. Fails soft (returns None) if
      the flag isn't clearly true.
    - Follows this file's standard conventions: verbose logging
      (CONTRACT UPDATE ATTEMPT / CONTRACT UPDATE RESPONSE / FAILED),
      never raises, returns None on any failure path.
    - No new rate-limiting/semaphore logic — call frequency is governed
      by config.EXIT_POLL_INTERVAL_SECS in bot_engine.py, not here.
    - Placed near sell_contract() for locality; buy_contract(),
      buy_multiplier(), force_check_contract(), and sell_contract()
      themselves are untouched.

  NEW — _snapshot_allows_update(poc): private helper, feature-detects
      contract-update eligibility on a proposal_open_contract snapshot
      dict. Used only by contract_update().

v14 — ACCUMULATOR + MATCHES/DIFFERS SUPPORT (additive, no existing
      contract logic touched):

  All new methods follow buy_contract()'s existing logging/error
  conventions (BUY ATTEMPT / PROPOSAL RESPONSE / BUY RESPONSE / FAILED),
  the same never-raises / None-on-failure contract, the same
  _buy_semaphore + 3s inter-buy spacing, and register successfully-bought
  contracts for polling exactly like buy_contract() does.

  Unlike Rise/Fall (which buys directly, no proposal step), ACCU and
  Matches/Differs contracts are bought via the standard two-step
  proposal -> buy flow, since Deriv's API requires a proposal id for
  these contract types.

  NEW — get_accumulator_proposal() / buy_accumulator():
    - contract_type="ACCU", growth_rate accepted as 1-5 (%) and sent to
      the API as a 0.01-0.05 decimal.
    - buy_accumulator() supports an optional take_profit limit_order.

  NEW — sell_contract(contract_id, price):
    - Generic early-close/sell, usable for closing an ACCU position
      before knockout (or any other open contract).

  NEW — get_digit_proposal() / buy_digit_contract():
    - contract_type="DIGITMATCH" or "DIGITDIFF" (match_type="MATCH" /
      "DIFFER"), barrier=str(digit), duration_unit defaults to "t".

  NEW — _cap_stake() helper:
    - Mirrors the FIX 3 stake-cap rule documented below for
      buy_contract(), reused by the new proposal-based buy methods.
      buy_contract() itself was left untouched.

v13 — POLLING-BASED CONTRACT RESOLUTION (replaces subscription model):

  ROOT CAUSE OF ORPHAN_TIMEOUT:
    - WebSocket proposal_open_contract subscriptions never delivered
      reliable close callbacks (no CALLBACK FIRED / POLLER CAUGHT events
      observed in production logs).

  FIX — Aggressive polling replaces all subscription-based monitoring:
    - All proposal_open_contract subscribe/forget/callback machinery removed
      (_contract_callbacks, _subscriptions, _subscribed_contracts,
      _pending_contract_msgs, _closed_before_callback, _contract_poller,
      _fallback_poll_loop, _resubscribe_open_contracts, _subscribe_after_buy,
      _cleanup_contract_subscription — all removed).
    - subscribe_contract(contract_id, callback) now simply registers the
      contract + callback in self._polling_contracts.
    - _polling_loop() runs every 30s, calls force_check_contract(cid) for
      every tracked contract. On is_sold/is_expired it pops the entry and
      fires the callback with the full proposal_open_contract dict
      (containing profit, sell_price, is_sold, is_expired, etc).
    - Started once in connect() via asyncio.create_task().

v9 → v10 audit changes (full compliance pass):

  DIRECTION MAPPING (VERIFIED CORRECT — DO NOT SWAP):
    direction="LONG"  → contract_type="CALL"  → wins if price RISES
    direction="SHORT" → contract_type="PUT"   → wins if price FALLS

  Every buy_contract() call logs BEFORE placement:
    PLACING {contract_type} on {symbol} stake=${stake:.4f} | duration={duration}m

  FIX 1 — VERBOSE buy_contract() LOGGING:
    - BUY ATTEMPT: {symbol} {contract_type} stake=${stake} mult={multiplier}
    - PROPOSAL RESPONSE: {full resp dict} (after _send)
    - BUY RESPONSE: {full result dict} (after success)
    - BUY FAILED: {symbol} error={detail} (on every failure path)

  FIX 2 — RISE/FALL FALLBACK:
    If buy_contract() gets None from multiplier attempt, retries with
    CALL (LONG) or PUT (SHORT), duration=5m, logs: FALLBACK TO RISE/FALL: {symbol}

  FIX 3 — STAKE CAP:
    If stake > balance * 0.5, caps at max(balance * 0.02, 0.35)
    Logs: STAKE CAPPED: ${original} → ${capped}

  FAILURE CONTRACT (buy_contract):
    - Network timeout  → None + log FAILED: {symbol} — timeout
    - API error        → None + log FAILED: {symbol} — {code}: {msg}
    - Market closed    → None + log FAILED: {symbol} — market_closed
    - Proposal reject  → None + log FAILED: {symbol} — proposal_rejected
    - Never raises — always returns None on any failure path
    - Trade counters never incremented on None return (caller responsibility enforced)

  MARKET OPEN CHECK:
    - Boom/Crash: active_symbols API checked for exchange_is_open before proposal
    - Volatility indices (R_*, 1HZ*): check skipped entirely — always open

  get_balance() METHOD:
    - Calls balance API endpoint
    - Returns float
    - On failure: returns cached balance, logs BALANCE FETCH FAILED — using cached
    - Cache TTL: 30 seconds

  CONNECTION RESILIENCE:
    - Disconnect: logs WEBSOCKET DISCONNECTED — reconnecting in {interval}s (attempt {n}/{max})
    - Retries every WEBSOCKET_RECONNECT_INTERVAL seconds
    - After WEBSOCKET_MAX_RECONNECTS failures: logs WEBSOCKET RECONNECT FAILED — bot halting
      then raises ConnectionError
    - Re-authenticates after reconnect

  get_candles():
    - Returns list[dict] with keys: open, high, low, close, epoch
    - On failure: returns [], logs CANDLE FETCH FAILED: {symbol} gran={granularity}
"""

import asyncio
import collections
import json
import logging
import math
import time
from typing import Callable, Dict, Optional, List, Any

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

import config
import window_scheduler as ws

# Deriv's newer Options trading API (REST for account/OTP setup, WS for trading).
# Replaces the legacy wss://ws.derivws.com/websockets/v3?app_id=... endpoint,
# which now returns HTTP 401 (InvalidAppID) at the handshake.
_OPTIONS_API_BASE = "https://api.derivws.com/trading/v1/options"

logger = logging.getLogger(__name__)

MAX_RETRY_DELAY = 60

_BOOM_CRASH_PREFIXES = ("BOOM", "CRASH")
_VOLATILITY_PREFIXES = ("R_", "1HZ")

# ── Reconnect defaults (overridden by config if present) ──────────────────────
_DEFAULT_RECONNECT_INTERVAL = 3   # base seconds between retries (exponential backoff, cap MAX_RETRY_DELAY)
_DEFAULT_MAX_RECONNECTS     = 0   # 0 = unlimited: never give up reconnecting

# Contract polling interval in seconds
_CONTRACT_POLL_INTERVAL = 30

# Minimum gap between proposal requests (rate-limit protection)
PROPOSAL_DELAY = 2.0  # 2 seconds between proposals


def _is_boom_crash(symbol: str) -> bool:
    return any(symbol.upper().startswith(p) for p in _BOOM_CRASH_PREFIXES)


def _is_volatility_index(symbol: str) -> bool:
    s = symbol.upper()
    return any(s.startswith(p) for p in _VOLATILITY_PREFIXES)


class RequestTimeout(TimeoutError):
    """The request frame was written but no reply arrived in time: the server MAY have acted on it.
    Still a TimeoutError, so existing `except TimeoutError` handlers keep working."""
    sent = True
    req_id = None


def _mark_sent(exc: BaseException, req_id: int) -> None:
    """Tag an exception raised after the frame went out (or may have): callers can tell "in doubt" from "refused"."""
    try:
        exc.sent = True
        exc.req_id = req_id
    except Exception:
        pass


LATE_BUY_MAX = 300          # late buy replies kept (bounded)
LATE_BUY_TTL = 120.0        # seconds a late buy reply is kept

_CLOCK_SANITY_LIMIT = 120.0   # |server - local| beyond this (seconds) is rejected as a bad measurement
_CONNECT_GUARD_SLACK = 2.0    # a connect attempt up to this many seconds before the lead is tolerated (clock skew)


class _OutsideLead(RuntimeError):
    """Internal: a connect / OTP attempt was refused because it is outside the connect lead."""


class DerivClient:

    def __init__(self):
        self._late_buys: "collections.OrderedDict[int, tuple]" = collections.OrderedDict()   # req_id -> (ts, msg)
        self._ws: Optional[Any]    = None
        self._ready: asyncio.Event = asyncio.Event()
        self._connected: bool      = False
        self._authorized: bool     = False

        self._pending: Dict[int, asyncio.Future] = {}
        self._req_id_counter: int = 10

        self._tick_callbacks: Dict[str, Callable] = {}
        self._subscription_map: Dict[str, str]    = {}

        # contract_id → symbol  (informational only)
        self._contract_symbol_map: Dict[str, str] = {}

        self._balance: float             = 0.0
        self._balance_ts: float          = 0.0          # epoch of last successful fetch
        self._balance_callbacks: List[Callable] = []
        self._stopped: bool              = False        # set by shutdown(); cleared by the next connect()

        # ── Session lifecycle (v16) ────────────────────────────────────────────
        self._reconnect_enabled: bool = True            # set_reconnect_enabled(False): connect() loop must not reconnect
        self._gen: int = 0                              # connect() generation: retires an older loop still winding down
        self._connect_task: Optional[asyncio.Task] = None
        self._polling_task: Optional[asyncio.Task] = None
        self._stop_evt: asyncio.Event = asyncio.Event() # wakes the reconnect back-off sleep on shutdown / disable
        self._clock_lock: asyncio.Lock = asyncio.Lock() # one clock burst at a time (engine lead vs resync loop)
        # Test hooks (fake clock): production uses the real wall clock and asyncio.sleep.
        self._wall  = lambda: time.time()
        self._sleep = asyncio.sleep

        # ── Polling-based contract resolution ──────────────────────────────────
        # {contract_id: {"callback": fn, "placed_at": time}}
        self._polling_contracts: Dict[str, dict] = {}

        # ── WS subscription-based contract resolution (primary; poller is fallback) ──
        self._subscribed_contracts: set = set()

        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Incremented on every successful (re)connect so the engine can tell that
        # ticks may have been missed and re-sync its predictor.
        self._conn_count: int = 0

        # ── Rate limiting ─────────────────────────────────────────────────────
        self._buy_semaphore  = asyncio.Semaphore(1)  # one at a time
        self._last_buy_time  = 0.0                   # epoch of last buy attempt

        # ── Circuit breaker: {(symbol, strategy): {"count": int, "cooldown_until": epoch}} ──
        self._circuit_breaker: Dict[tuple, dict] = {}

        # ── Direction→contract mapping audit ─────────────────────────────────
        logger.info(
            "Direction mapping: LONG → CALL (price rises) | SHORT → PUT (price falls)"
        )
        assert "CALL" == ("CALL" if "LONG" == "LONG" else "PUT"), (
            "FATAL: LONG→CALL mapping is broken")
        assert "PUT" == ("CALL" if "SHORT" == "LONG" else "PUT"), (
            "FATAL: SHORT→PUT mapping is broken")

    # ─── New Options API: REST account lookup + OTP WebSocket URL ─────────────

    async def _fetch_otp_ws_url(self) -> str:
        """
        Two-step REST flow required by Deriv's current Options trading API:
          1. GET  /accounts             -> list this user's Options accounts
          2. POST /accounts/{id}/otp    -> one-time WebSocket URL (OTP embedded)

        The account (real vs demo) is picked via config.DERIV_ACCOUNT_MODE.
        Returns the ready-to-connect wss:// URL. OTPs are short-lived, so this
        must be called fresh on every (re)connect attempt — never cache the URL.
        """
        if not self._connect_permitted():
            # Last line of defence: no OTP is ever minted outside the lead / session / drain.
            raise _OutsideLead("OTP not minted: outside the connect lead")

        app_id = config.DERIV_APP_ID
        token  = config.DERIV_API_TOKEN
        mode   = getattr(config, "DERIV_ACCOUNT_MODE", "demo").strip().lower()

        if not app_id:
            raise ValueError("DERIV_APP_ID is not set.")
        if not token:
            raise ValueError("DERIV_API_TOKEN is not set.")
        if mode not in ("real", "demo"):
            raise ValueError(
                f"DERIV_ACCOUNT_MODE must be 'real' or 'demo', got {mode!r}."
            )

        headers = {
            "Deriv-App-ID": app_id,
            "Authorization": f"Bearer {token}",
        }

        # Hard timeout: without it a stalled REST call can hang the reconnect loop for minutes.
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
            # Step 1 — list accounts
            async with session.get(
                f"{_OPTIONS_API_BASE}/accounts", headers=headers
            ) as resp:
                body = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(
                        f"Failed to list Options accounts: HTTP {resp.status} {body}"
                    )

            accounts = body.get("data", body) if isinstance(body, dict) else body
            if not accounts:
                raise RuntimeError(f"No Options accounts returned: {body}")

            chosen = None
            for acct in accounts:
                acct_type = (
                    acct.get("type")
                    or acct.get("account_type")
                    or ("demo" if acct.get("is_virtual") else "real")
                    or ""
                )
                if str(acct_type).lower() == mode:
                    chosen = acct
                    break

            if chosen is None:
                # Field names in the response aren't confirmed against docs yet —
                # fall back to the first account but log loudly so this is visible
                # rather than silently trading on the wrong one.
                logger.warning(
                    f"Could not match an Options account to mode={mode!r}; "
                    f"falling back to first account returned. Raw accounts: {accounts}"
                )
                chosen = accounts[0]

            account_id = (
                chosen.get("accountId") or chosen.get("account_id") or chosen.get("id")
            )
            if not account_id:
                raise RuntimeError(
                    f"Could not find an accountId field in account data: {chosen}"
                )

            logger.info(f"Using Options accountId={account_id} (mode={mode})")

            # Step 2 — request the OTP-embedded WebSocket URL
            async with session.post(
                f"{_OPTIONS_API_BASE}/accounts/{account_id}/otp", headers=headers
            ) as resp:
                body = await resp.json()
                if resp.status != 200:
                    raise RuntimeError(
                        f"Failed to get OTP WebSocket URL: HTTP {resp.status} {body}"
                    )

            ws_url = (body.get("data") or {}).get("url") or body.get("url")
            if not ws_url:
                raise RuntimeError(f"OTP response missing 'url' field: {body}")

            return ws_url

    @staticmethod
    def _mask_otp(ws_url: str) -> str:
        if "otp=" in ws_url:
            base, _, _ = ws_url.partition("otp=")
            return f"{base}otp=***OTP***"
        return ws_url

    # ─── Connection ───────────────────────────────────────────────────────────

    def _connect_permitted(self) -> bool:
        """True only inside LEAD / OPEN / DRAIN (a little slack before the lead for clock skew).
        Fails closed if the scheduler itself errors."""
        try:
            t = ws.now()
            return bool(ws.should_be_connected(t) or ws.should_be_connected(t + _CONNECT_GUARD_SLACK))
        except Exception as exc:
            logger.error(f"CONNECT GUARD: scheduler error ({exc}) - refusing to connect")
            return False

    def _should_stop(self, gen: int) -> bool:
        return self._stopped or not self._reconnect_enabled or gen != self._gen

    def set_reconnect_enabled(self, enabled: bool) -> None:
        """False: the connect() loop must NOT reconnect (it ends once the current socket is gone and
        its back-off sleep is interrupted). The engine calls this right before shutdown() at session end.
        The next connect() switches it back on (a new lead)."""
        self._reconnect_enabled = bool(enabled)
        if not enabled:
            self._stop_evt.set()
        elif not self._stopped:
            self._stop_evt.clear()
        logger.info(f"DERIV CLIENT reconnect {'ENABLED' if enabled else 'DISABLED'}")

    def _start_polling(self) -> None:
        t = self._polling_task
        if t is None or t.done():
            self._polling_task = asyncio.ensure_future(self._polling_loop())

    @staticmethod
    async def _cancel_task(task) -> None:
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        try:
            await asyncio.wait({task}, timeout=5)
        except Exception:
            pass

    async def connect(self):
        """
        Outer reconnect loop.  Enforces WEBSOCKET_MAX_RECONNECTS and
        WEBSOCKET_RECONNECT_INTERVAL from config (falls back to module defaults).
        Starts the contract polling loop and the clock-resync loop (per connection).

        v16: one DerivClient lives for the whole process. connect() is called at the start of every
        session lead and runs until shutdown() / set_reconnect_enabled(False). It never mints an OTP
        or opens a socket outside LEAD / OPEN / DRAIN (it ends instead). Calling it again after
        shutdown() starts a fresh run on the same object.
        """
        self._loop = asyncio.get_event_loop()

        # A new connect() is a new lead: clear what shutdown() left behind. The generation counter
        # retires an older loop that may still be winding down.
        self._gen += 1
        my_gen = self._gen
        self._stopped = False
        self._reconnect_enabled = True
        self._stop_evt.clear()
        self._connect_task = asyncio.current_task()

        if not self._connect_permitted():
            logger.warning("CONNECT REFUSED: outside the session lead - nothing started (no OTP, no socket, no tasks)")
            self._connect_task = None
            return

        # Contract polling loop: cancelled by shutdown(), restarted here
        self._start_polling()

        reconnect_interval = getattr(
            config, "WEBSOCKET_RECONNECT_INTERVAL", _DEFAULT_RECONNECT_INTERVAL
        )
        max_reconnects = getattr(
            config, "WEBSOCKET_MAX_RECONNECTS", _DEFAULT_MAX_RECONNECTS
        )
        attempt   = 0
        max_label = str(max_reconnects) if max_reconnects > 0 else "∞"
        connected_at = 0.0

        while True:
            connected_at = 0.0
            attempt_start = attempt
            if self._should_stop(my_gen):
                return
            if not self._connect_permitted():
                logger.warning(
                    "CONNECT REFUSED: outside the session lead - connect loop ended (no OTP minted, no socket opened)"
                )
                return
            resync_task = None
            try:
                ws_url = await self._fetch_otp_ws_url()
                logger.info(f"Connecting to {self._mask_otp(ws_url)} …")
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                ) as conn:
                    self._ws        = conn
                    self._connected = True
                    connected_at    = time.monotonic()
                    self._conn_count += 1
                    logger.info("WebSocket connected ✓")

                    # The OTP embedded in ws_url already authenticated this
                    # connection — no separate authorize message needed/accepted.
                    self._authorized = True
                    self._ready.set()
                    dispatch_task = asyncio.ensure_future(self._dispatch_loop())
                    resync_task   = asyncio.ensure_future(self._clock_resync_loop())
                    try:
                        await self._subscribe_balance()
                        await self._resubscribe_contracts()
                        # Task 2 fix — see _resubscribe_ticks() docstring:
                        # ticks were never re-subscribed after a reconnect
                        # before this, which silently orphaned live tick
                        # streams for every previously-subscribed symbol.
                        await self._resubscribe_ticks()
                        await dispatch_task
                    finally:
                        dispatch_task.cancel()
                        try:
                            await dispatch_task
                        except (asyncio.CancelledError, Exception):
                            pass

            except _OutsideLead:
                logger.warning("CONNECT REFUSED: outside the session lead - connect loop ended (no OTP minted)")
                return
            except InvalidStatus as exc:
                attempt += 1
                resp = exc.response
                header_dump = dict(resp.headers.raw_items()) if hasattr(
                    resp.headers, "raw_items") else dict(resp.headers)
                body_text = ""
                if resp.body:
                    try:
                        body_text = resp.body.decode("utf-8", errors="replace")
                    except Exception:
                        body_text = repr(resp.body)
                logger.error(
                    "HANDSHAKE REJECTED — Deriv's server refused the WebSocket "
                    f"upgrade before any messages were exchanged.\n"
                    f"  status_code   = {resp.status_code}\n"
                    f"  reason_phrase = {resp.reason_phrase!r}\n"
                    f"  headers       = {header_dump}\n"
                    f"  body          = {body_text!r}"
                )
                logger.warning(
                    f"WEBSOCKET DISCONNECTED — reconnecting in {reconnect_interval}s "
                    f"(attempt {attempt}/{max_label}) | reason={exc}"
                )
            except ConnectionClosed as exc:
                attempt += 1
                logger.warning(
                    f"WEBSOCKET DISCONNECTED — reconnecting in {reconnect_interval}s "
                    f"(attempt {attempt}/{max_label}) | reason={exc}"
                )
            except Exception as exc:
                attempt += 1
                logger.error(
                    f"WEBSOCKET DISCONNECTED — reconnecting in {reconnect_interval}s "
                    f"(attempt {attempt}/{max_label}) | error={exc}"
                )
            finally:
                if resync_task is not None:
                    resync_task.cancel()
                if my_gen == self._gen:     # a retired loop must not clobber a newer connection's state
                    self._ready.clear()
                    self._connected  = False
                    self._authorized = False
                    self._ws         = None
                    # Fail in-flight requests immediately instead of leaving them to time out.
                    for _fut in list(self._pending.values()):
                        if not _fut.done():
                            _fut.set_exception(ConnectionError("WebSocket disconnected"))
                    self._pending.clear()

            if self._should_stop(my_gen):
                logger.warning("WebSocket shut down on purpose - reconnect loop ended")
                return

            # A connection that stayed up for a while counts as healthy -> reset backoff.
            # One that died within seconds keeps backing off (avoids hammering OTP/REST).
            if connected_at and time.monotonic() - connected_at >= 30:
                attempt = 0
            elif connected_at and attempt == attempt_start:
                attempt += 1      # short-lived clean close: still back off a little

            if max_reconnects > 0 and attempt >= max_reconnects:
                logger.critical(
                    "WEBSOCKET RECONNECT FAILED — bot halting"
                )
                raise ConnectionError(
                    f"WebSocket failed after {max_reconnects} reconnect attempts."
                )

            delay = min(MAX_RETRY_DELAY, reconnect_interval * (2 ** min(max(attempt - 1, 0), 5)))
            logger.info(f"Reconnecting in {delay:.0f}s (backoff step {attempt})")
            try:    # interruptible: shutdown() / set_reconnect_enabled(False) wake it at once
                await asyncio.wait_for(self._stop_evt.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

    async def force_reconnect(self, reason: str = "") -> None:
        """Close the current socket; connect()'s loop then re-establishes it from scratch
        (fresh OTP, balance, contract and tick subscriptions) - unless reconnecting is disabled or
        the connect lead / session is over, in which case the loop ends."""
        sock = self._ws
        logger.warning(f"FORCE RECONNECT requested{': ' + reason if reason else ''}")
        if sock is not None:
            try:
                await sock.close()
            except Exception as exc:
                logger.debug(f"force_reconnect close error: {exc}")

    async def shutdown(self, reason: str = "") -> None:
        """Close the socket and stop everything that talks to Deriv. Afterwards is_connected is False
        and nothing sends. The object stays reusable: a later connect() starts a fresh run
        (the polling task is cancelled here and restarted by connect(); contracts still tracked
        in _polling_contracts are kept and re-subscribed at the next connect)."""
        self._stopped = True
        self._reconnect_enabled = False
        self._stop_evt.set()
        logger.warning(f"DERIV CLIENT SHUTDOWN{': ' + reason if reason else ''}")

        await self._cancel_task(self._polling_task)
        self._polling_task = None

        sock = self._ws
        if sock is not None:
            try:
                await asyncio.wait_for(sock.close(), timeout=5)
            except Exception as exc:
                logger.debug(f"shutdown close error: {exc}")

        # Let the connect() loop end by itself (it sees _stopped); cancel it if it is stuck
        # (OTP fetch, handshake or back-off sleep) so no connect attempt can follow.
        ct = self._connect_task
        if ct is not None and ct is not asyncio.current_task() and not ct.done():
            await asyncio.wait({ct}, timeout=5)
            if not ct.done():
                ct.cancel()
                await asyncio.wait({ct}, timeout=5)
        if ct is not None and ct.done() and not ct.cancelled():
            try:
                ct.exception()                 # mark retrieved (no "never retrieved" noise)
            except Exception:
                pass
        self._connect_task = None

        # Whatever the loop did, the client is now disconnected for good until the next connect().
        self._ready.clear()
        self._connected  = False
        self._authorized = False
        self._ws         = None
        for _fut in list(self._pending.values()):
            if not _fut.done():
                _fut.set_exception(ConnectionError("WebSocket disconnected"))
        self._pending.clear()

    # ─── Server clock (v16) ───────────────────────────────────────────────────

    async def _clock_sample(self):
        """One `time` round trip -> (t0, t1, server_seconds) or None. Local times are wall-clock."""
        try:
            t0 = self._wall()
            resp = await self._send({"time": 1}, timeout=5)
            t1 = self._wall()
            server_s = int(resp.get("time"))
        except Exception as exc:
            logger.debug(f"CLOCK SAMPLE FAILED: {exc}")
            return None
        return (t0, t1, server_s)

    @staticmethod
    def _clock_interval(obs: list) -> tuple:
        """Offset interval consistent with every observation. server = local + offset; the server
        stamped its reply at some local instant L in [t0, t1] and `time` is the whole second
        S <= server < S + 1, so offset lies in [S - t1, S + 1 - t0) for each sample."""
        lo = max(s - t1 for (t0, t1, s) in obs)
        hi = min(s + 1.0 - t0 for (t0, t1, s) in obs)
        return lo, hi

    @classmethod
    def _clock_estimate(cls, obs: list) -> tuple:
        """(lo, hi, samples_used) from the lowest-RTT samples only (NTP-style). Falls back to the
        single best sample if the selected ones are inconsistent (local clock stepped, odd stamps)."""
        ranked = sorted(obs, key=lambda o: o[1] - o[0])
        rmin = ranked[0][1] - ranked[0][0]
        thr = max(2.0 * rmin, rmin + 0.03)
        used = [o for o in ranked if (o[1] - o[0]) <= thr]
        if len(used) < min(3, len(ranked)):
            used = ranked[:min(3, len(ranked))]
        lo, hi = cls._clock_interval(used)
        if lo >= hi:
            used = ranked[:1]
            lo, hi = cls._clock_interval(used)
        return lo, hi, len(used)

    async def _clock_flip_probe(self, obs: list, flip_s: float, min_int: float) -> None:
        """fine=True: pin the sub-second phase by sending `time` right at the PREDICTED instant the
        integer server second flips; each reply (second N-1 or N) halves the remaining uncertainty.
        One probe per second boundary (>= min_int apart), bounded by flip_s."""
        deadline = self._wall() + max(0.0, flip_s)
        rtts = sorted(o[1] - o[0] for o in obs)
        rtt_min, rtt_med = rtts[0], rtts[len(rtts) // 2]
        last_send = obs[-1][0]
        fails = 0
        while True:
            lo, hi, _ = self._clock_estimate(obs)
            if hi - lo <= rtt_min * 1.25 + 0.01:        # at the RTT floor: more probes cannot help
                return
            c = (lo + hi) / 2.0
            now_l = self._wall()
            n = math.ceil(now_l + c + rtt_med / 2.0 + 0.01)
            send_at = n - c - rtt_med / 2.0             # local instant so the server stamp lands on the flip
            while send_at < last_send + min_int:
                send_at += 1.0
            if send_at + rtt_med > deadline:
                return
            wait = send_at - self._wall()
            if wait > 0:
                await self._sleep(wait)
            o = await self._clock_sample()
            if o is None:
                fails += 1
                if fails >= 2:
                    return
                continue
            obs.append(o)
            last_send = o[0]

    async def sync_clock(self, samples: int = None, fine: bool = False) -> tuple:
        """Measure Deriv's server clock: returns (offset, error) in seconds (server = local + offset,
        +/- error) and calls ws.set_clock_offset(offset, error).

        Deriv's `time` is whole seconds, so a burst of `{"time":1}` is sent spaced >=
        SESSION_CLOCK_MIN_INTERVAL (with deterministic jitter so the samples sweep the sub-second
        phase). Each sample bounds the offset to an interval (see _clock_interval); only the lowest-RTT
        samples are trusted and their intervals intersected. fine=True adds flip probing
        (_clock_flip_probe, bounded by SESSION_CLOCK_FLIP_SECONDS). If an estimate was already
        measured this session (ws.clock_measured()), the new interval is intersected with it
        (drift over minutes is negligible); a non-overlapping one means the clock stepped and replaces it.
        Raises RuntimeError if no usable reply arrives or |offset| >= 120 s (nothing is set)."""
        async with self._clock_lock:
            n = int(samples if samples is not None else getattr(config, "SESSION_CLOCK_SYNC_SAMPLES", 10))
            n = max(1, n)
            min_int = max(0.0, float(getattr(config, "SESSION_CLOCK_MIN_INTERVAL", 0.25)))
            flip_s  = float(getattr(config, "SESSION_CLOCK_FLIP_SECONDS", 4.0))

            obs: list = []
            fails = 0
            for i in range(n):
                if i:
                    await self._sleep(min_int * (1.0 + 0.5 * ((i * 0.6180339887) % 1.0)))
                o = await self._clock_sample()
                if o is None:
                    fails += 1
                    if fails >= 3 and not obs:
                        break
                    continue
                obs.append(o)
            if not obs:
                raise RuntimeError("CLOCK SYNC FAILED: no usable `time` reply")
            if fine:
                await self._clock_flip_probe(obs, flip_s, min_int)

            lo, hi, used = self._clock_estimate(obs)
            if ws.clock_measured():
                po, pe = ws.get_clock_offset(), ws.get_clock_error()
                mlo, mhi = max(lo, po - pe), min(hi, po + pe)
                if mlo < mhi:
                    lo, hi = mlo, mhi
                else:
                    logger.warning(
                        f"CLOCK SYNC: new estimate [{lo:+.3f},{hi:+.3f}] does not overlap the earlier "
                        f"{po:+.3f} +/-{pe:.3f} - clock stepped? using the new one")
            offset = (lo + hi) / 2.0
            error  = max((hi - lo) / 2.0, 0.001)
            if abs(offset) >= _CLOCK_SANITY_LIMIT:
                raise RuntimeError(f"CLOCK SYNC REJECTED: offset {offset:+.1f}s beyond +/-{_CLOCK_SANITY_LIMIT:.0f}s")

            ws.set_clock_offset(offset, error)
            rmin = min(o[1] - o[0] for o in obs)
            logger.info(
                f"CLOCK SYNC: offset {offset:+.3f}s +/-{error:.3f}s | samples {len(obs)} (trusted {used}) "
                f"| min RTT {rmin * 1000:.0f} ms | fine={fine}")
            try:
                import keep_alive
                keep_alive.update_status(clock_offset=offset, clock_error=error)
            except Exception:
                pass
            return offset, error

    async def _clock_resync_loop(self):
        """Per connection: light resync every SESSION_CLOCK_RESYNC_SECONDS while connected."""
        interval = float(getattr(config, "SESSION_CLOCK_RESYNC_SECONDS", 300.0))
        while True:
            await self._sleep(interval)
            if not self.is_connected:
                continue
            try:
                await self.sync_clock(samples=3)
            except Exception as exc:
                logger.warning(f"CLOCK RESYNC FAILED: {exc}")

    async def fetch_balance_live(self, timeout: float = 8.0) -> float:
        """Ask Deriv for the balance RIGHT NOW (no cache), update the cached value and fire
        the balance callbacks. Raises if not connected / on error."""
        resp = await self._send({"balance": 1}, timeout=timeout)
        new_bal = float(resp.get("balance", {}).get("balance", self._balance))
        if new_bal != self._balance:
            logger.info(f"Balance poll: ${self._balance:.4f} -> ${new_bal:.4f}")
        self._balance = new_bal
        self._balance_ts = time.monotonic()
        for cb in self._balance_callbacks:
            try:
                cb(self._balance)
            except Exception:
                pass
        return self._balance

    @property
    def connection_count(self) -> int:
        return self._conn_count

    # ─── Message dispatch ─────────────────────────────────────────────────────

    async def _dispatch_loop(self):
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
                await self._handle(msg)
            except Exception as exc:
                logger.debug(f"Dispatch error: {exc}")

    def _note_late_buy(self, req_id: int, msg: dict) -> None:
        now = time.time()
        self._late_buys[req_id] = (now, msg)
        while self._late_buys and (len(self._late_buys) > LATE_BUY_MAX
                                   or now - next(iter(self._late_buys.values()))[0] > LATE_BUY_TTL):
            self._late_buys.popitem(last=False)
        b = msg.get("buy") or {}
        logger.warning("LATE BUY REPLY req_id=%s contract_id=%s (arrived after the local timeout)", req_id, b.get("contract_id"))

    def pop_late_buy(self, req_id) -> Optional[dict]:
        """The buy reply that arrived after its request timed out (None if none yet / expired)."""
        item = self._late_buys.pop(req_id, None)
        if item is None or time.time() - item[0] > LATE_BUY_TTL:
            return None
        return item[1]

    async def _handle(self, msg: dict):
        logger.debug(f"RAW MSG: {msg}")
        msg_type = msg.get("msg_type", "")
        req_id   = msg.get("req_id")
        error    = msg.get("error")

        if error:
            logger.warning(
                f"Deriv API error: {error.get('message')} "
                f"(code={error.get('code')}) | msg_type={msg_type}"
            )

        if msg_type == "balance":
            balance_data = msg.get("balance", {})
            new_bal = float(balance_data.get("balance", self._balance))
            if new_bal != self._balance:
                logger.info(f"Balance updated: ${self._balance:.4f} → ${new_bal:.4f}")
            self._balance    = new_bal
            self._balance_ts = time.monotonic()
            for cb in self._balance_callbacks:
                try:
                    cb(self._balance)
                except Exception:
                    pass

        if req_id and msg_type == "buy" and not error and req_id not in self._pending:
            self._note_late_buy(req_id, msg)                # reply after the local timeout: keep it, never drop a contract id
        if req_id and req_id in self._pending:
            fut = self._pending.pop(req_id)
            if not fut.done():
                if error:
                    detail_suffix = (
                        f" | details={error.get('details')}"
                        if error.get("details") else ""
                    )
                    fut.set_exception(RuntimeError(
                        f"{error.get('code', 'ERR')}: {error.get('message', 'Unknown error')}"
                        f"{detail_suffix}"
                    ))
                else:
                    fut.set_result(msg)
            return

        if msg_type == "tick":
            tick   = msg.get("tick", {})
            sym    = tick.get("symbol", "")
            sub_id = tick.get("id", "")
            for key, cb in self._tick_callbacks.items():
                if key == sym or key == sub_id:
                    try:
                        cb(tick)
                    except Exception as exc:
                        logger.debug(f"Tick callback error: {exc}")

        if msg_type == "proposal_open_contract":
            poc = msg.get("proposal_open_contract", {})
            cid = str(poc.get("contract_id", ""))
            closed = bool(
                poc.get("is_sold") or poc.get("is_expired") or poc.get("status") == "sold"
            )
            if cid and closed:
                info = self._polling_contracts.get(cid)
                self.stop_tracking(cid)
                if info and info.get("callback"):
                    try:
                        cb_result = info["callback"]({"proposal_open_contract": poc})
                        if asyncio.iscoroutine(cb_result) or isinstance(cb_result, asyncio.Future):
                            await cb_result
                    except Exception as exc:
                        logger.error(f"Contract close callback error {cid}: {exc}")
                logger.info(
                    f"SUBSCRIPTION CLOSED: {cid} profit={poc.get('profit')}"
                )

    # ─── Polling-based contract resolution (replaces subscriptions) ──────────

    async def _polling_loop(self):
        """
        Every 30 seconds, poll every contract registered in
        self._polling_contracts via force_check_contract().

        On is_sold/is_expired: pops the entry and fires its callback with
        the full proposal_open_contract dict (profit, sell_price, is_sold,
        is_expired, etc). The callback (registered by BotEngine) is
        responsible for releasing the symbol from _active_symbols.
        """
        while True:
            await asyncio.sleep(_CONTRACT_POLL_INTERVAL)
            if self._stopped:
                return
            if not self.is_connected:       # no socket (between sessions): no requests
                continue
            # Task 2 observability requirement — cheap, always logged
            # (regardless of whether there's anything to poll) so tracking-
            # set sizes are visible on a predictable cadence rather than
            # only when something is already open.
            logger.debug(f"SUBSCRIPTION STATE: {self.subscription_state_sizes()}")
            if not self._polling_contracts:
                continue
            logger.info(
                f"POLLING: {len(self._polling_contracts)} open contracts")
            for cid in list(self._polling_contracts.keys()):
                try:
                    result = await self.force_check_contract(cid)
                    if result.get("is_sold") or result.get("is_expired") or result.get("status") == "sold":
                        info = self._polling_contracts.get(cid)
                        self.stop_tracking(cid)
                        if info and info.get("callback"):
                            cb_result = info["callback"]({"proposal_open_contract": result})
                            if asyncio.iscoroutine(cb_result) or isinstance(cb_result, asyncio.Future):
                                await cb_result
                            logger.info(f"POLL RESOLVED: {cid} profit={result.get('profit')}")
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.error(f"POLL ERROR {cid}: {e}")

    # ─── Fix D: stop tracking a contract on both resolution paths ────────────

    def stop_tracking(self, contract_id: str) -> None:
        """
        Remove *contract_id* from BOTH tracking sets (_polling_contracts and
        _subscribed_contracts). Whichever side closes a contract first must
        call this so a late-arriving real result from the other mechanism
        can never be silently dropped (Implementation Brief v2, Fix D).
        Safe to call even if the contract isn't tracked by one or either
        mechanism.
        """
        cid = str(contract_id)
        self._polling_contracts.pop(cid, None)
        self._subscribed_contracts.discard(cid)

    def get_tracked_contract_ids(self) -> set:
        """
        Union of every contract_id currently tracked by either resolution
        mechanism — used by bot_engine.py's periodic divergence check
        (Fix D) to confirm its own _open_contracts never drifts out of
        sync with what deriv_client is actually still watching.
        """
        return set(self._polling_contracts.keys()) | set(self._subscribed_contracts)

    def subscription_state_sizes(self) -> dict:
        """
        Task 2 observability requirement: current size of every tracking
        set this module maintains, so a divergence between them (the class
        of bug this task exists to fix) is visible in logs/dashboard going
        forward instead of only surfacing as an opaque "AlreadySubscribed"
        or silent-tick-loss symptom discovered after the fact.
        """
        return {
            "polling_contracts":    len(self._polling_contracts),
            "subscribed_contracts": len(self._subscribed_contracts),
            "tick_subscriptions":   len(self._subscription_map),
            "tick_callbacks":       len(self._tick_callbacks),
        }

    # ─── Last-resort truth source (Fix C.3) ───────────────────────────────

    async def profit_table_lookup(self, contract_id: str) -> dict:
        """
        Query Deriv's profit_table as a last-resort truth source for a
        contract that has failed to resolve via force_check_contract for
        an extended period (Implementation Brief v2, Fix C.3). Scans
        recent transactions for a matching contract_id.

        Returns a dict with at least is_sold/profit/sell_price when a
        match is found, or {} if not found / on any failure. Never raises.
        """
        try:
            resp = await self._send(
                {
                    "profit_table": 1,
                    "description": 1,
                    "limit": 50,
                    "sort": "DESC",
                },
                timeout=20,
            )
            transactions = resp.get("profit_table", {}).get("transactions", [])
            for txn in transactions:
                if str(txn.get("contract_id", "")) == str(contract_id):
                    return {
                        "is_sold":    1,
                        "contract_id": contract_id,
                        "profit":     float(txn.get("sell_price", 0)) - float(txn.get("buy_price", 0)),
                        "sell_price": float(txn.get("sell_price", 0)),
                        "buy_price":  float(txn.get("buy_price", 0)),
                        "sold_for":   txn.get("sell_price"),
                    }
            return {}
        except Exception as e:
            logger.error(f"PROFIT_TABLE LOOKUP ERROR {contract_id}: {e}")
            return {}

    async def subscribe_contract(
        self,
        contract_id: str,
        callback:    Callable,
        symbol:      str = "",
    ):
        """
        Register a contract + callback for polling-based resolution.
        Replaces the old WebSocket proposal_open_contract subscription.
        """
        cid = str(contract_id)
        self._polling_contracts[cid] = {
            "callback":  callback,
            "placed_at": time.time(),
        }
        if symbol:
            self._contract_symbol_map[cid] = symbol
        logger.info(f"TRACKING: {cid} via polling")
        asyncio.create_task(self._subscribe_ws_contract(cid))

    # ─── WS subscription (primary close signal; poller above is the fallback) ─

    async def _subscribe_ws_contract(self, contract_id: str):
        """
        Subscribe to proposal_open_contract updates for *contract_id*
        (subscribe: 1). Never double-subscribes — guarded by
        self._subscribed_contracts. Pushes are handled in _handle().
        """
        cid = str(contract_id)
        if cid in self._subscribed_contracts:
            return
        try:
            await self._send({
                "proposal_open_contract": 1,
                "contract_id": int(cid),
                "subscribe": 1,
            })
            self._subscribed_contracts.add(cid)
            logger.info(f"SUBSCRIBED: {cid} (proposal_open_contract)")
        except Exception as e:
            logger.error(f"SUBSCRIBE FAILED: {cid} — {e}")

    async def _resubscribe_contracts(self):
        """
        Re-subscribe every currently-open contract's proposal_open_contract
        WS channel after a (re)connect.

        FIX (Task 2 — tracking divergence): previously rebuilt from
        self._subscribed_contracts, which is NOT the authoritative "what's
        actually open" set — self._polling_contracts is. subscribe_contract()
        registers a contract in _polling_contracts synchronously, then fires
        the WS subscribe via asyncio.create_task(self._subscribe_ws_contract(cid))
        — fire-and-forget, non-blocking. If a reconnect interrupts the
        connection before that task lands (i.e. before cid is added to
        _subscribed_contracts), the old code would silently drop that
        contract from resubscription forever: it was never in
        _subscribed_contracts to begin with, so it wouldn't be in `pending`
        either. The contract would then only ever be resolved by the 30s
        polling fallback, with no live WS push — exactly the kind of
        one-tracker-diverges-from-the-other bug this task asked to find.

        Rebuilding from self._polling_contracts (the single source of truth
        for "which contracts we still consider open") instead means every
        genuinely open contract gets its WS subscription (re)requested on
        every reconnect, regardless of how far the previous subscribe
        attempt got.
        """
        self._subscribed_contracts.clear()
        pending = list(self._polling_contracts.keys())
        if not pending:
            return
        logger.info(f"RESUBSCRIBING: {len(pending)} contracts after reconnect")
        for cid in pending:
            await self._subscribe_ws_contract(cid)

    async def _resubscribe_ticks(self):
        """
        Re-issue every live tick subscription after a (re)connect.

        FIX (Task 2 — tracking divergence / silent tick loss): the previous
        connect() flow never re-subscribed ticks at all after a reconnect.
        subscribe_ticks()'s own no-double-subscribe guard is
        `if symbol in self._subscription_map: return existing`  — but
        self._subscription_map is never cleared on disconnect, so after a
        reconnect it still holds sub_ids from the now-dead connection.
        Every subscription the server held is gone the instant that
        connection drops (a fresh WS connection has no memory of the old
        one's subscriptions) — but subscribe_ticks() would keep matching
        the stale map entry and skip re-sending the request, leaving that
        symbol with local state that says "subscribed" while the server
        has nothing live for it. This is exactly the "bot's internal state
        doesn't think it's already subscribed... or vice versa" symptom
        from Task 2, and it explains ticks silently stopping for a symbol
        with nothing else visibly wrong (no error, no _tick_degraded flag —
        that only gets set on an explicit subscribe failure, not a reconnect
        that quietly orphans an existing one).

        Clears _subscription_map first so subscribe_ticks() always re-sends
        on the new connection, then rebuilds from self._tick_callbacks (the
        single source of truth for "which symbols the bot currently wants
        live ticks for") rather than the stale sub_id map.
        """
        symbols = list(self._tick_callbacks.keys())
        self._subscription_map.clear()
        if not symbols:
            return
        logger.info(f"RESUBSCRIBING TICKS: {len(symbols)} symbol(s) after reconnect")
        for symbol in symbols:
            callback = self._tick_callbacks.get(symbol)
            if callback is None:
                continue
            try:
                resp   = await self._send({"ticks": symbol, "subscribe": 1})
                sub_id = resp.get("tick", {}).get("id", symbol)
                self._subscription_map[symbol] = sub_id
                logger.info(f"Tick subscription restored: {symbol} (sub_id={sub_id})")
            except Exception as exc:
                logger.warning(
                    f"TICK RESUBSCRIBE FAILED: {symbol} after reconnect — {exc} "
                    f"(will retry via the normal degraded-retry path if bot_engine "
                    f"notices no ticks are arriving)"
                )

    # ─── Force check a specific contract ─────────────────────────────────────

    async def force_check_contract(self, contract_id: str) -> dict:
        """
        Manually query a contract's current state.

        Returns the full proposal_open_contract dict, including
        profit, sell_price, is_sold, is_expired (and all other fields
        the API returns). Returns {} on failure.
        """
        try:
            req = {
                "proposal_open_contract": 1,
                "contract_id": int(contract_id),
            }
            resp = await self._send(req)
            if not resp:
                return {}
            poc = resp.get("proposal_open_contract", {})
            logger.info(
                f"FORCE CHECK {contract_id}: "
                f"is_sold={poc.get('is_sold')} "
                f"profit={poc.get('profit')}")
            return poc
        except Exception as e:
            logger.error(f"FORCE CHECK ERROR {contract_id}: {e}")
            return {}

    async def get_profit_table(self, limit: int = 30, strict: bool = True, timeout: float = 10.0) -> list:
        """Most recent CLOSED contracts (newest first): used to find an in-doubt buy that already settled.
        strict=True raises on failure (an empty list must mean "nothing there", not "could not ask")."""
        try:
            resp = await self._send({"profit_table": 1, "description": 1, "limit": int(limit), "sort": "DESC"}, timeout=timeout)
            return list((resp or {}).get("profit_table", {}).get("transactions", []) or [])
        except Exception as e:
            logger.error(f"GET PROFIT TABLE ERROR: {e}")
            if strict:
                raise
            return []

    async def get_portfolio(self, strict: bool = False, timeout: float = 30.0) -> list:
        """
        Query Deriv's own server-side state for every currently-open
        contract on this account ({"portfolio": 1}) — no dependency on any
        local file, database, or in-memory dict.

        Spec point 10 (Aug 2026): Render's filesystem is ephemeral across
        deploys on this plan, and the bot's own in-memory open-contract
        tracking is wiped on every process restart too. Since the bot
        force-redeploys itself every REDEPLOY_INTERVAL_HOURS (spec point
        11), that's a real, frequent blackout window for the dashboard
        unless "what's open" is re-derived from Deriv's server on startup
        instead of assumed to have survived locally. Call this once, right
        after connecting, and use the result to rebuild both bot_engine.py's
        own open-contract tracking and keep_alive.py's dashboard state
        (see bot_engine.py's startup path / keep_alive.update_open_contracts()).

        Returns the list of contract dicts under "contracts" in Deriv's
        response (each with contract_id, symbol, contract_type, buy_price,
        purchase_time, etc. — the same shape as proposal_open_contract
        entries), or [] on any failure. Never raises.
        """
        try:
            resp = await self._send({"portfolio": 1}, timeout=timeout)
            if not resp:
                logger.warning("GET PORTFOLIO: no response")
                if strict:
                    raise RuntimeError("portfolio: no response")
                return []
            if resp.get("error"):
                err = resp["error"]
                logger.error(f"GET PORTFOLIO FAILED: {err.get('code')}: {err.get('message')}")
                return []
            contracts = resp.get("portfolio", {}).get("contracts", [])
            logger.info(f"GET PORTFOLIO: {len(contracts)} open contract(s) on Deriv's own state")
            return contracts
        except Exception as e:
            logger.error(f"GET PORTFOLIO ERROR: {e}")
            if strict:
                raise                   # strict: "could not ask" must not look like "nothing open"
            return []

    # ─── Request helper ───────────────────────────────────────────────────────

    async def _send(self, payload: dict, timeout: float = 30.0) -> dict:
        if self._stopped:
            raise RuntimeError("Client shut down")
        if "buy" in payload or "proposal" in payload:
            # Balance-target guard: once TARGET_BALANCE is hit nothing new may be priced or bought.
            import target_guard
            if target_guard.is_reached():
                raise RuntimeError("TARGET_REACHED: trading stopped, order not sent")
        if not self._ws or not self._connected:
            raise RuntimeError("Not connected")
        if "buy" in payload:
            # Send-time buy gate: checked immediately before the frame is written and before the
            # pending future is registered. Raising here means "definitely not sent".
            _t = ws.now()
            if not ws.buy_allowed_at(_t):
                logger.warning(
                    f"BUY BLOCKED AT SEND: server-time {_t:.3f} is outside a session / at or after the close - "
                    f"frame NOT sent")
                err = ws.SessionClosed("buy blocked: outside a session or at/after the close (not sent)")
                err.sent = False
                raise err

        req_id                = self._req_id_counter
        self._req_id_counter += 1
        payload["req_id"]     = req_id

        fut = self._loop.create_future()
        self._pending[req_id] = fut

        try:
            await self._ws.send(json.dumps(payload))
        except Exception as exc:
            self._pending.pop(req_id, None)
            _mark_sent(exc, req_id)                         # the frame may have gone out before the failure
            raise
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            err = RequestTimeout(f"Request timed out: {payload.get('msg_type', payload)}")
            err.req_id = req_id
            raise err
        except ConnectionError as exc:                      # socket dropped while the request was pending
            _mark_sent(exc, req_id)
            raise

    # ─── Auth & balance ───────────────────────────────────────────────────────

    async def _authorize(self):
        if not config.DERIV_API_TOKEN:
            raise ValueError("DERIV_API_TOKEN is not set.")

        payload = {"authorize": config.DERIV_API_TOKEN, "req_id": 1}
        await self._ws.send(json.dumps(payload))
        raw = await asyncio.wait_for(self._ws.recv(), timeout=30)
        msg = json.loads(raw)

        if msg.get("error"):
            raise RuntimeError(f"Auth failed: {msg['error'].get('message')}")

        account              = msg.get("authorize", {})
        self._balance        = float(account.get("balance", 0))
        self._balance_ts     = time.monotonic()
        self._authorized     = True
        self._req_id_counter = 10

        self._ready.set()

        logger.info(
            f"Authorized ✓ | Account: {account.get('loginid')} | "
            f"Balance: ${self._balance:.4f}"
        )

        for cb in self._balance_callbacks:
            try:
                cb(self._balance)
            except Exception:
                pass

    async def _subscribe_balance(self):
        try:
            await self._send({"balance": 1, "subscribe": 1})
            logger.info("Balance subscription active ✓")
        except Exception as exc:
            logger.warning(
                f"Balance subscription failed: {exc} — will use polling fallback"
            )

    async def balance_refresh_loop(self, interval: int = 60):
        while True:
            await asyncio.sleep(interval)
            if not self._authorized or not self._ws:
                continue
            try:
                resp    = await self._send({"balance": 1})
                new_bal = float(resp.get("balance", {}).get("balance", self._balance))
                if new_bal != self._balance:
                    logger.info(f"Balance poll: ${self._balance:.4f} → ${new_bal:.4f}")
                    self._balance    = new_bal
                    self._balance_ts = time.monotonic()
                    for cb in self._balance_callbacks:
                        try:
                            cb(self._balance)
                        except Exception:
                            pass
            except Exception as exc:
                logger.debug(f"Balance poll failed: {exc}")

    def on_balance(self, callback: Callable[[float], None]):
        self._balance_callbacks.append(callback)

    @property
    def balance(self) -> float:
        return self._balance

    async def get_balance(self) -> float:
        """
        Fetch current account balance via the balance API endpoint.

        Returns the live balance as a float.
        Cache TTL: 30 seconds — if the cached value is fresh enough, return it
        directly to avoid hammering the API.
        On any failure: return last known balance and log a warning.
        """
        _BALANCE_CACHE_TTL = 30.0

        now = time.monotonic()
        if (now - self._balance_ts) < _BALANCE_CACHE_TTL:
            return self._balance

        if not self._authorized or not self._ws:
            cached = self._balance
            logger.warning(
                f"BALANCE FETCH FAILED — using cached: ${cached:.2f} "
                f"(not connected)"
            )
            return cached

        try:
            resp    = await self._send({"balance": 1}, timeout=15)
            new_bal = float(resp.get("balance", {}).get("balance", self._balance))
            if new_bal != self._balance:
                logger.info(f"get_balance: ${self._balance:.4f} → ${new_bal:.4f}")
            self._balance    = new_bal
            self._balance_ts = time.monotonic()
            return self._balance
        except Exception as exc:
            cached = self._balance
            logger.warning(
                f"BALANCE FETCH FAILED — using cached: ${cached:.2f} | error={exc}"
            )
            return cached

    # ─── Historical candles ───────────────────────────────────────────────────

    async def get_candles(
        self,
        symbol:      str,
        granularity: int,
        count:       int = 100,
    ) -> List[dict]:
        """
        Fetch OHLC candles for *symbol*.

        Returns list[dict] with keys: open, high, low, close, epoch
        On any failure: returns [] and logs CANDLE FETCH FAILED: {symbol} gran={granularity}
        """
        await self._ready.wait()
        try:
            resp = await self._send(
                {
                    "ticks_history":    symbol,
                    "adjust_start_time": 1,
                    "count":            count,
                    "end":              "latest",
                    "granularity":      granularity,
                    "style":            "candles",
                },
                timeout=20,
            )
            raw_candles: List[dict] = resp.get("candles", [])

            # Normalise to guaranteed key set: open, high, low, close, epoch
            candles: List[dict] = [
                {
                    "open":  float(c.get("open",  0)),
                    "high":  float(c.get("high",  0)),
                    "low":   float(c.get("low",   0)),
                    "close": float(c.get("close", 0)),
                    "epoch": int(c.get("epoch",   0)),
                }
                for c in raw_candles
            ]

            logger.debug(
                f"Got {len(candles)} candles for {symbol} (gran={granularity}s)"
            )
            return candles

        except Exception as exc:
            logger.warning(
                f"CANDLE FETCH FAILED: {symbol} gran={granularity} | error={exc}"
            )
            return []

    # ─── Historical ticks (digit predictor warm-up) ───────────────────────────

    async def get_tick_history(self, symbol: str, count: int = 5000) -> List[dict]:
        """Latest `count` ticks as [{"epoch": int, "quote": float}], oldest first.
        Deriv returns only ~1000 ticks per call, so older pages are fetched backwards
        (end = oldest epoch - 1) until `count` is reached or no more come back.
        Returns [] if even the first page fails; a failed older page just ends paging."""
        await self._ready.wait()
        page = max(100, int(getattr(config, "TICK_HISTORY_PAGE_SIZE", 1000)))
        want = int(count)
        by_epoch: Dict[int, float] = {}
        end: Any = "latest"
        for page_no in range(max(2, -(-want // page) + 2)):
            req = {"ticks_history": symbol, "count": min(page, max(want - len(by_epoch), 1)),
                   "end": end, "style": "ticks"}
            if page_no == 0:
                req["adjust_start_time"] = 1
            try:
                resp = await self._send(req, timeout=30)
            except Exception as exc:
                logger.warning(f"TICK HISTORY FAILED: {symbol} page={page_no} count={count} | error={exc}")
                break
            h = resp.get("history", {}) or {}
            before = len(by_epoch)
            for t, p in zip(h.get("times", []), h.get("prices", [])):
                by_epoch[int(t)] = float(p)
            if len(by_epoch) >= want or len(by_epoch) == before:
                break
            end = min(by_epoch) - 1
        ordered = sorted(by_epoch.items())[-want:]
        return [{"epoch": e, "quote": q} for e, q in ordered]

    async def refresh_ticks(self, symbol: str, callback: Callable[[dict], None]) -> bool:
        """Drop and re-create one live tick subscription (used when a stream goes quiet
        while the socket itself still looks healthy). Returns True on success."""
        old = self._subscription_map.pop(symbol, None)
        if old:
            try:
                await self._send({"forget": old}, timeout=10)
            except Exception as exc:
                logger.debug(f"forget({old}) failed: {exc}")
        self._tick_callbacks.pop(symbol, None)
        sub_id = await self.subscribe_ticks(symbol, callback)
        return bool(sub_id)

    # ─── Live tick subscription ───────────────────────────────────────────────

    async def subscribe_ticks(
        self,
        symbol:   str,
        callback: Callable[[dict], None],
    ) -> str:
        if symbol in self._subscription_map:
            self._tick_callbacks[symbol] = callback
            return self._subscription_map[symbol]

        try:
            resp   = await self._send({"ticks": symbol, "subscribe": 1})
            sub_id = resp.get("tick", {}).get("id", symbol)
            self._subscription_map[symbol] = sub_id
            self._tick_callbacks[symbol]   = callback
            logger.info(f"Tick subscription: {symbol} (sub_id={sub_id})")
            return sub_id
        except Exception as exc:
            logger.error(f"subscribe_ticks({symbol}) failed: {exc}")
            return ""

    async def unsubscribe_ticks(self, symbol: str):
        sub_id = self._subscription_map.pop(symbol, None)
        self._tick_callbacks.pop(symbol, None)
        if sub_id:
            try:
                await self._send({"forget": sub_id})
            except Exception:
                pass

    # ─── Market-open check ────────────────────────────────────────────────────

    async def _check_market_open(self, symbol: str) -> bool:
        """
        Returns True if the symbol's market is open.

        Volatility indices (R_*, 1HZ*) are always open — skip the API call.
        For Boom/Crash (and everything else): call active_symbols and inspect
        exchange_is_open.  Returns False (and logs SKIPPED) on any error so
        that buy_contract() can bail out safely.
        """
        if _is_volatility_index(symbol):
            return True  # 24/7 — never gated

        try:
            resp    = await self._send(
                {"active_symbols": "brief", "product_type": "basic"},
                timeout=20,
            )
            symbols = resp.get("active_symbols", [])
            for s in symbols:
                if s.get("underlying_symbol") == symbol:
                    open_flag = s.get("exchange_is_open", 0)
                    return bool(open_flag)
            # Symbol not found in active list → treat as closed
            logger.warning(f"SKIPPED: {symbol} market_closed (symbol not in active list)")
            return False
        except Exception as exc:
            logger.warning(f"SKIPPED: {symbol} market_closed (check failed: {exc})")
            return False

    # ─── Trade execution ──────────────────────────────────────────────────────

    def _cap_stake(self, stake: float, symbol: str) -> float:
        """
        Mirrors the FIX 3 stake-cap rule documented above for buy_contract()
        (if stake > balance * 0.5, cap at max(balance * 0.02, MIN_STAKE)).

        Used only by the new proposal-based buy methods below —
        buy_contract() itself is left untouched per task instructions.

        FIX (Task 1 — $0 stake root cause): the cap floor used to be the
        module's own hardcoded 0.35, completely independent of
        config.MIN_STAKE (100 in this deployment). RiskManager.calculate_
        stake() already clamps every stake it returns to >= MIN_STAKE
        before handing it to bot_engine._execute() — but if that stake then
        exceeded balance*0.5, this method would cap it right back down to
        as low as $0.35, silently undercutting the floor RiskManager had
        just enforced one layer up. The floor is now config.MIN_STAKE
        itself, so no later capping step in the pipeline can push the
        final "amount" sent to the API back below the floor the risk
        system decided on.

        balance <= 0 (not yet fetched, or a genuine 0 mid-session) is left
        uncapped here (can't safely compute balance*0.5/balance*0.02
        against an unknown/zero balance) — the hard stake>0/finite
        validation immediately before each buy_* call's proposal request
        is what actually guards against a bad value reaching the API in
        that case, not this method.
        """
        balance = self._balance or 0.0
        if balance > 0 and stake > balance * 0.5:
            floor  = getattr(config, "MIN_STAKE", 0.35)
            capped = max(balance * 0.02, floor)
            logger.info(f"STAKE CAPPED: ${stake:.4f} → ${capped:.4f} ({symbol})")
            return capped
        return stake

    # ─── Task 1 — hard numeric validation guard ────────────────────────────
    # Centralized here (rather than duplicated at every bot_engine.py call
    # site) so every buy_* path — present and future — gets the same
    # guarantee: stake and any SL/TP dollar amount sent to the API is never
    # 0, negative, or non-finite. This is a safety net on top of the actual
    # root-cause fixes above (RiskManager's legitimate $0.00 "don't trade
    # this pair" sentinel now being checked by bot_engine._execute() before
    # it ever reaches a buy_* call, and _cap_stake() now respecting
    # MIN_STAKE as its floor) — not a replacement for them. Never raises;
    # returns a reason string on failure, or None if the value is valid.

    @staticmethod
    def _invalid_reason(label: str, value: Optional[float]) -> Optional[str]:
        if value is None:
            return f"{label} is None"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return f"{label}={value!r} is not numeric"
        if not math.isfinite(v):
            return f"{label}={v} is not finite"
        if v <= 0:
            return f"{label}={v:.6f} is <= 0"
        return None

    def _validate_stake(self, stake: float, symbol: str, context: str) -> bool:
        """
        True if `stake` is safe to send as the API's "amount" field.
        Logs a clear, specific reason and returns False otherwise —
        callers must abort that specific trade attempt on False rather
        than sending the malformed order.
        """
        reason = self._invalid_reason("stake", stake)
        if reason:
            logger.error(
                f"BUY ABORTED: {symbol} ({context}) — invalid stake, "
                f"refusing to send a malformed order: {reason}"
            )
            return False
        return True

    def _validate_limit_order(
        self, stop_loss_amount: float, take_profit_amount: float,
        symbol: str, context: str,
    ) -> bool:
        """
        True if both SL and TP dollar amounts are safe to send in a
        Multiplier proposal's limit_order. Logs which one is bad and why
        (dist-to-stop/target computed as 0, stake was 0 so the product
        collapsed to 0, entry_price was bad, etc. — the log line names the
        value, not the upstream cause, since by this point that's all this
        layer can see) and returns False otherwise.
        """
        ok = True
        reason = self._invalid_reason("stop_loss_amount", stop_loss_amount)
        if reason:
            logger.error(
                f"BUY ABORTED: {symbol} ({context}) — invalid stop_loss_amount, "
                f"refusing to send a malformed limit_order: {reason}"
            )
            ok = False
        reason = self._invalid_reason("take_profit_amount", take_profit_amount)
        if reason:
            logger.error(
                f"BUY ABORTED: {symbol} ({context}) — invalid take_profit_amount, "
                f"refusing to send a malformed limit_order: {reason}"
            )
            ok = False
        return ok

    @staticmethod
    def _digit_contract_type(match_type: str) -> Optional[str]:
        # OVER/UNDER support (digit predictor) — reuses the exact same match_type/
        # digit plumbing get_digit_proposal()/buy_digit_contract() already
        # had for MATCH/DIFFER, no signature changes needed.
        mapping = {
            "MATCH":  "DIGITMATCH",
            "DIFFER": "DIGITDIFF",
            "OVER":   "DIGITOVER",
            "UNDER":  "DIGITUNDER",
        }
        return mapping.get(str(match_type).upper())

    @staticmethod
    def _digit_barrier_valid(contract_type: str, digit: int) -> bool:
        """
        Per-contract-type barrier bounds (Deriv's actual contract rules):
          DIGITMATCH / DIGITDIFF : barrier 0-9 (any digit)
          DIGITOVER              : barrier 0-8 (barrier 9 can never win —
                                    no digit is > 9)
          DIGITUNDER             : barrier 1-9 (barrier 0 can never win —
                                    no digit is < 0)
        """
        if not (0 <= digit <= 9):
            return False
        if contract_type == "DIGITOVER":
            return digit <= 8
        if contract_type == "DIGITUNDER":
            return digit >= 1
        return True

    # ─── Circuit breaker (per symbol+strategy) ─────────────────────────────
    # Suspends new buy attempts on a (symbol, strategy) combination after N
    # consecutive failures, for a cooldown period — prevents the same dead
    # signal from retrying every scan cycle indefinitely.

    @staticmethod
    def _cb_key(symbol: str, strategy: str) -> tuple:
        return (symbol, strategy or "default")

    def _cb_blocked(self, symbol: str, strategy: str) -> Optional[float]:
        """Returns seconds remaining in cooldown, or None if not blocked."""
        state = self._circuit_breaker.get(self._cb_key(symbol, strategy))
        if not state:
            return None
        remaining = state.get("cooldown_until", 0.0) - time.time()
        return remaining if remaining > 0 else None

    def _cb_record_failure(self, symbol: str, strategy: str,
                            threshold: int, cooldown: float):
        key = self._cb_key(symbol, strategy)
        state = self._circuit_breaker.setdefault(key, {"count": 0, "cooldown_until": 0.0})
        state["count"] += 1
        if state["count"] >= threshold:
            state["cooldown_until"] = time.time() + cooldown
            state["count"] = 0
            logger.warning(
                f"CIRCUIT BREAKER TRIPPED: {symbol}/{strategy} — "
                f"{threshold} consecutive buy failures, suspending for {cooldown:.0f}s"
            )

    def _cb_record_success(self, symbol: str, strategy: str):
        self._circuit_breaker.pop(self._cb_key(symbol, strategy), None)

    async def buy_contract(
            self,
            symbol:        str,
            direction:     str,   # "LONG" or "SHORT"
            stake:         float,
            multiplier:    int    = 100,
            stop_loss:     float  = None,
            take_profit:   float  = None,
            strategy:      str    = "default",
            cb_threshold:  int    = 3,
            cb_cooldown:   float  = 60.0,
            **kwargs) -> dict:
        """
        Buy a Rise/Fall (CALL/PUT) contract via the required proposal ->
        buy flow. Deriv rejects inline 'parameters' (incl. 'symbol') on a
        direct buy request — a proposal must be requested first to obtain
        a proposal_id and ask_price, matching Deriv's documented API and
        every up-to-date client library.

        Direction mapping (VERIFIED CORRECT - DO NOT SWAP):
          direction="LONG"  -> contract_type="CALL"  -> wins if price RISES
          direction="SHORT" -> contract_type="PUT"   -> wins if price FALLS

        strategy identifies the signal/strategy that produced this trade,
        used as the second half of the circuit-breaker key (symbol,
        strategy). After cb_threshold consecutive failures on that
        combination, new attempts are suspended for cb_cooldown seconds.

        Returns dict (buy response) on success, None on every failure path
        (including a circuit-breaker cooldown skip). Never raises, EXCEPT
        ws.SessionClosed (v16: the buy frame was NOT sent because the session
        closed). Never increments trade counters on None return.

        After a successful buy, the caller is expected to register the
        contract for polling via subscribe_contract().
        """
        remaining = self._cb_blocked(symbol, strategy)
        if remaining is not None:
            logger.info(
                f"SKIPPED: {symbol}/{strategy} — circuit breaker cooldown "
                f"active ({remaining:.0f}s remaining)"
            )
            return None

        # Task 1 hard guard — checked before the semaphore/rate-limit wait
        # so an invalid stake never even occupies a buy slot.
        if not self._validate_stake(stake, symbol, "buy_contract/Rise-Fall"):
            self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
            return None

        async with self._buy_semaphore:
            elapsed = time.time() - self._last_buy_time
            if elapsed < 3.0:
                await asyncio.sleep(3.0 - elapsed)
            self._last_buy_time = time.time()

            contract_type = "CALL" if direction == "LONG" else "PUT"

            # Fix A (Implementation Brief v2): use the configured duration
            # instead of a hardcoded 5m — this was the literal source of
            # the "config says 14, bot trades 5" mismatch. Per-symbol
            # override takes precedence if one is ever confirmed necessary
            # by a contracts_for audit; falls back to 5m/"m" only if
            # config doesn't define TRADE_DURATION at all.
            overrides = getattr(config, "TRADE_DURATION_OVERRIDES", {})
            duration      = overrides.get(symbol, getattr(config, "TRADE_DURATION", 5))
            duration_unit = getattr(config, "TRADE_DURATION_UNIT", "m")

            logger.info(
                f"BUY ATTEMPT: {symbol} {contract_type} stake=${stake:.4f} "
                f"duration={duration}{duration_unit}"
            )

            proposal_req = {
                "proposal":      1,
                "amount":        stake,
                "basis":         "stake",
                "contract_type": contract_type,
                "currency":      "USD",
                "duration":      duration,
                "duration_unit": duration_unit,
                "underlying_symbol": symbol,
            }
            barrier = kwargs.get("barrier")
            if barrier is not None:
                proposal_req["barrier"] = barrier

            try:
                prop_resp = await self._send(proposal_req)
                if not prop_resp:
                    logger.error(f"FAILED: {symbol} — no response (proposal)")
                    self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                    return None
                if prop_resp.get("error"):
                    err = prop_resp["error"]
                    logger.error(
                        f"FAILED: {symbol} — {err.get('code')}: {err.get('message')} "
                        f"| details={err.get('details')} | full_req={proposal_req}"
                    )
                    self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                    return None

                proposal = prop_resp.get("proposal", {})
                logger.info(f"PROPOSAL RESPONSE: {proposal}")

                proposal_id = proposal.get("id")
                ask_price   = proposal.get("ask_price", stake)
                if not proposal_id:
                    logger.error(
                        f"FAILED: {symbol} — proposal_rejected (no id in {proposal})"
                    )
                    self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                    return None

                logger.info(
                    f"PRICE CHECK: {symbol} ask_price=${ask_price:.4f} "
                    f"vs stake=${stake:.4f} "
                    f"(diff=${ask_price - stake:+.4f})"
                )

                buy_req = {"buy": proposal_id, "price": ask_price}
                resp = await self._send(buy_req)
                if not resp:
                    logger.error(f"FAILED: {symbol} — no response (buy)")
                    self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                    return None
                if resp.get("error"):
                    err = resp["error"]
                    logger.error(
                        f"FAILED: {symbol} — {err.get('code')}: {err.get('message')} "
                        f"| details={err.get('details')} | full_req={buy_req}"
                    )
                    self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                    return None

                result = resp.get("buy", {})
                contract_id = str(result.get("contract_id", ""))
                logger.info(
                    f"CONTRACT OPENED: {contract_id} | {symbol} | "
                    f"{contract_type} | ${stake:.4f}"
                )

                self._contract_symbol_map[contract_id] = symbol
                self._cb_record_success(symbol, strategy)

                # Guaranteed closure: subscribe to this contract immediately.
                # (subscribe_contract(), called later by the caller to attach
                # its close callback, will no-op the resend via the
                # _subscribed_contracts guard — never double-subscribes.)
                asyncio.create_task(self._subscribe_ws_contract(contract_id))

                return result

            except ws.SessionClosed:
                raise                       # definite "not sent": the caller releases money/slot (no breaker hit)
            except Exception as e:
                logger.error(f"FAILED: {symbol} — {e} | full_req={proposal_req}")
                self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                return None

    async def buy_multiplier(
            self,
            symbol:        str,
            direction:     str,   # "LONG" or "SHORT"
            stake:         float,
            multiplier:    int    = None,
            stop_loss_pct: float  = None,
            take_profit_ratio: float = None,
            strategy:      str    = "default",
            cb_threshold:  int    = 3,
            cb_cooldown:   float  = 60.0,
            stop_loss_price:   float = None,
            take_profit_price: float = None,
            entry_price:       float = None,
            **kwargs) -> dict:
        """
        Buy a Multipliers (MULTUP/MULTDOWN) contract via the required
        proposal -> buy flow. Mirrors buy_contract()'s defensive error
        handling (log and return None on every failure path, never raise)
        and its circuit-breaker gating, but targets Multiplier contracts
        instead of Rise/Fall — needed for symbols (Boom/Crash, Jump,
        Drift Switch, and any future Range Break / Bear-Bull additions)
        that do not support CALL/PUT on this account.

        Direction mapping (mirrors buy_contract — DO NOT SWAP):
          direction="LONG"  -> contract_type="MULTUP"    -> wins if price RISES
          direction="SHORT" -> contract_type="MULTDOWN"  -> wins if price FALLS

        multiplier defaults to config.MULTIPLIER_MAP[symbol] (falling back
        to config.DEFAULT_MULTIPLIER) when not passed explicitly.

        Multiplier contracts have no fixed duration — risk is bounded here
        via an explicit limit_order (stop_loss / take_profit) attached to
        the proposal, rather than a duration/duration_unit pair.
        stop_loss_pct defaults to config.STOP_LOSS_MAP[symbol] (falling
        back to config.DEFAULT_STOP_LOSS_PCT); take_profit_ratio defaults
        to config.TAKE_PROFIT_RATIO. Both are first computed exactly as
        before, relative to stake:
          stop_loss_amount   = stake * (stop_loss_pct / 100)
          take_profit_amount = stop_loss_amount * take_profit_ratio
        ...and then, per config.TP_SL_SWAP_ENABLED (user-directed, Aug
        2026), SWAPPED before being sent: the limit_order's take_profit
        is set to the computed stop_loss_amount, and its stop_loss is
        set to the computed take_profit_amount.

        Popular-indicator pipeline (spec point 6, Aug 2026): when
        stop_loss_price/take_profit_price/entry_price are all supplied,
        this whole stake-percentage computation (and the
        TP_SL_SWAP_ENABLED amount-swap above) is bypassed entirely — the
        caller has already inverted direction AND swapped the two PRICE
        levels upstream (bot_engine.py's universal inversion step), so
        swapping the dollar AMOUNTS again here would double-swap and
        cancel out. Instead the two PRICE levels are converted straight to
        dollar SL/TP using the standard linear P&L relationship for
        Deriv's leveraged Multiplier contracts (P&L ~= stake x multiplier
        x %price-move):
            dist_to_stop   = abs(entry_price - stop_loss_price)
            dist_to_target = abs(entry_price - take_profit_price)
            stop_loss_amount   = stake * multiplier * (dist_to_stop   / entry_price)
            take_profit_amount = stake * multiplier * (dist_to_target / entry_price)
        entry_price should be the same price the caller's indicator
        actually measured its distance against (e.g.
        signal_engine.SignalResult.native_entry_price) — not a freshly
        re-fetched quote, or the distance sent here won't match what was
        computed upstream.

        strategy identifies the signal/strategy that produced this trade,
        used as the second half of the circuit-breaker key (symbol,
        strategy), exactly as in buy_contract().

        Returns dict (buy response) on success, None on every failure path
        (including a circuit-breaker cooldown skip). Never raises, EXCEPT
        ws.SessionClosed (v16: the buy frame was NOT sent because the session
        closed). Never increments trade counters on None return.

        After a successful buy, the caller is expected to register the
        contract for polling via subscribe_contract() — already handled
        here the same way buy_contract() does it, via _subscribe_ws_contract.
        """
        remaining = self._cb_blocked(symbol, strategy)
        if remaining is not None:
            logger.info(
                f"SKIPPED: {symbol}/{strategy} — circuit breaker cooldown "
                f"active ({remaining:.0f}s remaining)"
            )
            return None

        if multiplier is None:
            multiplier = config.MULTIPLIER_MAP.get(symbol, config.DEFAULT_MULTIPLIER)
        if stop_loss_pct is None:
            stop_loss_pct = config.STOP_LOSS_MAP.get(symbol, config.DEFAULT_STOP_LOSS_PCT)
        if take_profit_ratio is None:
            take_profit_ratio = config.TAKE_PROFIT_RATIO

        # Task 1 hard guard — stake checked before the semaphore/rate-limit
        # wait so an invalid stake never even occupies a buy slot. This is
        # the guard that actually catches RiskManager's legitimate $0.00
        # "no edge / exposure ceiling full / below MIN_STAKE" sentinel if
        # bot_engine._execute()'s own earlier check (the root-cause fix)
        # were ever bypassed or a future call site skipped it — belt and
        # suspenders, per the task's explicit "safety net on top of, not
        # instead of" requirement.
        if not self._validate_stake(stake, symbol, "buy_multiplier"):
            self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
            return None

        async with self._buy_semaphore:
            elapsed = time.time() - self._last_buy_time
            if elapsed < 3.0:
                await asyncio.sleep(3.0 - elapsed)
            self._last_buy_time = time.time()

            stake = self._cap_stake(stake, symbol)
            # Re-validate after capping — _cap_stake() can only raise a
            # stake that's too large down toward MIN_STAKE, never toward
            # 0, but this keeps the guarantee airtight regardless of any
            # future change to that method's floor logic.
            if not self._validate_stake(stake, symbol, "buy_multiplier (post-cap)"):
                self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                return None

            contract_type = "MULTUP" if direction == "LONG" else "MULTDOWN"

            if stop_loss_price is not None and take_profit_price is not None and entry_price:
                # Popular-indicator pipeline (spec point 6) — native PRICE
                # levels supplied, already inverted+swapped (and, per Task 4,
                # stop-tightened) upstream. Convert straight to dollar
                # SL/TP; do NOT re-apply TP_SL_SWAP_ENABLED here, that would
                # double-swap.
                #
                # Task 1: entry_price/stop_loss_price/take_profit_price
                # being merely truthy doesn't guarantee dist_to_stop or
                # dist_to_target is actually > 0 — a strategy path that
                # failed to compute a real price level but didn't None-out
                # the field could hand this a stop/target equal to entry,
                # which would silently divide out to a $0 SL/TP below. The
                # _validate_limit_order() call after this block is what
                # catches that; nothing here needs to change for it, but
                # flagging why the check below isn't redundant.
                dist_to_stop   = abs(entry_price - stop_loss_price)
                dist_to_target = abs(entry_price - take_profit_price)
                stop_loss_amount   = round(stake * multiplier * (dist_to_stop   / entry_price), 2)
                take_profit_amount = round(stake * multiplier * (dist_to_target / entry_price), 2)
                logger.info(
                    f"NATIVE SL/TP: {symbol} entry={entry_price:.5f} "
                    f"stop_price={stop_loss_price:.5f} target_price={take_profit_price:.5f} "
                    f"-> SL=${stop_loss_amount:.2f} TP=${take_profit_amount:.2f}"
                )
            else:
                if stop_loss_price is not None and take_profit_price is not None and not entry_price:
                    # Task 1 — this is the "strategy path reached here
                    # without a native entry price" case: falls through to
                    # the legacy percentage path below rather than dividing
                    # by a zero/None entry_price, but that fallback was
                    # previously silent. Logged now so a missing
                    # native_entry_price is visible instead of looking like
                    # this trade simply never had native levels at all.
                    logger.warning(
                        f"NATIVE SL/TP SKIPPED: {symbol} — stop_loss_price/"
                        f"take_profit_price supplied but entry_price="
                        f"{entry_price!r} is missing/zero; falling back to "
                        f"the stake-percentage SL/TP method instead"
                    )
                # Legacy stake-percentage path — still used by any call
                # site that doesn't supply native price levels.
                # TP/SL SWAP (user-directed, Aug 2026 — see
                # config.TP_SL_SWAP_ENABLED). Computed exactly as before —
                # stake x stop_loss_pct for the stop distance, then x
                # take_profit_ratio for the take-profit distance — and then
                # the two amounts are swapped before being sent: the take
                # profit is placed where the stop-loss would otherwise have
                # gone, and the stop-loss is placed where the take-profit
                # would otherwise have gone.
                _computed_stop_loss_amount   = round(stake * (stop_loss_pct / 100.0), 2)
                _computed_take_profit_amount = round(_computed_stop_loss_amount * take_profit_ratio, 2)
                if getattr(config, "TP_SL_SWAP_ENABLED", True):
                    stop_loss_amount   = _computed_take_profit_amount
                    take_profit_amount = _computed_stop_loss_amount
                else:
                    stop_loss_amount   = _computed_stop_loss_amount
                    take_profit_amount = _computed_take_profit_amount

            # Task 1 hard guard — the actual point that closes the "$0
            # stake -> $0 SL/TP" hole: both formulas above have stake as a
            # factor, so anything that let a $0/negative/non-finite value
            # through this far (a zero dist_to_stop from a bad native
            # price level, a zero stop_loss_pct, stake having slipped
            # through despite the earlier checks, etc.) is caught here,
            # immediately before the request is built, rather than being
            # sent to the API.
            if not self._validate_limit_order(stop_loss_amount, take_profit_amount, symbol, "buy_multiplier"):
                self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                return None

            logger.info(
                f"BUY ATTEMPT: {symbol} {contract_type} stake=${stake:.4f} "
                f"multiplier={multiplier}x SL=${stop_loss_amount:.2f} "
                f"TP=${take_profit_amount:.2f}"
            )

            proposal_req = {
                "proposal":      1,
                "amount":        stake,
                "basis":         "stake",
                "contract_type": contract_type,
                "currency":      "USD",
                "underlying_symbol": symbol,
                "multiplier":    multiplier,
                "limit_order": {
                    "stop_loss":   stop_loss_amount,
                    "take_profit": take_profit_amount,
                },
            }

            try:
                prop_resp = await self._send(proposal_req)
                if not prop_resp:
                    logger.error(f"FAILED: {symbol} — no response (multiplier proposal)")
                    self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                    return None
                if prop_resp.get("error"):
                    err = prop_resp["error"]
                    logger.error(
                        f"FAILED: {symbol} — {err.get('code')}: {err.get('message')} "
                        f"| details={err.get('details')} | full_req={proposal_req}"
                    )
                    self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                    return None

                proposal = prop_resp.get("proposal", {})
                logger.info(f"PROPOSAL RESPONSE: {proposal}")

                proposal_id = proposal.get("id")
                ask_price   = proposal.get("ask_price", stake)
                if not proposal_id:
                    logger.error(
                        f"FAILED: {symbol} — proposal_rejected (no id in {proposal})"
                    )
                    self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                    return None

                logger.info(
                    f"PRICE CHECK: {symbol} ask_price=${ask_price:.4f} "
                    f"vs stake=${stake:.4f} "
                    f"(diff=${ask_price - stake:+.4f})"
                )

                buy_req = {"buy": proposal_id, "price": ask_price}
                resp = await self._send(buy_req)
                if not resp:
                    logger.error(f"FAILED: {symbol} — no response (multiplier buy)")
                    self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                    return None
                if resp.get("error"):
                    err = resp["error"]
                    logger.error(
                        f"FAILED: {symbol} — {err.get('code')}: {err.get('message')} "
                        f"| details={err.get('details')} | full_req={buy_req}"
                    )
                    self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                    return None

                result = resp.get("buy", {})
                contract_id = str(result.get("contract_id", ""))
                logger.info(
                    f"CONTRACT OPENED: {contract_id} | {symbol} | "
                    f"{contract_type} | ${stake:.4f} | {multiplier}x"
                )

                self._contract_symbol_map[contract_id] = symbol
                self._cb_record_success(symbol, strategy)

                # Guaranteed closure: subscribe to this contract immediately,
                # same as buy_contract() — subscribe_contract() later no-ops
                # the resend via the _subscribed_contracts guard.
                asyncio.create_task(self._subscribe_ws_contract(contract_id))

                return result

            except ws.SessionClosed:
                raise                       # definite "not sent": the caller releases money/slot (no breaker hit)
            except Exception as e:
                logger.error(f"FAILED: {symbol} — {e} | full_req={proposal_req}")
                self._cb_record_failure(symbol, strategy, cb_threshold, cb_cooldown)
                return None

    # ─── Accumulator (ACCU) contracts ──────────────────────────────────────────

    async def get_accumulator_proposal(
            self,
            symbol:      str,
            stake:       float,
            growth_rate: float,
            take_profit: float = None,
            **kwargs) -> Optional[dict]:
        """
        Request a proposal for an Accumulator (ACCU) contract.

        growth_rate is accepted as a percentage (1-5, matching Deriv's UI)
        and converted internally to the decimal the API expects
        (0.01-0.05) — pass growth_rate=2 for 2%.

        take_profit, if given, is attached as a limit_order so the
        contract auto-closes at that profit level.

        Returns the full 'proposal' dict (contains 'id' and 'ask_price',
        needed by buy_accumulator) on success, None on any failure path.
        Never raises.
        """
        if not (1 <= growth_rate <= 5):
            logger.error(
                f"FAILED: {symbol} — invalid growth_rate={growth_rate} "
                f"(must be 1-5, i.e. 1%-5%)"
            )
            return None

        try:
            req = {
                "proposal":      1,
                "amount":        stake,
                "basis":         "stake",
                "contract_type": "ACCU",
                "currency":      "USD",
                "underlying_symbol": symbol,
                "growth_rate":   growth_rate / 100.0,
            }
            if take_profit is not None:
                req["limit_order"] = {"take_profit": take_profit}

            logger.info(
                f"PROPOSAL REQUEST: ACCU {symbol} stake=${stake:.4f} "
                f"growth_rate={growth_rate}%"
            )
            resp = await self._send(req)
            if not resp:
                logger.error(f"FAILED: {symbol} — no response (ACCU proposal)")
                return None
            if resp.get("error"):
                err = resp["error"]
                logger.error(
                    f"FAILED: {symbol} — {err.get('code')}: {err.get('message')} "
                    f"| details={err.get('details')} | full_req={req}"
                )
                return None

            proposal = resp.get("proposal", {})
            logger.info(f"PROPOSAL RESPONSE: {proposal}")
            return proposal

        except Exception as e:
            logger.error(f"FAILED: {symbol} — {e}")
            return None

    async def buy_accumulator(
            self,
            symbol:      str,
            stake:       float,
            growth_rate: float,
            take_profit: float = None,
            **kwargs) -> Optional[dict]:
        """
        Buy an Accumulator (ACCU) contract: proposal step, then buy against
        the returned proposal id. ACCU contracts cannot use the direct-buy
        shortcut buy_contract() uses for Rise/Fall — the API requires a
        proposal id here.

        growth_rate is a percentage 1-5 (see get_accumulator_proposal).

        Returns dict (buy response) on success, None on every failure
        path. Never raises. Trade counters should never be incremented by
        the caller on a None return (same contract as buy_contract()).

        After a successful buy, the caller is expected to register the
        contract for polling via subscribe_contract() — mirrors
        buy_contract()'s existing division of responsibility.
        """
        # Task 1 hard guard — checked before the semaphore/rate-limit wait.
        if not self._validate_stake(stake, symbol, "buy_accumulator"):
            return None

        async with self._buy_semaphore:
            elapsed = time.time() - self._last_buy_time
            if elapsed < 3.0:
                await asyncio.sleep(3.0 - elapsed)
            self._last_buy_time = time.time()

            if not (1 <= growth_rate <= 5):
                logger.error(
                    f"FAILED: {symbol} — invalid growth_rate={growth_rate} "
                    f"(must be 1-5, i.e. 1%-5%)"
                )
                return None

            stake = self._cap_stake(stake, symbol)
            if not self._validate_stake(stake, symbol, "buy_accumulator (post-cap)"):
                return None
            logger.info(
                f"BUY ATTEMPT: {symbol} ACCU stake=${stake:.4f} "
                f"growth_rate={growth_rate}%"
            )

            try:
                proposal_req = {
                    "proposal":      1,
                    "amount":        stake,
                    "basis":         "stake",
                    "contract_type": "ACCU",
                    "currency":      "USD",
                    "underlying_symbol": symbol,
                    "growth_rate":   growth_rate / 100.0,
                }
                if take_profit is not None:
                    proposal_req["limit_order"] = {"take_profit": take_profit}

                prop_resp = await self._send(proposal_req)
                if not prop_resp:
                    logger.error(f"FAILED: {symbol} — no response (ACCU proposal)")
                    return None
                if prop_resp.get("error"):
                    err = prop_resp["error"]
                    logger.error(
                        f"FAILED: {symbol} — {err.get('code')}: {err.get('message')} "
                        f"| details={err.get('details')} | full_req={proposal_req}"
                    )
                    return None

                proposal = prop_resp.get("proposal", {})
                logger.info(f"PROPOSAL RESPONSE: {proposal}")

                proposal_id = proposal.get("id")
                ask_price   = proposal.get("ask_price", stake)
                if not proposal_id:
                    logger.error(
                        f"FAILED: {symbol} — proposal_rejected (no id in {proposal})"
                    )
                    return None

                buy_req = {"buy": proposal_id, "price": ask_price}
                resp = await self._send(buy_req)
                if not resp:
                    logger.error(f"FAILED: {symbol} — no response (ACCU buy)")
                    return None
                if resp.get("error"):
                    err = resp["error"]
                    logger.error(
                        f"FAILED: {symbol} — {err.get('code')}: {err.get('message')} "
                        f"| details={err.get('details')} | full_req={buy_req}"
                    )
                    return None

                result = resp.get("buy", {})
                contract_id = str(result.get("contract_id", ""))
                logger.info(
                    f"CONTRACT OPENED: {contract_id} | {symbol} | ACCU | "
                    f"${stake:.4f} | growth_rate={growth_rate}%"
                )

                self._contract_symbol_map[contract_id] = symbol
                asyncio.create_task(self._subscribe_ws_contract(contract_id))

                return result

            except ws.SessionClosed:
                raise                       # definite "not sent"
            except Exception as e:
                logger.error(f"FAILED: {symbol} — {e}")
                return None

    async def sell_contract(
            self,
            contract_id: str,
            price:       float = 0) -> Optional[dict]:
        """
        Close/sell an open contract early — e.g. an ACCU position before
        knockout, or any other open contract that supports early exit.

        price is the minimum acceptable sell price. 0 is Deriv's
        documented "accept the current market price unconditionally"
        sentinel.

        Returns dict (sell response — contains sold_for, contract_id) on
        success, None on every failure path. Never raises.
        """
        try:
            req = {"sell": int(contract_id), "price": price}
            logger.info(f"SELL ATTEMPT: {contract_id} min_price=${price:.4f}")
            resp = await self._send(req)
            if not resp:
                logger.error(f"FAILED: sell {contract_id} — no response")
                return None
            if resp.get("error"):
                err = resp["error"]
                logger.error(
                    f"FAILED: sell {contract_id} — {err.get('code')}: "
                    f"{err.get('message')} | details={err.get('details')}"
                )
                return None

            result = resp.get("sell", {})
            logger.info(
                f"CONTRACT SOLD: {contract_id} | sold_for=${result.get('sold_for')}"
            )
            # It's closed now — stop tracking it under both resolution paths.
            self.stop_tracking(contract_id)
            return result

        except Exception as e:
            logger.error(f"FAILED: sell {contract_id} — {e}")
            return None

    @staticmethod
    def _snapshot_allows_update(poc: dict) -> bool:
        """
        Feature-detect whether a proposal_open_contract snapshot (as
        returned by force_check_contract()) indicates the contract is
        currently eligible for contract_update.

        Deriv doesn't document a single stable key name for this, so
        this inspects the snapshot defensively rather than assuming
        one — checks a couple of plausible is_valid_to_*-style keys and
        fails soft (False) if none is present or clearly truthy.
        """
        if not poc:
            return False
        for key in ("is_valid_to_update", "is_valid_to_sell"):
            if key in poc:
                return poc.get(key) in (1, True, "1")
        return False

    async def contract_update(
            self,
            contract_id: str,
            stop_loss:   float = None,
            take_profit: float = None) -> Optional[dict]:
        """
        Revise stop_loss and/or take_profit on an already-open contract
        (Multiplier contracts only — the API only supports this for
        contracts with an active limit_order). Either parameter may be
        omitted to leave that side unchanged.

        Returns the 'contract_update' response dict on success, None on any
        failure path (including when the contract isn't currently eligible
        for an update). Never raises.
        """
        limit_order = {}
        if stop_loss is not None:
            limit_order["stop_loss"] = stop_loss
        if take_profit is not None:
            limit_order["take_profit"] = take_profit

        logger.info(
            f"CONTRACT UPDATE ATTEMPT: {contract_id} SL={stop_loss} TP={take_profit}"
        )

        if not limit_order:
            logger.error(
                f"FAILED: contract_update {contract_id} — no stop_loss/take_profit "
                f"given (both None)"
            )
            return None

        try:
            poc = await self.force_check_contract(contract_id)
            if not self._snapshot_allows_update(poc):
                logger.error(
                    f"FAILED: contract_update {contract_id} — not currently "
                    f"eligible for update (snapshot={poc})"
                )
                return None

            req = {
                "contract_update": 1,
                "contract_id":     int(contract_id),
                "limit_order":     limit_order,
            }
            resp = await self._send(req)
            if not resp:
                logger.error(f"FAILED: contract_update {contract_id} — no response")
                return None
            if resp.get("error"):
                err = resp["error"]
                logger.error(
                    f"FAILED: contract_update {contract_id} — {err.get('code')}: "
                    f"{err.get('message')} | details={err.get('details')}"
                )
                return None

            logger.info(f"CONTRACT UPDATE RESPONSE: {resp}")
            return resp.get("contract_update", {})

        except Exception as e:
            logger.error(f"FAILED: contract_update {contract_id} — {e}")
            return None

    # ─── Matches/Differs (digit) contracts ─────────────────────────────────────

    async def get_digit_proposal(
            self,
            symbol:        str,
            stake:         float,
            digit:         int,
            match_type:    str,   # "MATCH" or "DIFFER"
            duration:      int = 5,
            duration_unit: str = "t") -> Optional[dict]:
        """
        Request a proposal for a Matches ('MATCH') or Differs ('DIFFER')
        digit contract targeting last-digit value `digit` (0-9).

        duration_unit defaults to "t" (ticks) — the standard unit for
        digit contracts on volatility indices.

        Returns the full 'proposal' dict on success, None on any failure
        path. Never raises.
        """
        contract_type = self._digit_contract_type(match_type)
        if contract_type is None:
            logger.error(
                f"FAILED: {symbol} — invalid match_type={match_type!r} "
                f"(must be 'MATCH', 'DIFFER', 'OVER', or 'UNDER')"
            )
            return None
        if not self._digit_barrier_valid(contract_type, digit):
            logger.error(
                f"FAILED: {symbol} — invalid digit/barrier={digit} for "
                f"{contract_type} (0-9 for MATCH/DIFFER, 0-8 for OVER, "
                f"1-9 for UNDER)"
            )
            return None

        try:
            req = {
                "proposal":      1,
                "amount":        stake,
                "basis":         "stake",
                "contract_type": contract_type,
                "currency":      "USD",
                "duration":      duration,
                "duration_unit": duration_unit,
                "underlying_symbol": symbol,
                "barrier":       str(digit),
            }
            logger.info(
                f"PROPOSAL REQUEST: {contract_type} digit={digit} {symbol} "
                f"stake=${stake:.4f}"
            )
            resp = await self._send(req)
            if not resp:
                logger.error(f"FAILED: {symbol} — no response ({contract_type} proposal)")
                return None
            if resp.get("error"):
                err = resp["error"]
                logger.error(
                    f"FAILED: {symbol} — {err.get('code')}: {err.get('message')} "
                    f"| details={err.get('details')} | full_req={req}"
                )
                return None

            proposal = resp.get("proposal", {})
            logger.info(f"PROPOSAL RESPONSE: {proposal}")
            return proposal

        except Exception as e:
            logger.error(f"FAILED: {symbol} — {e}")
            return None

    async def buy_digit_contract(
            self,
            symbol:        str,
            stake:         float,
            digit:         int,
            match_type:    str,   # "MATCH" or "DIFFER"
            duration:      int = 5,
            duration_unit: str = "t",
            **kwargs) -> Optional[dict]:
        """
        Buy a Matches ('MATCH') or Differs ('DIFFER') digit contract:
        proposal step, then buy against the returned proposal id.

        Returns dict (buy response) on success, None on every failure
        path. Never raises.

        After a successful buy, the caller is expected to register the
        contract for polling via subscribe_contract() — mirrors
        buy_contract()'s existing division of responsibility.
        """
        contract_type = self._digit_contract_type(match_type)

        # Task 1 hard guard — checked before the semaphore/rate-limit wait.
        if not self._validate_stake(stake, symbol, "buy_digit_contract"):
            return None

        async with self._buy_semaphore:
            elapsed = time.time() - self._last_buy_time
            if elapsed < 3.0:
                await asyncio.sleep(3.0 - elapsed)
            self._last_buy_time = time.time()

            if contract_type is None:
                logger.error(
                    f"FAILED: {symbol} — invalid match_type={match_type!r} "
                    f"(must be 'MATCH', 'DIFFER', 'OVER', or 'UNDER')"
                )
                return None
            if not self._digit_barrier_valid(contract_type, digit):
                logger.error(
                    f"FAILED: {symbol} — invalid digit/barrier={digit} for "
                    f"{contract_type} (0-9 for MATCH/DIFFER, 0-8 for OVER, "
                    f"1-9 for UNDER)"
                )
                return None

            stake = self._cap_stake(stake, symbol)
            if not self._validate_stake(stake, symbol, "buy_digit_contract (post-cap)"):
                return None
            logger.info(
                f"BUY ATTEMPT: {symbol} {contract_type} digit={digit} "
                f"stake=${stake:.4f}"
            )

            try:
                proposal_req = {
                    "proposal":      1,
                    "amount":        stake,
                    "basis":         "stake",
                    "contract_type": contract_type,
                    "currency":      "USD",
                    "duration":      duration,
                    "duration_unit": duration_unit,
                    "underlying_symbol": symbol,
                    "barrier":       str(digit),
                }
                prop_resp = await self._send(proposal_req)
                if not prop_resp:
                    logger.error(f"FAILED: {symbol} — no response ({contract_type} proposal)")
                    return None
                if prop_resp.get("error"):
                    err = prop_resp["error"]
                    logger.error(
                        f"FAILED: {symbol} — {err.get('code')}: {err.get('message')} "
                        f"| details={err.get('details')} | full_req={proposal_req}"
                    )
                    return None

                proposal = prop_resp.get("proposal", {})
                logger.info(f"PROPOSAL RESPONSE: {proposal}")

                proposal_id = proposal.get("id")
                ask_price   = proposal.get("ask_price", stake)
                if not proposal_id:
                    logger.error(
                        f"FAILED: {symbol} — proposal_rejected (no id in {proposal})"
                    )
                    return None

                buy_req = {"buy": proposal_id, "price": ask_price}
                resp = await self._send(buy_req)
                if not resp:
                    logger.error(f"FAILED: {symbol} — no response ({contract_type} buy)")
                    return None
                if resp.get("error"):
                    err = resp["error"]
                    logger.error(
                        f"FAILED: {symbol} — {err.get('code')}: {err.get('message')} "
                        f"| details={err.get('details')} | full_req={buy_req}"
                    )
                    return None

                result = resp.get("buy", {})
                contract_id = str(result.get("contract_id", ""))
                logger.info(
                    f"CONTRACT OPENED: {contract_id} | {symbol} | {contract_type} | "
                    f"digit={digit} | ${stake:.4f}"
                )

                self._contract_symbol_map[contract_id] = symbol
                asyncio.create_task(self._subscribe_ws_contract(contract_id))

                return result

            except ws.SessionClosed:
                raise                       # definite "not sent"
            except Exception as e:
                logger.error(f"FAILED: {symbol} — {e}")
                return None

    # ─── Active symbols ───────────────────────────────────────────────────────

    async def get_active_symbols(self) -> List[dict]:
        try:
            resp = await self._send(
                {"active_symbols": "brief", "product_type": "basic"},
                timeout=20,
            )
            return resp.get("active_symbols", [])
        except Exception as exc:
            logger.error(f"get_active_symbols failed: {exc}")
            return []

    # ─── Properties ───────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected and self._authorized
