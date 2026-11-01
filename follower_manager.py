"""
follower_manager.py - runs the follower accounts in their OWN thread with their OWN event loop.

Isolation: the collector only calls publish_signal() (non-blocking, never raises) and runs
collector_time_sync() (one tiny {"time":1} request a minute). If this module, the DB or any
follower is down, the collector is untouched.

Flow
  * Deriv server time (offset measured on the collector's connection) drives everything.
  * ONE zone snapshot per UTC hour is saved to the DB before the hour begins (30 s lead).
    Zones come only from that snapshot, so nothing can be fitted after the fact. A session
    (contiguous traded minutes) whose snapshot was saved after the session had started is
    never traded.
  * ~5 s before a session the enabled followers connect (fresh OTP); signals are mirrored
    from the session start; opening stops at the session end; followers disconnect once their
    open contracts have settled (30 s grace) and whenever no zone is active.
  * After a redeploy mid-session: the hour's snapshot is read back from the DB (never
    recomputed). Followers stake a flat amount (PREDICTOR_STAKE) on every buy.
Nothing here claims an edge exists; the 53-59 window is a hypothesis under test.
"""
import asyncio
import logging
import statistics
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import db
import follower_config as fc
import token_crypto
import zone_engine as ze
from follower_client import FollowerClient

logger = logging.getLogger("follower.manager")

_MGR: Optional["Manager"] = None
_LOOP: Optional[asyncio.AbstractEventLoop] = None
_server_offset = 0.0
_offsets: deque = deque(maxlen=5)

METRICS: Dict[str, object] = {
    "running": False, "signals": 0, "dropped_old": 0, "not_in_zone": 0, "buys_sent": 0,
    "buy_errors": 0, "connected": 0, "lag_avg_ms": 0.0, "lag_max_ms": 0.0,
    "delay_avg_ms": None, "delay_max_ms": None, "offset_s": 0.0, "plans": 0}
_lags: deque = deque(maxlen=120)
_delays: deque = deque(maxlen=100)


def server_now() -> float:
    return time.time() + _server_offset


# ── collector-side helpers (called from the collector's thread / loop) ───────
def publish_signal(sig: dict) -> None:
    """Thread-safe, non-blocking, never raises. No-op when the follower side is not running."""
    try:
        loop, mgr = _LOOP, _MGR
        if loop is None or mgr is None or not loop.is_running():
            return
        loop.call_soon_threadsafe(mgr.on_signal, sig)
    except Exception:
        pass


async def collector_time_sync(client) -> None:
    """Runs on the collector's loop: keeps a Deriv-server-time offset (median of 5 samples)."""
    global _server_offset
    while True:
        try:
            if client.is_connected:
                t0 = time.time()
                resp = await client._send({"time": 1}, timeout=5)
                t1 = time.time()
                st = float(resp.get("time"))
                off = st - (t0 + t1) / 2.0
                if abs(off) < 3600:
                    _offsets.append(off)
                    new = statistics.median(_offsets)
                    if abs(new - _server_offset) > 0.5:
                        logger.info("FOLLOWER server-time offset %+.2fs", new)
                    _server_offset = new
                    METRICS["offset_s"] = round(new, 3)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        await asyncio.sleep(60)


# ── hourly plan (from a saved snapshot) ──────────────────────────────────────
class HourPlan:
    def __init__(self, hour_key: int, snap_id: int, created_ts: float, plan: dict):
        self.hour_key = hour_key
        self.snap_id = snap_id
        self.created_ts = created_ts
        self.tags: List[Optional[str]] = plan.get("tags") or [None] * 60
        self.zones = plan.get("zones") or []
        self.sessions: List[dict] = []
        for s in ze.sessions_from_tags(self.tags):
            start_ts = hour_key * 3600 + s["start"] * 60
            end_ts = hour_key * 3600 + (s["end"] + 1) * 60
            # pre-registration: the snapshot must exist BEFORE the session opens
            self.sessions.append({"start": s["start"], "end": s["end"], "start_ts": start_ts,
                                  "end_ts": end_ts, "valid": created_ts <= start_ts})


# ── one follower account ─────────────────────────────────────────────────────
class FollowerRuntime:
    def __init__(self, mgr: "Manager", row: dict, token: str, mode: str):
        self.mgr = mgr
        self.id = int(row["id"])
        self.label = row.get("label") or f"#{self.id}"
        self.client = FollowerClient(self.label, token, row.get("app_id") or "", mode)
        self.last_balance: Optional[float] = row.get("last_balance")
        self.result_text: Optional[str] = row.get("last_window_result")
        self.ready = False
        self.failed = False
        self.disabled = False
        self._init_done = False
        self._task: Optional[asyncio.Task] = None
        self._pending = 0.0           # stakes sent in this burst but not yet acknowledged
        self._inflight = 0
        self._backoff_until = 0.0
        self._fails = 0
        self._stop_logged = False
        self.open: Dict[str, dict] = {}
        self._win: Dict[str, List[float]] = {}     # window_id -> [trades, wins, pnl]
        self._last_win: Optional[str] = None
        self._dirty = False
        self._persist_task: Optional[asyncio.Task] = None

    # -- lifecycle ----------------------------------------------------------
    @property
    def open_count(self) -> int:
        return len(self.open) + self._inflight

    async def start(self) -> None:
        c = self.client
        c.on_balance(self._on_balance)
        self._task = asyncio.create_task(c.connect())
        t0 = time.time()
        while time.time() - t0 < 25 and not c.is_connected and not self._task.done():
            await asyncio.sleep(0.1)
        if not c.is_connected:
            await self._fail("could not connect/authorize")
            return
        try:
            bal = await c.fetch_balance_live(timeout=8)
        except Exception as exc:
            await self._fail(f"live balance failed: {str(exc)[:80]}")
            return
        self.last_balance = bal
        self._init_done = True
        self.ready = True
        logger.info("FOLLOWER %s ready: balance=%.2f flat stake=%.2f", self.label, bal, fc.STAKE)

    async def ensure_reset(self) -> None:
        """Kept for the manager's call sites: just refreshes the cached balance."""
        if not self.ready:
            return
        try:
            self.last_balance = await self.client.fetch_balance_live(timeout=6)
        except Exception as exc:
            logger.warning("FOLLOWER %s balance refresh failed: %s", self.label, str(exc)[:80])

    async def stop(self) -> None:
        self.ready = False
        try:
            await self._persist_now()
        except Exception:
            pass
        try:
            await asyncio.wait_for(self.client.shutdown("session over"), timeout=6)
        except Exception:
            pass
        if self._task is not None:
            self._task.cancel()
        logger.info("FOLLOWER %s disconnected%s", self.label,
                    f" ({self.result_text})" if self.result_text else "")

    async def _fail(self, reason: str) -> None:
        self.failed = True
        self.ready = False
        logger.warning("FOLLOWER %s disabled for this session: %s", self.label, reason)
        try:
            await asyncio.wait_for(self.client.shutdown(reason), timeout=5)
        except Exception:
            pass
        if self._task is not None:
            self._task.cancel()

    # -- state ----------------------------------------------------------------
    def _on_balance(self, bal: float) -> None:
        self.last_balance = bal

    def _state_dict(self) -> dict:
        return {"p": None, "peak": None, "floor": None, "tier": None,
                "last_reset_date": None, "milestone_hit": False,
                "last_balance": self.last_balance, "last_window_result": self.result_text}

    async def _persist_now(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, db.save_follower_state, self.id,
                                                         self._state_dict())

    def _persist_soon(self) -> None:
        self._dirty = True
        if self._persist_task is None or self._persist_task.done():
            self._persist_task = asyncio.ensure_future(self._persist_loop())

    async def _persist_loop(self) -> None:
        while self._dirty:
            self._dirty = False
            try:
                await self._persist_now()
            except Exception as exc:
                logger.warning("FOLLOWER %s state persist error: %s", self.label, str(exc)[:80])

    # -- mirroring --------------------------------------------------------------
    async def mirror(self, sig: dict, ctx: dict) -> None:
        if not self.ready or self.disabled:
            return
        now = time.time()
        if now - sig["signal_time"] > fc.SIGNAL_MAX_AGE or now < self._backoff_until:
            METRICS["dropped_old"] = int(METRICS["dropped_old"]) + 1
            return
        if self._inflight >= fc.FOLLOWER_CONCURRENCY:
            return
        c = self.client
        self._inflight += 1
        reserved = 0.0
        try:
            if not c.is_connected:
                return
            if c.balance_age() > fc.BALANCE_MAX_AGE:
                try:
                    await c.fetch_balance_live(timeout=3)
                except Exception:
                    return                                   # cannot verify the balance -> no buy
                if time.time() - sig["signal_time"] > fc.SIGNAL_MAX_AGE:
                    METRICS["dropped_old"] = int(METRICS["dropped_old"]) + 1
                    return
            bal = c.balance - self._pending
            stake = fc.STAKE
            if bal < stake - 1e-9:                           # cannot pay for the buy: Deriv would reject it
                if not self._stop_logged and not self.open and self._pending == 0:
                    self._stop_logged = True
                    logger.info("FOLLOWER %s waiting: balance %.2f < stake %.2f", self.label, bal, stake)
                return
            self._stop_logged = False
            self._pending += stake
            reserved = stake
            params = {"amount": stake, "basis": "stake", "contract_type": sig["contract_type"],
                      "currency": fc.CURRENCY, "duration": int(sig["duration"]), "duration_unit": "t",
                      "underlying_symbol": sig["symbol"]}
            if sig.get("barrier") is not None:
                params["barrier"] = str(sig["barrier"])
            sent = time.time()
            METRICS["buys_sent"] = int(METRICS["buys_sent"]) + 1
            resp = await c._send({"buy": "1", "price": stake, "parameters": params}, timeout=8)
            buy = resp.get("buy") or {}
            cid = str(buy.get("contract_id") or "")
            if not cid:
                raise RuntimeError("buy returned no contract_id")
            if buy.get("balance_after") is not None:
                c.note_balance_after_buy(float(buy["balance_after"]))
            self._fails = 0
            price = float(buy.get("buy_price") or stake)
            payout = float(buy.get("payout") or 0.0)
            ratio = (payout / price) if payout and price else None
            _delays.append((sent - sig["signal_time"]) * 1000.0)
            self.open[cid] = {"stake": price, "wid": ctx["window_id"]}
            db.queue_op({"op": "ftrade_open", "follower_id": self.id, "contract_id": cid,
                         "signal_id": sig["signal_id"], "symbol": sig["symbol"],
                         "contract_type": sig["contract_type"],
                         "barrier": None if sig.get("barrier") is None else str(sig["barrier"]),
                         "stake": price, "payout_ratio": ratio,
                         "opened_at": float(buy.get("purchase_time") or sent), "source": ctx["source"],
                         "snapshot_id": ctx["snapshot_id"], "window_id": ctx["window_id"]})
            w = self._win.setdefault(ctx["window_id"], [0, 0, 0.0])
            w[0] += 1
            self._last_win = ctx["window_id"]
            try:
                await c.subscribe_contract(
                    cid, lambda msg, _c=cid: asyncio.ensure_future(self._on_settle(_c, msg)),
                    symbol=sig["symbol"])
            except Exception as exc:
                logger.warning("FOLLOWER %s subscribe %s failed (poller will settle): %s",
                               self.label, cid, str(exc)[:80])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            METRICS["buy_errors"] = int(METRICS["buy_errors"]) + 1
            self._fails += 1
            msg = str(exc)
            if "rate" in msg.lower():
                self._backoff_until = time.time() + fc.RATE_LIMIT_BACKOFF
            if "timed out" in msg.lower():
                c._balance_ts = 0.0                          # unknown outcome: force a fresh balance next time
            logger.warning("FOLLOWER %s buy failed (%d in a row): %s", self.label, self._fails, msg[:120])
            if self._fails >= fc.MAX_CONSECUTIVE_FAILS:
                self.disabled = True
                logger.error("FOLLOWER %s disabled for this session after %d failures", self.label, self._fails)
        finally:
            self._pending = max(0.0, self._pending - reserved)
            self._inflight -= 1

    async def _on_settle(self, cid: str, msg: dict) -> None:
        poc = msg.get("proposal_open_contract", msg)
        if not (poc.get("is_sold") or poc.get("is_expired") or poc.get("status") in ("sold", "won", "lost")):
            return
        info = self.open.pop(cid, None)
        if info is None:
            return
        try:
            self.client.stop_tracking(cid)
        except Exception:
            pass
        profit = float(poc.get("profit") or 0.0)
        db.queue_op({"op": "ftrade_settle", "follower_id": self.id, "contract_id": cid, "profit": profit})
        w = self._win.setdefault(info["wid"], [0, 0, 0.0])
        w[1] += 1 if profit > 0 else 0
        w[2] += profit
        self.result_text = f"{info['wid']}: {int(w[0])} trades, {int(w[1])} wins, pnl {w[2]:+.2f}"
        self._persist_soon()


# ── manager ──────────────────────────────────────────────────────────────────
class Manager:
    def __init__(self):
        self.plans: Dict[int, HourPlan] = {}
        self._plan_retry: Dict[int, float] = {}
        self.runtimes: Dict[int, FollowerRuntime] = {}
        self._attempted: set = set()
        self._connecting = False
        self._last_wanted = 0.0
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="follower-db")

    async def run(self) -> None:
        global _LOOP, _MGR
        loop = asyncio.get_running_loop()
        loop.set_default_executor(self.pool)
        _LOOP, _MGR = loop, self
        METRICS["running"] = True
        db.ensure_started()
        logger.info("FOLLOWER manager started (default zone %d..%d, pre-registered hypothesis)",
                    fc.DEFAULT_ZONE_START, fc.DEFAULT_ZONE_END)
        asyncio.ensure_future(self._lag_monitor())
        try:
            while True:
                try:
                    await self._tick()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("FOLLOWER tick error: %s", exc, exc_info=True)
                await asyncio.sleep(1.0)
        finally:
            METRICS["running"] = False
            _MGR = None

    async def _lag_monitor(self) -> None:
        while True:
            t = time.perf_counter()
            await asyncio.sleep(0.5)
            _lags.append(max(0.0, time.perf_counter() - t - 0.5) * 1000.0)
            METRICS["lag_avg_ms"] = round(sum(_lags) / len(_lags), 1)
            METRICS["lag_max_ms"] = round(max(_lags), 1)
            if _delays:
                METRICS["delay_avg_ms"] = round(sum(_delays) / len(_delays), 1)
                METRICS["delay_max_ms"] = round(max(_delays), 1)
            METRICS["connected"] = sum(1 for r in self.runtimes.values() if r.ready)

    # -- scheduling -----------------------------------------------------------
    async def _run_db(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)

    async def _tick(self) -> None:
        now = server_now()
        hk = int(now // 3600)
        await self._ensure_plan(hk, now)
        if now >= (hk + 1) * 3600 - fc.SNAPSHOT_LEAD_SECONDS:
            await self._ensure_plan(hk + 1, now)
        for k in [k for k in self.plans if k < hk - 1]:
            self.plans.pop(k, None)
        METRICS["plans"] = len(self.plans)
        active = self._wanted(now)
        if active is not None:
            self._last_wanted = now
            key = (active["hk"], active["start"])
            if key not in self._attempted and not self._connecting:
                self._attempted.add(key)
                self._connecting = True
                asyncio.ensure_future(self._connect_followers())
        else:
            await self._maybe_disconnect(now)
        if len(self._attempted) > 50:
            self._attempted = {k for k in self._attempted if k[0] >= hk - 1}

    async def _ensure_plan(self, hk: int, now: float) -> None:
        if hk in self.plans or now < self._plan_retry.get(hk, 0.0):
            return
        wid = f"H{hk}"
        try:
            snap = await self._run_db(db.get_snapshot, wid)       # resume: never recompute a saved one
            if snap is None:
                await self._run_db(db.flush_now, 3.0)
                rows = await self._run_db(db.load_minute_rows)
                plan = ze.build_plan(rows, fc.DEFAULT_ZONE_START, fc.DEFAULT_ZONE_END,
                                     fc.ZONE_MIN_TRADES, fc.ZONE_MAX_PER_HOUR, fc.SHRINK_K)
                saved = await self._run_db(db.save_snapshot, wid, plan, rows, "engine+default")
                if saved is None:
                    raise RuntimeError("snapshot not saved")
                snap = {"id": saved["id"], "created_ts": saved["created_ts"], "zones": plan}
                logger.info("FOLLOWER snapshot %s saved (id=%d): sessions=%s", wid, saved["id"],
                            [(s["start"], s["end"]) for s in plan["sessions"]])
            self.plans[hk] = HourPlan(hk, snap["id"], snap["created_ts"], snap["zones"])
        except Exception as exc:
            self._plan_retry[hk] = now + 5.0
            logger.warning("FOLLOWER snapshot %s unavailable (%s) - no trading in that hour until it is saved",
                           wid, str(exc)[:100])

    def _wanted(self, now: float) -> Optional[dict]:
        """A valid session that is active now or starts within PRE_CONNECT_SECONDS."""
        hk = int(now // 3600)
        for k in (hk, hk + 1):
            plan = self.plans.get(k)
            if not plan:
                continue
            for s in plan.sessions:
                if s["valid"] and s["start_ts"] - fc.PRE_CONNECT_SECONDS <= now < s["end_ts"]:
                    return {"hk": k, "start": s["start"]}
        return None

    def _session_ctx(self, ts: float) -> Optional[dict]:
        hk = int(ts // 3600)
        plan = self.plans.get(hk)
        if not plan:
            return None
        minute = int((ts % 3600) // 60)
        tag = plan.tags[minute]
        if tag is None:
            return None
        for s in plan.sessions:
            if s["valid"] and s["start"] <= minute <= s["end"]:
                return {"source": tag, "snapshot_id": plan.snap_id, "window_id": f"H{hk}-{s['start']}"}
        return None

    # -- followers -------------------------------------------------------------
    async def _connect_followers(self) -> None:
        try:
            rows = await self._run_db(db.list_followers, True, True)
            mode = await self._run_db(db.get_mode)
            new: List[FollowerRuntime] = []
            for r in rows:
                if r["id"] in self.runtimes:
                    continue
                token = token_crypto.decrypt(r["token_enc"]) if token_crypto.available() else None
                if not token:
                    logger.warning("FOLLOWER %s skipped: token unavailable", r.get("label"))
                    continue
                rt = FollowerRuntime(self, r, token, mode)
                self.runtimes[rt.id] = rt
                new.append(rt)
            existing = [r for r in self.runtimes.values() if r.ready and r not in new]
            if new:
                logger.info("FOLLOWER connecting %d follower(s) in %s mode", len(new), mode)
            await asyncio.gather(*(rt.start() for rt in new), *(rt.ensure_reset() for rt in existing),
                                 return_exceptions=True)
            for rid in [i for i, r in self.runtimes.items() if r.failed]:
                self.runtimes.pop(rid, None)
        except Exception as exc:
            logger.error("FOLLOWER connect error: %s", exc, exc_info=True)
        finally:
            self._connecting = False

    async def _maybe_disconnect(self, now: float) -> None:
        if not self.runtimes:
            return
        grace_over = now >= self._last_wanted + fc.SETTLE_GRACE_SECONDS
        for rid, rt in list(self.runtimes.items()):
            if rt.open_count == 0 or grace_over or rt.failed:
                self.runtimes.pop(rid, None)
                await rt.stop()

    # -- signals ----------------------------------------------------------------
    def on_signal(self, sig: dict) -> None:
        try:
            METRICS["signals"] = int(METRICS["signals"]) + 1
            if time.time() - float(sig["signal_time"]) > fc.SIGNAL_MAX_AGE:
                METRICS["dropped_old"] = int(METRICS["dropped_old"]) + 1
                return
            ctx = self._session_ctx(float(sig["signal_time"]) + _server_offset)
            if ctx is None:
                METRICS["not_in_zone"] = int(METRICS["not_in_zone"]) + 1
                return
            for rt in list(self.runtimes.values()):
                if rt.ready:
                    asyncio.ensure_future(rt.mirror(sig, ctx))
        except Exception as exc:
            logger.warning("FOLLOWER signal error: %s", exc)

    async def shutdown(self) -> None:
        for rt in list(self.runtimes.values()):
            await rt.stop()
        self.runtimes.clear()


# ── thread entry points ──────────────────────────────────────────────────────
_thread: Optional[threading.Thread] = None


def _thread_main() -> None:
    while True:
        try:
            asyncio.run(Manager().run())
        except Exception as exc:
            logger.error("FOLLOWER manager crashed: %s", exc, exc_info=True)
        logger.warning("FOLLOWER manager restarting in 5 s")
        time.sleep(5)


def start() -> bool:
    """Start the follower thread (own event loop). Safe to call once from main.py."""
    global _thread
    if _thread is not None:
        return True
    if not db.enabled():
        logger.info("FOLLOWER system OFF (DATABASE_URL not set or psycopg missing: %s)",
                    db.status().get("import_error"))
        return False
    _thread = threading.Thread(target=_thread_main, name="follower-manager", daemon=True)
    _thread.start()
    return True


def shutdown_blocking(timeout: float = 8.0) -> None:
    """Called from the SIGTERM handler: flush DB buffers (bounded)."""
    try:
        db.flush_now(timeout)
    except Exception:
        pass
