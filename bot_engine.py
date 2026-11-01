"""
bot_engine.py – lean engine: the walk-forward digit predictor is the ONLY strategy.

One long-lived process, two exact-second sessions per hour (window_scheduler): connected to
Deriv only from a short connect lead before a session until a bounded drain after it, fully
disconnected in between. run() is the session loop (IDLE > CONNECTING > WARMING > LIVE > DRAINING).
Self-healing inside a session is in-process: the Deriv client reconnects with backoff, a watchdog
re-subscribes / reconnects when a tick stream goes quiet, the predictor is re-synced after every
reconnect and re-fitted periodically, and the whole engine is restarted by main.py if anything
fatal escapes.

Replaces the old multi-strategy engine. There are no trading gates here: no kill
switch, no loss/drawdown/streak limits, no cooldowns, no news filter, no early
exits (digit contracts run to expiry). Only technical necessities remain:
reconnect (deriv_client), auth, rate-limit backoff, error logging, the session gate
(window_scheduler.buy_allowed_at) and draining open contracts after a session closes.

Per tick, per symbol:  rank() the tradable contexts -> live proposal (exact
contract, flat stake) -> payout/stake >= MIN_PAYOUT_RATIO and p_lower > 1/ratio
+ EDGE_MARGIN -> buy. Anything else = no trade.
"""
import asyncio
import collections
import logging
import statistics
import time
from typing import Dict, List, Optional

import config
import predictor as pred
import main_money
import predictor_stats
import target_guard
import window_scheduler as ws
from deriv_client import DerivClient

try:                                    # the single pre-A redeploy guard; a missing module never holds the connect
    import redeploy_guard as _redeploy_guard
except Exception:                       # pragma: no cover
    _redeploy_guard = None

try:                                    # follower system is optional; the main account never depends on it
    import db as _follower_db
except Exception:                       # pragma: no cover
    _follower_db = None
try:
    import follower_manager as _follower_mgr
except Exception:                       # pragma: no cover
    _follower_mgr = None

logger = logging.getLogger(__name__)


def _cfg(name, default):
    return getattr(config, name, default)


def _hold_preconnect() -> bool:
    """redeploy_guard.hold_preconnect(); False if the module is missing or raises."""
    if _redeploy_guard is None:
        return False
    try:
        return bool(_redeploy_guard.hold_preconnect())
    except Exception:
        return False


def _ts_or_zero(x) -> float:
    """Epoch seconds for the dashboard keys: 0 when unknown / infinite (schedule-off 'ALL' session)."""
    try:
        x = float(x)
    except Exception:
        return 0.0
    return x if x == x and abs(x) != float("inf") else 0.0


class BotEngine:
    def __init__(self):
        self.client = DerivClient()
        self.symbols: List[str] = list(_cfg("PREDICTOR_SYMBOLS", ["R_100"]))
        self.stake = float(_cfg("PREDICTOR_STAKE", 0.35))     # MINIMUM stake (env) - also the probe stake
        self.money = main_money.MainMoney()                  # profit-securing money rules (floor / stake sizing)
        self.duration = int(_cfg("CONTRACT_DURATION_TICKS", 1))
        self.stats = predictor_stats.get_stats()
        self.predictors: Dict[str, pred.Predictor] = {}
        self._decimals: Dict[str, int] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._busy: set = set()
        self._open: Dict[str, dict] = {}
        self._ratios: Dict[str, Dict[pred.Candidate, float]] = {}
        self._recent_epochs: Dict[str, List[float]] = {}
        self._last_epoch: Dict[str, float] = {}
        self._lag_obs: List[int] = []
        self._last_decision = "—"
        self.balance = 0.0
        self._target_done = asyncio.Event()   # set once the TARGET_BALANCE shutdown has finished
        self._session_over = False            # session closed: no buys until the next session starts
        self.session: Optional[ws.Session] = None   # the active session (None while IDLE)
        self._booted = False                  # True once the boot fit (_init_symbols) has succeeded in this process
        self._last_drain_end = 0.0            # drain_end_ts of the last session we handled (never repeat a session)
        self._lead_t0 = 0.0
        self._lead_seconds = 0.0
        self._lead_stage = ""
        self._feed_followers = True           # SetEngine turns this off (no follower stats / mirroring)
        # ── self-healing state ──
        self._last_tick_wall: Dict[str, float] = {}
        self._live: Dict[str, collections.deque] = {}      # recent (epoch, digit) seen live
        self._seen_conn = 0                                 # client.connection_count we are synced to
        self._last_forced_reconnect = 0.0
        # Deriv sometimes answers a buy-by-proposal-id with "Unknown contract proposal".
        # After the first such failure we buy with the contract parameters directly
        # (buy="1" + parameters), which does not depend on a proposal id at all.
        self._direct_buy = False
        self._worker_started: set = set()      # symbols whose worker an _init_symbols() already started
        self._early_tasks: list = []
        # ── performance tracking (log only; nothing here ever blocks a trade) ──
        self._n_trades = 0
        self._n_wins = 0
        self._be_sum = 0.0            # sum of break-even win probabilities (1/ratio) of trades taken
        self._fair_mean = 0.0         # sum of expected profit (in stakes) if every digit were fair
        self._fair_var = 0.0          # sum of variance (in stakes^2) under the same assumption
        self._total_pnl = 0.0
        self._stakes_won = 0.0        # sum of profit/stake (stakes vary with the money rules)

    # ── lifecycle: the session loop (contract section 4) ─────────────────────
    async def run(self):
        """IDLE -> CONNECTING -> WARMING -> LIVE -> DRAINING -> IDLE, once per session, forever. Returns only
        when the balance target is reached; a crash propagates so main.py rebuilds the engine."""
        logger.info("=" * 64)
        logger.info("  Digit predictor bot  –  min stake %.2f (PREDICTOR_STAKE), symbols %s", self.stake, self.symbols)
        logger.info("=" * 64)
        scheduled = bool(_cfg("SESSION_SCHEDULE_ENABLED", 1))
        if not scheduled:
            logger.warning("SESSION_SCHEDULE_ENABLED=0: one endless session 'ALL' (dev override), no schedule")
        first_pass = True
        while True:
            if target_guard.is_reached() or self._target_done.is_set():
                return                                      # target already reached: never connect again
            if scheduled and not ws.schedule_ok():
                logger.critical("SESSION SCHEDULE INCONSISTENT (ws.schedule_ok() is False): failing closed, "
                                "no connection")
                await asyncio.sleep(5)
                continue
            sess = self._pick_session(first_pass, scheduled)
            first_pass = False
            self._publish_idle(sess)
            if not await self._wait_for_lead(sess):
                continue
            if await self._run_session(sess):
                return

    def _pick_session(self, first_pass: bool, scheduled: bool) -> "ws.Session":
        now = ws.now()
        if not scheduled:
            return ws.Session("ALL", now, float("inf"), now)
        ref = max(now, self._last_drain_end)                # never repeat a session we already handled
        if first_pass:
            cur = ws.current_session(now)
            if cur is not None and not _cfg("RESUME_TRADING_MID_SESSION_AFTER_BOOT", False):
                logger.warning("BOOT INSIDE SESSION %s (closes in %.0fs): not trading it, waiting for the next "
                               "session", cur.name, cur.close_ts - now)
                ref = max(ref, cur.drain_end_ts)
        sess = ws.upcoming_session(ref)
        if sess.close_ts <= now:                            # its close has passed (drain only): skip it
            sess = ws.upcoming_session(max(ref, sess.drain_end_ts))
        return sess

    async def _wait_for_lead(self, sess: "ws.Session") -> bool:
        """IDLE: sleep (no socket, no Deriv traffic) until connect_ts; honour redeploy_guard.hold_preconnect()."""
        await self._sleep_until(sess.connect_ts)
        warned = 0.0
        while _hold_preconnect():
            if ws.now() >= sess.close_ts:
                logger.critical("SESSION %s MISSED: redeploy_guard held the connection until after the close",
                                sess.name)
                self._last_drain_end = max(self._last_drain_end, sess.drain_end_ts)
                return False
            if time.time() - warned >= 30:
                warned = time.time()
                logger.warning("SESSION %s: connect held by redeploy_guard (deploy in flight)", sess.name)
            await asyncio.sleep(1.0)
        return True

    async def _sleep_until(self, ts: float, watch: Optional[list] = None, connected: bool = False) -> Optional[str]:
        """Sleep until Deriv-server time `ts`: one computed sleep, then 5-50 ms steps inside the last 2 s
        (never a flat 5 s nap). While connected it also watches for a finished target shutdown ('target') and
        re-raises a crash of any watched task."""
        cap = 1.0 if (connected or watch) else 30.0
        while True:
            if watch:
                self._check_watch(watch)
            if connected and self._target_done.is_set():
                return "target"
            rem = ts - ws.now()
            if rem <= 0:
                return None
            if rem > 2.0:
                await asyncio.sleep(max(0.005, min(rem - 2.0, cap)))
            else:
                await asyncio.sleep(min(0.05, max(0.005, rem / 2.0)))

    @staticmethod
    def _check_watch(tasks: list):
        for t in tasks:
            if t.done() and not t.cancelled() and t.exception() is not None:
                raise t.exception()

    # ── status publishing (local only, never Deriv) ──────────────────────────
    def _publish(self, **kw):
        try:
            import keep_alive
            keep_alive.update_status(**kw)
        except Exception:
            pass

    def _sess_keys(self, sess: "ws.Session") -> dict:
        return {"session_name": sess.name, "session_open_ts": _ts_or_zero(sess.open_ts),
                "session_close_ts": _ts_or_zero(sess.close_ts), "session_connect_ts": _ts_or_zero(sess.connect_ts)}

    def _publish_idle(self, sess: "ws.Session"):
        self._publish(session_state="IDLE", session="IDLE - disconnected until next session",
                      session_lead_stage="", open_contracts_count=len(self._open), **self._sess_keys(sess))

    def _set_state(self, state: str, sess: "ws.Session", stage: Optional[str] = None):
        kw = self._sess_keys(sess)
        if stage is not None:
            self._lead_stage = stage
            kw["session_lead_stage"] = stage
        self._publish(session_state=state, session=f"{state} (session {sess.name})", **kw)

    def _set_stage(self, stage: str):
        self._lead_stage = stage
        self._publish(session_lead_stage=stage)

    # ── one session ──────────────────────────────────────────────────────────
    async def _run_session(self, sess: "ws.Session") -> bool:
        """Connect, warm, trade, drain, disconnect. True = the balance target was reached (run() returns)."""
        self.session = sess
        self._session_over = False
        self._lead_t0 = time.time()
        tasks: list = []
        ws_task: Optional[asyncio.Task] = None
        try:
            # ── 2. CONNECT ──
            self._set_state("CONNECTING", sess, stage="connect")
            ws.reset_clock_measured()
            self.client.set_reconnect_enabled(True)
            ws_task = await self._connect_in_lead(sess)
            if ws_task is None:
                return False                                # session ran out before we could connect
            self.balance = self.client.balance
            first = not self._booted
            # Target check BEFORE anything is warmed up: if the balance is already at/above TARGET_BALANCE
            # (e.g. service resumed without raising the target) never trade.
            if await self._target_gate(first):
                return True
            if first:
                try:
                    await self.money.start(self.client)     # saved floor/P, live balance, midnight reset
                except Exception as exc:
                    logger.error("MAIN MONEY could not start (%s) - the bot will not trade until it can", exc)
                self.client.on_balance(self._on_balance)
                self._seen_conn = self._conn_count()
            sup = asyncio.create_task(self._supervise_ws(ws_task))
            tasks.append(sup)

            # ── 3. WARMING ──
            self._set_state("WARMING", sess, stage="clock sync")
            warm = asyncio.create_task(self._warm(sess, first))
            tasks.append(warm)
            logged = False
            while not warm.done():
                self._check_watch([sup, warm])
                if self._target_done.is_set():
                    return True
                if not logged and sess.name != "ALL" and ws.now() >= sess.open_ts - 2.0:
                    logged = True                           # never open the gate early to compensate
                    pend = []
                    if not warm.done():
                        pend.append(f"lead stage '{self._lead_stage}'")
                    if not ws.clock_measured():
                        pend.append("clock sync")
                    if self._needs_resync():
                        pend.append("resync after reconnect")
                    if not self.client.is_connected:
                        pend.append("connection")
                    logger.critical("SESSION %s LEAD NOT FINISHED at open-2s: still pending: %s",
                                    sess.name, ", ".join(pend) or "?")
                await asyncio.wait({warm}, timeout=0.2)
            self._check_watch([warm])
            budget = float(_cfg("WINDOW_PRECONNECT_SECONDS", 120.0))
            self._lead_seconds = time.time() - self._lead_t0
            self._set_stage("ready")
            self._publish(session_lead_seconds=round(self._lead_seconds, 1))
            logger.info("LEAD DONE in %.1fs (budget %.0fs)", self._lead_seconds, budget)
            # in-lead self-healing runs before the open too: a drop between 'ready' and open is resynced
            for s in self.symbols:
                self._last_tick_wall[s] = time.time()
            tasks.append(asyncio.create_task(self._watchdog_loop()))
            tasks.append(asyncio.create_task(self._refit_loop()))

            # ── 4. LIVE ──
            if await self._sleep_until(sess.open_ts, watch=[sup], connected=True) == "target":
                return True
            self._set_state("LIVE", sess, stage="ready")
            if not ws.clock_measured():
                logger.critical("SESSION %s OPEN WITHOUT A CLOCK SAMPLE this session: falling back to the local "
                                "clock (offset 0, error assumed +/-1s)", sess.name)
                ws.set_clock_offset(0.0, 1.0)
            n_samples = int(getattr(self.client, "clock_samples", 0) or 0)
            logger.info("SESSION %s OPEN: offset %+.2fs, error +/-%.2fs, samples %d",
                        sess.name, ws.get_clock_offset(), ws.get_clock_error(), n_samples)
            self._publish(clock_offset=round(ws.get_clock_offset(), 3), clock_error=round(ws.get_clock_error(), 3))
            self._on_session_open(sess)
            for s in self.symbols:
                self._last_tick_wall[s] = time.time()
            tasks += [asyncio.create_task(self._worker(s)) for s in self.symbols if s not in self._worker_started]
            tasks += list(self._early_tasks)                # started by a boot fit during the lead
            tasks += [asyncio.create_task(self._settle_loop()),
                      asyncio.create_task(self._flush_loop()),
                      asyncio.create_task(self._dashboard_loop()),
                      asyncio.create_task(self._target_monitor_loop()),
                      asyncio.create_task(self._clock_resync_loop())]
            tasks += [asyncio.create_task(c) for c in self._extra_tasks()]

            # ── 5. CLOSE, DRAIN, DISCONNECT ──
            if await self._sleep_until(sess.close_ts, watch=tasks, connected=True) == "target":
                return True
            self._session_over = True                       # _blocked() is now True: no new proposals/buys
            drain = float(_cfg("SESSION_DRAIN_SECONDS", 10.0))
            logger.warning("SESSION %s CLOSED: no new trades; draining %d open contract(s) (max %.0fs)",
                           sess.name, len(self._open), drain)
            try:
                self._on_session_close(sess)
            except Exception as exc:
                logger.error("_on_session_close failed: %s", exc, exc_info=True)
            self._set_state("DRAINING", sess)
            drain_end = sess.close_ts + drain
            while (self._open or self._busy) and ws.now() < drain_end:   # also in-flight attempts: a buy frame
                #                                                         # already sent must not lose its reply
                self._check_watch(tasks)
                if self._target_done.is_set():
                    return True
                await asyncio.sleep(min(0.5, max(0.01, drain_end - ws.now())))
            try:
                await asyncio.wait_for(self._before_disconnect(sess), timeout=30)
            except asyncio.CancelledError:
                raise
            except Exception as exc:                        # incl. TimeoutError: bounded, never blocks the disconnect
                logger.error("_before_disconnect failed/timed out: %r", exc)
            try:
                await asyncio.wait_for(self.stats.flush(force=True), timeout=5)
            except Exception as exc:
                logger.warning("session end: stats flush skipped: %s", exc)
            return False
        finally:
            await self._disconnect(sess, tasks, ws_task)

    async def _connect_in_lead(self, sess: "ws.Session") -> Optional["asyncio.Task"]:
        """Start the client and wait until it is connected + authorised. Failures (token/OTP/Deriv down) are
        retried with backoff for as long as the session lasts; nothing is ever attempted after the close."""
        backoff = 1.0
        ws_task: Optional[asyncio.Task] = None
        while ws.now() < sess.close_ts:
            if ws_task is not None and ws_task.done():
                err = None if ws_task.cancelled() else ws_task.exception()
                logger.error("SESSION %s CONNECT failed (%r) - retrying in %.0fs", sess.name, err, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
                ws_task = None
                try:
                    await self.client.shutdown("connect retry")
                except Exception:
                    pass
                self.client.set_reconnect_enabled(True)
                if ws.now() >= sess.close_ts:
                    break
            if ws_task is None:
                ws_task = asyncio.create_task(self.client.connect())
            t_end = time.time() + 20.0
            while time.time() < t_end and not ws_task.done():
                if self.client.is_connected:
                    return ws_task
                await asyncio.sleep(0.1)
            if self.client.is_connected:
                return ws_task
            if not ws_task.done():
                logger.warning("SESSION %s CONNECT: not ready after 20s (client keeps retrying) - still waiting",
                               sess.name)
        logger.critical("SESSION %s MISSED: could not connect before the close", sess.name)
        if ws_task is not None:
            ws_task.cancel()
            await asyncio.gather(ws_task, return_exceptions=True)
        return None

    async def _target_gate(self, first: bool) -> bool:
        """True when the balance target is already reached (shutdown done): the session must not trade."""
        if target_guard.target() is None:
            if first:
                logger.info("TARGET GUARD: TARGET_BALANCE not set - no balance target")
            return False
        try:
            self.balance = await self.client.fetch_balance_live()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Initial live balance fetch failed (%s) - using subscribed value", exc)
        logger.info("TARGET GUARD: target=%.2f  live balance=%.2f", target_guard.target(), self.balance)
        if target_guard.should_stop(self.balance):
            if target_guard.mark_reached(self.balance):
                await self._shutdown_for_target(self.balance)
            return True
        return False

    async def _warm(self, sess: "ws.Session", first: bool):
        """Everything slow, finished BEFORE open_ts: clock burst, then boot fit (first session after process
        start) or the _lead_prepare hook (later sessions)."""
        await self._sync_clock_lead()
        if first:
            self._set_stage("refit")
            ready = await self._init_symbols()
            self.symbols = ready
            if not self.symbols:
                raise RuntimeError("no tradable symbol could be started")
            logger.info("SYMBOLS ACTIVE (%d): %s", len(self.symbols), ",".join(self.symbols))
            self._booted = True
        else:
            await self._lead_prepare(sess)
            if self._needs_resync():                        # hook did not resync (plain BotEngine): do it here
                await self._resync_in_lead()

    async def _sync_clock_lead(self):
        """The lead's clock burst (fine: also catch the integer-second flip). Retries; a total failure is only
        logged here - open_ts then logs CRITICAL and falls back to the local clock."""
        for attempt in (1, 2, 3):
            try:
                await self.client.sync_clock(fine=True)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("CLOCK SYNC attempt %d failed: %s", attempt, exc)
                await asyncio.sleep(1.0)

    async def _resync_in_lead(self):
        """Fallback gap resync for engines whose _lead_prepare does not do one: refit + re-subscribe every symbol."""
        self._set_stage("gap resync")
        conn_now = self._conn_count()
        ok_all = True
        for sym in self.symbols:
            try:
                ok_all = bool(await self._fit_symbol(sym)) and ok_all
                await self.client.refresh_ticks(sym, lambda tick, s=sym: self._on_tick(s, tick))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                ok_all = False
                logger.error("lead resync error %s: %s", sym, exc)
        if ok_all and self._conn_count() == conn_now:
            self._seen_conn = conn_now

    async def _clock_resync_loop(self):
        """Light clock resync while connected (per-session task)."""
        interval = float(_cfg("SESSION_CLOCK_RESYNC_SECONDS", 300.0))
        if interval <= 0:
            return
        samples = max(3, int(_cfg("SESSION_CLOCK_SYNC_SAMPLES", 10)) // 3)
        while True:
            await asyncio.sleep(interval)
            if not self.client.is_connected:
                continue
            try:
                await self.client.sync_clock(samples=samples)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("clock resync failed: %s", exc)

    async def _disconnect(self, sess: "ws.Session", tasks: list, ws_task: Optional["asyncio.Task"]):
        """Cancel EVERY session task, stop reconnecting, drop the socket. Runs on every exit path (also a crash)."""
        self._session_over = True
        pool = [t for t in tasks + list(self._early_tasks) if t is not None]
        for t in pool:
            t.cancel()
        try:
            await asyncio.gather(*pool, return_exceptions=True)
        finally:
            self._early_tasks = []
            self._worker_started.clear()
            self._busy.clear()
        try:
            self.client.set_reconnect_enabled(False)
        except Exception as exc:
            logger.warning("set_reconnect_enabled(False) failed: %s", exc)
        try:
            await self.client.shutdown("session closed")
        except Exception as exc:
            logger.warning("client shutdown error: %s", exc)
        if ws_task is not None:
            ws_task.cancel()
            await asyncio.gather(ws_task, return_exceptions=True)
        self.session = None
        self._last_drain_end = max(self._last_drain_end, sess.drain_end_ts)
        logger.warning("SESSION %s ENDED: disconnected from Deriv (%d contract(s) left to settle on Deriv's side)",
                       sess.name, len(self._open))

    async def _init_symbols(self) -> list:
        """Initialise all symbols one after another (MHEngine overrides this with a concurrent boot)."""
        ready = []
        for sym in list(self.symbols):
            ok = False
            for attempt in (1, 2):                  # one retry: a hiccup must not drop a good symbol
                try:
                    await self._init_symbol(sym)
                    ok = True
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("SYMBOL %s start-up attempt %d failed: %s", sym, attempt, exc)
                    await asyncio.sleep(3)
            if ok:
                ready.append(sym)
            else:
                logger.error("SYMBOL SKIPPED %s: no tick history / no digit contract could be priced "
                             "(not offered on this account?)", sym)
                for d in (self.predictors, self._events, self._live, self._ratios, self._decimals,
                          self._last_tick_wall, self._recent_epochs):
                    d.pop(sym, None)
        return ready

    def _extra_tasks(self) -> list:
        """Coroutines a subclass wants run next to the standard loops (none for the original engine).
        Called once per session at OPEN; every task made from them is cancelled at disconnect."""
        return []

    # ── session hooks (no-op defaults; MHEngine overrides) ───────────────────
    async def _lead_prepare(self, sess: "ws.Session") -> None:
        """Runs connected, inside the lead, before open: refit, gap resync, quotes (not for the first session
        after process start, which runs the boot fit instead)."""
        return

    async def _before_disconnect(self, sess: "ws.Session") -> None:
        """After the drain, right before the socket is dropped: reconcile in-doubt buys, ledger, flush."""
        return

    def _on_session_open(self, sess: "ws.Session") -> None:
        """At open_ts, before the workers start: reset per-session bookkeeping."""
        return

    def _on_session_close(self, sess: "ws.Session") -> None:
        """At close_ts, before the drain."""
        return

    def _pr_desc(self, pr) -> str:
        return f"tradable contexts={len(pr.selected)}"

    # ── warm-up ──────────────────────────────────────────────────────────────
    async def _init_symbol(self, sym: str):
        self._events[sym] = asyncio.Event()
        self._live[sym] = collections.deque(maxlen=300)
        self._last_tick_wall[sym] = time.time()
        await self._fit_symbol(sym, initial=True)
        if not self._recent_epochs.get(sym):
            raise RuntimeError("no tick history")
        if not self._ratios.get(sym):
            raise RuntimeError("no digit contract could be priced")
        await self.client.subscribe_ticks(sym, lambda tick, s=sym: self._on_tick(s, tick))
        pr = self.predictors[sym]
        logger.info("READY %s: decimals=%d horizon=%d %s",
                    sym, self._decimals[sym], pr.p.horizon, self._pr_desc(pr))

    async def _fit_symbol(self, sym: str, initial: bool = False) -> bool:
        """Download recent ticks, re-probe payouts and (re)fit the predictor. Used for the
        first warm-up and for every later re-sync/refit. On a refit that cannot get enough
        history the old predictor is kept untouched and False is returned."""
        n = int(_cfg("WARMUP_TICKS", 5000))
        ticks = await self.client.get_tick_history(sym, n)
        if len(ticks) < 500:
            logger.warning("WARM-UP %s: only %d ticks returned", sym, len(ticks))
        if not initial and len(ticks) < 300:
            return False
        quotes = [t["quote"] for t in ticks]
        if sym not in self._decimals:
            dec_map = _cfg("DIGIT_DECIMALS", {}) or {}
            dec = dec_map.get(sym)
            if dec is None:
                dec = pred.infer_decimals(quotes) if quotes else 2
            self._decimals[sym] = int(dec)
        digits = pred.digits_from_quotes(quotes, self._decimals[sym])
        last_ep = ticks[-1]["epoch"] if ticks else None
        self._recent_epochs[sym] = [t["epoch"] for t in ticks[-50:]]
        pr = self.predictors.get(sym)
        if pr is None:
            self._ratios.setdefault(sym, {})
            pr = pred.Predictor(sym, payout_ratio_fn=lambda c, s=sym: self._ratios.get(s, {}).get(c))
        await self._probe_ratios(sym, pr)
        pr.fit(digits, last_epoch=last_ep)
        # ticks that arrived live while we were downloading/probing are not in `ticks`: replay them
        if last_ep is not None:
            for ep, d in list(self._live.get(sym, ())):
                if ep > last_ep and pr.on_tick(d, ep):
                    self._recent_epochs[sym].append(ep)
        pred.register(sym, pr)
        self.predictors[sym] = pr
        return True

    async def _probe_ratios(self, sym: str, pr: pred.Predictor):
        """Live payout/stake for candidates that could clear the floor, so
        selection uses real break-evens (never a hardcoded payout)."""
        floor = pr.p.min_payout_ratio
        ceil_ = pr.p.max_payout_ratio
        cands = [c for c in pr.candidates
                 if pred.estimated_payout_ratio(c, pr.p.payout_factor) >= floor * 0.95
                 and (ceil_ <= 0 or pred.estimated_payout_ratio(c, pr.p.payout_factor) <= ceil_ * 1.05)]
        sem = asyncio.Semaphore(4)

        async def one(c):
            async with sem:
                prop = await self._propose(sym, c, self.stake)
                if prop and prop.get("payout"):
                    self._ratios[sym][c] = float(prop["payout"]) / self.stake
        await asyncio.gather(*(one(c) for c in cands), return_exceptions=True)
        logger.info("PAYOUT PROBE %s: %s", sym,
                    {f"{k[0]}{'' if k[1] is None else k[1]}": round(v, 3) for k, v in self._ratios[sym].items()})

    # ── ticks ────────────────────────────────────────────────────────────────
    def _on_tick(self, sym: str, tick: dict):
        try:
            q, ep = float(tick["quote"]), float(tick.get("epoch", time.time()))
        except Exception:
            return
        self._last_tick_wall[sym] = time.time()
        pr = self.predictors.get(sym)
        if pr is None:
            return
        digit = pred.last_digit(q, self._decimals[sym])
        if not pr.on_tick(digit, ep):
            return
        self._live[sym].append((ep, digit))
        self._last_epoch[sym] = ep
        eps = self._recent_epochs.setdefault(sym, [])
        eps.append(ep)
        del eps[:-50]
        self._events[sym].set()

    def _conn_count(self) -> int:
        return int(getattr(self.client, "connection_count", 0))

    def _needs_resync(self) -> bool:
        """True after a (re)connect until the predictor has been re-synced: ticks may have
        been missed, so the digit context would be wrong."""
        return self._conn_count() != self._seen_conn

    def _next_stake(self) -> Optional[float]:
        """Stake from the money rules for the next buy; None = stopped (floor reached) or not ready."""
        return self.money.next_stake()

    def _blocked(self) -> bool:
        # Only technical blocks remain: no socket, session over, outside the buy window (Deriv server clock),
        # balance target reached, or the predictor not yet re-synced after a reconnect (a few seconds).
        return (not self.client.is_connected or self._session_over or not ws.buy_allowed_at(ws.now())
                or target_guard.is_reached() or self._needs_resync())

    def _gate_buy(self):
        """Engine-side twin of the DerivClient send-time gate: called right before a buy frame is handed to the
        client. Raises ws.SessionClosed ('definitely not sent') when a buy would go out at/after the close."""
        if not ws.buy_allowed_at(ws.now()):
            raise ws.SessionClosed("buy not sent: outside the session buy window")

    # ── trading ──────────────────────────────────────────────────────────────
    async def _worker(self, sym: str):
        ev = self._events[sym]
        while True:
            await ev.wait()
            ev.clear()
            if self._blocked() or sym in self._busy:
                continue
            self._busy.add(sym)
            try:
                await self._try_trade(sym)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("trade attempt error %s: %s", sym, exc, exc_info=True)
            finally:
                self._busy.discard(sym)

    def _contract_params(self, sym: str, cand: pred.Candidate, stake: float, duration: Optional[int] = None) -> dict:
        p = {"amount": stake, "basis": "stake",
             "contract_type": pred.contract_type(cand), "currency": "USD",
             "duration": int(self.duration if duration is None else duration), "duration_unit": "t", "underlying_symbol": sym}
        if cand[1] is not None:
            p["barrier"] = str(cand[1])
        return p

    async def _buy_direct(self, sym: str, cand: pred.Candidate, price: float, stake: Optional[float] = None,
                          duration: Optional[int] = None, timeout: float = 10) -> dict:
        """Buy without a proposal id: {"buy": "1", "price": max_price, "parameters": {...}}."""
        self._gate_buy()                                    # a decision at 16:00.9 is never sent after 16:01.000
        return await self.client._send(
            {"buy": "1", "price": price, "parameters": self._contract_params(sym, cand, self.stake if stake is None else stake, duration)},
            timeout=timeout)

    async def _propose(self, sym: str, cand: pred.Candidate, stake: float, duration: Optional[int] = None) -> Optional[dict]:
        req = {"proposal": 1, **self._contract_params(sym, cand, stake, duration)}
        try:
            resp = await self.client._send(req, timeout=10)
            return resp.get("proposal") or None
        except Exception as exc:
            await self._on_api_error("proposal", exc)
            return None

    async def _on_api_error(self, what: str, exc: Exception):
        msg = str(exc)
        logger.warning("Deriv %s rejected: %s", what, msg[:300])
        if "rate" in msg.lower() or "RateLimit" in msg:
            await asyncio.sleep(2.0)                       # rate-limit backoff

    async def _try_trade(self, sym: str):
        pr = self.predictors[sym]
        picks = pr.rank()
        if not picks:
            return
        tick_epoch = self._last_epoch.get(sym)
        if tick_epoch is not None and not ws.in_window(float(tick_epoch)):
            return                                          # the signal tick itself is outside the session
        stake = self._next_stake()                          # money rules: None = stopped / floor reached
        if stake is None:
            self._last_decision = "stopped by money rules (floor reached) - resumes at next reset"
            return
        for pk in picks[: int(_cfg("PREDICTOR_MAX_PROPOSALS", 3))]:
            if self._blocked():
                return
            prop = await self._propose(sym, pk.cand, stake)
            if not prop:
                continue
            payout = float(prop.get("payout") or 0.0)
            ratio = payout / stake                          # live ratio, never hardcoded
            self._ratios[sym][pk.cand] = ratio
            dec = pr.accept(pk, ratio)
            tag = f"{pk.cand[0]}{'' if pk.cand[1] is None else pk.cand[1]}"
            if not dec.ok:
                self._last_decision = f"skip {tag}: {dec.reason}"
                logger.info("SKIP %s %s: %s", sym, tag, dec.reason)
                continue
            if self._blocked():
                return
            ask = float(prop.get("ask_price") or stake)
            if not self.money.reserve(stake):               # hard rule: balance - stake must stay >= floor
                self._last_decision = "skip: buying would breach the profit floor"
                return
            try:
                if self._direct_buy:
                    resp = await self._buy_direct(sym, pk.cand, ask, stake)
                else:
                    self._gate_buy()                        # before the frame is built/sent (retry path gates itself)
                    try:
                        resp = await self.client._send({"buy": prop["id"], "price": ask}, timeout=10)
                    except Exception as exc:
                        if "InvalidContractProposal" not in str(exc) and "Unknown contract proposal" not in str(exc):
                            raise
                        logger.warning("Buy by proposal id was refused (%s) - retrying with a direct buy "
                                       "(buy=1 + parameters)", str(exc)[:120])
                        resp = await self._buy_direct(sym, pk.cand, ask, stake)
                        self._direct_buy = True
                        logger.info("Direct buy worked - using it for all further trades")
            except ws.SessionClosed as exc:                 # definitely NOT sent: free the money, no error handling
                self.money.release(stake)
                self._last_decision = "buy blocked at session close"
                logger.warning("BUY BLOCKED AT CLOSE %s: %s", sym, exc)
                return
            except Exception as exc:
                self.money.release(stake)
                await self._on_api_error("buy", exc)        # error handling, not a gate
                return
            buy = resp.get("buy") or {}
            self.money.note_buy(buy)
            self.money.release(stake)
            cid = str(buy.get("contract_id", ""))
            if not cid:
                logger.warning("BUY returned no contract_id: %s", resp)
                return
            opened = float(buy.get("purchase_time") or time.time())
            if _follower_mgr is not None:    # mirror to follower accounts: non-blocking, never raises
                try:
                    _follower_mgr.publish_signal({
                        "signal_id": cid, "symbol": sym, "contract_type": pred.contract_type(pk.cand),
                        "barrier": pk.cand[1], "duration": int(self.duration), "signal_time": time.time()})
                except Exception:
                    pass
            self._open[cid] = {"sym": sym, "cand": pk.cand, "stake": stake, "ratio": ratio,
                               "opened_at": opened, "decision_epoch": tick_epoch, "t0": time.time(),
                               "ctx": pk.context, "p_lower": pk.p_lower}
            self._last_decision = f"BUY {tag} ctx={pk.context} lb={pk.p_lower:.3f} ratio={ratio:.3f}"
            logger.info("CONTRACT OPENED %s %s %s ctx=%s n=%d p=%.3f lb=%.3f ratio=%.3f be=%.4f edge=%.4f",
                        cid, sym, tag, pk.context, pk.n, pk.p_hat, pk.p_lower, ratio, dec.break_even, dec.edge)
            self.client._contract_symbol_map[cid] = sym
            try:
                await self.client.subscribe_contract(
                    cid, lambda msg, _c=cid: asyncio.create_task(self._on_contract_result(_c, msg)),
                    symbol=sym)
            except Exception as exc:
                logger.warning("subscribe_contract(%s) failed (settle loop will poll): %s", cid, exc)
            return

    # ── self-healing ─────────────────────────────────────────────────────────
    async def _supervise_ws(self, ws_task: "asyncio.Task"):
        """connect() runs for the whole session. If it ends while the session is still live, raise so
        main.py rebuilds the whole engine instead of leaving a zombie that is 'running' but never trades."""
        await asyncio.wait({ws_task})                       # not `await ws_task`: cancelling us must not cancel it
        if target_guard.is_reached() or self._session_over or self._target_done.is_set():
            return                                          # intentional shutdown, not a crash
        exc = None if ws_task.cancelled() else ws_task.exception()
        raise RuntimeError("Deriv WebSocket task ended unexpectedly") from exc

    # ── balance target ───────────────────────────────────────────────────────
    def _on_balance(self, bal: float):
        """Runs on EVERY live balance update from Deriv (inside the event loop)."""
        self.balance = bal
        self.money.on_balance(bal)
        if target_guard.should_stop(bal) and target_guard.mark_reached(bal):
            # flag is already set (no new trade can start); do the slow part as a task
            asyncio.get_running_loop().create_task(self._shutdown_for_target(bal))

    async def _target_monitor_loop(self):
        """Safety net next to the balance stream: poll Deriv's live balance directly."""
        while not target_guard.is_reached():
            await asyncio.sleep(target_guard.check_seconds())
            if target_guard.target() is None or not self.client.is_connected:
                continue
            try:
                await self.client.fetch_balance_live()     # fires _on_balance -> target check
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("target monitor: balance poll failed: %s", exc)

    async def _shutdown_for_target(self, bal: float):
        """Target hit: (flag already set) -> drop the Deriv socket -> flush stats -> suspend Render."""
        try:
            logger.warning("TARGET SHUTDOWN: balance %.2f, %d contract(s) still open on Deriv "
                           "(they settle on Deriv's side; not waiting)", bal, len(self._open))
            try:
                import keep_alive
                keep_alive.update_status(session="TARGET REACHED - STOPPED",
                                         open_contracts_count=len(self._open))
            except Exception:
                pass
            try:
                self.client.set_reconnect_enabled(False)
            except Exception:
                pass
            await self.client.shutdown("balance target reached")
            try:
                await asyncio.wait_for(self.stats.flush(force=True), timeout=5)
            except Exception as exc:
                logger.warning("TARGET SHUTDOWN: stats flush skipped: %s", exc)
            await asyncio.to_thread(target_guard.suspend_service)
        except Exception as exc:
            logger.error("TARGET SHUTDOWN error: %s", exc, exc_info=True)
        finally:
            self._target_done.set()

    async def _watchdog_loop(self):
        """Tick streams can die silently while the socket (and its pings) look fine."""
        stale = float(_cfg("STALE_TICK_SECONDS", 30))
        min_gap = float(_cfg("FORCE_RECONNECT_MIN_GAP", 90))
        strikes = {s: 0 for s in self.symbols}
        grace_until = {s: 0.0 for s in self.symbols}
        while True:
            await asyncio.sleep(5)
            now = time.time()
            if not self.client.is_connected:
                for s in self.symbols:                       # downtime is not "stale"
                    self._last_tick_wall[s] = now
                    strikes[s] = 0
                continue
            for sym in self.symbols:
                quiet = now - self._last_tick_wall.get(sym, now)
                if quiet < stale:                            # a real tick arrived: healthy
                    strikes[sym] = 0
                    continue
                if now < grace_until[sym]:                   # give the last fix time to work
                    continue
                try:
                    if strikes[sym] == 0:
                        logger.warning("WATCHDOG %s: no tick for %.0fs - re-subscribing", sym, quiet)
                        strikes[sym] = 1
                        grace_until[sym] = now + stale
                        await self.client.refresh_ticks(sym, lambda tick, s=sym: self._on_tick(s, tick))
                    elif now - self._last_forced_reconnect >= min_gap:
                        logger.warning("WATCHDOG %s: still no tick after re-subscribe - reconnecting", sym)
                        self._last_forced_reconnect = now
                        strikes[sym] = 0
                        grace_until[sym] = now + stale
                        await self.client.force_reconnect(f"no ticks for {sym}")
                except Exception as exc:
                    logger.error("watchdog error %s: %s", sym, exc)

    async def _refit_loop(self):
        """Replaces what the periodic redeploy did implicitly: fresh history, fresh payout
        probe, fresh selection - plus an immediate re-sync after any reconnect."""
        interval = float(_cfg("REFIT_INTERVAL_SECONDS", 300))
        last = time.time()
        while True:
            await asyncio.sleep(5)
            if not self.client.is_connected:
                continue
            resync = self._needs_resync()
            if not resync and not (interval > 0 and time.time() - last >= interval):
                continue
            conn_now = self._conn_count()
            ok_all = True
            for sym in self.symbols:
                try:
                    ok = await self._fit_symbol(sym)
                    ok_all = ok_all and ok
                    if ok:
                        pr = self.predictors[sym]
                        logger.info("%s %s: %s",
                                    "RESYNC" if resync else "REFIT", sym, self._pr_desc(pr))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    ok_all = False
                    logger.error("refit error %s: %s", sym, exc)
            if ok_all:
                if resync and self._conn_count() == conn_now:
                    self._seen_conn = conn_now
                last = time.time()

    # ── settlement ───────────────────────────────────────────────────────────
    async def _on_contract_result(self, cid: str, msg: dict):
        poc = msg.get("proposal_open_contract", msg)
        if not (poc.get("is_sold") or poc.get("is_expired") or poc.get("status") in ("sold", "won", "lost")):
            return
        info = self._open.pop(cid, None)
        if not info:
            return
        try:
            self.client.stop_tracking(cid)
        except Exception:
            pass
        profit = float(poc.get("profit") or 0.0)
        won = profit > 0
        self.stats.record(info["opened_at"], info["stake"], info["ratio"], profit)
        if _follower_db is not None and self._feed_followers:   # batched DB write; never blocks, never raises
            try:
                _follower_db.record_collector_trade(info["opened_at"], info["stake"], info["ratio"], profit)
            except Exception:
                pass
        tag = f"{info['cand'][0]}{'' if info['cand'][1] is None else info['cand'][1]}"
        logger.info("CONTRACT SETTLED %s %s %s %s profit=%.2f", cid, info["sym"], tag,
                    "WIN" if won else "LOSS", profit)
        self._note_lag(info, poc)
        self._after_trade(info, profit, won)
        try:
            import keep_alive
            keep_alive.record_trade(info["sym"], tag, info["stake"], profit,
                                    balance_after=self.client.balance, won=won, strategy="PREDICTOR")
        except Exception:
            pass

    # ── performance log (no limits, no pauses) ───────────────────────────────
    def _after_trade(self, info: dict, profit: float, won: bool):
        self._n_trades += 1
        self._n_wins += 1 if won else 0
        ratio = max(float(info.get("ratio") or 1.0), 1.0001)
        self._be_sum += 1.0 / ratio
        self._total_pnl += profit
        self._stakes_won += profit / max(float(info.get("stake") or self.stake), 1e-9)
        # Edge test: compare actual results with a FAIR game (true digit probabilities, real payout).
        # z near 0 = results are what luck alone produces; z > 3 over many trades = real edge.
        try:
            p_true = pred.true_probability(info["cand"])
            self._fair_mean += p_true * ratio - 1.0
            self._fair_var += p_true * (1.0 - p_true) * ratio * ratio
        except Exception:
            pass
        if self._n_trades % 50 == 0:
            n = self._n_trades
            stakes_won = self._stakes_won
            z = (stakes_won - self._fair_mean) / (self._fair_var ** 0.5) if self._fair_var > 0 else 0.0
            logger.info("PERFORMANCE after %d trades: win rate %.1f%% vs break-even %.1f%% | total P&L %+.2f "
                        "| expected for a fair game %+.2f | EDGE z-score %+.2f (needs > 3 to be an edge)",
                        n, 100.0 * self._n_wins / n, 100.0 * self._be_sum / n, self._total_pnl,
                        self._fair_mean * self.stake, z)

    def _note_lag(self, info: dict, poc: dict):
        """Observed entry delay in ticks vs the configured horizon (log only)."""
        try:
            entry = float(poc.get("entry_tick_time") or 0)
            dec = info.get("decision_epoch")
            eps = self._recent_epochs.get(info["sym"], [])
            if not entry or not dec or len(eps) < 5:
                return
            interval = statistics.median([b - a for a, b in zip(eps, eps[1:]) if b > a] or [1])
            gap = round((entry - dec) / interval)
            self._lag_obs.append(gap)
            # entry tick should be decision tick + 1 + ENTRY_DELAY_TICKS
            expected = 1 + self.predictors[info["sym"]].p.entry_delay_ticks
            if gap != expected:
                logger.warning("LAG CHECK: entry tick was %d ticks after the decision tick, model assumes %d. "
                               "Set PREDICTOR_ENTRY_DELAY_TICKS=%d if this persists.",
                               gap, expected, max(0, gap - 1))
        except Exception:
            pass

    async def _settle_loop(self):
        """Poll contracts that did not report their result (stragglers). Per-session task: stops at disconnect."""
        while True:
            await asyncio.sleep(2)
            if not self.client.is_connected:
                continue
            now = time.time()
            for cid, info in list(self._open.items()):         # poll stragglers
                if now - info["t0"] > 20:
                    poc = await self.client.force_check_contract(cid)
                    if poc:
                        await self._on_contract_result(cid, {"proposal_open_contract": poc})

    async def _flush_loop(self):
        while True:
            await asyncio.sleep(5)
            try:
                await self.stats.flush()
            except Exception as exc:
                logger.error("stats flush error: %s", exc)

    async def _dashboard_loop(self):
        while True:
            await asyncio.sleep(5)
            try:
                import keep_alive
                keep_alive.update_status(
                    balance=round(self.client.balance, 2), session="PREDICTOR",
                    current_symbol=",".join(self.symbols), last_signal=self._last_decision,
                    tradeable_count=sum(len(p.selected) for p in self.predictors.values()),
                    open_contracts_count=len(self._open),
                    target_balance=target_guard.target(), money=self.money.snapshot())
                keep_alive.set_active_trades(len(self._open))
            except Exception:
                pass
