"""
test_session_lifecycle.py - BotEngine session loop (contract sections 4 and 5), fake clock + fake client. No network.

Time is virtual: a custom asyncio loop jumps straight to the next timer, and time.time() is patched to the loop clock,
so a 70-minute schedule runs in a fraction of a second and every timing assertion is exact.

The fake DerivClient follows the contract: set_reconnect_enabled(), shutdown() leaves the object reusable, a buy is
refused with ws.SessionClosed outside the buy window, every call that talks to Deriv is recorded with its Deriv-server
timestamp. window_scheduler is the real module when it is importable; otherwise a small stand-in that implements the
contract's section 3 API (same names, same semantics) is installed so this file runs on its own.

Run:  python -m unittest test_session_lifecycle -v
"""
import asyncio
import collections
import contextlib
import heapq
import importlib
import logging
import sys
import time
import types
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

SKEW = 0.3                                   # the fake Deriv server clock runs this far ahead of the local clock
H0 = (1_760_000_000 // 3600) * 3600          # an exact hour boundary (UTC)


# ── modules the engine imports: use the real ones if present, else minimal stubs ─────────────────────────────────────
def _ensure_module(name, **attrs):
    try:
        importlib.import_module(name)
    except Exception:
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m


def _make_ws_standin():
    """Contract section 3 (names and semantics), pure functions over time.time() + offset."""
    m = types.ModuleType("window_scheduler")
    LEAD, DRAIN, CUTOFF, EXTRA = 120.0, 10.0, 0.0, 1.0
    SPECS = (("A", 360, 960), ("B", 1260, 1740))
    st = {"offset": 0.0, "error": 0.0, "measured": False}

    class SessionClosed(RuntimeError):
        pass

    @dataclass(frozen=True)
    class Session:
        name: str
        open_ts: float
        close_ts: float
        connect_ts: float

        @property
        def drain_end_ts(self):
            return self.close_ts + DRAIN

    def _all(ts):
        hour = (ts // 3600) * 3600
        out = []
        for h in (hour - 3600, hour, hour + 3600, hour + 7200):
            for name, o, e in SPECS:
                out.append(Session(name, h + o, h + e + EXTRA, h + o - LEAD))
        return sorted(out, key=lambda s: s.open_ts)

    m.SessionClosed, m.Session = SessionClosed, Session
    m.set_clock_offset = lambda offset, error=0.0: st.update(offset=float(offset), error=float(error), measured=True)
    m.get_clock_offset = lambda: st["offset"]
    m.get_clock_error = lambda: st["error"]
    m.reset_clock_measured = lambda: st.update(measured=False)
    m.clock_measured = lambda: st["measured"]
    m.now = lambda: time.time() + st["offset"]
    m.schedule_ok = lambda: True

    def current_session(ts=None):
        ts = m.now() if ts is None else ts
        return next((s for s in _all(ts) if s.open_ts <= ts < s.close_ts), None)

    def upcoming_session(ts=None):
        ts = m.now() if ts is None else ts
        return next(s for s in _all(ts) if s.drain_end_ts > ts)

    def phase(ts=None):
        ts = m.now() if ts is None else ts
        for s in _all(ts):
            if s.open_ts <= ts < s.close_ts:
                return "OPEN"
            if s.close_ts <= ts < s.drain_end_ts:
                return "DRAIN"
            if s.connect_ts <= ts < s.open_ts:
                return "LEAD"
        return "IDLE"

    def buy_allowed_at(send_ts=None):
        send_ts = m.now() if send_ts is None else send_ts
        s = current_session(send_ts)
        return s is not None and send_ts < s.close_ts - CUTOFF

    m.current_session, m.upcoming_session, m.phase = current_session, upcoming_session, phase
    m.buy_allowed_at = buy_allowed_at
    m.in_window = lambda ts=None: current_session(ts) is not None
    m.should_be_connected = lambda ts=None: phase(ts) != "IDLE"
    return m


_ensure_module("config")
_ensure_module("predictor", Predictor=object, Candidate=tuple)
_ensure_module("main_money")
_ensure_module("predictor_stats")
_ensure_module("target_guard")
try:
    ws = importlib.import_module("window_scheduler")
except Exception:
    ws = _make_ws_standin()
    sys.modules["window_scheduler"] = ws
_ensure_module("deriv_client", DerivClient=object)      # the REAL one when importable (needs aiohttp + websockets)
import deriv_client  # noqa: E402
REAL_CLIENT = hasattr(deriv_client.DerivClient, "sync_clock")

_KA = SimpleNamespace(status=[])             # the fake keep_alive appends to _KA.status
_fake_keep_alive = types.ModuleType("keep_alive")
_fake_keep_alive.update_status = lambda **kw: _KA.status.append((ws.now(), kw))
_fake_keep_alive.record_trade = lambda *a, **k: None
_fake_keep_alive.set_active_trades = lambda n: None
_REAL_KEEP_ALIVE = sys.modules.get("keep_alive")
sys.modules["keep_alive"] = _fake_keep_alive

import bot_engine  # noqa: E402  (after the stubs)

logging.getLogger("bot_engine").addHandler(logging.NullHandler())


def tearDownModule():
    if _REAL_KEEP_ALIVE is not None:
        sys.modules["keep_alive"] = _REAL_KEEP_ALIVE
    else:
        sys.modules.pop("keep_alive", None)


# ── virtual time ─────────────────────────────────────────────────────────────────────────────────────────────────────
class VLoop(asyncio.SelectorEventLoop):
    """Event loop whose clock jumps to the next scheduled timer whenever nothing is ready to run."""

    def __init__(self, start):
        super().__init__()
        self._vt = float(start)
        self._clock_resolution = 1e-6        # > float ulp at epoch magnitude, so a timer due 'now' fires

    def time(self):
        return self._vt

    def _run_once(self):
        if not self._ready:
            while self._scheduled and self._scheduled[0]._cancelled:
                h = heapq.heappop(self._scheduled)
                h._scheduled = False
            if self._scheduled and self._scheduled[0]._when > self._vt:
                self._vt = self._scheduled[0]._when
        super()._run_once()


# ── fakes ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class FakeClient:
    """Contract-faithful fake DerivClient. `ops` = every call that talks to Deriv, with its server timestamp."""

    def __init__(self):
        self.is_connected = False
        self.balance = 1000.0
        self.connection_count = 0
        self._contract_symbol_map = {}
        self.ops, self.shutdowns, self.sent_buys, self.bg = [], [], [], []
        self.connect_calls = 0
        self.reconnect_enabled = True
        self.gate_enabled = True             # the client-side send-time buy gate (contract section 5)
        self.raise_connects = 0
        self.clock_fail = False
        self.proposal_latency = 0.0
        self.settle_delay = 1.0
        self.clock_samples = 0
        self._stop = False
        self._drop = None
        self._cb_balance = None

    def _net(self, name):
        self.ops.append((ws.now(), name))

    def set_reconnect_enabled(self, enabled):
        self.reconnect_enabled = bool(enabled)

    async def connect(self):
        self.connect_calls += 1
        self._stop = False
        self._drop = asyncio.Event()
        while not self._stop:
            self._net("connect")
            if self.raise_connects > 0:
                self.raise_connects -= 1
                await asyncio.sleep(0.5)
                raise ConnectionError("simulated OTP / connect failure")
            await asyncio.sleep(1.0)
            self._net("authorise")
            self.is_connected = True
            self.connection_count += 1
            await self._drop.wait()
            self.is_connected = False
            if self._stop or not self.reconnect_enabled:
                return
            await asyncio.sleep(1.0)                      # in-session reconnect (same session)
            self._drop = asyncio.Event()

    def drop(self):
        self._drop.set()

    async def shutdown(self, reason=""):
        self._stop = True
        self.is_connected = False
        self.shutdowns.append((ws.now(), reason))
        if self._drop is not None:
            self._drop.set()

    async def sync_clock(self, samples=None, fine=False):
        self._net("sync_clock")
        if self.clock_fail:
            raise RuntimeError("no usable time samples")
        await asyncio.sleep(2.5 + (4.0 if fine else 0.0))
        ws.set_clock_offset(SKEW, 0.05)
        self.clock_samples = 5
        return SKEW, 0.05

    async def _send(self, req, timeout=10):
        if not self.is_connected:
            raise ConnectionError("not connected")
        if "buy" in req:
            if self.gate_enabled and not ws.buy_allowed_at(ws.now()):
                raise ws.SessionClosed("client gate: buy at/after close")
            self._net("buy")
            self.sent_buys.append((ws.now(), req))
            return {"buy": {"contract_id": 1000 + len(self.sent_buys), "purchase_time": time.time()}}
        self._net(next(iter(req)))
        if "proposal" in req:
            if self.proposal_latency:
                await asyncio.sleep(self.proposal_latency)
            return {"proposal": {"id": "p1", "payout": 0.7, "ask_price": 0.35}}
        return {}

    async def fetch_balance_live(self):
        self._net("balance")
        if self._cb_balance:
            self._cb_balance(self.balance)
        return self.balance

    def on_balance(self, cb):
        self._cb_balance = cb

    async def get_tick_history(self, sym, n):
        self._net("history")
        return []

    async def subscribe_ticks(self, sym, cb):
        self._net("subscribe_ticks")

    async def refresh_ticks(self, sym, cb):
        self._net("refresh_ticks")

    async def subscribe_contract(self, cid, cb, symbol=None):
        self._net("subscribe_contract")

        async def later():                                # Deriv settles the contract on its side
            await asyncio.sleep(self.settle_delay)
            cb({"proposal_open_contract": {"is_sold": True, "profit": 0.3}})

        self.bg.append(asyncio.get_running_loop().create_task(later()))

    async def force_check_contract(self, cid):
        self._net("poc")
        return None

    async def force_reconnect(self, reason=""):
        self._net("force_reconnect")
        self.drop()

    def stop_tracking(self, cid):
        pass


class FakeMoney:
    def __init__(self):
        self.started = 0
        self.reserved = 0
        self.released = 0

    async def start(self, client):
        self.started += 1

    def next_stake(self):
        return 0.35

    def reserve(self, stake):
        self.reserved += 1
        return True

    def release(self, stake):
        self.released += 1

    def note_buy(self, buy):
        pass

    def on_balance(self, bal):
        pass

    def snapshot(self):
        return {}


class FakeStats:
    def record(self, *a, **k):
        pass

    async def flush(self, force=False):
        pass


class FakeTarget:
    def __init__(self):
        self.tgt, self.reached, self.suspended = None, False, False

    def target(self):
        return self.tgt

    def is_reached(self):
        return self.reached

    def should_stop(self, bal):
        return self.tgt is not None and bal >= self.tgt

    def mark_reached(self, bal):
        if self.reached:
            return False
        self.reached = True
        return True

    def check_seconds(self):
        return 2.0

    def suspend_service(self):
        self.suspended = True


class FakePredictor:
    selected = []

    def rank(self):
        return [SimpleNamespace(cand=("DIGITDIFF", 3), context="c", n=100, p_hat=0.9, p_lower=0.8)]

    def accept(self, pk, ratio):
        return SimpleNamespace(ok=True, reason="", break_even=0.5, edge=0.1)


class TEngine(bot_engine.BotEngine):
    """The real BotEngine with only the predictor/boot parts faked (_try_trade, buy paths, gate, loops are real)."""

    def __init__(self, h):
        self.h = h
        super().__init__()
        self.symbols = ["R_100"]

    async def _init_symbols(self):
        self.h.counts["boot_fit"] += 1
        await asyncio.sleep(20.0)
        sym = "R_100"
        self._events[sym] = asyncio.Event()
        self._live[sym] = collections.deque(maxlen=10)
        self._last_tick_wall[sym] = time.time()
        self._ratios[sym] = {}
        self.predictors[sym] = FakePredictor()
        await self.client.subscribe_ticks(sym, lambda tick: None)
        return [sym]

    async def _lead_prepare(self, sess):
        self.h.counts["lead_prepare"] += 1
        await asyncio.sleep(10.0)
        await self.client.get_tick_history("R_100", 10)
        self._seen_conn = self._conn_count()              # what MHEngine's gap resync does

    async def _fit_symbol(self, sym, initial=False):
        self.h.counts["fit"] += 1
        await self.client.get_tick_history(sym, 10)
        return True

    def _contract_params(self, sym, cand, stake, duration=None):
        return {"amount": stake, "contract_type": cand[0], "underlying_symbol": sym}

    def _extra_tasks(self):
        return [self._statement_poll()]

    async def _statement_poll(self):                      # stands in for account_history.run
        while True:
            await asyncio.sleep(1.0)
            if self.client.is_connected:
                await self.client._send({"statement": 1})


class Harness:
    def __init__(self, start, ticker=None, target=None, config=None):
        self.start, self.ticker_every = start, ticker
        ws.set_clock_offset(0.0, 0.0)
        ws.reset_clock_measured()
        _KA.status = []
        self.loop = VLoop(start)
        self.counts = collections.Counter()
        self.client, self.money, self.target = FakeClient(), FakeMoney(), FakeTarget()
        if target is not None:
            self.target.tgt = target
        self.config = config or {}
        self.config.setdefault("cfg", {}).setdefault("STALE_TICK_SECONDS", 1e9)   # the fake feeds no ticks: keep the watchdog quiet
        self.eng = self.engine_task = None
        self.leaked = None

    @property
    def status(self):
        return _KA.status

    def live_events(self):
        return [(ts, kw) for ts, kw in self.status if kw.get("session_state") == "LIVE"]

    def fire(self):
        self.eng._events["R_100"].set()

    async def at(self, server_ts):
        while ws.now() < server_ts:                       # re-evaluated often: the offset changes at the clock sync
            await asyncio.sleep(max(0.001, min(0.5, server_ts - ws.now())))

    def run(self, until, driver=None):
        with contextlib.ExitStack() as st:
            st.enter_context(mock.patch("time.time", lambda: self.loop._vt))
            st.enter_context(mock.patch.object(bot_engine, "DerivClient", lambda: self.client))
            st.enter_context(mock.patch.object(bot_engine, "main_money", SimpleNamespace(MainMoney=lambda: self.money)))
            st.enter_context(mock.patch.object(bot_engine, "predictor_stats", SimpleNamespace(get_stats=FakeStats)))
            st.enter_context(mock.patch.object(bot_engine, "target_guard", self.target))
            st.enter_context(mock.patch.object(bot_engine, "pred", SimpleNamespace(
                contract_type=lambda c: c[0], true_probability=lambda c: 0.9, Candidate=tuple)))
            st.enter_context(mock.patch.object(bot_engine, "_follower_mgr", None))
            st.enter_context(mock.patch.object(bot_engine, "_follower_db", None))
            st.enter_context(mock.patch.object(bot_engine, "_redeploy_guard", self.config.get("redeploy_guard")))

            async def _to_thread(fn, *a, **k):
                return fn(*a, **k)
            st.enter_context(mock.patch("asyncio.to_thread", _to_thread))
            for name, val in self.config.get("cfg", {}).items():
                st.enter_context(mock.patch.object(bot_engine.config, name, val, create=True))
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_until_complete(self._main(until, driver))
            finally:
                for t in asyncio.all_tasks(self.loop):
                    t.cancel()
                self.loop.run_until_complete(asyncio.sleep(0))
                self.loop.close()
                asyncio.set_event_loop(None)
        return self

    async def _main(self, until, driver):
        self.eng = TEngine(self)
        self.engine_task = asyncio.create_task(self.eng.run())
        main = self._main_task = asyncio.current_task()
        side = self.side = []
        if self.ticker_every:
            async def ticker():
                while True:
                    await asyncio.sleep(self.ticker_every)
                    ev = self.eng._events.get("R_100")
                    if ev is not None:
                        ev.set()
            side.append(asyncio.create_task(ticker()))
        if driver is not None:
            side.append(asyncio.create_task(driver(self)))
        while ws.now() < until and not self.engine_task.done():
            await asyncio.sleep(min(1.0, max(0.001, until - ws.now())))
        if self.engine_task.done() and not self.engine_task.cancelled() and self.engine_task.exception():
            raise self.engine_task.exception()
        self.engine_ended = self.engine_task.done()
        for t in side:
            if t.done() and not t.cancelled() and t.exception():
                raise t.exception()
            t.cancel()
        self.engine_task.cancel()
        await asyncio.gather(self.engine_task, *side, return_exceptions=True)

    def leaked_tasks(self):
        """Tasks still alive that are neither the engine run loop, the driver/ticker, nor the fake server's own."""
        keep = {self.engine_task, self._main_task, asyncio.current_task(), *self.side, *self.client.bg}
        return [t for t in asyncio.all_tasks(self.loop) if t not in keep and not t.done()]


def sched():
    a = ws.upcoming_session(H0)
    b = ws.upcoming_session(a.drain_end_ts)
    a2 = ws.upcoming_session(b.drain_end_ts)
    return a, b, a2


# ── tests ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class SessionLifecycleTests(unittest.TestCase):

    def test_idle_has_zero_deriv_traffic_and_client_is_reusable(self):
        """Two consecutive sessions on ONE client object: no connect/send/authorise/balance/statement call while
        phase() is IDLE, shutdown leaves the client reusable, no task survives a disconnect."""
        A, B, A2 = sched()
        leak_probe = {}

        async def drv(h):
            await h.at(B.drain_end_ts + 60)               # deep in the idle gap after session B
            leak_probe["tasks"] = [t.get_coro().__qualname__ for t in h.leaked_tasks()]
            leak_probe["connected"] = h.client.is_connected
            leak_probe["ops_n"] = len(h.client.ops)

        h = Harness(A.drain_end_ts + 30, ticker=5.0, target=5000).run(A2.drain_end_ts + 30, drv)
        idle_ops = [(ts - H0, op) for ts, op in h.client.ops if ws.phase(ts) == "IDLE"]
        self.assertEqual(idle_ops, [], "Deriv traffic while phase() is IDLE")
        names = {op for _, op in h.client.ops}
        self.assertTrue({"connect", "authorise", "balance", "statement", "proposal", "buy"} <= names, names)
        self.assertGreaterEqual(h.client.ops[0][0], B.connect_ts)       # first connect exactly at the lead
        self.assertEqual([kw["session_name"] for _, kw in h.live_events()], ["B", "A"])
        self.assertEqual(h.client.connect_calls, 2)                      # same object connected twice
        self.assertEqual(len(h.client.shutdowns), 2)
        self.assertFalse(h.client.is_connected)
        self.assertIsNone(h.eng.session)
        self.assertEqual((h.counts["boot_fit"], h.counts["lead_prepare"]), (1, 1))   # boot fit once, then lead prepare
        self.assertEqual(leak_probe["tasks"], [], "tasks leaked past the disconnect")
        self.assertFalse(leak_probe["connected"])
        self.assertEqual(h.client.ops[-1][0] < A2.drain_end_ts + 1e-6, True)

    def test_buy_decided_before_close_is_not_sent_after_it(self):
        """A buy decided at 16:00.9 whose frame would go out at 16:01.2 is refused by the ENGINE gate even when the
        client gate is off and _blocked() is bypassed (stale decision); nothing is transmitted at/after close."""
        A, _, _ = sched()
        seen = {}

        async def drv(h):
            await h.at(A.close_ts - 3.0)
            h.fire()                                                      # (a) control: a normal buy goes out
            await h.at(A.close_ts - 0.1)
            h.client.gate_enabled = False                                 # only the engine gate may protect now
            h.client.proposal_latency = 0.3                               # decision 16:00.9, frame due 16:01.2
            h.eng._blocked = lambda: False                                # a decision already past every check
            h.fire()
            await h.at(A.close_ts + 0.5)
            del h.eng._blocked                                            # (c) the real _blocked() after close
            seen["blocked_after_close"] = h.eng._blocked()

        with self.assertLogs("bot_engine", "WARNING") as cm:
            h = Harness(A.connect_ts - 20).run(A.drain_end_ts + 20, drv)
        self.assertEqual(len(h.client.sent_buys), 1)
        self.assertLess(h.client.sent_buys[0][0], A.close_ts)
        self.assertTrue(all(ts < A.close_ts for ts, _ in h.client.sent_buys))
        self.assertTrue(any("BUY BLOCKED AT CLOSE" in line for line in cm.output), cm.output)
        self.assertEqual(h.money.reserved, h.money.released)              # money freed on the blocked buy
        self.assertEqual(h.eng._last_decision, "buy blocked at session close")
        self.assertTrue(seen["blocked_after_close"])

    def test_direct_buy_path_is_gated_too(self):
        A, _, _ = sched()
        out = {}

        async def drv(h):
            await h.at(A.close_ts + 0.2)
            h.client.gate_enabled = False
            try:
                await h.eng._buy_direct("R_100", ("DIGITDIFF", 3), 0.35, 0.35)
                out["raised"] = False
            except ws.SessionClosed:
                out["raised"] = True

        h = Harness(A.connect_ts - 20).run(A.drain_end_ts + 5, drv)
        self.assertTrue(out["raised"])
        self.assertEqual(h.client.sent_buys, [])

    def test_boot_mid_session_does_not_trade_until_next_open(self):
        A, B, _ = sched()
        h = Harness(A.open_ts + 120, ticker=5.0).run(B.drain_end_ts + 20)   # default RESUME flag (off)
        self.assertGreaterEqual(h.client.ops[0][0], B.connect_ts)           # nothing at all during the rest of A
        self.assertEqual([kw["session_name"] for _, kw in h.live_events()], ["B"])
        self.assertTrue(h.client.sent_buys)
        self.assertTrue(all(ts >= B.open_ts for ts, _ in h.client.sent_buys))
        self.assertEqual(h.counts["boot_fit"], 1)                           # the boot fit runs in B's lead

    def test_resume_flag_trades_the_rest_of_the_session(self):
        A, _, _ = sched()
        start = A.open_ts + 120
        h = Harness(start, ticker=5.0, config={"cfg": {"RESUME_TRADING_MID_SESSION_AFTER_BOOT": True}}
                    ).run(A.drain_end_ts + 20)
        self.assertLessEqual(h.client.ops[0][0], start + 2)
        self.assertEqual([kw["session_name"] for _, kw in h.live_events()], ["A"])
        self.assertTrue(h.client.sent_buys)
        self.assertTrue(all(ts < A.close_ts for ts, _ in h.client.sent_buys))

    def test_gate_opens_exactly_at_open_with_lead_finished(self):
        A, _, _ = sched()
        probe = {}

        async def drv(h):
            await h.at(A.open_ts - 0.2)
            probe["before"] = h.eng._blocked()
            await h.at(A.open_ts + 0.2)
            probe["after"] = h.eng._blocked()

        h = Harness(A.connect_ts - 30, ticker=0.5).run(A.open_ts + 20, drv)
        live_ts = h.live_events()[0][0]
        ready_ts = next(ts for ts, kw in h.status if kw.get("session_lead_stage") == "ready"
                        and kw.get("session_state") != "LIVE")
        self.assertLessEqual(ready_ts, A.open_ts)                           # lead finished before the open
        self.assertGreaterEqual(live_ts, A.open_ts)
        self.assertLess(live_ts - A.open_ts, 0.1)                           # ... and the gate opens on the instant
        self.assertTrue(probe["before"])
        self.assertFalse(probe["after"])
        self.assertGreaterEqual(h.client.sent_buys[0][0], A.open_ts)        # ticks during the lead bought nothing
        self.assertLess(h.client.sent_buys[0][0] - A.open_ts, 1.0)

    def test_in_session_reconnect_continues_the_same_session(self):
        A, _, _ = sched()
        probe = {}

        async def drv(h):
            await h.at(A.open_ts + 60)
            probe["session"] = h.eng.session
            h.client.drop()
            await h.at(A.open_ts + 60.5)
            probe["blocked_right_after"] = h.eng._blocked()
            await h.at(A.open_ts + 80)
            probe["blocked_later"] = h.eng._blocked()
            probe["same"] = h.eng.session is probe["session"]
            probe["shutdowns"] = len(h.client.shutdowns)
            probe["fits"] = h.counts["fit"]

        h = Harness(A.connect_ts - 10, ticker=5.0).run(A.close_ts - 5, drv)
        self.assertTrue(probe["blocked_right_after"])                       # until the predictor is re-synced
        self.assertFalse(probe["blocked_later"])                            # then trading continues
        self.assertTrue(probe["same"])
        self.assertGreaterEqual(probe["fits"], 1)                           # the resync refit ran
        self.assertEqual([kw["session_name"] for _, kw in h.live_events()], ["A"])   # no second session opened
        self.assertEqual(probe["shutdowns"], 0)                             # no disconnect in between
        self.assertEqual((h.counts["boot_fit"], h.counts["lead_prepare"]), (1, 0))

    def test_connect_failures_are_retried_inside_the_lead(self):
        A, _, _ = sched()
        h = Harness(A.connect_ts - 5)
        h.client.raise_connects = 2
        h.run(A.open_ts + 30)
        self.assertEqual(h.client.connect_calls, 3)
        self.assertGreaterEqual(h.client.ops[0][0], A.connect_ts)
        live_ts = h.live_events()[0][0]
        self.assertTrue(A.open_ts <= live_ts < A.open_ts + 0.1)

    def test_no_clock_sample_is_critical_but_the_session_still_opens(self):
        A, _, _ = sched()
        h = Harness(A.connect_ts - 5)
        h.client.clock_fail = True
        with self.assertLogs("bot_engine", "CRITICAL") as cm:
            h.run(A.open_ts + 10)
        self.assertTrue(any("WITHOUT A CLOCK SAMPLE" in line for line in cm.output), cm.output)
        self.assertEqual([kw["session_name"] for _, kw in h.live_events()], ["A"])

    def test_hold_preconnect_delays_the_connect(self):
        A, _, _ = sched()
        hold_until = A.connect_ts + 30
        guard = SimpleNamespace(hold_preconnect=lambda: time.time() < hold_until)
        h = Harness(A.connect_ts - 5, config={"redeploy_guard": guard}).run(A.open_ts + 10)
        self.assertGreaterEqual(h.client.ops[0][0], hold_until)
        self.assertEqual([kw["session_name"] for _, kw in h.live_events()], ["A"])

    def test_target_reached_ends_run_and_never_reconnects(self):
        A, _, _ = sched()
        h = Harness(A.connect_ts - 5, target=1000)
        h.client.balance = 1500.0
        h.run(H0 + 7200)
        self.assertTrue(h.engine_ended)
        self.assertIsNone(h.engine_task.exception())
        self.assertEqual(h.client.connect_calls, 1)
        self.assertTrue(h.target.suspended)
        self.assertEqual(h.client.sent_buys, [])
        self.assertLess(h.client.ops[-1][0], A.open_ts)                     # nothing after the target shutdown

    def test_drain_is_bounded_and_unsettled_contracts_are_left_to_deriv(self):
        A, _, _ = sched()

        async def drv(h):
            await h.at(A.close_ts - 3.0)
            h.client.settle_delay = 1000.0                                  # this contract will not settle in time
            h.fire()

        h = Harness(A.connect_ts - 20).run(A.drain_end_ts + 20, drv)
        self.assertEqual(len(h.eng._open), 1)
        down_ts = h.client.shutdowns[0][0]
        self.assertTrue(A.drain_end_ts <= down_ts < A.drain_end_ts + 0.6, down_ts - A.close_ts)
        self.assertFalse(any(ts > down_ts for ts, _ in h.client.ops))      # nothing sent after the disconnect


# ── the REAL DerivClient against a fake websocket server (virtual time) ─────────────────────────────────────────────
import json  # noqa: E402

SERVER_OFFSET = 0.37                          # the fake Deriv server clock is this far ahead of the local clock


class FakeSocket:
    def __init__(self, srv):
        self.srv, self.q, self.closed = srv, asyncio.Queue(), False

    async def send(self, raw):
        if self.closed:
            raise ConnectionError("socket closed")
        msg = json.loads(raw)
        self.srv.frames.append((time.time(), msg))
        self.srv.respond(self, msg)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.q.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def close(self):
        self.closed = True
        self.q.put_nowait(None)


class FakeConn:
    def __init__(self, srv):
        self.srv = srv

    async def __aenter__(self):
        self.sock = FakeSocket(self.srv)
        self.srv.socks.append(self.sock)
        return self.sock

    async def __aexit__(self, *exc):
        self.sock.closed = True
        return False


class FakeServer:
    def __init__(self, loop):
        self.loop, self.frames, self.socks, self.otp_calls = loop, [], [], 0

    def frames_of(self, key):
        return [m for _, m in self.frames if key in m]

    def respond(self, sock, msg):
        def arrive():                                      # request reaches the server 25 ms later
            body = {"req_id": msg.get("req_id")}
            if "time" in msg:
                body.update(msg_type="time", time=int(time.time() + SERVER_OFFSET))   # whole seconds, as Deriv
            elif "balance" in msg:
                body.update(msg_type="balance", balance={"balance": 1000.0})
            elif "buy" in msg:
                body.update(msg_type="buy", buy={"contract_id": 777, "purchase_time": int(time.time())})
            elif "ticks" in msg:
                body.update(msg_type="tick", tick={"id": "sub1", "symbol": msg["ticks"]})
            else:
                body.update(msg_type="other")
            self.loop.call_later(0.025, sock.q.put_nowait, json.dumps(body))
        self.loop.call_later(0.025, arrive)


def run_real_client(start, body):
    loop = VLoop(start)
    srv = FakeServer(loop)
    ws.set_clock_offset(0.0, 0.0)
    ws.reset_clock_measured()
    with contextlib.ExitStack() as st:
        st.enter_context(mock.patch("time.time", lambda: loop._vt))
        st.enter_context(mock.patch.object(deriv_client.websockets, "connect", lambda url, **kw: FakeConn(srv)))
        st.enter_context(mock.patch.dict(sys.modules, {"target_guard": SimpleNamespace(is_reached=lambda: False)}))
        asyncio.set_event_loop(loop)
        try:
            async def main():
                client = deriv_client.DerivClient()

                async def otp():
                    srv.otp_calls += 1
                    return "wss://fake.example/ws?otp=1"
                client._fetch_otp_ws_url = otp
                await body(srv, client)
            loop.run_until_complete(main())
        finally:
            for t in asyncio.all_tasks(loop):
                t.cancel()
            loop.run_until_complete(asyncio.sleep(0))
            loop.close()
            asyncio.set_event_loop(None)


async def _connected(client, limit=60.0):
    t_end = time.time() + limit
    while not client.is_connected:
        assert time.time() < t_end, "client did not connect"
        await asyncio.sleep(0.1)


def _others():
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]


@unittest.skipUnless(REAL_CLIENT, "real deriv_client.py (with sync_clock) not importable here")
class RealClientTests(unittest.TestCase):

    def test_shutdown_then_connect_on_the_same_object_leaks_no_task(self):
        A, _, _ = sched()
        out = {}

        async def body(srv, client):
            for round_no in (1, 2):
                client.set_reconnect_enabled(True)
                task = asyncio.create_task(client.connect())
                await _connected(client)
                self.assertIsNotNone(client._polling_task)               # (re)started by connect()
                self.assertFalse(client._polling_task.done())
                await client.shutdown("test")
                await asyncio.wait_for(task, 10)                         # connect() returns by itself
                self.assertFalse(client.is_connected)
                self.assertIsNone(client._polling_task)                  # cancelled by shutdown()
                await asyncio.sleep(0.1)
                out[round_no] = [t.get_coro().__qualname__ for t in _others()]
            with self.assertRaises(RuntimeError):                       # nothing sends after shutdown
                await client._send({"balance": 1})
            out["otp"] = srv.otp_calls
            out["conn"] = client.connection_count

        run_real_client(A.connect_ts, body)
        self.assertEqual(out[1], [])
        self.assertEqual(out[2], [])
        self.assertEqual((out["otp"], out["conn"]), (2, 2))

    def test_reconnect_disabled_means_no_otp_and_no_socket(self):
        A, _, _ = sched()
        out = {}

        async def body(srv, client):
            task = asyncio.create_task(client.connect())
            await _connected(client)
            # a drop with reconnect still enabled reconnects (in-session behaviour)
            srv.socks[-1].q.put_nowait(None)
            await asyncio.sleep(10)
            out["after_drop_enabled"] = (srv.otp_calls, client.connection_count)
            # now disable and drop again: the loop must end without minting an OTP
            client.set_reconnect_enabled(False)
            srv.socks[-1].q.put_nowait(None)
            await asyncio.wait_for(task, 10)
            await asyncio.sleep(300)
            out["otp"] = srv.otp_calls
            out["connected"] = client.is_connected
            # connect() itself is a no-op while disabled
            await client.connect()
            out["otp_after_connect_call"] = srv.otp_calls
            out["frames_after"] = len(srv.frames)

        run_real_client(A.connect_ts, body)
        self.assertEqual(out["after_drop_enabled"], (2, 2))
        self.assertEqual(out["otp"], 2)
        self.assertEqual(out["otp_after_connect_call"], 2)
        self.assertFalse(out["connected"])

    def test_nothing_is_sent_in_the_gap_even_with_a_tracked_contract(self):
        A, _, _ = sched()
        out = {}

        async def body(srv, client):
            task = asyncio.create_task(client.connect())
            await _connected(client)
            await client.subscribe_contract("555", lambda m: None, symbol="R_100")   # an orphan left open
            await asyncio.sleep(1)
            await client.shutdown("session closed")
            await asyncio.wait_for(task, 10)
            n = len(srv.frames)
            await asyncio.sleep(600)                                    # ten minutes of "idle gap"
            out["new_frames"] = len(srv.frames) - n
            out["otp"] = srv.otp_calls

        run_real_client(A.connect_ts, body)
        self.assertEqual(out["new_frames"], 0)
        self.assertEqual(out["otp"], 1)

    def test_send_time_buy_gate(self):
        A, _, _ = sched()
        out = {}

        async def body(srv, client):
            task = asyncio.create_task(client.connect())
            await _connected(client)
            await asyncio.sleep(A.close_ts - 1.0 - time.time())          # 16:00.0
            r = await client._send({"buy": "p1", "price": 0.35})
            out["early"] = r["buy"]["contract_id"]
            await asyncio.sleep(A.close_ts - 0.05 - time.time())         # 16:00.95
            await client._send({"proposal": 1})                          # proposals are NOT gated
            await asyncio.sleep(A.close_ts + 0.05 - time.time())         # 16:01.05
            n_pending = len(client._pending)
            with self.assertRaises(_sched_closed()):
                await client._send({"buy": "p1", "price": 0.35})
            with self.assertRaises(_sched_closed()):
                await client._send({"buy": "1", "price": 0.35, "parameters": {}})   # the direct-buy shape
            out["buys_on_wire"] = len(srv.frames_of("buy"))
            out["pending_unchanged"] = len(client._pending) == n_pending           # no future was registered
            out["proposal_ok"] = len(srv.frames_of("proposal")) == 1
            await client.shutdown("x")
            await asyncio.wait_for(task, 10)

        run_real_client(A.open_ts + 30, body)
        self.assertEqual(out["early"], 777)
        self.assertEqual(out["buys_on_wire"], 1)
        self.assertTrue(out["pending_unchanged"])
        self.assertTrue(out["proposal_ok"])

    def test_sync_clock_recovers_the_server_offset(self):
        A, _, _ = sched()
        out = {}

        async def body(srv, client):
            task = asyncio.create_task(client.connect())
            await _connected(client)
            out["coarse"] = await client.sync_clock()
            out["fine"] = await client.sync_clock(fine=True)
            out["spacing_ok"] = all(b - a >= 0.25 - 1e-6 for a, b in zip(
                [t for t, m in srv.frames if "time" in m], [t for t, m in srv.frames if "time" in m][1:]))
            out["samples"] = client.clock_samples
            out["ws_offset"] = ws.get_clock_offset()
            out["measured"] = ws.clock_measured()
            await client.shutdown("x")
            await asyncio.wait_for(task, 10)

        run_real_client(A.connect_ts, body)
        for key in ("coarse", "fine"):
            off, err = out[key]
            self.assertLessEqual(abs(off - SERVER_OFFSET), err + 1e-9, (key, off, err))   # truth inside the bound
        self.assertLess(out["coarse"][1], 0.2)
        self.assertLess(out["fine"][1], 0.06)                      # fine mode narrows the bound
        self.assertTrue(out["spacing_ok"])                         # SESSION_CLOCK_MIN_INTERVAL respected
        self.assertAlmostEqual(out["ws_offset"], out["fine"][0], places=6)
        self.assertTrue(out["measured"])
        self.assertGreaterEqual(out["samples"], 3)


def _sched_closed():
    return deriv_client._sched.SessionClosed


if __name__ == "__main__":
    unittest.main(verbosity=2)
