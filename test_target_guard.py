"""
test_target_guard.py – offline tests for the profit cycle: TARGET_PROFIT parsing, net profit, the boot decision matrix,
the low-balance rule, the once-only terminal claim, verified persistence, the target-hit and low-balance flows.
Every Render and Deriv network call is mocked. Run: python test_target_guard.py   (or pytest test_target_guard.py)
"""
import asyncio
import importlib
import os
import sys
import time
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DERIV_API_TOKEN", "x" * 20)

import target_guard as tg
import env_persist
import profit_cycle as pc
import window_scheduler as ws

_ENV = ("TARGET_PROFIT", "TARGET_BALANCE", "RENDER_API_KEY", "RENDER_SERVICE_ID", "RENDER_DEPLOY_HOOK_URL",
        "CYCLE_STARTING_BALANCE", "CYCLE_COUNT", "CYCLE_LIFETIME_PROFIT", "CYCLE_DRAIN_TIMEOUT_SECONDS",
        "CYCLE_COOLDOWN_SECONDS", "PREDICTOR_STAKE")
_REAL_SLEEP = asyncio.sleep


def _reset():
    for k in _ENV:
        os.environ.pop(k, None)
    importlib.reload(tg)
    importlib.reload(pc)


# ── a fake Render API (env vars, deploys, suspend) ───────────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeRender:
    def __init__(self, env=None):
        self.env = dict(env or {})        # what Render stores
        self.calls = []                   # (method, url, body)
        self.put_fail = {}                # key -> number of PUTs that still fail (HTTP 500)
        self.readback_wrong = set()       # keys whose read-back disagrees
        self.deploys = [{"id": "dep-old", "status": "live", "createdAt": "2020-01-01T00:00:00Z"}]
        self.deploy_posts = 0
        self.deploy_status = 201
        self.suspends = 0

    def put(self, url, json=None, headers=None, timeout=None):
        key = url.rsplit("/", 1)[-1]
        self.calls.append(("PUT", key, json))
        if self.put_fail.get(key, 0) > 0:
            self.put_fail[key] -= 1
            return _Resp(500, "boom")
        self.env[key] = json["value"]
        return _Resp(200, {"key": key, "value": json["value"]})

    def get(self, url, headers=None, params=None, timeout=None):
        if "/env-vars/" in url:
            key = url.rsplit("/", 1)[-1]
            self.calls.append(("GET", key, None))
            if key not in self.env:
                return _Resp(404, "nf")
            val = "WRONG" if key in self.readback_wrong else self.env[key]
            return _Resp(200, {"envVar": {"key": key, "value": val}})
        self.calls.append(("GET", "deploys", None))
        return _Resp(200, [{"deploy": d} for d in self.deploys])

    def post(self, url, headers=None, json=None, timeout=None):
        if url.endswith("/suspend"):
            self.calls.append(("POST", "suspend", None))
            self.suspends += 1
            return _Resp(200, {})
        self.calls.append(("POST", "deploy", json))
        self.deploy_posts += 1
        if self.deploy_status in (200, 201, 202):
            self.deploys.insert(0, {"id": "dep-new", "status": "live", "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return _Resp(self.deploy_status, {"id": "dep-new"})

    def order(self, method, key):
        return [i for i, c in enumerate(self.calls) if c[0] == method and c[1] == key]

    def put_keys(self):
        return {c[1] for c in self.calls if c[0] == "PUT"}


def _setup(start=None, tp=None, creds=True, env=None, **render_kw):
    _reset()
    if creds:
        os.environ["RENDER_API_KEY"], os.environ["RENDER_SERVICE_ID"] = "rnd_x", "srv-abc"
    if start is not None:
        os.environ["CYCLE_STARTING_BALANCE"] = str(start)
    if tp is not None:
        os.environ["TARGET_PROFIT"] = str(tp)
    fr = FakeRender(env or ({"CYCLE_STARTING_BALANCE": str(start)} if start is not None else {}))
    for k, v in render_kw.items():
        setattr(fr, k, v)
    return fr


def _patched(fr):
    import contextlib
    st = contextlib.ExitStack()
    st.enter_context(mock.patch("requests.put", fr.put))
    st.enter_context(mock.patch("requests.get", fr.get))
    st.enter_context(mock.patch("requests.post", fr.post))
    return st


def _assert_only_bot_keys(fr):
    bad = fr.put_keys() - set(pc.BOT_KEYS)
    assert not bad, f"the bot wrote user-owned variables: {bad}"
    assert "TARGET_PROFIT" not in fr.put_keys()


# ── pure rules ───────────────────────────────────────────────────────────────────────────────────────────────────
def test_target_profit_parsing():
    _reset()
    assert tg.target_profit() is None                       # unset = OFF
    for raw, want in (("50", 50.0), (" 5,000.50 ", 5000.5), ("$250", 250.0), ("$ 1,250", 1250.0), ("0", None),
                      ("0.00", None), ("", None), ("  ", None),
                      ("-5", None), ("abc", None), ("nan", None), ("inf", None), ("0.001", None)):   # invalid -> OFF, no crash
        os.environ["TARGET_PROFIT"] = raw
        assert tg.target_profit() == want, (raw, tg.target_profit())
    os.environ["TARGET_PROFIT"] = "10"                      # read from the environment every call, never cached
    assert tg.target_profit() == 10.0
    os.environ["TARGET_PROFIT"] = "25"
    assert tg.target_profit() == 25.0


def test_invalid_target_profit_logs_error_and_is_off():
    _reset()
    import logging
    seen = []

    class H(logging.Handler):
        def emit(self, rec):
            seen.append((rec.levelname, rec.getMessage()))
    h = H()
    logging.getLogger("target_guard").addHandler(h)
    try:
        os.environ["TARGET_PROFIT"] = "fifty"
        assert tg.target_profit() is None
    finally:
        logging.getLogger("target_guard").removeHandler(h)
    assert any(lvl == "ERROR" and "TARGET_PROFIT" in m for lvl, m in seen), seen


def test_target_balance_is_deprecated_and_ignored():
    _reset()
    os.environ["TARGET_BALANCE"] = "5000"
    assert tg.target_profit() is None and tg.deprecated_target_balance_set()
    assert not tg.target_hit(1e9, 100.0)


def test_net_profit_and_target_balance():
    assert tg.net_profit(150.0, 100.0) == 50.0
    assert tg.net_profit(99.5, 100.0) == -0.5
    assert tg.net_profit(100.004, 100.0) == 0.0             # cent rounding
    assert tg.net_profit(150.0, 0) is None and tg.net_profit(None, 100.0) is None and tg.net_profit("x", 100) is None
    assert tg.net_profit(float("nan"), 100.0) is None
    assert tg.target_balance_for(100.0, 50.0) == 150.0
    assert tg.target_balance_for(100.0, None) is None and tg.target_balance_for(0, 50.0) is None


def test_target_hit_rule_with_tolerance():
    assert tg.target_hit(150.0, 100.0, 50.0)
    assert tg.target_hit(150.01, 100.0, 50.0)
    assert not tg.target_hit(149.99, 100.0, 50.0)
    assert tg.target_hit(100.1 + 49.9, 100.1, 49.9)         # float noise (150.0 vs 150.00000000000003) must not matter
    assert tg.target_hit(0.3 + 0.1, 0.1, 0.3)
    assert not tg.target_hit(1e9, 0, 50.0)                  # no starting balance recorded -> never a hit
    assert not tg.target_hit(1e9, 100.0, None)              # profit target off
    _reset()
    assert not tg.target_hit(1e9, 100.0)                    # unset TARGET_PROFIT -> off


def test_low_balance_rule_never_while_contracts_are_open():
    assert tg.low_balance(0.20, 0.35, True)
    assert tg.low_balance(0.34, 0.35, True)
    assert not tg.low_balance(0.35, 0.35, True)             # exactly the minimum stake can still buy
    assert not tg.low_balance(0.36, 0.35, True)
    assert not tg.low_balance(0.20, 0.35, False)            # an open contract: pending payouts make the balance look low
    assert not tg.low_balance(0.20, 0.35, None)             # could not ask Deriv -> not flat
    assert not tg.low_balance(None, 0.35, True) and not tg.low_balance(-1.0, 0.35, True)
    assert not tg.low_balance(float("nan"), 0.35, True)
    assert not tg.low_balance(0.0, 0.35, True)              # an unconfirmed zero is never acted on
    assert tg.low_balance(0.0, 0.35, True, zero_confirmed=True)
    assert not tg.low_balance(0.20, 0, True)


def test_boot_decision_matrix():
    D = tg.boot_decision
    # saved cycle at/above target -> straight to the target-hit path
    assert D(100.0, 150.0, 50.0, 0.35, True) == tg.TARGET_HIT
    assert D(100.0, 150.0, 50.0, 0.35, False) == tg.TARGET_HIT          # contracts open: the drain handles them
    assert D(100.0, 400.0, 50.0, 0.35, True) == tg.TARGET_HIT           # TARGET_PROFIT lowered below current net profit
    # saved cycle below target -> resume
    assert D(100.0, 149.99, 50.0, 0.35, True) == tg.RESUME
    assert D(100.0, 80.0, 50.0, 0.35, True) == tg.RESUME
    assert D(100.0, 1e6, None, 0.35, True) == tg.RESUME                 # profit target off: never a target hit
    # empty / 0 starting balance -> record (only on a flat portfolio)
    assert D(None, 500.0, 50.0, 0.35, True) == tg.RECORD
    assert D(0, 500.0, 50.0, 0.35, True) == tg.RECORD
    assert D(0.0, 500.0, None, 0.35, True) == tg.RECORD                 # profit target off still records a start
    assert D(0, 500.0, 50.0, 0.35, False) == tg.WAIT_FLAT
    assert D(0, 500.0, 50.0, 0.35, None) == tg.WAIT_FLAT
    # below the minimum stake and flat -> low-balance exit, at any point
    assert D(100.0, 0.20, 50.0, 0.35, True) == tg.LOW_BALANCE
    assert D(None, 0.20, 50.0, 0.35, True) == tg.LOW_BALANCE
    assert D(None, 0.20, None, 0.35, True) == tg.LOW_BALANCE
    assert D(100.0, 0.20, 50.0, 0.35, False) == tg.RESUME               # not flat: never judged low
    assert D(0, 0.20, 50.0, 0.35, False) == tg.WAIT_FLAT
    # unusable balances are never acted on
    for bad in (None, float("nan"), -3.0, 0.0):
        assert D(100.0, bad, 50.0, 0.35, True) == tg.WAIT_BALANCE, bad
    assert D(100.0, 0.0, 50.0, 0.35, True, zero_confirmed=True) == tg.LOW_BALANCE


def test_no_redeploy_loop_after_a_completed_cycle():
    """After the clear the next boot has no starting balance: it records instead of seeing the target met again, also when
    TARGET_PROFIT is tiny (anything below one cent is rejected, so net 0.00 can never satisfy it)."""
    _reset()
    for raw in ("0.01", "1", "50"):
        os.environ["TARGET_PROFIT"] = raw
        tp = tg.target_profit()
        assert tg.boot_decision(0, 1500.0, tp, 0.35, True) == tg.RECORD
        assert not tg.target_hit(1500.0, 1500.0, tp)
    os.environ["TARGET_PROFIT"] = "0.004"
    assert tg.target_profit() is None


def test_claim_is_once_only_and_terminal_actions_are_mutually_exclusive():
    _reset()
    assert not tg.is_reached()
    assert tg.mark_reached(150.0) is True
    assert tg.mark_reached(151.0) is False
    assert tg.mark_low_balance(0.1) is False                # redeploy path won: the suspend path is inert
    assert tg.is_reached() and tg.terminal_kind() == tg.TARGET
    assert tg.release(tg.LOW) is False and tg.is_reached()  # a target claim can never be released
    _reset()
    assert tg.mark_low_balance(0.1) is True
    assert tg.mark_reached(500.0) is False                  # suspend path won: the redeploy path is inert
    assert tg.release(tg.LOW) is True and not tg.is_reached()


def test_claim_once_only_across_threads():
    _reset()
    import threading
    wins = []
    ts = [threading.Thread(target=lambda i=i: wins.append(tg.claim(tg.TARGET if i % 2 else tg.LOW, 1.0))) for i in range(40)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert wins.count(True) == 1


def test_send_blocks_buy_and_proposal_after_claim():
    for claimer in ("mark_reached", "mark_low_balance"):
        _reset()
        from deriv_client import DerivClient
        c = DerivClient()
        getattr(tg, claimer)(100.0)
        for payload in ({"buy": "1", "price": 1}, {"proposal": 1}):
            try:
                asyncio.run(c._send(payload))
            except RuntimeError as exc:
                assert "TARGET_REACHED" in str(exc), exc
            else:
                raise AssertionError("order was not blocked")


def test_suspend_service_without_api_key_is_safe():
    _reset()
    assert tg.suspend_service() is False
    assert "no RENDER_API_KEY" in tg.snapshot()["suspend_result"]


def test_suspend_service_calls_render_api():
    fr = _setup()
    seen = {}

    class R:
        status_code = 202
        text = ""

    def fake_post(url, headers=None, timeout=None):
        seen["url"], seen["auth"] = url, headers["Authorization"]
        return R()
    with mock.patch("requests.post", fake_post):
        assert tg.suspend_service() is True
    assert seen == {"url": "https://api.render.com/v1/services/srv-abc/suspend", "auth": "Bearer rnd_x"}


# ── persistence ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_bot_never_writes_target_profit_or_other_user_variables():
    fr = _setup(tp=50)
    with _patched(fr):
        for key in ("TARGET_PROFIT", "RENDER_API_KEY", "DERIV_API_TOKEN", "TARGET_BALANCE"):
            assert env_persist.put_env_var_verified_sync(key, "1") is False
            assert env_persist.put_env_vars_sync({key: "1"}) is False
            assert asyncio.run(env_persist.put_env_vars_async({key: "1"})) is False
        try:
            pc._write("TARGET_PROFIT", "1")
        except ValueError:
            pass
        else:
            raise AssertionError("profit_cycle._write accepted TARGET_PROFIT")
    assert fr.calls == [] and os.environ["TARGET_PROFIT"] == "50"


def test_verified_write_needs_200_and_a_matching_readback():
    fr = _setup()
    with _patched(fr):
        assert env_persist.put_env_var_verified_sync("CYCLE_COUNT", "3") is True
        assert fr.env["CYCLE_COUNT"] == "3" and os.environ["CYCLE_COUNT"] == "3"
        fr.put_fail["CYCLE_COUNT"] = 1
        assert env_persist.put_env_var_verified_sync("CYCLE_COUNT", "4") is False      # HTTP 500
        assert os.environ["CYCLE_COUNT"] == "3"                                          # in-process env only after success
        fr.readback_wrong.add("CYCLE_COUNT")
        assert env_persist.put_env_var_verified_sync("CYCLE_COUNT", "5") is False      # PUT ok, read-back disagrees
        assert os.environ["CYCLE_COUNT"] == "3"
    _reset()
    assert env_persist.put_env_var_verified_sync("CYCLE_COUNT", "1") is False          # no credentials


def test_record_start_persists_and_mirrors_in_process():
    fr = _setup()
    with _patched(fr):
        assert pc.record_start(1234.5) is True
    assert fr.env["CYCLE_STARTING_BALANCE"] == "1234.50" and pc.starting_balance() == 1234.5
    for bad in (0.0, -1.0, None, float("nan")):
        with _patched(fr):
            assert pc.record_start(bad) is False
    _assert_only_bot_keys(fr)


def test_clear_failure_blocks_redeploy():
    fr = _setup(start=100)
    fr.put_fail["CYCLE_STARTING_BALANCE"] = 99
    with _patched(fr):
        assert pc.clear_start() is False
        assert pc.starting_balance() == 0.0               # this process never resumes the finished cycle
        ok, why = pc.trigger_redeploy()
    assert ok is False and "not verified" in why and fr.deploy_posts == 0
    assert pc.dashboard_fields()["cycle_alert_clear_unverified"] is True
    fr.put_fail.clear()
    fr.readback_wrong.add("CYCLE_STARTING_BALANCE")        # PUT 200 but read-back disagrees: still not verified
    with _patched(fr):
        assert pc.clear_start() is False
        assert pc.trigger_redeploy()[0] is False and fr.deploy_posts == 0
    fr.readback_wrong.clear()
    with _patched(fr):
        assert pc.clear_start() is True and pc.dashboard_fields()["cycle_alert_clear_unverified"] is False
        ok, did = pc.trigger_redeploy()
    assert ok is True and fr.deploy_posts == 1 and did == "dep-new"


def test_redeploy_loop_brake_and_no_double_deploy():
    fr = _setup(start=100)
    with _patched(fr):
        assert pc.clear_start() is True
        fr.deploys[0] = {"id": "dep-young", "status": "live", "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 60))}
        ok, why = pc.trigger_redeploy()                    # the latest deploy is only a minute old (before the clear)
        assert ok is False and "loop brake" in why and fr.deploy_posts == 0
        fr.deploys[0] = {"id": "dep-auto", "status": "build_in_progress",   # a deploy already started since the clear
                         "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 1))}
        ok, did = pc.trigger_redeploy()
        assert ok is True and did == "dep-auto" and fr.deploy_posts == 0


def test_missing_credentials_never_persist_or_redeploy():
    fr = _setup(start=100, tp=50, creds=False)
    with _patched(fr):
        assert pc.record_start(500.0) is False
        assert pc.clear_start() is False
        assert pc.record_completed_cycle(50.0) is False
        assert pc.trigger_redeploy()[0] is False
        assert tg.suspend_service() is False
    assert fr.calls == [] and pc.dashboard_fields()["cycle_alert_persistence_missing"] is True
    os.environ["RENDER_DEPLOY_HOOK_URL"] = "https://hook.example/x"      # a hook alone cannot verify the clear: still no redeploy
    with _patched(fr):
        assert pc.trigger_redeploy()[0] is False
    assert fr.calls == []


# ── engine flows (fake Deriv client, fake Render) ────────────────────────────────────────────────────────────────
class FakeStats:
    async def flush(self, force=False):
        pass

    def record(self, *a, **k):
        pass


class FakeClient:
    def __init__(self, bal, portfolio=(0,)):
        self.balance = bal
        self.is_connected = True
        self.connection_count = 1
        self._cbs, self.pf, self.pf_calls = [], list(portfolio), 0
        self.shutdown_called, self.reconnect = False, True
        self.live_fail = 0

    def on_balance(self, cb):
        self._cbs.append(cb)

    async def fetch_balance_live(self, timeout=8.0):
        if self.live_fail > 0:
            self.live_fail -= 1
            raise RuntimeError("balance unavailable")
        for cb in self._cbs:
            cb(self.balance)
        return self.balance

    async def get_portfolio(self, strict=False, timeout=30.0):
        self.pf_calls += 1
        n = self.pf.pop(0) if len(self.pf) > 1 else self.pf[0]
        if n is None:
            raise RuntimeError("portfolio unavailable")
        return [{"contract_id": str(i)} for i in range(n)]

    def set_reconnect_enabled(self, enabled):
        self.reconnect = enabled

    async def shutdown(self, reason=""):
        self.shutdown_called = True
        self.is_connected = False


def _engine(bal, portfolio=(0,)):
    import bot_engine
    eng = bot_engine.BotEngine()
    eng.client = FakeClient(bal, portfolio)
    eng.stats = FakeStats()
    eng.client.on_balance(eng._on_balance)
    return eng


def _run(coro_fn, fr, slow_sleep=False):
    """Run an async scenario with Render mocked and asyncio.sleep shortened (the flows poll every few seconds)."""
    async def fast(delay, *a, **k):
        await _REAL_SLEEP(0.25 if slow_sleep and delay >= 1 else 0.001)     # slow_sleep: real-time-ish polling, capped
    with _patched(fr), mock.patch.object(asyncio, "sleep", fast), \
            mock.patch.object(pc, "BACKOFF_BASE_SECONDS", 0.001):
        return asyncio.run(coro_fn())


def _sess():
    return SimpleNamespace(name="A", close_ts=ws.now() + 600)


def test_target_hit_flow_drains_records_clears_then_redeploys_once():
    fr = _setup(start=100, tp=50)
    os.environ["CYCLE_COUNT"], os.environ["CYCLE_LIFETIME_PROFIT"] = "2", "75.00"
    fr.env.update(CYCLE_COUNT="2", CYCLE_LIFETIME_PROFIT="75.00")

    async def go():
        eng = _engine(100.0, portfolio=(2, 1, 0))          # two contracts open on Deriv when the target is hit
        eng.client.balance = 149.99
        await eng.client.fetch_balance_live()
        assert not tg.is_reached()
        eng.client.balance = 150.0
        await eng.client.fetch_balance_live()
        assert tg.is_reached() and eng._blocked()          # stopped synchronously, inside the balance callback
        eng.client.balance = 151.0
        await eng.client.fetch_balance_live()              # more balance updates while draining: inert (once-only)
        await asyncio.wait_for(eng._target_done.wait(), 10)
        assert eng.client.shutdown_called and eng.client.pf_calls >= 3
        return eng
    eng = _run(go, fr)
    assert fr.deploy_posts == 1 and fr.suspends == 0
    assert fr.env["CYCLE_STARTING_BALANCE"] == "0"
    assert fr.env["CYCLE_COUNT"] == "3" and fr.env["CYCLE_LIFETIME_PROFIT"] == "126.00"     # 75 + (151-100)
    first_put = fr.order("PUT", "CYCLE_COUNT")[0]
    clear_put = fr.order("PUT", "CYCLE_STARTING_BALANCE")[-1]
    assert first_put < clear_put < fr.order("POST", "deploy")[0]           # stats -> verified clear -> redeploy
    assert fr.order("GET", "CYCLE_STARTING_BALANCE")                          # the clear was read back
    assert pc.dashboard_fields()["cycle_phase"] == pc.REDEPLOYING
    _assert_only_bot_keys(fr)


def test_clear_not_verified_retries_and_never_redeploys_until_verified():
    fr = _setup(start=100, tp=50)
    fr.put_fail["CYCLE_STARTING_BALANCE"] = 3

    async def go():
        eng = _engine(100.0)
        eng.client.balance = 160.0
        await eng.client.fetch_balance_live()
        await asyncio.wait_for(eng._target_done.wait(), 10)
    _run(go, fr)
    assert fr.put_fail["CYCLE_STARTING_BALANCE"] == 0
    assert len(fr.order("PUT", "CYCLE_STARTING_BALANCE")) == 4              # three failures, then the verified success
    assert fr.deploy_posts == 1 and fr.order("POST", "deploy")[0] > fr.order("PUT", "CYCLE_STARTING_BALANCE")[-1]
    assert pc.dashboard_fields()["cycle_alert_clear_unverified"] is False


def test_clear_never_verifiable_means_no_redeploy_and_no_suspend():
    fr = _setup(start=100, tp=50)
    fr.put_fail["CYCLE_STARTING_BALANCE"] = 10 ** 6

    async def go():
        eng = _engine(100.0)
        eng.client.balance = 160.0
        await eng.client.fetch_balance_live()
        await _REAL_SLEEP(0.5)                                              # many retries happen
        assert not eng._target_done.is_set()
        assert pc.dashboard_fields()["cycle_alert_clear_unverified"] is True
        eng._terminal_task.cancel()
    _run(go, fr)
    assert fr.deploy_posts == 0 and fr.suspends == 0


def test_redeploy_request_failure_is_retried_with_backoff():
    fr = _setup(start=100, tp=50, deploy_status=500)

    async def go():
        eng = _engine(100.0)
        eng.client.balance = 160.0
        await eng.client.fetch_balance_live()
        await _REAL_SLEEP(0.3)
        assert pc.dashboard_fields()["cycle_alert_redeploy_retrying"] is True and not eng._target_done.is_set()
        fr.deploy_status = 201
        await asyncio.wait_for(eng._target_done.wait(), 10)
    _run(go, fr)
    assert fr.deploy_posts >= 2 and pc.dashboard_fields()["cycle_alert_redeploy_retrying"] is False


def test_drain_timeout_is_critical_flagged_and_continues():
    fr = _setup(start=100, tp=50)
    os.environ["CYCLE_DRAIN_TIMEOUT_SECONDS"] = "1"

    async def go():
        eng = _engine(100.0, portfolio=(1,))               # a contract that never settles
        eng.client.balance = 160.0
        await eng.client.fetch_balance_live()
        await asyncio.wait_for(eng._target_done.wait(), 10)
        assert eng._drain_finished.is_set()
    _run(go, fr, slow_sleep=True)
    assert pc.dashboard_fields()["cycle_alert_drain_timeout"] is True
    assert fr.deploy_posts == 1 and fr.env["CYCLE_STARTING_BALANCE"] == "0"


def test_profit_target_off_never_redeploys_but_still_records_a_start():
    fr = _setup(start=100)                                  # TARGET_PROFIT unset
    async def go():
        eng = _engine(100.0)
        eng.client.balance = 10 ** 6
        await eng.client.fetch_balance_live()
        assert not tg.is_reached()
        return await eng._cycle_gate(_sess(), True)
    assert _run(go, fr) == "go" and fr.deploy_posts == 0
    fr = _setup()                                            # no saved start: recorded even though the target is off
    async def go2():
        return await _engine(500.0)._cycle_gate(_sess(), True)
    assert _run(go2, fr) == "go" and fr.env["CYCLE_STARTING_BALANCE"] == "500.00"


def test_low_balance_exit_clears_then_suspends_and_never_redeploys():
    fr = _setup(start=100, tp=50)

    async def go():
        eng = _engine(0.20, portfolio=(0,))
        await eng._low_balance_check()
        assert tg.is_reached() and tg.terminal_kind() == tg.LOW
        assert tg.mark_reached(999.0) is False             # a profit-target claim arriving now is inert
        await asyncio.wait_for(eng._target_done.wait(), 10)
        assert eng.client.shutdown_called
    _run(go, fr)
    assert fr.env["CYCLE_STARTING_BALANCE"] == "0" and fr.suspends == 1 and fr.deploy_posts == 0
    assert fr.order("PUT", "CYCLE_STARTING_BALANCE")[-1] < fr.order("POST", "suspend")[0]     # suspend only AFTER the clear
    assert "CYCLE_COUNT" not in fr.put_keys()                                                   # not a completed cycle
    assert pc.dashboard_fields()["cycle_phase"] == pc.SUSPENDED
    _assert_only_bot_keys(fr)


def test_low_balance_is_not_judged_while_contracts_are_open():
    fr = _setup(start=100, tp=50)

    async def go():
        eng = _engine(0.20, portfolio=(1,))                # Deriv still has one open contract: pending payout
        await eng._low_balance_check()
        assert not tg.is_reached()
        eng.client.pf = [0]
        eng._open["123"] = {"sym": "R_100"}                # tracked locally as open: also not judged
        await eng._low_balance_check()
        assert not tg.is_reached()
        eng._open.clear()
        eng._busy.add("R_100")                             # a buy in flight: not flat
        await eng._low_balance_check()
        assert not tg.is_reached()
        eng._busy.clear()
        eng.client.balance = 0.35                          # exactly the minimum stake is still tradable
        await eng._low_balance_check()
        assert not tg.is_reached()
        eng.client.balance = 0.20
        await eng._low_balance_check()
        assert tg.is_reached()
        eng._terminal_task.cancel()
    _run(go, fr)
    assert fr.suspends == 0


def test_low_balance_uses_the_engines_own_stake_source():
    fr = _setup(start=100)
    os.environ["PREDICTOR_STAKE"] = "2.50"

    async def go():
        eng = _engine(2.0)
        assert eng._min_stake() == 2.5 == eng.money.next_stake()
        await eng._low_balance_check()
        assert tg.is_reached()
        eng._terminal_task.cancel()
    _run(go, fr)


def test_low_balance_without_api_key_stops_disconnects_and_stays_idle():
    fr = _setup(start=100, tp=50, creds=False)

    async def go():
        eng = _engine(0.20)
        await eng._low_balance_check()
        await asyncio.wait_for(eng._target_done.wait(), 10)
        assert eng.client.shutdown_called
    _run(go, fr)
    f = pc.dashboard_fields()
    assert f["cycle_phase"] == pc.IDLE and f["cycle_alert_idle_not_suspended"] and "NOT suspended" in f["cycle_message"]
    assert fr.calls == []


def test_balance_that_recovers_after_the_claim_releases_the_low_claim():
    fr = _setup(start=100, tp=50)

    async def go():
        eng = _engine(0.20)
        await eng._low_balance_check()
        eng.client.balance = 5.0                           # a straggler win lands during the re-verification
        await _REAL_SLEEP(0.2)
        await eng._await_terminal()
        return eng
    eng = _run(go, fr)
    assert not tg.is_reached() and not eng._target_done.is_set()
    assert fr.suspends == 0 and fr.env.get("CYCLE_STARTING_BALANCE") == "100"


def test_gate_fresh_start_waits_for_a_flat_portfolio_then_records():
    fr = _setup(tp=50)

    async def go():
        eng = _engine(500.0, portfolio=(2, 1, 0))
        res = await eng._cycle_gate(_sess(), True)
        return res, eng
    res, eng = _run(go, fr)
    assert res == "go" and fr.env["CYCLE_STARTING_BALANCE"] == "500.00" and eng.client.pf_calls >= 3
    assert pc.dashboard_fields()["cycle_target_balance"] == 550.0 and pc.dashboard_fields()["cycle_phase"] == pc.TRADING


def test_gate_never_records_while_the_portfolio_is_not_flat_or_unreadable():
    for pf in ((3,), (None,)):
        fr = _setup(tp=50)

        async def go():
            eng = _engine(500.0, portfolio=pf)
            eng._cycle_gate.__func__                       # (bound method exists)
            sess = SimpleNamespace(name="A", close_ts=ws.now() + 6)
            with mock.patch.object(tg, "drain_timeout_seconds", lambda: 0.3):
                return await eng._cycle_gate(sess, True)
        assert _run(go, fr, slow_sleep=True) == "skip" and "CYCLE_STARTING_BALANCE" not in fr.env, pf


def test_gate_resume_keeps_the_saved_start_and_does_not_write():
    fr = _setup(start=100, tp=50)

    async def go():
        return await _engine(120.0)._cycle_gate(_sess(), True)
    assert _run(go, fr) == "go" and fr.put_keys() == set() and pc.starting_balance() == 100.0


def test_gate_boot_target_met_goes_to_target_path_and_next_boot_records():
    fr = _setup(start=100, tp=20)                           # TARGET_PROFIT lowered mid-cycle below the current net profit (+30)

    async def go():
        eng = _engine(130.0, portfolio=(1, 0))
        res = await eng._cycle_gate(_sess(), True)
        await eng._await_terminal()
        return res, eng
    res, eng = _run(go, fr)
    assert res == "stop" and eng._target_done.is_set() and fr.deploy_posts == 1 and fr.env["CYCLE_STARTING_BALANCE"] == "0"
    assert fr.env["CYCLE_COUNT"] == "1" and fr.env["CYCLE_LIFETIME_PROFIT"] == "30.00"
    # the redeployed process: fresh module state, cleared start, same balance -> records, does not loop
    importlib.reload(tg)
    importlib.reload(pc)
    os.environ["CYCLE_STARTING_BALANCE"], os.environ["TARGET_PROFIT"] = fr.env["CYCLE_STARTING_BALANCE"], "20"
    assert tg.boot_decision(pc.starting_balance(), 130.0, tg.target_profit(), 0.35, True) == tg.RECORD


def test_gate_boot_low_balance_exits_with_suspend():
    fr = _setup(start=100, tp=50)

    async def go():
        eng = _engine(0.10)
        res = await eng._cycle_gate(_sess(), True)
        await eng._await_terminal()
        return res
    assert _run(go, fr) == "stop"
    assert fr.suspends == 1 and fr.deploy_posts == 0 and fr.env["CYCLE_STARTING_BALANCE"] == "0"


def test_gate_confirms_a_zero_balance_and_ignores_failed_reads():
    fr = _setup(start=100, tp=50)

    async def go():
        eng = _engine(0.0)
        eng.client.live_fail = 2                           # first reads fail: never acted on, retried
        res = await eng._cycle_gate(_sess(), True)
        await eng._await_terminal()
        return res
    assert _run(go, fr) == "stop" and fr.suspends == 1


def test_gate_without_credentials_does_not_trade_or_call_render():
    fr = _setup(tp=50, creds=False)

    async def go():
        return await _engine(500.0)._cycle_gate(_sess(), True)
    assert _run(go, fr) == "skip" and fr.calls == []
    assert pc.dashboard_fields()["cycle_alert_persistence_missing"] is True


def test_dashboard_fields():
    fr = _setup(start=100, tp=50)
    os.environ["CYCLE_COUNT"], os.environ["CYCLE_LIFETIME_PROFIT"] = "4", "210.5"
    f = pc.dashboard_fields(130.0, 0.35)
    assert f["target_profit"] == 50.0 and f["cycle_starting_balance"] == 100.0 and f["cycle_target_balance"] == 150.0
    assert f["cycle_net_profit"] == 30.0 and f["min_stake_used"] == 0.35
    assert f["cycle_count"] == 4 and f["lifetime_profit"] == 210.5
    _reset()
    f = pc.dashboard_fields(None, None)
    assert f["target_profit"] is None and f["cycle_target_balance"] is None and f["cycle_starting_balance"] is None


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                import traceback; traceback.print_exc()
                fails += 1; print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if fails else 0)
