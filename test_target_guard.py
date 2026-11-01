"""
test_target_guard.py – offline tests for the TARGET_BALANCE stop. Run: python test_target_guard.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DERIV_API_TOKEN", "x" * 20)

import importlib
import target_guard as tg


def _reset():
    importlib.reload(tg)
    for k in ("TARGET_BALANCE", "RENDER_API_KEY", "RENDER_SERVICE_ID"):
        os.environ.pop(k, None)


def test_target_parsing():
    _reset()
    assert tg.target() is None
    for raw, want in (("5000", 5000.0), (" 5,000.50 ", 5000.5), ("$250", 250.0), ("0", None),
                      ("-5", None), ("abc", None), ("", None)):
        os.environ["TARGET_BALANCE"] = raw
        assert tg.target() == want, (raw, tg.target())
    os.environ["TARGET_BALANCE"] = "100"
    assert tg.should_stop(100.0) and tg.should_stop(100.01) and not tg.should_stop(99.99)
    os.environ.pop("TARGET_BALANCE")
    assert not tg.should_stop(1e9)                    # feature off


def test_mark_reached_runs_once():
    _reset()
    os.environ["TARGET_BALANCE"] = "100"
    assert tg.mark_reached(101.0) is True
    assert tg.mark_reached(102.0) is False
    assert tg.is_reached()


def test_send_blocks_buy_and_proposal_after_target():
    _reset()
    os.environ["TARGET_BALANCE"] = "100"
    from deriv_client import DerivClient
    c = DerivClient()
    tg.mark_reached(100.0)
    for payload in ({"buy": "1", "price": 1}, {"proposal": 1}):
        try:
            asyncio.run(c._send(payload))
        except RuntimeError as exc:
            assert "TARGET_REACHED" in str(exc), exc
        else:
            raise AssertionError("order was not blocked")


class FakeClient:
    def __init__(self, bal):
        self.balance = bal
        self.is_connected = True
        self._cbs = []
        self.shutdown_called = False
        self.connection_count = 1

    def on_balance(self, cb):
        self._cbs.append(cb)

    async def fetch_balance_live(self, timeout=8.0):
        for cb in self._cbs:
            cb(self.balance)
        return self.balance

    async def shutdown(self, reason=""):
        self.shutdown_called = True
        self.is_connected = False

    async def connect(self):
        await asyncio.sleep(3600)


def _engine(bal):
    import bot_engine
    eng = bot_engine.BotEngine()
    eng.client = FakeClient(bal)
    return eng


def test_live_balance_update_triggers_stop_with_open_trades():
    _reset()
    os.environ["TARGET_BALANCE"] = "1000"
    calls = []
    tg.suspend_service = lambda: calls.append("suspend") or True

    async def go():
        eng = _engine(900.0)
        eng._open["123"] = {"sym": "R_100"}              # an open trade must NOT prevent the stop
        eng.client.on_balance(eng._on_balance)
        assert not eng._blocked() or eng.client.is_connected is True
        eng.client.balance = 999.99
        await eng.client.fetch_balance_live()
        assert not tg.is_reached()
        eng.client.balance = 1000.0
        await eng.client.fetch_balance_live()
        assert tg.is_reached() and eng._blocked()        # blocked instantly, same tick
        await asyncio.wait_for(eng._target_done.wait(), 5)
        assert eng.client.shutdown_called and calls == ["suspend"]
    asyncio.run(go())


def test_already_above_target_at_boot_never_trades():
    _reset()
    os.environ["TARGET_BALANCE"] = "500"
    calls = []
    tg.suspend_service = lambda: calls.append("suspend") or True
    import bot_engine

    class Boot(FakeClient):
        pass

    async def go():
        eng = bot_engine.BotEngine()
        eng.client = Boot(750.0)
        inited = []
        async def no_init(sym): inited.append(sym)
        eng._init_symbol = no_init
        await asyncio.wait_for(eng.run(), 10)
        assert inited == [], "symbols must not even be warmed up"
        assert tg.is_reached() and eng.client.shutdown_called and calls == ["suspend"]
    asyncio.run(go())


def test_suspend_service_without_api_key_is_safe():
    _reset()
    assert tg.suspend_service() is False
    assert "no RENDER_API_KEY" in tg.snapshot()["suspend_result"]


def test_suspend_service_calls_render_api():
    _reset()
    os.environ["RENDER_API_KEY"] = "rnd_x"
    os.environ["RENDER_SERVICE_ID"] = "srv-abc"
    seen = {}

    class R:
        status_code = 202
        text = ""

    def fake_post(url, headers=None, timeout=None):
        seen["url"], seen["auth"] = url, headers["Authorization"]
        return R()
    tg.requests.post = fake_post
    assert tg.suspend_service() is True
    assert seen == {"url": "https://api.render.com/v1/services/srv-abc/suspend", "auth": "Bearer rnd_x"}


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
