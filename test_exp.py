"""test_exp.py - pre-registered test pipeline checks (run: python test_exp.py).

  1. negative control : a pure uniform random digit stream must show NO confirmed edge and ~the base rate
  2. parity           : batch vs online predictions agree (test_mh.test_batch_equals_online is re-run here)
  3. alignment        : realized_h labelling (pure + through MHEngine._inspect_settlement and the ledger)
  + frozen-config hash, zone rule, append-only ledger + restore, stopping rule, cache-clear safety
"""
import asyncio
import os
import sqlite3
import sys
import tempfile
import time
import types

import numpy as np

for m in ("aiohttp", "psycopg", "psycopg2", "pytz", "websockets", "websockets.exceptions"):
    try:
        __import__(m)
    except Exception:
        sys.modules[m] = types.ModuleType(m)
ws = sys.modules["websockets.exceptions"]
for n in ("ConnectionClosed", "InvalidStatus"):
    ws.__dict__.setdefault(n, type(n, (Exception,), {}))

import config
import exp_mode
import exp_stats
import mh_engine as me
import mh_predictor as mp
import predictor as pred
import verdict


# ── 1. negative control ──────────────────────────────────────────────────────
def test_negative_control_uniform_stream():
    rng = np.random.default_rng(2024)
    D = rng.integers(0, 10, 15000 + 4000)
    sym = mp.MHSymbol("NEG", 1, 1)
    asyncio.run(sym.fit(D[:15000], 1.0))
    hs = sym.hs[1]
    assert hs.cal is not None
    ci_o, ci_u = mp.COVERS.index(exp_mode.OVER5), mp.COVERS.index(exp_mode.UNDER4)
    exp = exp_stats.Experiment()
    thr, ratio, stake = 0.40, 2.38, 1.0            # lowered threshold so the noise stream produces enough trades
    top = n = 0
    for t in range(15000, len(D)):
        p = hs.cur[1]
        zd = exp_mode.zone_decision(p, lambda x: hs.cal[ci_o](x)[0], lambda x: hs.cal[ci_u](x)[0], thr)
        d = int(D[t])
        top += int(int(np.argmax(p)) == d)
        n += 1
        if zd.fire:
            won = exp_mode.zone_hit("over" if zd.cand == exp_mode.OVER5 else "under", d)
            exp.add_settlement({"arm": "test", "sym": "NEG", "won": won, "stake": stake, "ratio": ratio, "align": "ALIGNED",
                                "profit": stake * (ratio - 1) if won else -stake, "h": 1, "realized_h": 1})
        sym.on_tick(d)
    v = exp.verdict()
    assert v["n"] > 800, v
    assert v["state"] != verdict.PROFIT, v                                   # no confirmed edge on pure noise
    se = (0.4 * 0.6 / v["n"]) ** 0.5
    assert abs(v["win_rate"] - exp_mode.BASE_RATE) < 4 * se, v               # about the base rate
    assert abs(top / n - 0.10) < 4 * (0.09 / n) ** 0.5, top / n              # shadow top-1 ~ chance
    b = exp.balance_path()                                                   # actual path stays inside a wide band
    assert abs(b["actual"][-1] - b["expected"][-1]) < (b["hi"][-1] - b["lo"][-1])


# ── 2. parity ────────────────────────────────────────────────────────────────
def test_batch_equals_online_parity():
    import test_mh
    test_mh.test_batch_equals_online()


# ── 3. alignment ─────────────────────────────────────────────────────────────
def test_alignment_labels_pure():
    L = exp_mode.label_alignment
    assert L(1, 1) == "ALIGNED" and L(2, 1) == "LATE" and L(5, 1) == "LATE"
    assert L(None, 1) == "UNKNOWN" and L(0, 1) == "ANOMALY"


class _Cal:
    def __call__(self, x):
        return 0.55, 0.5


def _engine(path):
    import exp_ledger
    e = me.MHEngine()
    e.ledger = exp_ledger.Ledger(path=path, url="").open()
    e._decimals["R_10"] = 3
    e._ep2idx["R_10"] = {1000 + i: 100 + i for i in range(10)}               # epoch -> tick idx
    e.mh = {"R_10": types.SimpleNamespace(hs={2: types.SimpleNamespace(cal=[_Cal() for _ in mp.COVERS])})}
    return e


def _settle(e, cid, exit_idx, profit):
    p2 = [0.04] * 10
    for d in (6, 7, 8, 9):
        p2[d] = 0.14
    info = {"sym": "R_10", "h": 1, "cand": exp_mode.OVER5, "sig_idx": 100, "p_cal": 0.5, "arm": "test", "stake": 1.0,
            "ratio": 2.4, "p_by_h": {1: [0.1] * 10, 2: p2}}
    poc = {"profit": profit, "exit_tick_time": None if exit_idx is None else 1000 + (exit_idx - 100),
           "contract_type": pred.contract_type(exp_mode.OVER5), "barrier": "5", "exit_tick": 123.457}
    e._inspect_settlement(cid, info, poc)


def test_alignment_through_engine_and_ledger():
    exp_stats.EXP.reset()
    path = os.path.join(tempfile.mkdtemp(), "l.sqlite")
    e = _engine(path)
    _settle(e, "c1", 101, 1.4)          # exit on the intended tick: ALIGNED
    _settle(e, "c2", 102, -1.0)         # one tick later: LATE, re-scored against h=2
    _settle(e, "c3", None, -1.0)        # exit epoch not found: UNKNOWN
    e.ledger.flush()
    rows = e.ledger.settlements()
    assert [r["align"] for r in rows] == ["ALIGNED", "LATE", "UNKNOWN"]
    assert [r["realized_h"] for r in rows] == [1, 2, None]
    assert rows[1]["p_rescored"] == 0.55 and rows[0]["p_rescored"] is None
    s = exp_stats.EXP.summary()["three_numbers"]
    assert s["aligned"]["n"] == 1 and s["late"]["n"] == 1 and s["unlabelled"] == 1
    # ── ledger is append-only, and a fresh engine restores counters from it ──
    con = sqlite3.connect(path)
    try:
        con.execute("UPDATE exp_settlements SET won = 1")
        raise AssertionError("update must be refused")
    except sqlite3.DatabaseError:
        pass
    con.close()
    exp_stats.EXP.reset()
    e.ledger.event("mute", sym="R_10", z=-2.5, n=40)
    e.ledger.flush()
    e2 = me.MHEngine()
    e2.ledger = e.ledger
    e2._exp_restore()
    assert e2._n_trades == 3 and e2._n_wins == 1 and "R_10" in e2._muted and exp_stats.EXP.n_test() == 3


# ── frozen config / zone rule / stopping / cache safety ──────────────────────
def test_zone_rule_and_hash():
    cal = lambda x: x
    p = np.full(10, 0.04)
    p[[6, 7, 8, 9]] = 0.14                                   # P(6-9) = 0.56, P(0-3) = 0.16, argmax is a 6-9 digit
    d = exp_mode.zone_decision(p, cal, cal, 0.45)
    assert d.fire and d.cand == exp_mode.OVER5 and abs(d.zo - 0.56) < 1e-9
    assert not exp_mode.zone_decision(p, cal, cal, 0.60).fire                      # below the fixed threshold
    q = p.copy(); q[5] = 0.30; q /= q.sum()
    assert exp_mode.zone_decision(q, cal, cal, 0.0).reason == "skip_45_dominant"   # 4/5-dominant ticks skipped
    assert exp_mode.zone_decision(p, None, None, 0.45).reason == "no_calibration"
    h1 = exp_mode.config_hash()
    old = config.TEST_ZONE_THRESHOLD
    config.TEST_ZONE_THRESHOLD = 0.5
    try:
        assert exp_mode.config_hash() != h1                  # any change to the frozen config changes the hash
    finally:
        config.TEST_ZONE_THRESHOLD = old
    assert exp_mode.config_hash() == h1


def test_stopping_rule_and_demo_only():
    old = config.TEST_MAX_TRADES
    config.TEST_MAX_TRADES = 3
    try:
        e = exp_stats.Experiment()
        for i in range(3):
            assert e.stop_reason(0) == "" or i == 2
            e.add_settlement({"arm": "test", "sym": "X", "won": False, "stake": 1, "ratio": 2.4, "profit": -1.0})
        assert "TEST_MAX_TRADES" in e.stop_reason(0)
        assert e.finalize("done")["verdict"]["state"] == verdict.NOT_ENOUGH and e.is_done()
    finally:
        config.TEST_MAX_TRADES = old
    import set_engine
    config.TEST_MODE, mode = True, config.DERIV_ACCOUNT_MODE
    config.DERIV_ACCOUNT_MODE, real = "real", config.SET_ALLOW_REAL
    config.SET_ALLOW_REAL = True
    try:
        assert set_engine.account_allowed()[0] is False      # SET_ALLOW_REAL never honoured in test mode
    finally:
        config.TEST_MODE, config.DERIV_ACCOUNT_MODE, config.SET_ALLOW_REAL = False, mode, real


def test_redeploy_never_clears_cache():
    import redeploy_guard
    assert redeploy_guard.NO_CLEAR == {"clearCache": "do_not_clear"}
    src = open("restart_scheduler.py").read() + open("redeploy_guard.py").read()
    assert '"clearCache": "clear"' not in src and "clearCache='clear'" not in src      # no request body that clears the cache
    assert config.AUTO_REDEPLOY_SECONDS == 420 and config.AUTO_REDEPLOY_ENABLED     # 7-minute cycle, always on


def test_redeploy_cycle_repeats_forever():
    """The scheduler must raise 'pending' again after every redeploy, cycle after cycle."""
    import asyncio, restart_scheduler as rs, redeploy_guard as rg
    saved = (config.REDEPLOY_INTERVAL_SECONDS, config.REDEPLOY_BUY_BLOCK_SECONDS, rg.preflight)
    config.REDEPLOY_INTERVAL_SECONDS, config.REDEPLOY_BUY_BLOCK_SECONDS = 0.3, 0.0
    rg.preflight = lambda: (True, "")
    fired = []
    async def go():
        t = asyncio.ensure_future(rs.run_scheduler())
        for _ in range(3):                                   # three full cycles
            for _ in range(200):
                if rs.is_redeploy_pending():
                    break
                await asyncio.sleep(0.05)
            assert rs.is_redeploy_pending()
            fired.append(time.time())
            rs._pending = False                              # what trigger_redeploy() does once the deploy is requested
            await asyncio.sleep(0.05)
        t.cancel()
    try:
        asyncio.run(go())
    finally:
        config.REDEPLOY_INTERVAL_SECONDS, config.REDEPLOY_BUY_BLOCK_SECONDS, rg.preflight = saved
        rs._pending = False
    assert len(fired) == 3 and fired[1] - fired[0] >= 0.25


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            fn()
            print("ok", name, f"({time.time() - t:.1f}s)")
