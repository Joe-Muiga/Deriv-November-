"""
test_predictor.py – no network. Run:  python test_predictor.py   (or pytest)
"""
import math
import os
import re
import subprocess
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import predictor as P
from predictor import Predictor, PredictorParams, Pick, break_even, wilson_lower

RATIO = 1.95
FAIR_CANDS = dict(barriers=(4, 5))        # OVER4 / UNDER5 are the ~50% barrier contracts


def params(**kw):
    base = dict(barriers=(4, 5), min_context_samples=300, confirm_fraction=0.2,
                edge_margin=0.01, wilson_z=2.0, duration_ticks=1, entry_delay_ticks=0,
                min_payout_ratio=1.8, context_sizes=(1, 2, 3), alpha=0.05)
    base.update(kw)
    return PredictorParams(**base)


def run_boots(digits, warmup, live, stride, p, ratio=RATIO):
    """Full flow: fit on a warm-up window, then walk the next `live` ticks making
    decisions with only digits <= t; settle against digit t+horizon."""
    h = p.horizon
    trades = wins = 0
    exp_p = exp_var = 0.0
    picks = []
    pos = 0
    boots = 0
    while pos + warmup + live + h < len(digits):
        pr = Predictor("SIM", p, payout_ratio_fn=lambda c: ratio)
        pr.fit(digits[pos:pos + warmup])
        boots += 1
        for t in range(pos + warmup, pos + warmup + live):
            pr.on_tick(int(digits[t]))
            best = pr.best_with_ratios(lambda c: ratio)
            if best:
                pk, d = best
                won = P.wins_digit(pk.cand, int(digits[t + h]))
                trades += 1
                wins += int(won)
                tp = P.true_probability(pk.cand)
                exp_p += tp
                exp_var += tp * (1 - tp)
                picks.append((pk.cand, pk.context))
        pos += warmup + live
    return dict(boots=boots, trades=trades, wins=wins, exp_wins=exp_p, exp_var=exp_var, picks=picks)


# ── 1. fair data ─────────────────────────────────────────────────────────────
def test_fair_data():
    rng = np.random.default_rng(12345)
    digits = rng.integers(0, 10, 200_000)
    p = params()
    r = run_boots(digits, warmup=5000, live=1000, stride=None, p=p)
    live_ticks = r["boots"] * 1000
    print(f"[fair] boots={r['boots']} live_ticks={live_ticks} trades={r['trades']} wins={r['wins']}")
    assert r["trades"] <= 0.005 * live_ticks, "predictor should trade rarely or never on fair data"
    if r["trades"]:
        z = (r["wins"] - r["exp_wins"]) / math.sqrt(max(r["exp_var"], 1e-9))
        print(f"[fair] hit-rate={r['wins']/r['trades']:.4f} expected={r['exp_wins']/r['trades']:.4f} z={z:.2f}")
        assert abs(z) < 4.0
    else:
        print("[fair] no trades taken (expected on a fair generator)")


# ── 2. planted bias ──────────────────────────────────────────────────────────
def planted_digits(n, seed=7, p_even_after3=0.60):
    rng = np.random.default_rng(seed)
    d = np.empty(n, dtype=np.int64)
    d[0] = rng.integers(0, 10)
    u = rng.random(n)
    base = rng.integers(0, 10, n)
    ev = rng.integers(0, 5, n) * 2
    od = rng.integers(0, 5, n) * 2 + 1
    for t in range(1, n):
        if d[t - 1] == 3:
            d[t] = ev[t] if u[t] < p_even_after3 else od[t]
        else:
            d[t] = base[t]
    return d


def test_biased_data_is_found_and_traded():
    digits = planted_digits(60_000)
    p = params()
    # 20k warm-up so the planted context clears Bonferroni (see CHANGES.md: with
    # only 5000 ticks a 0.60 bias is borderline against the corrected threshold).
    pr = Predictor("SIM", p, payout_ratio_fn=lambda c: RATIO)
    rep = pr.fit(digits[:20_000])
    print(f"[biased] fit report: {rep}")
    ci = pr.candidates.index(("EVEN", None))
    assert (1, 3, ci) in pr.selected, "planted context (prev digit 3 -> EVEN) must be tradable"
    r = run_boots(digits[20_000:], warmup=20_000, live=5000, stride=None, p=p)
    print(f"[biased] trades={r['trades']} wins={r['wins']} hit={r['wins']/max(r['trades'],1):.3f}")
    assert r["trades"] > 0
    planted = [x for x in r["picks"] if x == (("EVEN", None), (3,))]
    assert len(planted) >= 0.9 * len(r["picks"]), "trades should be (almost) all the planted context"
    hit = r["wins"] / r["trades"]
    assert hit > break_even(RATIO)


# ── 3. no look-ahead + lag handling ──────────────────────────────────────────
def brute_counts(digs, k, h):
    out = np.zeros((10 ** k, 10), dtype=np.int64)
    for j in range(h + k - 1, len(digs)):
        i = j - h
        ctx = 0
        for x in digs[i - k + 1:i + 1]:
            ctx = ctx * 10 + int(x)
        out[ctx, digs[j]] += 1
    return out


def test_no_lookahead_and_lag_counts():
    rng = np.random.default_rng(3)
    digits = rng.integers(0, 10, 2500)
    for delay in (0, 1):                               # horizon 1 and 2
        p = params(entry_delay_ticks=delay)
        h = p.horizon
        pr = Predictor("SIM", p)
        pr.fit(digits[:1000])
        for t in range(1000, 1400):
            pr.on_tick(int(digits[t]))
            assert pr.context(2) == tuple(int(x) for x in digits[t - 1:t + 1]), "context must end at tick t"
            if t in (1000, 1001, 1100, 1399):
                for k in p.context_sizes:
                    assert (pr.counts_all[k] == brute_counts(digits[:t + 1], k, h)).all(), \
                        f"counts at t={t} (k={k}, h={h}) must equal a recount from digits[:t+1] only"
    print("[lookahead] online counts == brute-force recount from digits <= t, for horizon 1 and 2")
    # decisions at t are independent of anything after t
    a = planted_digits(30_000, seed=11)
    b = a.copy()
    b[25_001:] = np.random.default_rng(1).integers(0, 10, len(b) - 25_001)
    outs = []
    for dd in (a, b):
        pr = Predictor("SIM", params(), payout_ratio_fn=lambda c: RATIO)
        pr.fit(dd[:20_000])
        seq = []
        for t in range(20_000, 25_001):
            pr.on_tick(int(dd[t]))
            seq.append(tuple((x.cand, x.context) for x in pr.rank()))
        outs.append(seq)
    assert outs[0] == outs[1], "changing FUTURE digits changed past predictions"
    print("[lookahead] predictions identical when only future digits differ")


def test_lag_learns_the_right_horizon():
    # parity of d[t] follows parity of d[t-2] (90%); the two interleaved chains are independent,
    # so digit t+1 says nothing about t, while digit t+2 does.
    rng = np.random.default_rng(5)
    n = 40_000
    q = np.zeros(n, dtype=np.int64)
    q[0], q[1] = 0, 1
    flip = rng.random(n) < 0.1
    for t in range(2, n):
        q[t] = q[t - 2] ^ int(flip[t])
    d = rng.integers(0, 5, n) * 2 + q                 # parity = q
    h1 = Predictor("SIM", params(entry_delay_ticks=0), payout_ratio_fn=lambda c: RATIO)
    h2 = Predictor("SIM", params(entry_delay_ticks=1), payout_ratio_fn=lambda c: RATIO)
    h1.fit(d[:30_000]); h2.fit(d[:30_000])
    print(f"[lag] horizon1 tradable={len(h1.selected)}  horizon2 tradable={len(h2.selected)}")
    assert len(h1.selected) == 0, "next-tick target must show no edge in this generator"
    assert len(h2.selected) > 0, "settlement-at-t+2 target must be learned when horizon=2"
    # horizon 2 predicts SAME parity as the context digit
    eo = [(k, ctx, h2.candidates[ci][0]) for (k, ctx, ci) in h2.selected if h2.candidates[ci][0] in ("EVEN", "ODD")]
    assert eo, "Even/Odd contexts must be learned at horizon 2"
    for (k, ctx, kind) in eo:
        assert (kind == "EVEN") == (ctx % 2 == 0)


def test_default_horizon_matches_existing_buy_flow():
    """Existing flow: see tick t -> proposal -> buy (< one tick interval) -> entry tick =
    first tick AFTER the purchase -> a 1-tick contract settles on that tick => t+1."""
    assert PredictorParams().horizon == 1, "default horizon must be 1 (duration 1 + entry delay 0)"
    epochs = [100 + 2 * i for i in range(50)]                 # 2 s ticks (R_100)
    t = 10
    seen_at = epochs[t] + 0.05                                # tick received
    fast = seen_at + 0.40                                     # proposal+buy round trip 0.4 s
    assert P.settlement_index(epochs, t, fast, 1) == t + PredictorParams().horizon
    slow = seen_at + 2.5                                      # round trip longer than a tick
    assert P.settlement_index(epochs, t, slow, 1) == t + 2    # would need ENTRY_DELAY_TICKS=1
    assert P.settlement_index(epochs, t, fast, 5) == t + 5    # duration 5: exit = entry + 4
    print("[lag] default horizon 1 == settlement tick of the real buy flow (round trip < tick interval)")


# ── 4. payout floor / break-even ─────────────────────────────────────────────
def test_payout_floor():
    pr = Predictor("SIM", params())
    pk = Pick(("EVEN", None), 1, (3,), 2000, 0.70, 0.68)
    stake = 1000.0
    assert not pr.accept(pk, 1790.0 / stake).ok, "1.79 must be skipped"
    assert pr.accept(pk, 1800.0 / stake).ok, "1.80 must be accepted"


def test_break_even_values():
    assert abs(break_even(1.8) - 0.5556) < 1e-4
    assert abs(break_even(1.95) - 0.5128) < 1e-4
    pr = Predictor("SIM", params())
    pk = Pick(("EVEN", None), 1, (3,), 2000, 0.60, 0.575)
    d = pr.accept(pk, 1.95)
    assert abs(d.break_even - 0.5128) < 1e-4 and d.ok
    d = pr.accept(pk, 1.8)                                    # be .5556 + .01 = .5656 < .575 still ok
    assert abs(d.break_even - 0.5556) < 1e-4
    pk2 = Pick(("EVEN", None), 1, (3,), 2000, 0.60, 0.56)     # .56 < .5656 -> reject at 1.8
    assert not pr.accept(pk2, 1.8).ok


def test_wilson():
    assert wilson_lower(60, 100, 2.0) < 0.6
    assert wilson_lower(0, 0) == 0.0


# ── 4b. engine flow with a fake Deriv client (no network) ────────────────────
def test_engine_flow_with_fake_client():
    import asyncio, time
    import config
    import bot_engine, restart_scheduler, predictor_stats

    digits = planted_digits(23_000, seed=21)
    WARM = 20_000
    config.WARMUP_TICKS = WARM
    config.PREDICTOR_SYMBOLS = ["R_100"]
    # this scenario plants its edge in EVEN/ODD (1.95x): pin the payout window the test was
    # written for (production defaults are floor 2.0 / ceiling 2.5)
    config.MIN_PAYOUT_RATIO = 1.8
    config.MAX_PAYOUT_RATIO = 0.0

    class Fake:
        balance = 10_000.0
        is_connected = True
        def __init__(self):
            self._contract_symbol_map = {}
            self.tick_cb = None; self.sent = []; self.pending = []; self.cid = 0
        def on_balance(self, cb): pass
        async def connect(self): await asyncio.sleep(3600)
        async def get_tick_history(self, sym, n):
            return [{"epoch": 1000 + 2 * i, "quote": round(1234.50 + int(d) * 0.01, 2)} for i, d in enumerate(digits[:WARM])]
        async def subscribe_ticks(self, sym, cb): self.tick_cb = cb
        async def subscribe_contract(self, cid, cb, symbol=""): self.pending.append((cid, cb))
        def stop_tracking(self, cid): pass
        async def force_check_contract(self, cid): return {}
        async def _send(self, req, timeout=0):
            self.sent.append(req)
            if "proposal" in req:
                cand = {"DIGITEVEN": ("EVEN", None), "DIGITODD": ("ODD", None)}.get(req["contract_type"])
                if cand is None:
                    kind = "OVER" if req["contract_type"] == "DIGITOVER" else "UNDER"
                    cand = (kind, int(req["barrier"]))
                ratio = 1.95 if cand[0] in ("EVEN", "ODD") else 0.975 / P.true_probability(cand)
                return {"proposal": {"id": "p", "payout": req["amount"] * ratio, "ask_price": req["amount"]}}
            if "buy" in req:
                self.cid += 1
                return {"buy": {"contract_id": str(self.cid), "purchase_time": time.time()}}
            raise RuntimeError("unexpected request")

    async def scenario():
        bot_engine.DerivClient = Fake
        predictor_stats._stats = predictor_stats.MinuteStats(local_file=os.devnull)
        eng = bot_engine.BotEngine()
        await eng._init_symbol("R_100")
        fake = eng.client
        assert len(eng.predictors["R_100"].selected) >= 1
        worker = asyncio.create_task(eng._worker("R_100"))
        settle = asyncio.create_task(eng._settle_loop())
        buys = 0
        for t in range(WARM, WARM + 2500):
            ep = 1000 + 2 * t
            fake.tick_cb({"quote": round(1234.50 + int(digits[t]) * 0.01, 2), "epoch": ep})
            # settle contracts bought on the previous tick against THIS tick's digit
            for cid, cb in fake.pending:
                info = eng._open.get(cid)
                if info:
                    won = P.wins_digit(info["cand"], int(digits[t]))
                    profit = info["stake"] * (info["ratio"] - 1) if won else -info["stake"]
                    await eng._on_contract_result(cid, {"proposal_open_contract": {"is_sold": 1, "profit": profit}})
            fake.pending.clear()
            await asyncio.sleep(0.002)
        buys = fake.cid
        snap = eng.stats.snapshot()
        buy_reqs = [r for r in fake.sent if "proposal" in r]
        print(f"[engine] contracts bought={buys} stats_trades={snap['t'][0]} wins={snap['t'][1]} net={snap['t'][3]:.0f}")
        assert buys > 0 and snap["t"][0] == buys and not eng._open
        assert all(r["amount"] == 1000.0 for r in fake.sent if "proposal" in r), "flat 1000.00 stake"
        # redeploy drain: pending flag -> no new trades -> trigger fired once, stats flushed
        calls = []
        restart_scheduler.trigger_redeploy = lambda: calls.append(1)
        restart_scheduler._pending = True
        await asyncio.sleep(3.5)
        assert calls == [1], calls
        before = fake.cid
        fake.tick_cb({"quote": 1234.53, "epoch": 10**7})
        await asyncio.sleep(0.05)
        assert fake.cid == before, "no new trades while a redeploy is pending/halted"
        worker.cancel(); settle.cancel()
        restart_scheduler._pending = False
    asyncio.run(scenario())


# ── 4c. minute stats + dashboard endpoint ────────────────────────────────────
def test_stats_analysis_and_endpoint():
    import predictor_stats as PS
    st = PS.MinuteStats(local_file=os.devnull)
    a = PS.analyze(st.snapshot())
    assert a["window"]["status"] == "insufficient data" and a["overall"]["trades"] == 0
    base = 3600 * 5                                     # 05:00 UTC
    for minute in (55, 56, 57, 58, 59, 0, 1, 2, 3, 4):  # window wrapping past :59
        for i in range(40):
            st.record(base + minute * 60 + 5 if minute < 55 else base - 3600 + minute * 60 + 5,
                      1000.0, 1.95, 950.0 if i % 10 < 7 else -1000.0)
    a = PS.analyze(st.snapshot(), min_trades_per_minute=30, window_minutes=10)
    w = a["window"]
    assert (w["start"], w["end"]) == (55, 4) and w["trades"] == 400, w
    assert a["minutes"][57]["reliable"] and not a["minutes"][20]["reliable"]
    assert abs(a["overall"]["break_even"] - 1 / 1.95) < 1e-6
    assert w["beats_break_even"] is True            # 70% win rate, LB ~0.65 > 0.5128
    assert len(st.serialize()) < 3000               # compact enough for an env var
    import keep_alive, config
    cl = keep_alive.app.test_client()
    pin = keep_alive._expected_pin()
    if pin:
        assert cl.get("/api/predictor_stats").status_code == 403
    r = cl.get("/api/predictor_stats", headers={"X-Pin": pin})
    assert r.status_code == 200 and "overall" in r.get_json()
    assert cl.get("/predictor").status_code == 200
    assert cl.get("/kill-switch").status_code == 404


# ── 5. smoke import + grep ───────────────────────────────────────────────────
def test_digit_helpers_match_signal_engine():
    import signal_engine as se
    for q, dec in ((123.45, 2), (9876.5432, 4), (1.1, 3), (0.007, 3)):
        assert P.last_digit(q, dec) == se._last_digit(q, dec)
    assert P.infer_decimals([100.12, 100.5, 101.07]) == 2


def test_smoke_imports_and_no_legacy_references():
    root = os.path.dirname(os.path.abspath(__file__))
    for m in ("config", "signal_engine", "bot_engine", "keep_alive", "main", "predictor",
              "predictor_stats", "restart_scheduler"):
        __import__(m)
    pat = re.compile(r"donkey|fixed_cycle", re.I)
    hits = []
    for fn in sorted(os.listdir(root)):
        if fn.endswith((".py", ".md", ".yaml", ".html", ".txt")) and fn not in ("test_predictor.py", "apply_patch.py", "CHANGES.md"):
            for i, line in enumerate(open(os.path.join(root, fn), errors="ignore"), 1):
                if pat.search(line):
                    hits.append(f"{fn}:{i}: {line.strip()[:80]}")
    print("[grep donkey|fixed_cycle] hits:", hits if hits else "none")
    assert not hits, hits


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                fails += 1; print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if fails else 0)
