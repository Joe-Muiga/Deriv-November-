"""rng_eval.run_eval_more - A2 (legacy comparison), A3 (exact), A4 (robustness), A5 (performance). See run_eval.py for the CLI."""
from __future__ import annotations

import asyncio
import math
import os
import resource
import sys
import time
from multiprocessing import Pool
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rng_core import diag as dg                              # noqa: E402
from rng_core.exact import gens                               # noqa: E402
from rng_core.exact.base import ExactMonitor                  # noqa: E402
from rng_core.exact.gens import GENS, MAPS                    # noqa: E402
from rng_core.exact.runner import run_exact                   # noqa: E402
from rng_core.orchestrator import SymbolOrchestrator, V2Config   # noqa: E402
from rng_eval import zoo                                      # noqa: E402
from rng_eval.run_eval import save, LIFTS                     # noqa: E402
from rng_eval.scoring import top4_mask                        # noqa: E402

T0 = zoo.T0


# ── A2: the legacy predictor on the same streams ────────────────────────────────────────────────
def _legacy_job(a):
    """Fit the legacy MHSymbol on a prefix, then walk forward `hold` ticks. Returns gate (any tradable horizon) + held-out top-4 / top-1 at h=1."""
    import mh_predictor as mp
    kind, lift, s, n_fit, hold = a
    n = n_fit + hold
    if kind == "null":
        d = zoo.null_pcg64(9000 + s, n)
    else:
        d = zoo.SIGNALS[kind](7000 + s, n, lift)
    d = np.asarray(d, np.int64)

    async def go():
        m = mp.MHSymbol("X", 5, 10, 0.998)
        t = time.time()
        await m.fit(d[:n_fit], float(T0 + n_fit), deadline=0.0)
        fit_s = time.time() - t
        trad = [h for h, hs in m.hs.items() if hs.tradable]
        k4 = k1 = cnt = 0
        for i in range(n_fit, n):
            hs = m.hs.get(1)
            if hs is not None and hs.cur is not None:
                P = np.asarray(hs.cur[1])
                k4 += int(top4_mask(P[None, :])[0, d[i]])
                k1 += int(int(np.argmax(P)) == d[i])
                cnt += 1
            m.on_tick(int(d[i]), float(T0 + i + 1))
        return {"kind": kind, "lift": lift, "s": s, "n_fit": n_fit, "gate": bool(trad), "tradable_h": trad, "top4": k4, "top1": k1,
                "n": cnt, "fit_s": round(fit_s, 1)}

    return asyncio.run(go())


def task_legacy(streams=6, procs=2, hold=3000, **_):
    t0 = time.time()
    jobs = []
    for nf in (10000, 30000):
        for s in range(streams * 4):
            jobs.append(("null", 0.0, s, nf, hold))
        for fam in ("bias", "markov2", "lag17"):
            for l in LIFTS:
                for s in range(streams):
                    jobs.append((fam, l, s, nf, hold))
    with Pool(procs) as p:
        rs = p.map(_legacy_job, jobs, chunksize=1)
    out = {"seconds": round(time.time() - t0, 1), "hold": hold, "rows": rs}
    agg = {}
    for r in rs:
        k = f"{r['kind']}@{r['lift']}@n{r['n_fit']}"
        a = agg.setdefault(k, {"streams": 0, "gate": 0, "top4": 0, "top1": 0, "n": 0})
        a["streams"] += 1
        a["gate"] += int(r["gate"])
        a["top4"] += r["top4"]
        a["top1"] += r["top1"]
        a["n"] += r["n"]
    out["agg"] = agg
    save("legacy", out)
    return out


# ── A3: exact solvers ───────────────────────────────────────────────────────────────────────────
LADDER_SMALL = (60, 100, 150, 200, 300, 500, 800, 1200, 2000, 4000)
LADDER_MT = (10000, 16000, 24000, 32000, 48000)
MT_LIKE = ("mt19937", "pyrandom")


def _exact_job(a):
    name, mp = a
    seed = 123457
    ladder = LADDER_MT if name in MT_LIKE else LADDER_SMALL
    N = max(ladder) + 10000
    d = zoo.weak_digits(name, mp, seed, N).astype(np.int64)
    rec = {"gen": name, "map": mp, "solved": False, "tried_n": list(ladder), "seed": seed}
    for n in ladder:
        t = time.time()
        r = run_exact(d[:n], None, 60.0, False, name in MT_LIKE, {})
        if r["fit"]:
            f = r["fit"]
            mon = ExactMonitor(f, [int(x) for x in d[:n]])
            hit = ins = tot = 0
            for x in d[n:n + 10000]:
                pr = mon.predict(1)
                tot += 1
                if pr is not None:
                    if pr[0] == "digit":
                        hit += int(pr[1] == int(x))
                    else:
                        ins += int(int(x) in pr[1])
                mon.observe(int(x))
            rec.update({"solved": True, "n_required": n, "fit_name": f["name"], "kind": f["kind"], "fit_map": f["map"],
                        "n_fit": f["n_fit"], "solve_s": round(time.time() - t, 2), "top1_next10000": hit / tot,
                        "set_hit_next10000": ins / tot})
            break
        rec["last_s"] = round(time.time() - t, 2)
    return rec


def _time_job(a):
    name, mp = a
    out = {"gen": name, "map": mp, "solved": False}
    for offset in (3,):
        seed = T0 - offset
        spec = GENS[name]
        d = gens.digits_from_words(spec, mp, spec.fn(seed, 1500)).astype(np.int64)
        for n in (100, 200, 400, 800, 1500):
            t = time.time()
            r = run_exact(d[:n], float(T0), 60.0, False, False, {}, order=["seed_scan"])
            if r["fit"]:
                out.update({"solved": True, "n_required": n, "fit_name": r["fit"]["name"], "solve_s": round(time.time() - t, 2)})
                break
    return out


def task_exact(procs=2, **_):
    t0 = time.time()
    combos = [(g, m) for g in GENS for m in MAPS]
    with Pool(procs) as p:
        rows = p.map(_exact_job, combos, chunksize=1)
    tcombos = [(g, m) for g in ("minstd", "randu", "glibc_lcg", "msvc", "nr_lcg", "borland", "java", "mt19937", "pyrandom") for m in MAPS]
    with Pool(procs) as p:
        trows = p.map(_time_job, tcombos, chunksize=1)
    out = {"seconds": round(time.time() - t0, 1), "arbitrary_seed": rows, "time_seeded": trows}
    save("exact", out)
    return out


def _mt_job(a):
    """State-level MT19937 recovery (GF(2) elimination of the 19937-bit state) with a realistic budget; run ALONE (no contention)."""
    name, mp = a
    seed = 123457
    rec = {"gen": name, "map": mp, "solved": False, "rungs": []}
    d = zoo.weak_digits(name, mp, seed, 60000).astype(np.int64)
    for n in (16000, 24000):
        t = time.time()
        r = run_exact(d[:n], None, 600.0, False, True, {}, order=["gf2_mt"])
        rec["rungs"].append({"n": n, "s": round(time.time() - t, 1), "ok": bool(r["fit"])})
        if r["fit"]:
            f = r["fit"]
            mon = ExactMonitor(f, [int(x) for x in d[:n]])
            hit = tot = 0
            for x in d[n:n + 10000]:
                pr = mon.predict(1)
                tot += 1
                hit += int(pr is not None and pr[0] == "digit" and pr[1] == int(x))
                mon.observe(int(x))
            rec.update({"solved": True, "n_required": n, "n_fit": f["n_fit"], "fit_name": f["name"], "solve_s": rec["rungs"][-1]["s"],
                        "top1_next10000": hit / tot})
            break
    if (name, mp) == ("mt19937", "top"):                  # bit-level model (BM) timing at 48000 ticks, measured once
        t = time.time()
        r = run_exact(d[:48000], None, 400.0, False, False, {}, order=["bm_bits"])
        rec["bm_bits_48000"] = {"s": round(time.time() - t, 1), "ok": bool(r["fit"]), "name": (r["fit"] or {}).get("name")}
    return rec


def _load_partial(name):
    import json
    p = os.path.join(ROOT, "docs", "v2", "results", name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def task_mt(procs=1, **_):
    t0 = time.time()
    combos = [(g, m) for g in MT_LIKE for m in MAPS]
    prev = _load_partial("mt")
    rows = list(prev["rows"]) if prev else []
    done = {(r["gen"], r["map"]) for r in rows}
    for c in combos:
        if c in done:
            continue
        rows.append(_mt_job(c))
        print(rows[-1], flush=True)
        save("mt", {"rows": rows, "seconds": round(time.time() - t0, 1)})
    return {"rows": rows}


def task_time(procs=2, **_):
    tcombos = [(g, m) for g in ("minstd", "randu", "glibc_lcg", "msvc", "nr_lcg", "borland", "java", "mt19937", "pyrandom") for m in MAPS]
    with Pool(procs) as p:
        trows = p.map(_time_job, tcombos, chunksize=1)
    out = {"time_seeded": trows}
    save("time_seeded", out)
    return out


def _false_job(a):
    i, n, with_scan = a
    d = zoo.null_pcg64(40_000 + i, n).astype(np.int64)
    r = run_exact(d, float(T0) if with_scan else None, 60.0, False, False, {})
    return bool(r["fit"]), (r["fit"] or {}).get("name")


def task_false_exact(streams=1000, procs=2, **_):
    """Three settings, processed in chunks and saved after every chunk (resumable)."""
    t0 = time.time()
    plan = {"n3000_no_scan": [(i, 3000, False) for i in range(streams)],
            "n3000_with_seed_scan": [(streams + i, 3000, True) for i in range(200)],
            "n20000": [(2 * streams + i, 20000, False) for i in range(100)]}
    prev = _load_partial("false_exact_parts") or {}
    res = {k: list(prev.get(k, [])) for k in plan}
    with Pool(procs) as p:
        for k, jobs in plan.items():
            todo = jobs[len(res[k]):]
            for i in range(0, len(todo), 50):
                res[k] += [list(x) for x in p.map(_false_job, todo[i:i + 50], chunksize=2)]
                save("false_exact_parts", res)
                print(k, len(res[k]), "/", len(jobs), f"{time.time() - t0:.0f}s", flush=True)
    out = {"seconds": round(time.time() - t0, 1), "n3000_no_scan": [sum(x[0] for x in res["n3000_no_scan"]), len(res["n3000_no_scan"])],
           "n3000_with_seed_scan": [sum(x[0] for x in res["n3000_with_seed_scan"]), len(res["n3000_with_seed_scan"])],
           "n20000": [sum(x[0] for x in res["n20000"]), len(res["n20000"])],
           "false_names": [x[1] for k in res for x in res[k] if x[0]]}
    save("false_exact", out)
    return out


# ── drivers for the real online orchestrator (A4, A5) ───────────────────────────────────────────
def drive(d, epochs=None, quotes=None, cfg=None, solve=True, diag_every_run=True):
    """Feed a digit stream to one SymbolOrchestrator, running diag / exact requests synchronously. Returns (orch, timeline)."""
    cfg = cfg or V2Config(diag_first=3000, B_compress=40, B_mc=39)
    o = SymbolOrchestrator("T", cfg)
    n = len(d)
    epochs = np.arange(n, dtype=float) + T0 if epochs is None else epochs
    quotes = np.full(n, np.nan) if quotes is None else quotes
    tl = []
    pend_pred = None
    pend_top4 = None
    hit = tot = hit4 = 0
    for i in range(n):
        if o.status == "EXACT_STATE" and pend_pred is not None:
            tot += 1
            hit += int(pend_pred == int(d[i]))
            hit4 += int(pend_top4 is not None and int(d[i]) in pend_top4)
        idx = o.on_tick(float(epochs[i]), float(quotes[i]), int(d[i]))
        pr = o.prediction(1) if idx >= 0 else None
        pend_pred = int(np.argmax(pr.dist)) if (pr is not None and pr.status == "EXACT_STATE") else None
        pend_top4 = tuple(pr.top4) if (pr is not None and pr.status == "EXACT_STATE") else None
        req = o.diag_request()
        if req:
            rep = dg.run_diagnostics(req["digits"], req["run_index"], req["alpha"], req["quotes"], req["decimals"], 0,
                                     req["B_compress"], req["B_mc"])
            o.apply_diag(req, rep)
        rq = o.exact_request() if solve else None
        if rq:
            res = run_exact(np.array(rq["digits"]), rq["epoch0"], 20.0, False, rq["try_mt"], rq["progress"])
            o.apply_exact(rq, {"ok": True, **res})
        if i % 50 == 0 or (tl and tl[-1][1] != o.status):
            if not tl or tl[-1][1] != o.status:
                tl.append((i, o.status))
    return o, tl, (hit, tot, hit4)


def _robust_cases():
    cases = []
    base = ("xorshift32", "top")
    n = 6000
    d = zoo.weak_digits(*base, 5, n).astype(np.int64)
    ep = np.arange(n, dtype=float) + T0
    q = zoo.quotes_for_digits(d, 5)
    for rate in (0.001, 0.01):
        e2, q2, d2, k = zoo.hostile_missed(ep, q, d, rate, 7)
        cases.append((f"missed {rate:.1%} ({k} ticks)", d2, e2, q2))
    e2, q2, d2, k = zoo.hostile_dups(ep, q, d, 0.01, 8)
    cases.append((f"duplicates 1% ({k} dups)", d2, e2, q2))
    dr, pos = zoo.hostile_reseed(*base, 5, n, 2000)
    cases.append((f"reseed every 2000 (at {pos})", dr.astype(np.int64), ep, zoo.quotes_for_digits(dr, 5)))
    di = zoo.hostile_interleaved(*base, 5, n, 0, 2).astype(np.int64)
    cases.append(("2 symbols share one generator (symbol 0)", di, ep, zoo.quotes_for_digits(di, 5)))
    for pw in (0.9, 0.5):
        dm = zoo.hostile_mixed(*base, 5, n, pw).astype(np.int64)
        cases.append((f"CSPRNG mix: weak with p={pw}", dm, ep, zoo.quotes_for_digits(dm, 5)))
    dn = zoo.null_pcg64(3, n).astype(np.int64)
    cases.append(("control: null", dn, ep, zoo.quotes_for_digits(dn, 3)))
    return cases


def task_robust(procs=1, **_):
    t0 = time.time()
    rows = []
    for name, d, ep, q in _robust_cases():
        o, tl, (hit, tot, hit4) = drive(d, ep, None)
        s = o.snapshot()
        rows.append({"case": name, "n": len(d), "final_status": o.status, "timeline": tl[:12], "gaps": o.n_gaps, "dups": s["dups"],
                     "exact_drops": o.n_exact_drops, "first_exact_n": o.first_exact_n,
                     "top1_while_exact": (hit / tot if tot else None), "top4_while_exact": (hit4 / tot if tot else None), "n_exact_ticks": tot, "scoreboard": s["scoreboard"],
                     "events": [e for _, e in list(o.events)[-6:]]})
        print(rows[-1]["case"], rows[-1]["final_status"], rows[-1]["top1_while_exact"], flush=True)
    out = {"seconds": round(time.time() - t0, 1), "rows": rows}
    save("robust", out)
    return out


# ── A5: performance ─────────────────────────────────────────────────────────────────────────────
def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def task_perf(**_):
    from rng_core.adapter import make_symbol
    import types
    out = {}
    cfgm = types.SimpleNamespace(V2_ALPHA=0.01, V2_HMAX=10, V2_STORE_CAPACITY=300000, V2_DIAG_FIRST=10 ** 9, V2_VALIDATE_N=24,
                                 V2_EXACT_MASS=0.999, V2_ACT_ON_NO_STRUCTURE=0, V2_SOLVER_BUDGET_SECONDS=20, V2_SOLVER_MAX_MB=400,
                                 V2_EXHAUSTIVE=0, V2_RANK_MODE="off", V2_MIN_EDGE=0.0)

    async def per_tick(status_stream, label, n=6000, hist=5000):
        d = status_stream
        q = zoo.quotes_for_digits(d, 1)
        m = make_symbol("P", 5, 2, cfgm)
        ticks = [{"epoch": T0 + i, "quote": float(q[i])} for i in range(hist)]
        await m.fit_ticks(ticks, d[:hist], 2)
        tt_core, tt_engine = [], []
        for i in range(hist, n):
            a = time.perf_counter()
            m.on_tick(int(d[i]), T0 + i, float(q[i]))
            b = time.perf_counter()
            for h in range(1, 6):                              # what the engine reads per tick and horizon
                hs = m.hs[h]
                _ = hs.cur
                _ = hs.tradable
            c = time.perf_counter()
            tt_core.append(b - a)
            tt_engine.append(c - a)
        r = {"core_ms_mean": 1e3 * float(np.mean(tt_core)), "core_ms_p99": 1e3 * float(np.percentile(tt_core, 99)),
             "with_engine_reads_ms_mean": 1e3 * float(np.mean(tt_engine)), "with_engine_reads_ms_p99": 1e3 * float(np.percentile(tt_engine, 99)),
             "max_ms": 1e3 * float(np.max(tt_engine))}
        out[label] = r
        print(label, r, flush=True)

    asyncio.run(per_tick(zoo.null_pcg64(1, 6000).astype(np.int64), "null (NO_STRUCTURE)"))
    asyncio.run(per_tick(zoo.sig_bias(2, 6000, 0.2).astype(np.int64), "bias (STAT_SIGNAL)"))

    # loop blocking while a 100k-digit diagnostic runs in the thread, and memory with a full 300k store
    async def lag_and_mem():
        d15 = zoo.null_pcg64(4, 15_000).astype(np.int64)
        q15 = zoo.quotes_for_digits(d15, 2)
        b15 = rss_mb()
        t = time.time()
        m15 = make_symbol("M15", 5, 2, cfgm)
        await m15.fit_ticks([{"epoch": T0 + i, "quote": float(q15[i])} for i in range(len(d15))], d15, 2)
        out["fit_15k"] = {"seconds": round(time.time() - t, 1), "peak_rss_mb_before": round(b15, 1), "peak_rss_mb_after": round(rss_mb(), 1)}
        print(out["fit_15k"], flush=True)
        d = zoo.null_pcg64(5, 100_000).astype(np.int64)
        lag, lag_t = [], []
        stop = asyncio.Event()

        async def probe():
            while not stop.is_set():
                t = time.perf_counter()
                await asyncio.sleep(0.005)
                lag.append(time.perf_counter() - t - 0.005)
                lag_t.append(time.perf_counter())

        pt = asyncio.create_task(probe())
        t = time.time()
        t_start = time.perf_counter()
        await asyncio.to_thread(dg.run_diagnostics, d, 1, 0.005, None, 2, 0, 200, 99)
        el = time.time() - t
        stop.set()
        await pt
        top3 = np.argsort(lag)[-3:][::-1]
        out["diag_100k_thread"] = {"seconds": round(el, 1), "max_loop_lag_ms": 1e3 * max(lag), "p99_loop_lag_ms": 1e3 * float(np.percentile(lag, 99)),
                                   "worst3_ms_at_s": [[round(1e3 * lag[i], 1), round(lag_t[i] - t_start, 2)] for i in top3]}
        print(out["diag_100k_thread"], flush=True)
        before = rss_mb()
        m = make_symbol("M", 5, 2, cfgm)
        q = zoo.quotes_for_digits(d, 2)
        t = time.time()
        await m.fit_ticks([{"epoch": T0 + i, "quote": float(q[i])} for i in range(len(d))], d, 2)
        out["fit_100k"] = {"seconds": round(time.time() - t, 1), "rss_mb_before": round(before, 1), "rss_mb_after": round(rss_mb(), 1)}
        print(out["fit_100k"], flush=True)

    asyncio.run(lag_and_mem())
    save("perf", out)
    return out


def run(task: str, **kw):
    fn = {"legacy": task_legacy, "exact": task_exact, "false_exact": task_false_exact, "robust": task_robust, "perf": task_perf,
          "mt": task_mt, "time": task_time}[task]
    return fn(**kw)
