"""rng_eval.run_eval - evaluation harness for the v2 predictor (acceptance criteria A1-A5).

    python -m rng_eval.run_eval null        [--n 12000 --streams 1000]      A1  null calibration
    python -m rng_eval.run_eval null_long   [--n 60000 --streams 60]        A1  alpha-spending check on long streams
    python -m rng_eval.run_eval power       [--n 60000 --streams 16]        A2  detection / accuracy vs lift (bias, markov2, lag17)
    python -m rng_eval.run_eval legacy      [--streams 6]                   A2  the LEGACY predictor on the same streams
    python -m rng_eval.run_eval exact                                       A3  exact solvers per generator x map
    python -m rng_eval.run_eval false_exact [--streams 300]                 A3  false EXACT_STATE on nulls
    python -m rng_eval.run_eval robust                                      A4  hostile conditions
    python -m rng_eval.run_eval perf                                        A5  speed / loop blocking / memory

Results -> docs/v2/results/<name>.json.  Seeds are fixed (documented per task).  EMULATION: for the statistical path the harness runs the
SAME code the live system runs (batch_horizon is equal to the online model to 1e-16, verified in the unit tests) and applies the same
state machine (e-process threshold, geometric diagnostic schedule with alpha-spending, status at publication time), but without
per-tick Python overhead; the online orchestrator itself is exercised in `perf`, `robust` and the adapter tests.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "docs", "v2", "results")

from rng_core import diag as dg                        # noqa: E402
from rng_core.decide import BIN_EDGES                  # noqa: E402
from rng_core.stat_model import batch_horizon          # noqa: E402
from rng_core.types import V_BIAS, V_SERIAL, V_CHANGEPOINT   # noqa: E402
from rng_eval import zoo                               # noqa: E402
from rng_eval.scoring import top4_mask                 # noqa: E402

HMAX = 10
N_MIN = 600
ALPHA = 0.01
CHECKPOINTS = (5000, 10000, 20000, 40000, 60000, 100000)


def schedule(n: int, first: int = 5000, factor: float = 2.0) -> List[int]:
    out, k = [], first
    while k <= n:
        out.append(k)
        k = int(k * factor)
    return out


def emulate(d: np.ndarray, alpha: float = ALPHA, hmax: int = HMAX, B_compress: int = 200, B_mc: int = 99, diag: bool = True,
            quotes: Optional[np.ndarray] = None, keep_hits: bool = False, seed: int = 0) -> dict:
    """Run the statistical path of the orchestrator on a digit stream and score every published prediction prequentially."""
    d = np.asarray(d, np.int64)
    n = len(d)
    thr = math.log(hmax / (alpha / 2.0))
    cross, hits, ll, p4raw = {}, {}, {}, {}
    for h in range(1, hmax + 1):
        o = batch_horizon(d, h)
        inc = o["log_e_incr"]
        cs = np.cumsum(inc)
        idx = np.nonzero(cs >= thr)[0]
        cross[h] = int(idx[0]) + h + 1 if len(idx) else None              # tick count at which the e-process crossed
        P = o["prank"][: n - h]
        m = top4_mask(P)
        hits[h] = m[np.arange(n - h), d[h:]]                              # hit of the prediction made at t (target t+h)
        ll[h] = -np.log(np.maximum(o["pmix"][np.arange(n - h), d[h:]], 1e-12))
        p4raw[h] = np.where(m, P, 0.0).sum(axis=1)
    t_e = min((v for v in cross.values() if v is not None), default=None)
    t_d, diag_log, any_flag = None, [], False
    if diag:
        for j, nk in enumerate(schedule(n), 1):
            lo = max(0, nk - 100_000)
            q = None if quotes is None else quotes[lo:nk]
            rep = dg.run_diagnostics(d[lo:nk], j, alpha / 2.0, q, 2, seed, B_compress, B_mc)
            fams = set(rep.families)
            diag_log.append((nk, rep.verdict, sorted(fams)))
            if rep.verdict != "NONE":
                any_flag = True
            if t_d is None and (fams & {V_BIAS, V_SERIAL}):
                t_d = nk
    cands = [(t, k) for t, k in ((t_e, "e-process"), (t_d, "diagnostics")) if t is not None]
    t_stat, trig = min(cands) if cands else (None, None)
    out = {"n": n, "t_stat": t_stat, "trigger": trig, "t_e": t_e, "t_d": t_d, "any_diag_flag": any_flag, "diag": diag_log}
    # scoring: prediction made at tick t (0-based) has status STAT iff t+1 >= t_stat
    S = {"all": [0, 0], "pre": [0, 0], "post": [0, 0], "last25": [0, 0]}
    per_h = {}
    ll_post, n_post = 0.0, 0
    p4_sum, p4_n = 0.0, 0
    for h in range(1, hmax + 1):
        k = len(hits[h])
        t = np.arange(k)
        valid = t >= N_MIN
        post = valid & (t_stat is not None) & (t + 1 >= (t_stat or 10 ** 12))
        pre = valid & ~post
        for key, msk in (("all", valid), ("pre", pre), ("post", post), ("last25", t >= int(0.75 * n))):
            S[key][0] += int(hits[h][msk].sum())
            S[key][1] += int(msk.sum())
        per_h[h] = [int(hits[h][post].sum()), int(post.sum())]
        ll_post += float(ll[h][post].sum())
        n_post += int(post.sum())
        p4_sum += float(p4raw[h][valid].sum())
        p4_n += int(valid.sum())
    out.update({"score": S, "post_by_h": per_h, "ll_post": ll_post, "n_post": n_post, "p4raw_mean": p4_sum / max(p4_n, 1),
                "e_cross": cross})
    # raw bins for the cross-stream calibrator emulation (status NO_STRUCTURE / STAT per prediction)
    bins_ns = np.zeros((2, len(BIN_EDGES)))
    bins_k = np.zeros((2, len(BIN_EDGES)))
    for h in range(1, hmax + 1):
        k = len(hits[h])
        t = np.arange(k)
        valid = t >= N_MIN
        stat = valid & (t_stat is not None) & (t + 1 >= (t_stat or 10 ** 12))
        b = np.minimum(np.searchsorted(BIN_EDGES, p4raw[h], side="left"), len(BIN_EDGES) - 1)
        for si, msk in ((0, valid & ~stat), (1, stat)):
            np.add.at(bins_ns[si], b[msk], 1.0)
            np.add.at(bins_k[si], b[msk], hits[h][msk].astype(float))
    out["bins_n"], out["bins_k"] = bins_ns.tolist(), bins_k.tolist()
    if keep_hits:
        out["hits"] = {h: hits[h] for h in hits}
    return out


# ── A1 ──────────────────────────────────────────────────────────────────────────────────────────
def _null_stream(i: int, n: int) -> np.ndarray:
    k = i % 10
    if k < 8:
        return zoo.null_pcg64(1000 + i, n).astype(np.int64)
    if k == 8:
        return zoo.null_drbg(1000 + i, n).astype(np.int64)
    return zoo.null_urandom(1000 + i, n).astype(np.int64)


def _null_job(a):
    i, n = a
    r = emulate(_null_stream(i, n))
    r["i"] = i
    return r


def task_null(n=12000, streams=1000, procs=2, **_):
    t0 = time.time()
    rs = []
    with Pool(procs) as p:
        for k, r in enumerate(p.imap_unordered(_null_job, [(i, n) for i in range(streams)], chunksize=2), 1):
            rs.append(r)
            if k % 50 == 0:
                print(f"null {k}/{streams} {time.time() - t0:.0f}s", flush=True)
                save("null_rows", [{x: y for x, y in q.items() if x not in ("diag",)} for q in rs])
    rs.sort(key=lambda r: r["i"])
    left = [r for r in rs if r["t_stat"] is not None]
    flag = [r for r in rs if r["any_diag_flag"]]
    # pooled top-4 (all predictions, regardless of status), stream-level SE
    per = np.array([r["score"]["all"][0] / r["score"]["all"][1] for r in rs])
    hit_all = sum(r["score"]["all"][0] for r in rs)
    n_all = sum(r["score"]["all"][1] for r in rs)
    # cross-stream calibrator emulation: calibrator state at the start of each stream = counts of the previous streams
    from rng_core.decide import Top4Calibrator
    cal = Top4Calibrator()
    cal_means = []
    for r in rs:
        bn, bk = np.array(r["bins_n"]), np.array(r["bins_k"])
        tot, wsum = 0.0, 0.0
        for si, st in ((0, "NO_STRUCTURE"), (1, "STAT_SIGNAL")):
            if bn[si].sum() == 0:
                continue
            mean, _nn, _ = cal._post(st)
            tot += float((mean * bn[si]).sum())
            wsum += float(bn[si].sum())
        cal_means.append(tot / wsum)
        for si, st in ((0, "NO_STRUCTURE"), (1, "STAT_SIGNAL")):
            cal.n[st] += bn[si]
            cal.k[st] += bk[si]
    w = np.array([sum(sum(x) for x in r["bins_n"]) for r in rs])
    out = {"streams": streams, "n": n, "seconds": round(time.time() - t0, 1),
           "frac_left_no_structure": len(left) / streams, "left": [(r["i"], r["trigger"], r["t_stat"]) for r in left],
           "frac_any_diag_flag": len(flag) / streams,
           "pooled_top4": hit_all / n_all, "stream_mean_top4": float(per.mean()), "stream_se": float(per.std(ddof=1) / math.sqrt(streams)),
           "pooled_se_indep": math.sqrt(0.24 / n_all), "mean_p4_cal": float((np.array(cal_means) * w).sum() / w.sum()),
           "mean_p4raw": float(np.mean([r["p4raw_mean"] for r in rs])),
           "ll_pre_mean": None}
    save("null", out)
    return out


def task_null_long(n=60000, streams=60, procs=2, **_):
    t0 = time.time()
    with Pool(procs) as p:
        rs = p.map(_null_job, [(5000 + i, n) for i in range(streams)], chunksize=2)
    left = [r for r in rs if r["t_stat"] is not None]
    out = {"streams": streams, "n": n, "diag_runs_per_stream": len(rs[0]["diag"]), "seconds": round(time.time() - t0, 1),
           "frac_left": len(left) / streams, "left": [(r["i"], r["trigger"], r["t_stat"]) for r in left],
           "frac_any_diag_flag": float(np.mean([r["any_diag_flag"] for r in rs]))}
    save("null_long", out)
    return out


# ── A2 ──────────────────────────────────────────────────────────────────────────────────────────
LIFTS = (0.005, 0.01, 0.02, 0.05)


def _power_job(a):
    fam, lift, s, n = a
    d = zoo.SIGNALS[fam](7000 + s, n, lift).astype(np.int64)
    r = emulate(d)
    r.update({"fam": fam, "lift": lift, "s": s})
    return r


def task_power(n=60000, streams=16, procs=2, **_):
    t0 = time.time()
    jobs = [(f, l, s, n) for f in ("bias", "markov2", "lag17") for l in LIFTS for s in range(streams)]
    with Pool(procs) as p:
        rs = p.map(_power_job, jobs, chunksize=2)
    cells = {}
    for f in ("bias", "markov2", "lag17"):
        for l in LIFTS:
            c = [r for r in rs if r["fam"] == f and r["lift"] == l]
            det = {str(k): float(np.mean([(r["t_stat"] is not None and r["t_stat"] <= k) for r in c])) for k in CHECKPOINTS if k <= n}
            kp, npst = sum(r["score"]["post"][0] for r in c), sum(r["score"]["post"][1] for r in c)
            k1, n1 = sum(r["post_by_h"][1][0] for r in c), sum(r["post_by_h"][1][1] for r in c)
            ka, na = sum(r["score"]["last25"][0] for r in c), sum(r["score"]["last25"][1] for r in c)
            cells[f"{f}@{l}"] = {"detect_by_n": det, "median_t_stat": (float(np.median([r["t_stat"] for r in c if r["t_stat"]]))
                                                                         if any(r["t_stat"] for r in c) else None),
                                 "triggers": {k: sum(1 for r in c if r["trigger"] == k) for k in ("e-process", "diagnostics")},
                                 "top4_post_all_h": [kp, npst], "top4_post_h1": [k1, n1], "top4_last25_all_h": [ka, na],
                                 "target_top4": 0.4 + l}
    out = {"n": n, "streams_per_cell": streams, "seconds": round(time.time() - t0, 1), "cells": cells}
    save("power", out)
    return out


def save(name: str, obj) -> None:
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


TASKS = {"null": task_null, "null_long": task_null_long, "power": task_power}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--n", type=int)
    ap.add_argument("--streams", type=int)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    kw = {k: v for k, v in (("n", a.n), ("streams", a.streams), ("procs", a.procs)) if v is not None}
    if a.task in TASKS:
        print(json.dumps(TASKS[a.task](**kw), default=str)[:3000])
    else:
        from rng_eval import run_eval_more
        print(json.dumps(run_eval_more.run(a.task, **kw), default=str)[:3000])
