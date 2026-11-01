"""rng_core.exact.runner - run the solver ladder on one segment under a time budget (used by the worker process and by tests)."""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import solvers
from .gens import LCGS, MAPS

SMALL_LINEAR = ("xorshift32", "lfsr32", "xorshift128")
BRUTE_GENS = ("minstd", "randu", "glibc_lcg", "msvc", "nr_lcg", "borland")
NOT_IMPLEMENTED = ["lattice (4c)", "smt (4d)", "glibc TYPE_3 / PCG / middle-square / 48-bit LCG state recovery"]


def run_exact(D: Sequence[int], epoch0: Optional[float] = None, budget_s: float = 20.0, enable_brute: bool = False,
              try_mt: bool = False, progress: Optional[dict] = None, V: int = 24, order: Optional[Sequence[str]] = None) -> dict:
    """Try the cheap solvers first. Returns {"fit": dict|None, "tried": [...], "elapsed": s, "not_implemented": [...]}.
    State-level fits beat bit-level fits (bit-level is returned only when nothing better was found)."""
    t0 = time.time()
    d = np.asarray(D, np.int64)
    tried: List[dict] = []
    bits_fit = None

    def left() -> float:
        return budget_s - (time.time() - t0)

    def record(name, t, res):
        tried.append({"solver": name, "s": round(time.time() - t, 3), "ok": res is not None})
        return res

    plan = list(order or ["cycle", "gf2_small", "seed_scan", "bm_bits", "gf2_mt", "brute"])
    for step in plan:
        if left() <= 0:
            tried.append({"solver": step, "skipped": "budget"})
            continue
        t = time.time()
        if step == "cycle":
            r = record("cycle", t, solvers.solve_cycle(d, V, budget_s=min(5.0, left())))
            if r:
                return {"fit": r, "tried": tried, "elapsed": time.time() - t0, "not_implemented": NOT_IMPLEMENTED}
        elif step == "gf2_small":
            for g in SMALL_LINEAR:
                for mp in MAPS:
                    if left() <= 0:
                        break
                    t = time.time()
                    r = record(f"gf2:{g}:{mp}", t, solvers.solve_gf2(d, g, mp, V, budget_s=min(10.0, left())))
                    if r:
                        return {"fit": r, "tried": tried, "elapsed": time.time() - t0, "not_implemented": NOT_IMPLEMENTED}
        elif step == "seed_scan" and epoch0 is not None:
            r = record("seed_scan", t, solvers.solve_seed_scan(d, epoch0, V, budget_s=min(20.0, left())))
            if r:
                return {"fit": r, "tried": tried, "elapsed": time.time() - t0, "not_implemented": NOT_IMPLEMENTED}
        elif step == "bm_bits":
            bits_fit = record("bm_bits", t, solvers.solve_bm_bits(d, V, budget_s=min(10.0, left())))
        elif step == "gf2_mt" and try_mt:
            for g in ("mt19937", "pyrandom"):
                for mp in MAPS:
                    if left() <= 0:
                        break
                    t = time.time()
                    r = record(f"gf2:{g}:{mp}", t, solvers.solve_gf2(d, g, mp, V, budget_s=left()))
                    if r:
                        return {"fit": r, "tried": tried, "elapsed": time.time() - t0, "not_implemented": NOT_IMPLEMENTED}
        elif step == "brute" and enable_brute:
            progress = progress if progress is not None else {}
            for g in BRUTE_GENS:
                if left() <= 0:
                    break
                if progress.get(g, 0) >= (LCGS[g].m if g not in ("randu",) else LCGS[g].m // 2):
                    continue
                t = time.time()
                r = record(f"brute:{g}", t, solvers.solve_brute(d, g, V, budget_s=left(), progress=progress))
                if r:
                    return {"fit": r, "tried": tried, "elapsed": time.time() - t0, "not_implemented": NOT_IMPLEMENTED, "progress": progress}
    return {"fit": bits_fit, "tried": tried, "elapsed": time.time() - t0, "not_implemented": NOT_IMPLEMENTED, "progress": progress}
