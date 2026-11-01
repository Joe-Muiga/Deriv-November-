"""A3 (long part): 32-bit LCG brute-force scan timings. Written to docs/v2/results/brute.json.
For each LCG: (a) a FULL 2^32 scan on a null stream (time + no false positive), (b) found-state cases per map with the true state at a
known fraction of the scan range (found time scales with that fraction)."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from rng_core.exact import solvers
from rng_core.exact.gens import GENS, LCGS, MAPS, digits_from_words
from rng_eval import zoo

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs/v2/results/brute.json")
res = {"full_scan_null": {}, "found": []}
def save(): json.dump(res, open(OUT, "w"), indent=1)

for gen in ("minstd", "randu", "glibc_lcg", "msvc", "nr_lcg", "borland"):
    lcg, spec = LCGS[gen], GENS[gen]
    null = zoo.null_pcg64(99, 200).astype(np.int64)
    t = time.time()
    r = solvers.solve_brute(null, gen, budget_s=3000, progress={})
    res["full_scan_null"][gen] = {"s": round(time.time() - t, 1), "false_positive": r is not None}
    print(gen, "null full scan", res["full_scan_null"][gen], flush=True); save()
    for mp in MAPS:
        stride = 2 if gen == "randu" else 1
        frac = 0.05 + 0.04 * MAPS.index(mp)
        seed = int(frac * (lcg.m // stride)) * stride + (1 if gen in ("randu", "minstd") else 0)
        words = spec.fn(seed, 200)
        d = digits_from_words(spec, mp, words).astype(np.int64)
        t = time.time()
        r = solvers.solve_brute(d, gen, budget_s=3000, progress={})
        el = time.time() - t
        ok = r is not None
        acc = None
        if ok:
            d2 = digits_from_words(spec, mp, spec.fn(seed, 5200)).astype(np.int64)
            from rng_core.exact.base import ExactMonitor
            mon = ExactMonitor(r, [int(x) for x in d2[:200]])
            hit = 0
            for x in d2[200:5200]:
                pr = mon.predict(1)
                hit += int(pr is not None and pr[0] == "digit" and pr[1] == int(x))
                mon.observe(int(x))
            acc = hit / 5000
        res["found"].append({"gen": gen, "map": mp, "state_fraction": frac, "found": ok, "s": round(el, 1),
                             "fit": (r or {}).get("name"), "fit_map": (r or {}).get("map"), "top1_next5000": acc})
        print(res["found"][-1], flush=True); save()
print("DONE", flush=True)
