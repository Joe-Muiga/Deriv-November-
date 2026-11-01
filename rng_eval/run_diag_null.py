"""rng_eval.run_diag_null - calibration of every diagnostic on null streams: per-test p-value uniformity and false-positive rate.

usage: python -m rng_eval.run_diag_null N_STREAMS N_DIGITS [B_compress] [B_mc] [out.json]
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from rng_core import diag
from rng_eval import zoo


def one(args):
    seed, n, bc, bm = args
    d = zoo.null_pcg64(seed, n) if seed % 2 == 0 else zoo.null_drbg(seed, n)
    rep = diag.run_diagnostics(d, run_index=1, B_compress=bc, B_mc=bm, seed=seed)
    return {f"{r.name}@{r.stream}": (r.p if r.ok else None) for r in rep.results}, rep.verdict


def main():
    N, n = int(sys.argv[1]), int(sys.argv[2])
    bc = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    bm = int(sys.argv[4]) if len(sys.argv) > 4 else 49
    out = sys.argv[5] if len(sys.argv) > 5 else None
    t = time.time()
    with Pool(2) as pool:
        rows = pool.map(one, [(s, n, bc, bm) for s in range(N)])
    names = list(rows[0][0].keys())
    table = {}
    for nm in names:
        ps = np.array([r[0][nm] for r in rows if r[0][nm] is not None])
        if len(ps) == 0:
            table[nm] = None
            continue
        ps = np.sort(ps)
        ks = float(np.max(np.abs(ps - (np.arange(len(ps)) + 0.5) / len(ps))))
        table[nm] = {"n": len(ps), "fp@.05": float((ps < .05).mean()), "fp@.01": float((ps < .01).mean()), "min_p": float(ps[0]), "ks": ks}
    verdicts = {}
    for r in rows:
        verdicts[r[1]] = verdicts.get(r[1], 0) + 1
    print(f"streams={N} n={n} time={time.time() - t:.0f}s verdicts={verdicts}")
    for nm, v in table.items():
        print(f"{nm:34s}", v)
    if out:
        json.dump({"table": table, "verdicts": verdicts, "N": N, "n": n}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
