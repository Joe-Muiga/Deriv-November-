"""rng_core.exact.solvers - Layer 2 solvers (numpy + Python ints only). Each returns a FIT dict (see base.py) or None.

    solve_cycle        exact periodic suffix (any generator that has entered a cycle)
    solve_bm_bits      Berlekamp-Massey on parity / (digit >= 5) bit streams -> bit-level model (top-4 hit 0.8)
    solve_gf2          GF(2) Gaussian elimination for xorshift32, xorshift128, LFSR32, MT19937 x {top, mod10, scaled}
    solve_seed_scan    time-seed scan (seed = epoch seconds / milliseconds +- window, small stream offset) over the catalogue
    solve_brute        chunked vectorised state scan of the <= 32-bit LCGs (resumable, time budgeted)
    solve_lattice      STUB (4c): truncated-LCG lattice recovery, see tools/lattice_lcg.py
    solve_smt          STUB (4d): Z3 for xorshift128+ style generators, offline only

Every solver trusts nothing: a candidate is accepted only if it reproduces EVERY observed tick of the segment and at least
V (=24) of those ticks were not used by the solve.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import gens
from .base import GenModel, V_DEFAULT
from .gens import GENS, LCGS, M32, MAPS, known_bits
from .gf2 import GF2System, SYM_GENS, berlekamp_massey, bm_predict


def _fit(kind, name, gen, mp, params, n_fit, n_check, t0, **extra) -> dict:
    d = {"kind": kind, "name": name, "gen": gen, "map": mp, "params": params, "n_fit": int(n_fit), "n_check": int(n_check),
         "solve_s": round(time.time() - t0, 3)}
    d.update(extra)
    return d


# ── cycle ───────────────────────────────────────────────────────────────────────────────────────
def _lags_by_autocorr(d: np.ndarray, max_lag: int, top: int = 8) -> List[int]:
    n = len(d)
    max_lag = min(max_lag, n // 2)
    if max_lag < 1:
        return []
    m = 1 << (2 * n - 1).bit_length()
    ac = np.zeros(max_lag + 1)
    for c in range(10):
        x = (d == c).astype(np.float64)
        F = np.fft.rfft(x, m)
        ac += np.fft.irfft(F * np.conj(F), m)[:max_lag + 1]
    lags = np.arange(1, max_lag + 1)
    frac = ac[1:] / (n - lags)
    order = np.argsort(-frac)[:top]
    return sorted(int(lags[i]) for i in order if frac[i] > 0.5)


def solve_cycle(D: Sequence[int], V: int = V_DEFAULT, max_lag: int = 20000, budget_s: float = 5.0) -> Optional[dict]:
    t0 = time.time()
    d = np.asarray(D, np.int64)
    n = len(d)
    if n < V + 8:
        return None
    cands = set(_lags_by_autocorr(d, max_lag))
    tail = d[-min(n, 4096):]
    cands |= set(_lags_by_autocorr(tail, min(max_lag, len(tail) // 2)))
    for p in sorted(cands):
        # longest suffix on which d[i] == d[i-p]
        eq = d[p:] == d[:-p]
        bad = np.nonzero(~eq)[0]
        suf = len(eq) if len(bad) == 0 else len(eq) - 1 - int(bad[-1])
        if suf >= V and n - p >= V:                  # the cycle vector = last p digits; `suf` ticks confirm the lag-p repetition
            k0 = n - p
            return _fit("cycle", "cycle", "cycle", "digit", {"vec": d[k0:].tolist(), "k0": int(k0), "period": int(p)}, p, suf, t0)
        if time.time() - t0 > budget_s:
            break
    return None


# ── bit-level (Berlekamp-Massey) ─────────────────────────────────────────────────────────────────
def solve_bm_bits(D: Sequence[int], V: int = V_DEFAULT, max_bits: int = 40000, budget_s: float = 10.0) -> Optional[dict]:
    t0 = time.time()
    d = np.asarray(D, np.int64)[-max_bits:]
    n = len(d)
    if n < 2 * V + 64:
        return None
    best = None
    for stream in ("ge5", "parity"):
        bits = ((d >= 5).astype(np.int64) if stream == "ge5" else (d & 1)).tolist()
        fitn = n - V
        L, C = berlekamp_massey(bits[:fitn])
        if L > fitn // 2 - 16:                       # no short recurrence
            continue
        pred = bm_predict(bits[:fitn], L, C, V)
        if pred != bits[fitn:]:
            continue
        L2, C2 = berlekamp_massey(bits)              # refit on everything (same recurrence if the hypothesis is right)
        if L2 > L:
            continue
        cand = _fit("bits", f"bm_{stream}", "linear_bits", stream, {"stream": stream, "L": int(L2), "C": int(C2)}, fitn, V, t0)
        if best is None or L2 < best["params"]["L"]:
            best = cand
        if time.time() - t0 > budget_s:
            break
    return best


# ── GF(2) state recovery ─────────────────────────────────────────────────────────────────────────
def _state_from_solution(gen: str, sol: int):
    if gen in ("xorshift32", "lfsr32"):
        return sol & M32
    if gen == "xorshift128":
        return [(sol >> (32 * i)) & M32 for i in range(4)]
    if gen in ("mt19937", "pyrandom"):
        return [(sol >> (32 * i)) & M32 for i in range(624)]
    raise ValueError(gen)


def solve_gf2(D: Sequence[int], gen: str, mp: str, V: int = V_DEFAULT, budget_s: float = 30.0, max_ticks: int = 60000) -> Optional[dict]:
    t0 = time.time()
    d = np.asarray(D, np.int64)
    n = len(d)
    spec = GENS[gen]
    symgen = "mt19937" if gen == "pyrandom" else gen
    sym = SYM_GENS[symgen]()
    nvars = sym.nvars
    sysm = GF2System(nvars)
    wpt = spec.words_per_tick
    allowed_free = set(range(31)) if symgen == "mt19937" else set()
    used = 0
    limit = min(n - V, max_ticks)
    if limit <= 0:
        return None
    extra_needed = 24
    extra = 0
    determined = False
    for k in range(limit):
        words = [sym.next() for _ in range(wpt)]
        for wo, bit, val in known_bits(spec, mp, int(d[k])):
            r = sysm.add(words[wo][bit], val)
            if r == "bad":
                return None                           # hypothesis (generator, map) is wrong: inconsistent equations
            if r == "dep" and determined:
                extra += 1
        used = k + 1
        if not determined:
            free = [v for v in range(nvars) if v not in sysm.piv]
            if sysm.rank >= nvars - len(allowed_free) and set(free) <= allowed_free:
                determined = True
        if determined and extra >= extra_needed:
            break
        if (k & 63) == 0 and time.time() - t0 > budget_s:
            return None
    if not determined or sysm.inconsistent:
        return None
    sol = sysm.solve()
    if sol is None:
        return None
    state = _state_from_solution(gen, sol)
    model = GenModel(gen, mp, state)
    # EVERY tick of the segment must be reproduced - except tick 0 of an MT-type stream: the 31 low bits of the virtual word 0 are
    # free in the solution (they influence that single output only, never the next block) and the observation maps constrain
    # only a prefix of its bits, so the digit of tick 0 is not determined by the equations. All later ticks are.
    for k in range(1 if symgen == "mt19937" else 0, n):
        if model.digit(k) != int(d[k]):
            return None
    return _fit("gen", f"gf2_{gen}", gen, mp, {"state": state}, used, n - used, t0)


# ── time-seed scan ───────────────────────────────────────────────────────────────────────────────
def solve_seed_scan(D: Sequence[int], epoch0: float, V: int = V_DEFAULT, budget_s: float = 20.0, window_s: int = 7200,
                    window_ms: int = 20000, max_skip: int = 32, gens_: Optional[Sequence[str]] = None) -> Optional[dict]:
    """Seed = epoch (seconds) +- window_s, or epoch in milliseconds +- window_ms, generator already `skip` ticks into its stream.
    For every generator the word matrix of ALL candidate seeds is built once and read through the three observation maps."""
    t0 = time.time()
    d = np.asarray(D, np.int64)
    n = len(d)
    K = 20
    if n < K + V:
        return None
    base = int(epoch0)
    seeds = list(range(base - window_s, base + window_s + 1)) + list(range(base * 1000 - window_ms, base * 1000 + window_ms + 1))
    for gen in list(gens_ or (list(LCGS) + ["mt19937", "pyrandom"])):
        if time.time() - t0 > budget_s:
            return None
        if gen == "pyrandom":                                  # random.random(): 2 words / tick, generator at most a few ticks in
            r = _scan_pyrandom(d, seeds, K, V, n, t0, budget_s)
            if r is not None:
                return r
            continue
        words, sd = _seed_words(gen, seeds, K + max_skip, t0, budget_s)
        if words is None:
            return None
        spec = GENS[gen]
        for mp in MAPS:
            M = _digits_vec(spec, mp, words)
            for skip in range(max_skip + 1):
                hit = np.nonzero((M[:, skip:skip + K] == d[:K][None, :]).all(axis=1))[0]
                for i in hit:
                    model = _seeded_model(gen, mp, sd[int(i)], skip)
                    if all(model.digit(k) == int(d[k]) for k in range(n)):
                        return _fit("mtseed" if gen == "mt19937" else "gen", f"seed_{gen}", gen, mp,
                                    _seeded_params(gen, sd[int(i)], skip), K, n - K, t0)
    return None


def _scan_pyrandom(d: np.ndarray, seeds: List[int], K: int, V: int, n: int, t0: float, budget_s: float, max_skip: int = 4) -> Optional[dict]:
    """Seed scan for Python's random.random() (53-bit float from two 32-bit words): digit = floor(10 f) ('top' and 'scaled') or the
    53-bit integer mod 10 ('mod10')."""
    import random
    cols = 2 * (K + max_skip)
    rows = []
    for i, sd in enumerate(seeds):
        if (i & 4095) == 0 and time.time() - t0 > budget_s:
            return None
        rg = random.Random(sd)
        rows.append([rg.getrandbits(32) for _ in range(cols)])
    W = np.array(rows, np.uint64)
    v = ((W[:, 0::2] >> np.uint64(5)) << np.uint64(26)) | (W[:, 1::2] >> np.uint64(6))
    for mp in MAPS:
        M = (v % np.uint64(10)).astype(np.int64) if mp == "mod10" else ((v * np.uint64(10)) >> np.uint64(53)).astype(np.int64)
        for skip in range(max_skip + 1):
            hit = np.nonzero((M[:, skip:skip + K] == d[:K][None, :]).all(axis=1))[0]
            for i in hit:
                model = GenModel("pyrandom", mp, None, skip_words=2 * skip, seed=seeds[int(i)])
                if all(model.digit(k) == int(d[k]) for k in range(n)):
                    return _fit("mtseed", "seed_pyrandom", "pyrandom", mp, {"seed": int(seeds[int(i)]), "skip_words": 2 * skip}, K, n - K, t0)
    return None


def _seed_words(gen: str, seeds: List[int], cols: int, t0: float, budget_s: float):
    """(n_seeds, cols) uint64 matrix of the first `cols` output words of `gen` for every candidate seed."""
    import random
    if gen == "mt19937":
        rows = []
        for i, s in enumerate(seeds):
            if (i & 4095) == 0 and time.time() - t0 > budget_s:
                return None, None
            rg = random.Random(s)
            rows.append([rg.getrandbits(32) for _ in range(cols)])
        return np.array(rows, np.uint64), seeds
    lcg = LCGS[gen]
    st = [lcg.seed_fn(s) for s in seeds]
    if lcg.vec_ok:
        x = np.array(st, np.uint64)
        out = []
        for _ in range(cols):
            x = (x * np.uint64(lcg.a) + np.uint64(lcg.c)) % np.uint64(lcg.m)
            out.append((x >> np.uint64(lcg.shift)) & np.uint64((1 << lcg.w) - 1))
        return np.stack(out, axis=1), seeds
    rows = []
    for i, x in enumerate(st):
        if (i & 4095) == 0 and time.time() - t0 > budget_s:
            return None, None
        row = []
        for _ in range(cols):
            x = lcg.step(x)
            row.append(lcg.out(x))
        rows.append(row)
    return np.array(rows, np.uint64), seeds


def _seeded_model(gen: str, mp: str, seed: int, skip: int):
    if gen == "mt19937":
        return GenModel(gen, mp, None, skip_words=skip, seed=seed)
    lcg = LCGS[gen]
    x = lcg.seed_fn(seed)
    for _ in range(skip):
        x = lcg.step(x)
    return GenModel(gen, mp, x)


def _seeded_params(gen: str, seed: int, skip: int) -> dict:
    if gen == "mt19937":
        return {"seed": int(seed), "skip_words": int(skip)}
    lcg = LCGS[gen]
    x = lcg.seed_fn(seed)
    for _ in range(skip):
        x = lcg.step(x)
    return {"state": int(x), "seed": int(seed), "skip": int(skip)}


def _digits_vec(spec, mp: str, out: np.ndarray) -> np.ndarray:
    """Vectorised gens.digit_of for stride-1 power-of-two generators (out is a uint64 array of words)."""
    if mp == "mod10":
        return (out % np.uint64(10)).astype(np.int64)
    s, wb = gens._window(spec, mp)
    v = (out >> np.uint64(s)) & np.uint64((1 << wb) - 1)
    return ((v * np.uint64(10)) >> np.uint64(wb)).astype(np.int64)


# ── brute force over small LCG state spaces ──────────────────────────────────────────────────────
def solve_brute(D: Sequence[int], gen: str, V: int = V_DEFAULT, budget_s: float = 60.0, progress: Optional[dict] = None,
                chunk: int = 1 << 20, K: int = 14) -> Optional[dict]:
    """Scan every state of an LCG with <= 32-bit state. `progress` (dict) stores the next state index per generator so a later
    call resumes where this one stopped (time budget). Digit equality on the first K ticks filters; survivors are verified on
    the whole segment. Maps are tested together (the output word is computed once)."""
    t0 = time.time()
    lcg, spec = LCGS[gen], GENS[gen]
    if not lcg.vec_ok:
        raise NotImplementedError(f"{gen}: state space too large for a scan (needs the lattice solver, 4c)")
    d = np.asarray(D, np.int64)
    n = len(d)
    if n < K + V:
        return None
    progress = progress if progress is not None else {}
    if gen == "randu":
        stride, off, total = 2, 1, lcg.m // 2          # odd states only
    elif gen == "minstd":
        stride, off, total = 1, 1, lcg.m - 1
    else:
        stride, off, total = 1, 0, lcg.m
    a, c, m = np.uint64(lcg.a), np.uint64(lcg.c), np.uint64(lcg.m)
    mask_w = np.uint64((1 << lcg.w) - 1)
    pw2 = (lcg.m & (lcg.m - 1)) == 0
    start = progress.get(gen, 0)
    for lo in range(start, total, chunk):
        hi = min(lo + chunk, total)
        x0 = np.arange(lo, hi, dtype=np.uint64) * np.uint64(stride) + np.uint64(off)
        for mp in MAPS:
            x = x0
            idx = np.arange(len(x0))
            for k in range(K):
                x = (x * a + c) & np.uint64(lcg.m - 1) if pw2 else (x * a + c) % m
                out = (x >> np.uint64(lcg.shift)) & mask_w
                keep = _digits_vec(spec, mp, out) == d[k]
                if not keep.any():
                    break
                x, idx = x[keep], idx[keep]
            else:
                for j in idx:
                    st = int(x0[int(j)])
                    model = GenModel(gen, mp, st)
                    if all(model.digit(k) == int(d[k]) for k in range(n)):
                        progress[gen] = total
                        return _fit("gen", f"brute_{gen}", gen, mp, {"state": st}, K, n - K, t0)
        progress[gen] = hi
        if time.time() - t0 > budget_s:
            return None
    progress[gen] = total
    return None


# ── stubs (declared, not faked) ──────────────────────────────────────────────────────────────────
def solve_lattice(*a, **k):
    raise NotImplementedError("4c truncated-LCG lattice recovery (LLL/Babai) is not implemented; see tools/lattice_lcg.py")


def solve_smt(*a, **k):
    raise NotImplementedError("4d SMT (Z3) recovery for xorshift128+-style generators is not implemented (offline only)")
