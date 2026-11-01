"""test_mh.py - MH model checks (run: python test_mh.py  or  pytest test_mh.py)."""
import asyncio
import collections
import time

import numpy as np

import mh_predictor as mp
import predictor as P


def _shift_seq(n=4000, h=2, k=3, seed=0):
    """Deterministic: D[i] = D[i-h] + k (mod 10) -> D[t+h] is exactly d_t + k."""
    r = np.random.default_rng(seed)
    D = list(r.integers(0, 10, h))
    for i in range(h, n):
        D.append((D[i - h] + k) % 10)
    return np.array(D, dtype=np.int64)


def _noisy_seq(n=6000, h=2, k=3, prob=0.6, seed=0):
    r = np.random.default_rng(seed)
    D = list(r.integers(0, 10, n))
    for i in range(h, n):
        if r.random() < prob:
            D[i] = (D[i - h] + k) % 10
    return np.array(D, dtype=np.int64)


def test_horizon_alignment():
    D = _shift_seq(h=2, k=3)
    res = mp.walk_batch(D, 2, 0.998)
    want = (D[-1] + 3) % 10                      # prediction made at the last tick is for tick n-1+2
    assert res["PB"][-1].argmax() == want and res["PB"][-1][want] > 0.9


def test_scored_target_is_t_plus_h():
    D = _noisy_seq()
    hit = {}
    for h in (2, 5):
        res = mp.walk_batch(D, h, 0.998)
        ok, t = res["ok"], res["t"]
        hit[h] = (res["PB"][ok].argmax(1) == D[t[ok] + h]).mean()   # prediction at t is scored against D[t+h]
    assert hit[2] > 0.5 and hit[2] > hit[5] + 0.2, hit


def test_cover_scoring():
    assert len(mp.COVERS) == 18
    i, j = mp.COVERS.index(("OVER", 6)), mp.COVERS.index(("UNDER", 3))
    assert list(np.nonzero(mp.W[:, i])[0]) == [7, 8, 9] and list(np.nonzero(mp.W[:, j])[0]) == [0, 1, 2]
    dist = np.full(10, 0.05)
    dist[[7, 8, 9]] = [0.3, 0.2, 0.1]
    assert abs((dist @ mp.W)[i] - 0.6) < 1e-12


def test_calibration_monotone():
    r = np.random.default_rng(1)
    p = r.random(3000)
    y = (r.random(3000) < p ** 2).astype(float)
    c = mp.Calibrator(p, y)
    assert np.all(np.diff(c.p) >= -1e-12) and np.all(c.lb <= c.p + 1e-12)
    lo, hi = c(0.05)[0], c(0.95)[0]
    assert lo < hi


def test_batch_equals_online():
    r = np.random.default_rng(3)
    D = r.integers(0, 10, 1500)
    for i in range(2, 1500):
        if r.random() < 0.3:
            D[i] = (D[i - 2] + 1) % 10
    for h in (1, 4):
        res = mp.walk_batch(D, h, 0.998)
        hs, buf = mp.HState(h, 0.998), collections.deque(maxlen=64)
        for j, d in enumerate(D):
            buf.append(int(d))
            hs.update(buf, j)
        assert np.abs(res["PB"][-1] - hs.cur[1]).max() < 1e-9
        assert np.abs(res["state"]["L"] - hs.L).max() < 1e-6


def test_fair_stream_not_tradable():
    D = np.random.default_rng(5).integers(0, 10, 15000)
    s = mp.MHSymbol("X")
    asyncio.run(s.fit(D, 1.0, deadline=time.time() + 30))
    assert not any(hs.tradable for hs in s.hs.values())          # nothing beats uniform log-loss on noise
    assert s.screen(1, 0.1, 0.0) == []


def test_late_mixing_and_screen():
    D = _shift_seq(n=15000, h=2, k=3)
    s = mp.MHSymbol("Y")
    asyncio.run(s.fit(D, 1.0, deadline=time.time() + 30))
    assert s.hs[2].tradable
    for d in D[-3:]:
        s.on_tick(int(d))
    a, b = s.screen(2, 0.0, 0.0), s.screen(2, 0.5, 0.0)
    assert a
    same = [x for x in b if x[0] == a[0][0]]
    assert not same or same[0][1] < a[0][1]                      # mixing with pattern-free h=3 lowers p_eff
    assert P.estimated_payout_ratio(("OVER", 6)) > 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
