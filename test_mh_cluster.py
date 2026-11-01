"""test_mh_cluster.py - cluster mode checks (run: python test_mh_cluster.py)."""
import asyncio, sys, time, types
import numpy as np
for m in ("aiohttp", "psycopg", "psycopg2", "pytz", "websockets", "websockets.exceptions"):
    try:
        __import__(m)
    except Exception:
        sys.modules[m] = types.ModuleType(m)
ws = sys.modules["websockets.exceptions"]
for n in ("ConnectionClosed", "InvalidStatus"):
    ws.__dict__.setdefault(n, type(n, (Exception,), {}))
import mh_engine as me
import mh_predictor as mp

TOP = lambda *ds: np.array([0.14 if d in ds else 0.02 for d in range(10)])

def names(cl, w=5, widths=(3, 4, 5), pick="widest"):
    return [f"{c[0]}{c[1]}" for c, _ in me.pick_fits(cl, w, widths, pick)]

def test_top3_ties():
    p = np.full(10, 0.1)
    assert me.top_cluster(p, 3) == (0, 1, 2)                    # ties -> lower digits
    p[9] = 0.2
    assert me.top_cluster(p, 3) == (0, 1, 9)

def test_covers():
    assert names((7, 8, 9)) == ["OVER4", "OVER5", "OVER6"] and names((7, 8, 9), pick="tightest") == ["OVER6", "OVER5", "OVER4"]
    assert names((0, 1, 2), pick="tightest") == ["UNDER3", "UNDER4", "UNDER5"]
    assert names((5, 6, 7)) == ["OVER4"] and names((2, 3, 4)) == ["UNDER5"]
    assert names((1, 5, 9)) == [] and names((4, 5, 6)) == []
    assert names((7, 8, 9), widths=(3, 5)) == ["OVER4", "OVER6"]

def test_lag_mix():
    ph, pn = TOP(1, 2, 3), TOP(7, 8, 9)
    pn[0] = 0.0
    p = me.p_eff_mix(ph, pn, 0.3)
    assert abs(p.sum() - 1) < 1e-12 and p.min() >= 1e-4 / 1.01 and me.p_eff_mix(ph, None, 0.3) is ph

class FakeCal:
    def __init__(self, p, lb):
        self.p, self.lb = p, lb
    def __call__(self, x):
        return self.p, self.lb

class FakeHS:
    """Cluster mode now needs a tradable horizon with calibrators: (p_cal, Wilson lower bound) per cover."""
    def __init__(self, p, lb=0.6, tradable=True, muted=False):
        self.cur = (None, np.asarray(p) / np.sum(p))
        self.cal = [FakeCal(0.62, lb) for _ in mp.COVERS]
        self.tradable, self._muted = tradable, muted
    def muted(self):
        return self._muted

class FakeMoney:
    reserve = release = lambda self, s: True

def make(cluster=True):
    e = me.MHEngine()
    e.cluster = cluster
    e.min_ratio, e.max_per_target, e.hmax, e.set_size = 1.9, 4, 4, 3     # set_size 3: the tests are written for top-3 clusters
    e.money, e._next_stake, e._blocked = FakeMoney(), (lambda: 1.0), (lambda: False)
    e._T["R_10"], e._rtt, e.symbols = 2.0, 0.1, ["R_10"]
    e.bought = []
    async def buy(c, stake):
        e.bought.append((c.h, mp.COVERS[c.ci]))
        e._inflight -= 1
    e._buy_one = buy
    return e

def run_scan(e, sig_idx, p, ratios):
    mhs = types.SimpleNamespace(hs={h: FakeHS(p) for h in range(1, 6)})
    cands, info = {}, {}
    for h in range(1, 5):
        out = e._cluster_cands(mhs, h, 0.0)
        if out:
            cands[h], info[h] = out
    e._idx["R_10"] = sig_idx
    e._sig["R_10"] = me.MHSignal("R_10", sig_idx, time.time(), 0, cands, info)
    for k, v in ratios.items():
        e._ratio_cache[("R_10",) + k] = (v, time.time())
    asyncio.run(e._scan_once())
    return cands

def test_engine():
    e = make()
    ci = {n: mp.COVERS.index(c) for n, c in (("O4", ("OVER", 4)), ("O5", ("OVER", 5)), ("O6", ("OVER", 6)))}
    full = {(ci[n], h): 2.0 for n in ci for h in range(1, 5)}
    cands = run_scan(e, 100, TOP(7, 8, 9), full)
    assert e._cl_formed == 4 and cands[1][0][0] == ci["O4"] and len(cands[1]) == 3   # widest first, 3 covers
    assert len(e.bought) == 4 and all(c == ("OVER", 4) for _, c in e.bought)         # one per (sig, h)
    e = make()
    low = dict(full); low.update({(ci["O4"], h): 1.5 for h in range(1, 5)})
    run_scan(e, 100, TOP(7, 8, 9), low)                                              # OVER4 quote too thin -> OVER5
    assert {c for _, c in e.bought} == {("OVER", 5)} and e._cl_floor == 4
    e = make()
    run_scan(e, 100, TOP(1, 5, 9), full)                                             # spread out
    assert e._cl_spread == 4 and not e.bought
    e = make(); e._tcount[("R_10", 104)] = 3                                         # per-target cap 4
    run_scan(e, 103, TOP(7, 8, 9), full)
    n1, n2 = sum(1 for h, _ in e.bought if h == 1), sum(1 for h, _ in e.bought if h == 2)
    run_scan(e, 102, TOP(7, 8, 9), full)
    assert n1 == 1 and e._tcount[("R_10", 104)] == 4 and sum(1 for h, _ in e.bought if h == 2) == n2   # 2nd scan's h=2 blocked
    e = make()                                                                       # safeguards restored in cluster mode
    ratios = {(ci[n], h): 2.0 for n in ci for h in range(1, 5)}
    mhs = types.SimpleNamespace(hs={h: FakeHS(TOP(7, 8, 9), tradable=False) for h in range(1, 6)})
    assert all(e._cluster_cands(mhs, h, 0.0) is None for h in range(1, 5)) and e._cl_gate == 4     # not significant
    mhs = types.SimpleNamespace(hs={h: FakeHS(TOP(7, 8, 9), muted=True) for h in range(1, 6)})
    assert e._cluster_cands(mhs, 1, 0.0) is None                                                  # horizon muted
    e = make()
    mhs = types.SimpleNamespace(hs={h: FakeHS(TOP(7, 8, 9), lb=0.45) for h in range(1, 6)})        # lb*2.0-1 < 0
    e._idx["R_10"] = 100
    out = e._cluster_cands(mhs, 1, 0.0)
    e._sig["R_10"] = me.MHSignal("R_10", 100, time.time(), 0, {1: out[0]}, {1: out[1]})
    for k, v in ratios.items():
        e._ratio_cache[("R_10",) + k] = (v, time.time())
    asyncio.run(e._scan_once())
    assert not e.bought and e._cl_edge == 3                                                       # Wilson edge check applies
    e = make(cluster=False)                                                          # legacy path untouched
    e._sig["R_10"] = me.MHSignal("R_10", 5, time.time(), 0, {}); e._idx["R_10"] = 5
    asyncio.run(e._scan_once())
    assert not e.bought and e._cl_formed == 0

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
