"""rng_core.adapter - glue between the engine (mh_engine.py) and the RNG-agnostic core.

V2Symbol is a duck-typed stand-in for mh_predictor.MHSymbol: the engine's existing code (`_cluster_cands`, workers, readiness checks)
runs on it unchanged. Each horizon view (`hs[h]`) exposes
    cur      (target_idx, calibrated distribution)   - the published distribution shrunk towards uniform so that its top-4 mass
                                                      equals the CALIBRATED probability p4_cal (ranking is unchanged by shrinkage)
    cal      list of callables  p_cover -> (p, lower bound)  (lower bound scaled by p4_lb / p4_cal)
    tradable bool  - Selector.admit(Prediction) and orchestrator.tradable()
    muted()  bool  - rolling top-4 hit rate of this horizon significantly below the fair 0.40
No held-out `oos_at` / `pooled_lb_at` calibrator methods are provided, so the engine's legacy OOS gate is skipped (those calibrators
do not exist in v2; its calibration is prequential by construction).

Heavy work never runs on the event loop: boot-time model fit and diagnostics use asyncio.to_thread, exact solvers run in a
budgeted child process (rng_core.exact.worker).
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from . import diag as dg
from .decide import Top4Calibrator
from .orchestrator import SymbolOrchestrator, V2Config
from .selector import Selector
from .types import EXACT_STATE, NO_STRUCTURE, STAT_SIGNAL

NCOVERS = 20                                    # mh_predictor.COVERS (OVER 0..8, UNDER 1..9); only the count matters here
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED_CAL: Optional[Top4Calibrator] = None
_BUILD_LOCKS: Dict[int, asyncio.Lock] = {}


def _build_lock() -> asyncio.Lock:
    """One model build at a time per event loop: the batch fit needs ~3 KB of transient memory per history tick, so concurrent
    symbol boots must not stack (512 MB container)."""
    k = id(asyncio.get_running_loop())
    if k not in _BUILD_LOCKS:
        _BUILD_LOCKS.clear()
        _BUILD_LOCKS[k] = asyncio.Lock()
    return _BUILD_LOCKS[k]



def _release_memory() -> None:
    """Give the batch fit's freed memory back to the OS (glibc keeps it otherwise); a no-op where unavailable."""
    try:
        import ctypes
        import gc
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def shared_calibrator() -> Top4Calibrator:
    """One calibrator pooled across every symbol of the process (brief: pooled per status)."""
    global _SHARED_CAL
    if _SHARED_CAL is None:
        _SHARED_CAL = Top4Calibrator()
    return _SHARED_CAL


class _V2H:
    """Per-horizon view with the attributes the engine reads from HState."""

    def __init__(self, owner: "V2Symbol", h: int):
        self.o, self.h = owner, h
        self.roll: deque = deque(maxlen=1)         # kept for engine compatibility (carry-over of mutes is not used in v2)
        self._key = -1
        self._cur = None
        self._pred = None
        self._shrink = 1.0
        self.cal = [self._cover_cal] * NCOVERS     # same callable for every cover: the distribution is already calibrated

    def _refresh(self) -> None:
        key = self.o.orch.store.n_total
        if key == self._key:
            return
        self._key = key
        p = self.o.orch.prediction(self.h)
        self._pred = p
        if p is None:
            self._cur = None
            return
        base = np.asarray(p.dist_alt if p.dist_alt is not None else p.dist, np.float64)
        span = p.p4_raw - 0.4
        lam = 0.0 if span <= 1e-9 else float(min(1.0, max(0.0, (p.p4_cal - 0.4) / span)))
        d = 0.1 + lam * (base - 0.1)
        self._cur = (p.target_idx, d / d.sum())
        self._shrink = (p.p4_lb / p.p4_cal) if p.p4_cal > 0 else 1.0

    def _cover_cal(self, pc: float):
        return float(pc), float(max(0.0, pc * min(1.0, self._shrink)))

    @property
    def cur(self):
        self._refresh()
        return self._cur

    @property
    def prediction(self):
        self._refresh()
        return self._pred

    @property
    def tradable(self) -> bool:
        self._refresh()
        return bool(self._pred is not None and self.o.orch.tradable() and self.o.selector.admit(self._pred))

    def muted(self) -> bool:
        r = self.o.orch.board.rollh[self.h]
        n = len(r)
        return bool(n >= 300 and sum(r) / n < 0.40 - 2.33 * math.sqrt(0.24 / n) and self.o.orch.status != EXACT_STATE)

    @property
    def val_ll(self) -> Optional[float]:
        s = self.o.orch.board.a[:, self.h, :].sum(axis=0)
        return float(s[3] / s[0]) if s[0] else None

    @property
    def n_scored(self) -> int:
        return int(self.o.orch.board.a[:, self.h, 0].sum())


class V2Symbol:
    is_v2 = True

    def __init__(self, symbol: str, cfg: Optional[V2Config] = None, decimals: int = 2, rank_mode: str = "off",
                 calibrator: Optional[Top4Calibrator] = None, min_edge: float = 0.0):
        self.symbol = symbol
        self.cfg = cfg or V2Config()
        self.orch = SymbolOrchestrator(symbol, self.cfg, calibrator or shared_calibrator(), decimals)
        self.selector = Selector(rank_mode, min_edge, self.cfg.act_on_no_structure)
        self.hs: Dict[int, _V2H] = {}
        self._tick_n = 0
        self._tasks: set = set()
        self._stat_task: Optional[asyncio.Task] = None
        self._diag_s = 0.0
        self._solver_s = 0.0

    # ── MHSymbol surface ─────────────────────────────────────────────────────
    @property
    def n(self) -> int:
        return self.orch.store.n_total

    @property
    def last_epoch(self) -> Optional[float]:
        s = self.orch.store
        return float(s.epochs()[-1]) if s.n_total else None

    def ready(self) -> bool:
        return bool(self.hs) and self.orch.n_seg >= self.cfg.n_min

    def screen(self, h, q, min_edge):                 # non-cluster mode is not supported by v2
        return None

    async def fit(self, digits, last_epoch, deadline: float = 0.0, offload: bool = False) -> dict:
        raise RuntimeError("V2Symbol is built from ticks: use fit_ticks(ticks, digits, decimals)")

    async def fit_ticks(self, ticks, digits, decimals: int) -> dict:
        """History hand-off: raw store on the loop (fast, numpy), model fit in a thread, then replay what arrived meanwhile."""
        eps = [float(t["epoch"]) for t in ticks]
        qs = [float(t["quote"]) for t in ticks]
        self.orch.decimals = decimals
        self.orch.load_history(eps, qs, [int(x) for x in digits])
        covered = self.orch.store.n_total
        _, seg = self.orch.store.segment_digits()
        async with _build_lock():
            model = await asyncio.to_thread(self.orch.build_stat, np.asarray(seg, np.int64))
        self.orch.install_stat(model, covered)
        _release_memory()
        self.hs = {h: _V2H(self, h) for h in range(1, self.cfg.hmax + 1)}
        if self.orch.store.n_total:
            self.orch._publish(self.orch.store.n_total - 1)
        return {"h": {"status": self.orch.status, "n": self.orch.store.n_total, "n_seg": self.orch.n_seg,
                      "segments": len(self.orch.store.seg_starts)}}

    def on_tick(self, digit: int, ep: float, quote: Optional[float] = None) -> bool:
        idx = self.orch.on_tick(float(ep), float("nan") if quote is None else float(quote), int(digit))
        if idx < 0:
            return False
        self._tick_n += 1
        if self.orch.pending_restart is not None:
            self._restart_stat()
        if self._tick_n % 5 == 0:
            self._kick()
        return True

    # ── background jobs ──────────────────────────────────────────────────────
    def _spawn(self, coro) -> None:
        try:
            t = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()                                # no running loop (tests / offline harness): jobs are driven explicitly
            return
        self._tasks.add(t)
        t.add_done_callback(self._tasks.discard)

    def _kick(self) -> None:
        o = self.orch
        req = o.diag_request()
        if req is not None:
            self._spawn(self._run_diag(req))
        req = o.exact_request()
        if req is not None:
            self._spawn(self._run_exact(req))

    async def _run_diag(self, req: dict) -> None:
        try:
            rep = await asyncio.to_thread(dg.run_diagnostics, req["digits"], req["run_index"], req["alpha"], req["quotes"],
                                          req["decimals"], 0, req["B_compress"], req["B_mc"])
            self.orch.apply_diag(req, rep)
        except Exception as exc:
            self.orch.diag_inflight = False
            self.orch._event(f"diag failed: {type(exc).__name__}: {exc}")

    async def _run_exact(self, req: dict) -> None:
        try:
            res = await run_solver_async(req)
        except Exception as exc:
            res = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self.orch.apply_exact(req, res)

    def _restart_stat(self) -> None:
        """Change point: rebuild the statistical model on the data after it (thread), then swap in."""
        start = self.orch.restart_request()
        if start is None or (self._stat_task is not None and not self._stat_task.done()):
            return
        o = self.orch
        o._seg_start = max(start, o.store.first_idx)
        lo, seg = o.store.segment_digits()
        seg = np.asarray(seg, np.int64)[max(0, o._seg_start - lo):]
        covered = o.store.n_total

        async def job():
            m = await asyncio.to_thread(o.build_stat, seg)
            o.install_stat(m, covered)

        try:
            self._stat_task = asyncio.get_running_loop().create_task(job())
        except RuntimeError:
            o.install_stat(o.build_stat(seg), covered)

    def brief(self) -> str:
        o = self.orch
        s = f"{self.symbol}:{o.status[:4]}"
        if o.status == EXACT_STATE and o.monitor is not None:
            s += f"[{o.monitor.fit['name']}]"
        rt = o.board.rolling_top4()
        return s + (f" t4={100 * rt:.1f}%" if rt is not None else "") + f" n={o.store.n_total}"

    def snapshot(self) -> dict:
        return self.orch.snapshot()


async def run_solver_async(req: dict, python: Optional[str] = None) -> dict:
    """Run the exact-solver ladder in a child process with CPU / memory limits and a wall-clock kill."""
    budget = float(req.get("budget_s", 20.0))
    payload = json.dumps({k: req[k] for k in ("digits", "epoch0", "budget_s", "max_mb", "enable_brute", "try_mt", "progress")})
    proc = await asyncio.create_subprocess_exec(python or sys.executable, "-m", "rng_core.exact.worker", cwd=_ROOT,
                                                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(payload.encode()), timeout=budget + 15.0)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"ok": False, "error": "wall-clock timeout"}
    if proc.returncode != 0 or not out:
        return {"ok": False, "error": f"worker exit {proc.returncode}"}
    return json.loads(out.decode())


def make_symbol(symbol: str, hmax: int, decimals: int, cfg_module=None):
    """Factory used by the engine when MH_PRED_V2=1 (imported lazily, so MH_PRED_V2=0 never loads rng_core)."""
    if cfg_module is None:
        import config as cfg_module
    c = V2Config.from_module(cfg_module)
    c.hmax = max(int(hmax), int(c.hmax))
    return V2Symbol(symbol, c, decimals, rank_mode=str(getattr(cfg_module, "V2_RANK_MODE", "off")).lower(),
                    min_edge=float(getattr(cfg_module, "V2_MIN_EDGE", 0.0)))
