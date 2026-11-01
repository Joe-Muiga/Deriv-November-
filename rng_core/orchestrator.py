"""rng_core.orchestrator - per-symbol state machine.

    COLLECT (n < n_min) -> DIAGNOSE on a geometric schedule ->
        NO_STRUCTURE : stat model with a uniform-dominant prior, re-diagnose at the next scheduled n
        STAT_SIGNAL  : e-process crossed 1/alpha_e on some horizon, or diagnostics rejected BIAS/SERIAL tests
        EXACT_STATE  : a solver reproduced the segment and >= V held-out ticks; live monitor (first miss -> suspect, two misses in 50 ticks -> drop)
        CHANGEPOINT  : stat model restarts on the data after the change point; diagnosis continues

The class is synchronous and does no heavy work in on_tick (O(E) numpy per horizon). Heavy jobs are handed out as REQUESTS
(diag_request / exact_request / stat build) that the adapter runs in a thread or a worker process and hands back through
apply_* / install_*; the orchestrator never blocks on them.

Every prediction of every horizon is stored before its digit exists and scored when it arrives ('shadow scoring'), whatever the
status, so the scoreboard is honest whether or not the engine acts on it.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from . import diag as dg
from . import mathutil as mu
from .decide import Top4Calibrator, make_prediction
from .exact.base import ExactMonitor
from .observe import TickStore
from .stat_model import StatModel
from .types import (EXACT_STATE, NO_STRUCTURE, STAT_SIGNAL, V_BIAS, V_CHANGEPOINT, V_LINEAR, V_SERIAL, Prediction, finalize_dist,
                    top4_of)

STATUS_IDX = {NO_STRUCTURE: 0, STAT_SIGNAL: 1, EXACT_STATE: 2}


@dataclass
class V2Config:
    alpha: float = 0.01                       # total type-I budget: half to diagnostics (alpha-spending), half to the e-process
    hmax: int = 10
    store_capacity: int = 300_000
    diag_first: int = 5000
    diag_factor: float = 2.0
    diag_max_n: int = 100_000                 # diagnostics look at the newest this-many digits of the segment
    n_min: int = 600                          # below this the symbol is in COLLECT
    exact_first: Tuple[int, ...] = (300, 1200)
    validate_n: int = 24
    exact_mass: float = 0.999
    act_on_no_structure: bool = False
    solver_budget: float = 20.0
    solver_max_mb: int = 400
    exhaustive: bool = False                  # also run the multi-minute solvers (brute-force LCG scans, MT19937 elimination)
    B_compress: int = 200
    B_mc: int = 99
    nominal_gap: Optional[float] = None
    stat_cfg: Optional[dict] = None

    @classmethod
    def from_module(cls, m) -> "V2Config":
        g = lambda k, d: getattr(m, k, d)
        return cls(alpha=float(g("V2_ALPHA", 0.01)), hmax=int(g("V2_HMAX", 10)), store_capacity=int(g("V2_STORE_CAPACITY", 300_000)),
                   diag_first=int(g("V2_DIAG_FIRST", 5000)), validate_n=int(g("V2_VALIDATE_N", 24)),
                   exact_mass=float(g("V2_EXACT_MASS", 0.999)), act_on_no_structure=bool(int(g("V2_ACT_ON_NO_STRUCTURE", 0))),
                   solver_budget=float(g("V2_SOLVER_BUDGET_SECONDS", 20.0)), solver_max_mb=int(g("V2_SOLVER_MAX_MB", 400)),
                   exhaustive=bool(int(g("V2_EXHAUSTIVE", 0))))


class Scoreboard:
    """Prequential scoring of the PUBLISHED predictions: per status and horizon n, top-1, top-4, log-loss; rolling top-4 per horizon."""

    def __init__(self, hmax: int, roll: int = 500):
        self.hmax = hmax
        self.a = np.zeros((3, hmax + 1, 4))               # n, hit1, hit4, ll_sum
        self.rollh: Dict[int, Deque[int]] = {h: deque(maxlen=roll) for h in range(1, hmax + 1)}

    def add(self, status: str, h: int, hit1: int, hit4: int, ll: float) -> None:
        r = self.a[STATUS_IDX[status], h]
        r[0] += 1
        r[1] += hit1
        r[2] += hit4
        r[3] += ll
        self.rollh[h].append(hit4)

    def summary(self) -> dict:
        out = {}
        for s, i in STATUS_IDX.items():
            n = self.a[i, :, 0].sum()
            if n:
                out[s] = {"n": int(n), "top1": float(self.a[i, :, 1].sum() / n), "top4": float(self.a[i, :, 2].sum() / n),
                          "logloss": float(self.a[i, :, 3].sum() / n)}
        return out

    def rolling_top4(self, h: Optional[int] = None) -> Optional[float]:
        hs = [h] if h else list(self.rollh)
        v = [x for k in hs for x in self.rollh[k]]
        return (sum(v) / len(v)) if len(v) >= 100 else None


class SymbolOrchestrator:
    def __init__(self, symbol: str, cfg: Optional[V2Config] = None, calibrator: Optional[Top4Calibrator] = None, decimals: int = 2):
        self.symbol = symbol
        self.cfg = cfg or V2Config()
        self.decimals = decimals
        self.cal = calibrator or Top4Calibrator()
        self.store = TickStore(self.cfg.store_capacity, nominal_gap=self.cfg.nominal_gap)
        self.stat: Optional[StatModel] = StatModel(self.cfg.hmax, self.cfg.stat_cfg)
        self.board = Scoreboard(self.cfg.hmax)
        self.monitor: Optional[ExactMonitor] = None
        self.stat_signal = False
        self.stat_trigger: Optional[str] = None
        self.e_thr = math.log(self.cfg.hmax / (self.cfg.alpha / 2.0))
        self.last_report = None
        self.diag_runs = 0
        self.diag_inflight = False
        self.exact_inflight = False
        self.exact_progress: dict = {}
        self.exact_from_idx = 0                      # absolute index from which solvers look (moves after an exact drop)
        self._seg_start = 0
        self._diag_next = self.cfg.diag_first
        self._exact_sched: List[int] = list(self.cfg.exact_first)
        self._exact_next_n = self._exact_sched[0] if self._exact_sched else None
        self._gap_seen = 0
        self._missed_seen = 0
        self.pending_restart: Optional[int] = None
        self._pend: Dict[int, Deque[tuple]] = {h: deque() for h in range(1, self.cfg.hmax + 1)}
        self.n_dup = 0
        self.n_gaps = 0
        self.n_exact_drops = 0
        self.events: Deque[tuple] = deque(maxlen=200)    # (idx, text) status changes
        self.first_stat_n: Optional[int] = None
        self.first_exact_n: Optional[int] = None
        self._first_miss_idx: Optional[int] = None
        self.t_tick_ms = 0.0

    # ── status ───────────────────────────────────────────────────────────────
    @property
    def status(self) -> str:
        if self.monitor is not None and not self.monitor.dropped:
            return EXACT_STATE
        return STAT_SIGNAL if self.stat_signal else NO_STRUCTURE

    @property
    def phase(self) -> str:
        return "COLLECT" if self.n_seg < self.cfg.n_min else ("EXACT" if self.status == EXACT_STATE else
                                                              ("STAT" if self.stat_signal else "NO_STRUCTURE"))

    @property
    def n_seg(self) -> int:
        return self.store.n_total - max(self._seg_start, self.store.first_idx)

    def tradable(self) -> bool:
        return self.n_seg >= self.cfg.n_min and (self.status != NO_STRUCTURE or self.cfg.act_on_no_structure)

    def _event(self, text: str) -> None:
        self.events.append((self.store.n_total, text))

    def _set_stat_signal(self, trigger: str) -> None:
        if not self.stat_signal:
            self.stat_signal = True
            self.stat_trigger = trigger
            self.first_stat_n = self.store.n_total
            self._event(f"STAT_SIGNAL via {trigger}")

    # ── boot / history ───────────────────────────────────────────────────────
    def load_history(self, epochs, quotes, digits) -> int:
        """Store raw history (fast). The statistical model is built separately (build_stat in a thread, then install_stat)."""
        n = self.store.extend(epochs, quotes, digits)
        self._seg_start = self.store.segment(0)[0]
        self._diag_next = self.cfg.diag_first
        while self._diag_next <= self.n_seg:           # history that already crossed the schedule: diagnose at the NEXT point
            self._diag_next = int(self._diag_next * self.cfg.diag_factor)
        if self.n_seg >= self.cfg.diag_first:
            self._diag_next = self.cfg.diag_first      # run the first diagnostic right away on the loaded history
        self._exact_sched_reset()
        return n

    def _exact_sched_reset(self) -> None:
        sched = list(self.cfg.exact_first)
        k = self.cfg.diag_first
        while k < 4_000_000:
            sched.append(k)
            k = int(k * self.cfg.diag_factor)
        self._exact_sched = sched
        n = self.n_seg
        later = [x for x in sched if x > n]
        self._exact_next_n = later[0] if later else None
        if n >= sched[0] and (self._exact_next_n is None or self._exact_next_n > sched[0]):
            self._exact_next_n = min(self._exact_next_n or n, n)      # attempt once on the loaded history

    def build_stat(self, digits: np.ndarray) -> StatModel:
        """Heavy: batch-fit a StatModel on `digits` (call in a thread; touches no orchestrator state)."""
        m = StatModel(self.cfg.hmax, self.cfg.stat_cfg)
        m.load(np.asarray(digits, np.int64))
        return m

    def install_stat(self, model: StatModel, covered_end_idx: int) -> None:
        """Swap in a freshly built model, replaying the digits that arrived after the snapshot (store indices >= covered_end_idx)."""
        lo = max(covered_end_idx, self.store.first_idx)
        d = self.store.digits()
        base = self.store.first_idx
        for i in range(lo, self.store.n_total):
            model.on_digit(int(d[i - base]))
        self.stat = model
        self._pend = {h: deque() for h in range(1, self.cfg.hmax + 1)}     # old pending predictions refer to the old model

    # ── per tick ─────────────────────────────────────────────────────────────
    def on_tick(self, epoch: float, quote: float, digit: int) -> int:
        t0 = time.perf_counter()
        idx = self.store.append(epoch, quote, digit)
        if idx < 0:
            self.n_dup += 1
            return -1
        st = self.stat
        if self.store.gap_events > self._gap_seen:                       # ticks were missed before this one
            missed = self.store.missed - self._missed_seen
            self._gap_seen, self._missed_seen = self.store.gap_events, self.store.missed
            self.n_gaps += 1
            if st is not None:
                st.gap()
            for q in self._pend.values():
                q.clear()
            if self.monitor is not None:
                self.monitor.skip(missed)
            self._seg_start = idx
            self._diag_next = self.cfg.diag_first
            self._exact_sched_reset()
            self._event(f"gap: {missed} tick(s) missed -> new segment")
        for h, q in self._pend.items():                                  # shadow-score everything published for this tick
            while q and q[0][0] < idx:
                q.popleft()
            while q and q[0][0] == idx:
                _, status, top4, p4raw, dist = q.popleft()
                top = int(np.argmax(dist))
                hit4 = int(digit in top4)
                self.board.add(status, h, int(top4[0] == digit), hit4, -math.log(max(float(dist[digit]), 1e-12)))
                self.cal.add(status, p4raw, hit4)
        if st is not None:
            st.on_digit(int(digit))
            if not self.stat_signal:
                for h in range(1, self.cfg.hmax + 1):
                    if st.H[h].log_e >= self.e_thr:
                        self._set_stat_signal(f"e-process h={h} log_e={st.H[h].log_e:.1f}")
                        break
        if self.monitor is not None and not self.monitor.dropped:
            res = self.monitor.observe(int(digit))
            if res == "miss" and self._first_miss_idx is None:
                self._first_miss_idx = idx
            elif res == "hit" and not self.monitor.miss:
                self._first_miss_idx = None
            elif res == "drop":
                self.n_exact_drops += 1
                self.exact_from_idx = self._first_miss_idx if self._first_miss_idx is not None else idx - 1
                self._event(f"EXACT dropped ({self.monitor.misses} misses) -> re-solve on data after idx {self.exact_from_idx}")
                self.monitor = None
                self._first_miss_idx = None
                self._exact_next_n = self.n_seg + 150
        self._publish(idx)
        self.t_tick_ms += (time.perf_counter() - t0) * 1e3
        return idx

    def _publish(self, idx: int) -> None:
        """Store the top-4 of every horizon for tick idx+h (before the digit exists)."""
        st = self.stat
        status = self.status
        for h in range(1, self.cfg.hmax + 1):
            alt = st.dist_alt(h) if st is not None else None
            mix = st.dist(h) if st is not None else None
            if alt is None:
                alt = mix = np.full(10, 0.1)
            dist, rank = mix, alt
            if status == EXACT_STATE:
                pr = self.monitor.predict(h)
                if pr is not None:
                    dist = self._exact_dist(pr, alt)
                    rank = dist
            t4 = top4_of(rank)
            p4raw = float(sum(rank[d] for d in t4))
            self._pend[h].append((idx + h, status, t4, p4raw, dist))

    def _exact_dist(self, pr, alt: np.ndarray) -> np.ndarray:
        mass = self.cfg.exact_mass if not self.monitor.suspect else 0.90
        p = np.zeros(10)
        kind, v = pr
        if kind == "digit":
            rest = alt.copy()
            rest[v] = 0.0
            rest = rest / rest.sum() if rest.sum() > 0 else np.full(10, 1.0 / 9)
            p = (1.0 - mass) * rest
            p[v] = mass
        else:                                         # bit-level: mass spread over the digits consistent with the predicted bit
            p[:] = (1.0 - mass) / (10 - len(v))
            for d in v:
                p[d] = mass / len(v)
        return finalize_dist(p)

    # ── predictions for the engine ───────────────────────────────────────────
    def prediction(self, h: int) -> Optional[Prediction]:
        """The newest published Prediction for horizon h (target = newest tick + h), or None before any tick."""
        q = self._pend.get(h)
        if not q:
            return None
        target, status, t4, p4raw, dist = q[-1]
        st = self.stat
        log_e = st.log_e(h) if st is not None else 0.0
        n_sc = st.H[h].n_scored if st is not None else 0
        alt = st.dist_alt(h) if st is not None else None
        exact_digit = None
        if status == EXACT_STATE:
            pr = self.monitor.predict(h) if self.monitor else None
            exact_digit = pr[1] if (pr and pr[0] == "digit") else None
        src = "exact:" + self.monitor.fit["name"] if status == EXACT_STATE and self.monitor else ("stat_mixture" if status == STAT_SIGNAL else "uniform_null")
        p = make_prediction(self.symbol, target, h, dist, status, log_e, n_sc, src, self.cal, exact_digit, self.cfg.exact_mass,
                            dist_alt=alt if status != EXACT_STATE else None)
        if status == EXACT_STATE:
            p.top4, p.p4_raw = t4, p4raw
        return p

    # ── heavy-job requests ───────────────────────────────────────────────────
    def diag_request(self) -> Optional[dict]:
        if self.diag_inflight or self.n_seg < self._diag_next:
            return None
        lo, d = self.store.segment_digits()
        d = d[max(0, self._seg_start - lo):][-self.cfg.diag_max_n:]
        q = np.array(self.store.segment_quotes()[-len(d):], np.float64)
        if np.isnan(q).any():                       # no usable quotes (digit-only feed): skip the observation-map streams
            q = None
        self.diag_inflight = True
        self.diag_runs += 1
        return {"digits": np.array(d, np.int64), "quotes": q, "run_index": self.diag_runs, "alpha": self.cfg.alpha / 2.0,
                "decimals": self.decimals, "n_end_idx": self.store.n_total, "B_compress": self.cfg.B_compress, "B_mc": self.cfg.B_mc,
                "offset_idx": self.store.n_total - len(d)}

    def apply_diag(self, req: dict, report) -> None:
        self.diag_inflight = False
        self.last_report = report
        while self._diag_next <= self.n_seg:
            self._diag_next = int(self._diag_next * self.cfg.diag_factor)
        fams = set(report.families)
        stat_f = fams & {V_BIAS, V_SERIAL}
        other = fams - {V_BIAS, V_SERIAL, V_CHANGEPOINT}
        self._event(f"diag #{report.run_index} n={report.n}: {report.verdict} rejected={report.rejected[:4]}")
        if V_CHANGEPOINT in fams and report.changepoint_idx is not None:
            self.pending_restart = req["offset_idx"] + int(report.changepoint_idx)
            self.stat_signal = False
            self.stat_trigger = None
            self._event(f"CHANGEPOINT at idx {self.pending_restart}: stat model restarts after it")
            return
        if stat_f:
            self._set_stat_signal(f"diag {report.verdict}")
        if other or self.cfg.exhaustive:
            self._exact_next_n = self.n_seg                               # try solvers now
            self._want_mt = V_LINEAR in other or self.cfg.exhaustive

    _want_mt = False

    def exact_request(self) -> Optional[dict]:
        if self.exact_inflight or self.monitor is not None or self._exact_next_n is None or self.n_seg < self._exact_next_n:
            return None
        lo = max(self.store.segment(0)[0], self.exact_from_idx, self._seg_start, self.store.first_idx)
        base = self.store.first_idx
        d = self.store.digits()[lo - base:]
        if len(d) < 100:
            return None
        ep0 = float(self.store.epochs()[lo - base])
        self.exact_inflight = True
        later = [x for x in self._exact_sched if x > self.n_seg]
        self._exact_next_n = later[0] if later else None
        return {"digits": np.array(d, np.int64).tolist(), "epoch0": ep0, "budget_s": self.cfg.solver_budget, "max_mb": self.cfg.solver_max_mb,
                "enable_brute": self.cfg.exhaustive, "try_mt": bool(self._want_mt or self.cfg.exhaustive), "progress": self.exact_progress,
                "start_idx": int(lo), "end_idx": int(self.store.n_total)}

    def apply_exact(self, req: dict, result: dict) -> None:
        self.exact_inflight = False
        if not result or not result.get("ok", True):
            self._event(f"exact solver failed: {(result or {}).get('error')}")
            return
        if result.get("progress"):
            self.exact_progress = result["progress"]
        fit = result.get("fit")
        if not fit:
            return
        lo, hi = req["start_idx"], req["end_idx"]
        base = self.store.first_idx
        allv = self.store.digits()
        seg = [int(x) for x in allv[lo - base:hi - base]]
        mon = ExactMonitor(fit, seg)
        # catch up on the ticks that arrived while the solver ran; any miss here means the fit does not hold -> discard
        for i in range(hi, self.store.n_total):
            if mon.observe(int(allv[i - base])) != "hit":
                self._event(f"exact fit {fit['name']} failed catch-up replay: discarded")
                return
        self.monitor = mon
        self._first_miss_idx = None
        if self.first_exact_n is None:
            self.first_exact_n = self.store.n_total
        for _ in range(self.cfg.validate_n):                              # seed the EXACT pool with the validation evidence
            self.cal.add(EXACT_STATE, 0.9993 if fit["kind"] != "bits" else 0.7992, 1)
        self._event(f"EXACT_STATE via {fit['name']} (map {fit['map']}, fit {fit['n_fit']} ticks, {fit['n_check']} held-out, "
                    f"{fit['solve_s']}s)")

    def restart_request(self) -> Optional[int]:
        """Absolute index after which the stat model should be rebuilt (change point), or None."""
        r, self.pending_restart = self.pending_restart, None
        return r

    def snapshot(self) -> dict:
        s = self.board.summary()
        st = self.stat
        return {"symbol": self.symbol, "status": self.status, "phase": self.phase, "n": self.store.n_total, "n_seg": self.n_seg,
                "log_e_max": max((st.H[h].log_e for h in st.H), default=0.0) if st else 0.0, "e_thr": round(self.e_thr, 2),
                "stat_trigger": self.stat_trigger, "exact": (self.monitor.fit["name"] if self.monitor else None),
                "exact_hits": (self.monitor.hits if self.monitor else 0), "exact_drops": self.n_exact_drops,
                "roll_top4": self.board.rolling_top4(), "scoreboard": s, "gaps": self.n_gaps, "dups": self.n_dup + self.store.dups,
                "ms_per_tick": round(self.t_tick_ms / max(self.store.n_total, 1), 3), "verdict": getattr(self.last_report, "verdict", None),
                "store": self.store.info()}
