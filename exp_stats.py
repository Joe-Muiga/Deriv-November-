"""
exp_stats.py - statistics + reporting for the pre-registered test (thread-safe; engine thread writes, Flask reads).

  * one observation per signal (test arm), profit in stakes (win = ratio-1, loss = -1), verdict.py states
  * break-even = 1/avg quoted ratio; expected "no edge" balance path = sum(stake_i * (p0*ratio_i - 1)) with a +/-2 SE band
  * three numbers reported separately: shadow hit rate, ALIGNED win rate, LATE win rate (+ "execution loss")
  * stopping rule TEST_MAX_TRADES; optional SPRT (TEST_SPRT=1)
Per-symbol / per-horizon tables are EXPLORATORY.
"""
import collections
import math
import threading
import time
from typing import Dict, List, Optional

import config
import verdict
from exp_mode import BASE_RATE, config_hash, frozen_config


def _wr(w, n):
    return (w / n) if n else None


class Experiment:
    def __init__(self):
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        with self.lock:
            self.t0 = time.time()
            self.start_balance: Optional[float] = None
            self.test: List[dict] = []                      # settled test-arm trades (compact rows)
            self.ctl: List[dict] = []                       # settled control-arm trades
            self.shadow = collections.Counter()             # ticks, top1, zone_all_n/h, fired_n/h
            self.shadow_sym: Dict[str, collections.Counter] = {}
            self.skipped_late = collections.Counter()
            self.rtt: Dict[str, dict] = {}
            self.epoch: Dict[str, dict] = {}
            self.diag: dict = {}
            self.ledger_stats: dict = {}
            self.cycle: dict = {}
            self.engine: dict = {}
            self.final: Optional[dict] = None
            self.done_reason = ""
            self.sprt_llr = 0.0
            self.sprt_state = "running"
            self._latest: dict = {}

    # ── recording ────────────────────────────────────────────────────────────
    def add_shadow(self, sym: str, top_hit: bool, zone_hit: bool, fired: bool) -> None:
        with self.lock:
            for c in (self.shadow, self.shadow_sym.setdefault(sym, collections.Counter())):
                c["ticks"] += 1
                c["top1"] += int(top_hit)
                c["zone_n"] += 1
                c["zone_h"] += int(zone_hit)
                if fired:
                    c["fired_n"] += 1
                    c["fired_h"] += int(zone_hit)

    def load_shadow(self, snap: dict) -> None:
        with self.lock:
            self.shadow = collections.Counter(snap.get("total", {}))
            self.shadow_sym = {s: collections.Counter(v) for s, v in (snap.get("per_symbol") or {}).items()}

    def shadow_snapshot(self) -> dict:
        with self.lock:
            return {"total": dict(self.shadow), "per_symbol": {s: dict(c) for s, c in self.shadow_sym.items()}}

    def add_settlement(self, row: dict) -> None:
        """row: arm, sym, won, profit, stake, ratio, align, h, realized_h, p_cal, p_rescored, ts"""
        with self.lock:
            (self.test if row.get("arm", "test") == "test" else self.ctl).append(row)
            if row.get("arm", "test") == "test" and getattr(config, "TEST_SPRT", False):
                self._sprt(bool(row["won"]))

    def _sprt(self, won: bool) -> None:
        if self.sprt_state != "running":
            return
        p0, p1 = BASE_RATE, float(config.TEST_SPRT_P1)
        self.sprt_llr += math.log(p1 / p0) if won else math.log((1 - p1) / (1 - p0))
        a, b = float(config.TEST_SPRT_ALPHA), float(config.TEST_SPRT_BETA)
        if self.sprt_llr >= math.log((1 - b) / a):
            self.sprt_state = "accept H1 (edge)"
        elif self.sprt_llr <= math.log(b / (1 - a)):
            self.sprt_state = "accept H0 (no edge)"

    # ── stopping rule ────────────────────────────────────────────────────────
    def n_test(self) -> int:
        with self.lock:
            return len(self.test)

    def stop_reason(self, open_test: int = 0) -> str:
        """'' while trading may continue; otherwise why it must stop."""
        with self.lock:
            if self.final is not None:
                return self.done_reason or "finished"
            if self.n_test() + open_test >= int(config.TEST_MAX_TRADES):
                return f"TEST_MAX_TRADES={config.TEST_MAX_TRADES} reached"
            if getattr(config, "TEST_SPRT", False) and self.sprt_state != "running":
                return f"SPRT: {self.sprt_state}"
            return ""

    def is_done(self) -> bool:
        with self.lock:
            return self.final is not None

    # ── verdict ──────────────────────────────────────────────────────────────
    @staticmethod
    def _obs(rows: List[dict]):
        return [((float(r["profit"]) / max(float(r["stake"]), 1e-9)), float(r["ratio"])) for r in rows]

    def verdict(self, rows: Optional[List[dict]] = None) -> dict:
        with self.lock:
            rows = self.test if rows is None else rows
            return verdict.summarize(self._obs(rows), int(config.TEST_VERDICT_MIN))

    def finalize(self, reason: str) -> dict:
        with self.lock:
            if self.final is None:
                v = self.verdict()
                self.final = {"reason": reason, "ts": time.time(), "verdict": v, "config_hash": config_hash(), "n": len(self.test)}
                self.done_reason = reason
            return self.final

    # ── balance chart ────────────────────────────────────────────────────────
    def balance_path(self, max_points: int = 400) -> dict:
        with self.lock:
            rows = list(self.test)
            sb = self.start_balance if self.start_balance is not None else 0.0
        act = exp = var = 0.0
        pts = []
        for i, r in enumerate(rows, 1):
            st, ra = float(r["stake"]), float(r["ratio"])
            act += float(r["profit"])
            exp += st * (BASE_RATE * ra - 1.0)
            var += (st * ra) ** 2 * BASE_RATE * (1 - BASE_RATE)
            se = math.sqrt(var)
            pts.append((i, sb + act, sb + exp, sb + exp - 2 * se, sb + exp + 2 * se))
        if len(pts) > max_points:
            step = len(pts) / max_points
            pts = [pts[min(len(pts) - 1, int(k * step))] for k in range(max_points)] + [pts[-1]]
        return {"start_balance": sb, "n": [p[0] for p in pts], "actual": [round(p[1], 4) for p in pts],
                "expected": [round(p[2], 4) for p in pts], "lo": [round(p[3], 4) for p in pts],
                "hi": [round(p[4], 4) for p in pts]}

    # ── three numbers + breakdowns ───────────────────────────────────────────
    def _arm_block(self, rows: List[dict]) -> dict:
        n = len(rows)
        w = sum(1 for r in rows if r["won"])
        return {"n": n, "wins": w, "win_rate": _wr(w, n), "pnl": round(sum(float(r["profit"]) for r in rows), 4)}

    def summary(self) -> dict:
        with self.lock:
            test, ctl = list(self.test), list(self.ctl)
            sh = dict(self.shadow)
            shs = {s: dict(c) for s, c in self.shadow_sym.items()}
        aligned = [r for r in test if r.get("align") == "ALIGNED"]
        late = [r for r in test if r.get("align") == "LATE"]
        other = [r for r in test if r.get("align") not in ("ALIGNED", "LATE")]
        shadow_fired = _wr(sh.get("fired_h", 0), sh.get("fired_n", 0))
        a_blk, l_blk = self._arm_block(aligned), self._arm_block(late)
        ex_wr = _wr(sum(1 for r in test if r["won"]), len(test))
        resc = [r for r in late if r.get("p_rescored") is not None]
        by_sym: Dict[str, dict] = {}
        for r in test:
            b = by_sym.setdefault(r["sym"], {"n": 0, "wins": 0, "pnl": 0.0})
            b["n"] += 1
            b["wins"] += int(bool(r["won"]))
            b["pnl"] += float(r["profit"])
        by_sym = {s: {"n": b["n"], "win_rate": _wr(b["wins"], b["n"]), "pnl": round(b["pnl"], 3)} for s, b in by_sym.items()}
        return {
            "three_numbers": {
                "shadow_hit_rate": shadow_fired, "shadow_fired_n": sh.get("fired_n", 0),
                "aligned": a_blk, "late": l_blk, "unlabelled": len(other),
                "execution_loss": (shadow_fired - ex_wr) if (shadow_fired is not None and ex_wr is not None) else None,
            },
            "shadow": {"ticks": sh.get("ticks", 0), "top1_hit": _wr(sh.get("top1", 0), sh.get("ticks", 0)), "top1_chance": 0.10,
                       "zone_hit_all": _wr(sh.get("zone_h", 0), sh.get("zone_n", 0)), "zone_chance": BASE_RATE},
            "late_rescored": {"n": len(resc), "win_rate": _wr(sum(1 for r in resc if r["won"]), len(resc)),
                              "mean_p_rescored": (sum(float(r["p_rescored"]) for r in resc) / len(resc)) if resc else None},
            "control": {**self._arm_block(ctl), "verdict": verdict.summarize(self._obs(ctl), int(config.TEST_VERDICT_MIN))},
            "aligned_only_verdict": verdict.summarize(self._obs(aligned), int(config.TEST_VERDICT_MIN)),
            "per_symbol_EXPLORATORY": by_sym,
            "per_horizon_EXPLORATORY": {"1": self._arm_block(test)},
        }

    # ── publication ──────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        v = self.verdict()
        with self.lock:
            out = {
                "ts": time.time(), "config_hash": config_hash(), "frozen_config": frozen_config(),
                "test_mode": bool(config.TEST_MODE), "max_trades": int(config.TEST_MAX_TRADES),
                "n_test": len(self.test), "verdict": v,
                "state": v["state"], "break_even": v["break_even"], "base_rate": BASE_RATE,
                "sprt": {"enabled": bool(config.TEST_SPRT), "llr": round(self.sprt_llr, 3), "state": self.sprt_state},
                "final": self.final, "balance": self.balance_path(),
                "skipped_late": dict(self.skipped_late), "rtt": self.rtt, "epoch": self.epoch,
                "diag": self.diag, "ledger": self.ledger_stats, "cycle": self.cycle, "engine": self.engine,
            }
        out.update(self.summary())
        return out

    def publish(self, snap: Optional[dict] = None) -> None:
        s = snap or self.snapshot()
        with self.lock:
            self._latest = s

    def latest(self) -> dict:
        with self.lock:
            return dict(self._latest) if self._latest else {"test_mode": bool(config.TEST_MODE), "config_hash": config_hash(),
                                                            "state": "no data yet"}

    def banner(self) -> str:
        f = self.final or {}
        v = f.get("verdict") or self.verdict()
        def pc(x):
            return "n/a" if x is None else f"{100 * x:.2f}%"
        return ("\n" + "=" * 78 + f"\nFINAL VERDICT ({f.get('reason', 'in progress')}) | config {config_hash()}\n"
                f"trades {v['n']} | win rate {pc(v['win_rate'])} vs break-even {pc(v['break_even'])} "
                f"(base rate {pc(BASE_RATE)}) | mean profit/stake "
                f"{'n/a' if v['mean'] is None else format(v['mean'], '+.4f')} "
                f"CI [{'n/a' if v['ci_low'] is None else format(v['ci_low'], '+.4f')}, "
                f"{'n/a' if v['ci_high'] is None else format(v['ci_high'], '+.4f')}]\n"
                f"STATE: {v['state'].upper()}\n" + "=" * 78)


EXP = Experiment()
