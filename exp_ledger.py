"""
exp_ledger.py - append-only evidence ledger that survives redeploys.

Backend: Postgres when DATABASE_URL is set (psycopg), otherwise SQLite on disk (LEDGER_PATH, default
./exp_ledger.sqlite - NOTE: Render's disk is wiped on every deploy, so with the 13-minute redeploy cycle only
Postgres keeps evidence across cycles).  If a write fails, rows are spilled to a JSONL file and replayed later.
Only INSERT and SELECT are ever issued (SQLite additionally has triggers that abort UPDATE/DELETE).

Tables
  exp_predictions  every tick's prediction, written BEFORE the outcome (ts, sym, idx, h, probs[10], zone probs, threshold, decision)
  exp_outcomes     shadow outcome of each prediction (digit that arrived, top-1 / zone hit)
  exp_settlements  every settled contract (test and control arm), realized_h, ALIGNED/LATE label
  exp_events       everything else (buy, skipped_late, mute, shadow_snapshot, deploy_triggered, ready, donkey_guard_state, ...)
Writes go through one background thread so the trading loop never blocks.
"""
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

import config

logger = logging.getLogger("ledger")

COLS = {
    "exp_predictions": ["ts", "epoch", "sym", "idx", "h", "probs", "zone_over", "zone_under", "cal_over", "cal_under",
                        "threshold", "decision", "cfg_hash"],
    "exp_outcomes": ["ts", "sym", "idx", "digit", "top_hit", "zone", "zone_hit", "fired"],
    "exp_settlements": ["ts", "cid", "sym", "sig_idx", "h", "realized_h", "align", "arm", "contract", "stake", "ratio",
                        "profit", "won", "exit_digit", "p_cal", "p_rescored", "rtt", "cfg_hash"],
    "exp_events": ["ts", "kind", "data"],
}
TYPES = {"ts": "R", "epoch": "R", "sym": "T", "idx": "I", "h": "I", "probs": "T", "zone_over": "R", "zone_under": "R",
         "cal_over": "R", "cal_under": "R", "threshold": "R", "decision": "T", "cfg_hash": "T", "digit": "I",
         "top_hit": "I", "zone": "T", "zone_hit": "I", "fired": "I", "cid": "T", "sig_idx": "I", "realized_h": "I",
         "align": "T", "arm": "T", "contract": "T", "stake": "R", "ratio": "R", "profit": "R", "won": "I",
         "exit_digit": "I", "p_cal": "R", "p_rescored": "R", "rtt": "R", "kind": "T", "data": "T"}


def _ddl(pg: bool) -> List[str]:
    out = []
    for t, cols in COLS.items():
        pk = "id BIGSERIAL PRIMARY KEY" if pg else "id INTEGER PRIMARY KEY AUTOINCREMENT"
        ty = {"R": "DOUBLE PRECISION" if pg else "REAL", "T": "TEXT", "I": "BIGINT" if pg else "INTEGER"}
        out.append(f"CREATE TABLE IF NOT EXISTS {t} ({pk}, " + ", ".join(f"{c} {ty[TYPES[c]]}" for c in cols) + ")")
    out.append("CREATE INDEX IF NOT EXISTS ix_exp_set_ts ON exp_settlements (ts)")
    out.append("CREATE INDEX IF NOT EXISTS ix_exp_ev_kind ON exp_events (kind, ts)")
    if not pg:
        for t in COLS:
            out.append(f"CREATE TRIGGER IF NOT EXISTS {t}_no_upd BEFORE UPDATE ON {t} "
                       f"BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END")
            out.append(f"CREATE TRIGGER IF NOT EXISTS {t}_no_del BEFORE DELETE ON {t} "
                       f"BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END")
    return out


def _db_url() -> str:
    return (os.environ.get("DATABASE_URL") or getattr(config, "DATABASE_URL", "") or "").strip()


class Ledger:
    def __init__(self, path: Optional[str] = None, url: Optional[str] = None):
        self.url = _db_url() if url is None else url
        base = os.path.dirname(os.path.abspath(__file__))
        self.path = path or getattr(config, "LEDGER_PATH", "") or os.path.join(base, "exp_ledger.sqlite")
        self.spill_path = self.path + ".spill.jsonl"
        self.pg = False
        self.backend = "none"
        self._conn = None
        self._q: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.errors = 0
        self.written = 0
        self.last_error = ""

    # ── connection ───────────────────────────────────────────────────────────
    def _connect(self):
        if self.url:
            try:
                import psycopg
                url = self.url if "sslmode=" in self.url else self.url + (("&" if "?" in self.url else "?") + "sslmode=require")
                c = psycopg.connect(url, autocommit=True, connect_timeout=15, prepare_threshold=None)
                self.pg, self.backend = True, "postgres"
                return c
            except Exception as exc:
                logger.error("LEDGER: Postgres unavailable (%s) - falling back to SQLite", exc)
        c = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        c.execute("PRAGMA journal_mode=WAL")
        self.pg, self.backend = False, "sqlite"
        return c

    def open(self) -> "Ledger":
        with self._lock:
            if self._conn is not None:
                return self
            self._conn = self._connect()
            cur = self._conn.cursor()
            for stmt in _ddl(self.pg):
                cur.execute(stmt)
            if not self.pg:
                self._conn.commit()
            self._replay_spill()
            if self._thread is None:
                self._thread = threading.Thread(target=self._writer, name="ledger-writer", daemon=True)
                self._thread.start()
        logger.info("LEDGER open: backend=%s%s", self.backend,
                    "" if self.pg else f" path={self.path} (wiped on every Render deploy - set DATABASE_URL to keep evidence)")
        return self

    def _ph(self) -> str:
        return "%s" if self.pg else "?"

    # ── write path ───────────────────────────────────────────────────────────
    def append(self, table: str, row: Dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("ts", time.time())
        self._q.put((table, row))

    def prediction(self, **kw) -> None:
        self.append("exp_predictions", kw)

    def outcome(self, **kw) -> None:
        self.append("exp_outcomes", kw)

    def settlement(self, **kw) -> None:
        self.append("exp_settlements", kw)

    def event(self, kind: str, **data) -> None:
        self.append("exp_events", {"kind": kind, "data": json.dumps(data, default=str)})

    def _insert(self, table: str, rows: List[dict]) -> None:
        cols = COLS[table]
        ph = ", ".join([self._ph()] * len(cols))
        sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({ph})"
        cur = self._conn.cursor()
        cur.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])
        if not self.pg:
            self._conn.commit()

    def _writer(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [item]
            while len(batch) < 1000:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            flush_marks = [b for b in batch if b[0] == "__flush__"]
            by: Dict[str, List[dict]] = {}
            for t, r in batch:
                if t != "__flush__":
                    by.setdefault(t, []).append(r)
            for t, rows in by.items():          # one table at a time, order inside a table preserved
                try:
                    self._insert(t, rows)
                    self.written += len(rows)
                except Exception as exc:
                    self.errors += 1
                    self.last_error = str(exc)[:200]
                    logger.error("LEDGER write failed (%s) - spilling %d row(s) to %s", exc, len(rows), self.spill_path)
                    self._spill(t, rows)
                    self._reconnect()
            for _, ev in flush_marks:
                ev.set()
            for _ in batch:
                self._q.task_done()

    def _reconnect(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
        try:
            self._conn = self._connect()
        except Exception as exc:
            logger.error("LEDGER reconnect failed: %s", exc)

    def _spill(self, table: str, rows: List[dict]) -> None:
        try:
            with open(self.spill_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps({"t": table, "r": r}, default=str) + "\n")
        except Exception as exc:
            logger.critical("LEDGER spill failed too: %s", exc)

    def _replay_spill(self) -> None:
        if not os.path.exists(self.spill_path):
            return
        try:
            by: Dict[str, List[dict]] = {}
            with open(self.spill_path, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    by.setdefault(d["t"], []).append(d["r"])
            for t, rows in by.items():
                self._insert(t, rows)
            os.remove(self.spill_path)
            logger.warning("LEDGER: replayed %d spilled row(s)", sum(len(v) for v in by.values()))
        except Exception as exc:
            logger.error("LEDGER spill replay failed: %s", exc)

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until everything queued so far is written (call from a thread or via to_thread)."""
        ev = threading.Event()
        self._q.put(("__flush__", ev))
        return ev.wait(timeout)

    # ── read path (boot restore, reports) ────────────────────────────────────
    def select(self, table: str, where: str = "", params: Iterable = (), order: str = "id", limit: int = 0) -> List[dict]:
        cols = ["id"] + COLS[table]
        sql = f"SELECT {', '.join(cols)} FROM {table}"
        if where:
            sql += f" WHERE {where.replace('?', self._ph())}"
        sql += f" ORDER BY {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        conn = self._connect()                          # own connection: never shares a cursor with the writer thread
        try:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        finally:
            conn.close()
        return [dict(zip(cols, r)) for r in rows]

    def events(self, kind: str, limit: int = 0, newest_first: bool = False) -> List[dict]:
        rows = self.select("exp_events", "kind = ?", (kind,), order="id DESC" if newest_first else "id", limit=limit)
        for r in rows:
            try:
                r["data"] = json.loads(r["data"] or "{}")
            except Exception:
                r["data"] = {}
        return rows

    def last_event(self, kind: str) -> Optional[dict]:
        r = self.events(kind, limit=1, newest_first=True)
        return r[0] if r else None

    def settlements(self) -> List[dict]:
        return self.select("exp_settlements")

    def stats(self) -> dict:
        return {"backend": self.backend, "written": self.written, "errors": self.errors, "queued": self._q.qsize(),
                "last_error": self.last_error}


_LEDGER: Optional[Ledger] = None
_LOCK = threading.Lock()


def get_ledger() -> Ledger:
    global _LEDGER
    with _LOCK:
        if _LEDGER is None:
            _LEDGER = Ledger().open()
        return _LEDGER
