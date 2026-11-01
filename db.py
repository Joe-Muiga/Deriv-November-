"""
db.py - Neon Postgres layer for the follower system.

* psycopg v3 + psycopg_pool, SSL, prepared statements OFF (Neon's pooled URL is pgbouncer).
* Tables are created on startup (idempotent).
* The collector only calls record_collector_trade(): it adds to an in-memory buffer and
  returns.  A dedicated daemon thread (no event loop) flushes every ~30 s and on shutdown.
* If the DB is unreachable the buffered writes are appended to a local JSONL file and are
  replayed (oldest first) once the DB answers again.  Nothing here can block the collector.
* Every function that touches the DB is synchronous: call it from a thread / executor.
"""
import json
import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import follower_config as fc

logger = logging.getLogger("follower.db")

try:
    import psycopg
    from psycopg_pool import ConnectionPool
    _IMPORT_ERROR: Optional[str] = None
except Exception as _exc:                                  # pragma: no cover
    psycopg = None            # type: ignore
    ConnectionPool = None     # type: ignore
    _IMPORT_ERROR = str(_exc)

# ── buffers (collector side only ever touches these under _lock) ──────────────
_lock = threading.Lock()
_minute_delta: Dict[int, List[float]] = {}      # minute -> [trades, wins, psu, stake_sum, ratio_sum]
_ops: List[dict] = []                           # ordered follower-trade ops
_flush_lock = threading.Lock()
_file_lock = threading.Lock()

_pool = None
_pool_lock = threading.Lock()
_started = False
_start_lock = threading.Lock()
_inited = False
_wake = threading.Event()
_db_ok = False
_last_error = ""

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS minute_stats(
        minute SMALLINT PRIMARY KEY,
        trades BIGINT NOT NULL DEFAULT 0,
        wins BIGINT NOT NULL DEFAULT 0,
        profit_stake_units DOUBLE PRECISION NOT NULL DEFAULT 0,
        stake_sum DOUBLE PRECISION NOT NULL DEFAULT 0,
        payout_ratio_sum DOUBLE PRECISION NOT NULL DEFAULT 0)""",
    "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)",
    "CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT)",
    """CREATE TABLE IF NOT EXISTS followers(
        id SERIAL PRIMARY KEY,
        label TEXT NOT NULL,
        token_enc TEXT NOT NULL,
        token_last4 TEXT,
        app_id TEXT NOT NULL,
        enabled BOOLEAN NOT NULL DEFAULT TRUE,
        p_base DOUBLE PRECISION,
        peak DOUBLE PRECISION,
        floor_amt DOUBLE PRECISION,
        tier TEXT,
        last_reset_date TEXT,
        milestone_hit BOOLEAN NOT NULL DEFAULT FALSE,
        last_balance DOUBLE PRECISION,
        last_window_result TEXT,
        state_ts TIMESTAMPTZ,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
    """CREATE TABLE IF NOT EXISTS zone_snapshots(
        id SERIAL PRIMARY KEY,
        window_id TEXT UNIQUE,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        zones TEXT NOT NULL,
        minute_stats TEXT NOT NULL,
        source TEXT)""",
    """CREATE TABLE IF NOT EXISTS follower_trades(
        id BIGSERIAL PRIMARY KEY,
        follower_id INTEGER NOT NULL,
        contract_id TEXT NOT NULL,
        signal_id TEXT,
        symbol TEXT,
        contract_type TEXT,
        barrier TEXT,
        stake DOUBLE PRECISION,
        payout_ratio DOUBLE PRECISION,
        opened_at TIMESTAMPTZ,
        profit DOUBLE PRECISION,
        source TEXT,
        snapshot_id INTEGER,
        window_id TEXT,
        UNIQUE(follower_id, contract_id))""",
    "CREATE INDEX IF NOT EXISTS ft_signal_idx ON follower_trades(signal_id)",
    "CREATE INDEX IF NOT EXISTS ft_follower_idx ON follower_trades(follower_id, opened_at)",
]

UPSERT_MIN = """INSERT INTO minute_stats(minute,trades,wins,profit_stake_units,stake_sum,payout_ratio_sum)
VALUES (%s,%s,%s,%s,%s,%s)
ON CONFLICT (minute) DO UPDATE SET
  trades=minute_stats.trades+EXCLUDED.trades,
  wins=minute_stats.wins+EXCLUDED.wins,
  profit_stake_units=minute_stats.profit_stake_units+EXCLUDED.profit_stake_units,
  stake_sum=minute_stats.stake_sum+EXCLUDED.stake_sum,
  payout_ratio_sum=minute_stats.payout_ratio_sum+EXCLUDED.payout_ratio_sum"""


# ── pool / helpers ───────────────────────────────────────────────────────────
def enabled() -> bool:
    return fc.db_enabled() and psycopg is not None


def status() -> dict:
    return {"enabled": enabled(), "db_ok": _db_ok, "last_error": _last_error,
            "import_error": _IMPORT_ERROR}


def _conninfo() -> str:
    url = fc.DATABASE_URL
    if "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


def _get_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                _conninfo(), min_size=1, max_size=5, open=False, timeout=20,
                max_idle=240, max_lifetime=1800, name="follower-db",
                check=ConnectionPool.check_connection,
                kwargs={"connect_timeout": 15, "prepare_threshold": None})
            _pool.open(wait=False)
        return _pool


def _mark(ok: bool, exc: Optional[Exception] = None) -> None:
    global _db_ok, _last_error
    _db_ok = ok
    if exc is not None:
        _last_error = f"{type(exc).__name__}: {str(exc)[:160]}"


def run(fn: Callable[[Any], Any], retries: int = 2) -> Any:
    """Run fn(conn) in one transaction with retries (Neon may be waking from suspend)."""
    if not enabled():
        raise RuntimeError("database not configured")
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with _get_pool().connection(timeout=20) as conn:
                out = fn(conn)
            _mark(True)
            return out
        except Exception as exc:
            last = exc
            _mark(False, exc)
            if attempt < retries:
                time.sleep(min(1.5 * (attempt + 1), 4.0))
    raise last  # type: ignore[misc]


# ── collector-facing API (never blocks, never raises) ────────────────────────
def record_collector_trade(opened_at: float, stake: float, ratio: float, profit: float) -> None:
    if not fc.db_enabled():
        return
    try:
        minute = int((float(opened_at) % 3600) // 60)
        units = (profit / stake) if stake > 0 else 0.0
        with _lock:
            v = _minute_delta.setdefault(minute, [0, 0, 0.0, 0.0, 0.0])
            v[0] += 1
            v[1] += 1 if profit > 0 else 0
            v[2] += units
            v[3] += stake
            v[4] += ratio
        ensure_started()
    except Exception as exc:
        logger.debug("FOLLOWER db record skipped: %s", exc)


def queue_op(op: dict) -> None:
    """Ordered follower-trade op (ftrade_open / ftrade_settle) for the batched writer."""
    with _lock:
        _ops.append(op)
    ensure_started()


def queue_stats() -> dict:
    with _lock:
        pending = len(_ops) + len(_minute_delta)
    spilled = 0
    try:
        if os.path.exists(fc.QUEUE_FILE):
            with open(fc.QUEUE_FILE, "rb") as f:
                spilled = sum(1 for _ in f)
    except Exception:
        pass
    return {"pending": pending, "spilled_lines": spilled, "db_ok": _db_ok}


def ensure_started() -> None:
    global _started
    if _started or not enabled():
        return
    with _start_lock:
        if _started:
            return
        _started = True
        threading.Thread(target=_writer_main, name="follower-db-writer", daemon=True).start()


# ── schema + writer thread ───────────────────────────────────────────────────
def init_schema() -> None:
    def fn(conn):
        for stmt in SCHEMA:
            conn.execute(stmt)
    run(fn, retries=3)


def _writer_main() -> None:
    global _inited
    backoff = 5.0
    while not _inited:
        try:
            init_schema()
            import stats_import
            stats_import.maybe_import(sys.modules[__name__])
            _inited = True
            logger.info("FOLLOWER db ready (tables ensured)")
        except Exception as exc:
            logger.warning("FOLLOWER db init failed (%s) - retry in %.0fs", str(exc)[:160], backoff)
            _flush()                                  # spills memory to the local file
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
    while True:
        _wake.wait(fc.DB_FLUSH_SECONDS)
        _wake.clear()
        try:
            _flush()
        except Exception as exc:
            logger.warning("FOLLOWER db flush error: %s", exc)


def _spill(md: Dict[int, List[float]], ops: List[dict]) -> None:
    try:
        with _file_lock:
            if os.path.exists(fc.QUEUE_FILE) and os.path.getsize(fc.QUEUE_FILE) > fc.QUEUE_FILE_MAX_BYTES:
                logger.error("FOLLOWER db queue file is full - dropping this batch")
                return
            with open(fc.QUEUE_FILE, "a") as f:
                f.write(json.dumps({"md": {str(k): v for k, v in md.items()}, "ops": ops},
                                   separators=(",", ":")) + "\n")
        logger.warning("FOLLOWER db unreachable - %d minute rows + %d ops queued locally",
                       len(md), len(ops))
    except Exception as exc:
        logger.error("FOLLOWER db could not spill to local file: %s", exc)


def _apply_op(cur, op: dict) -> None:
    kind = op.get("op")
    if kind == "ftrade_open":
        cur.execute(
            """INSERT INTO follower_trades(follower_id,contract_id,signal_id,symbol,contract_type,barrier,
               stake,payout_ratio,opened_at,source,snapshot_id,window_id)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,to_timestamp(%s),%s,%s,%s)
               ON CONFLICT (follower_id, contract_id) DO NOTHING""",
            (op["follower_id"], op["contract_id"], op.get("signal_id"), op.get("symbol"),
             op.get("contract_type"), op.get("barrier"), op.get("stake"), op.get("payout_ratio"),
             op.get("opened_at"), op.get("source"), op.get("snapshot_id"), op.get("window_id")))
    elif kind == "ftrade_settle":
        cur.execute("UPDATE follower_trades SET profit=%s WHERE follower_id=%s AND contract_id=%s",
                    (op["profit"], op["follower_id"], op["contract_id"]))
    elif kind == "fstate":
        cur.execute(
            """UPDATE followers SET p_base=%s, peak=%s, floor_amt=%s, tier=%s, last_reset_date=%s,
               milestone_hit=%s, last_balance=%s, last_window_result=%s, state_ts=to_timestamp(%s)
               WHERE id=%s AND (state_ts IS NULL OR state_ts <= to_timestamp(%s))""",
            (op.get("p"), op.get("peak"), op.get("floor"), op.get("tier"), op.get("last_reset_date"),
             bool(op.get("milestone_hit")), op.get("last_balance"), op.get("last_window_result"),
             op["ts"], op["follower_id"], op["ts"]))


def _apply(conn, md: Dict[int, List[float]], ops: List[dict]) -> None:
    with conn.cursor() as cur:
        if md:
            cur.executemany(UPSERT_MIN, [(int(m), int(v[0]), int(v[1]), float(v[2]), float(v[3]), float(v[4]))
                                         for m, v in md.items()])
        for op in ops:
            _apply_op(cur, op)


def _replay() -> None:
    """Apply the local JSONL queue (oldest first). Raises if the DB is still down."""
    with _file_lock:
        if not os.path.exists(fc.QUEUE_FILE):
            return
        items: List[Tuple[Dict[int, List[float]], List[dict]]] = []
        with open(fc.QUEUE_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    items.append(({int(k): v for k, v in (d.get("md") or {}).items()}, d.get("ops") or []))
                except Exception:
                    logger.warning("FOLLOWER db queue: skipped a corrupt line")
        if items:
            def fn(conn):
                for md, ops in items:
                    _apply(conn, md, ops)
            run(fn, retries=1)
            logger.info("FOLLOWER db queue replayed (%d batches)", len(items))
        os.remove(fc.QUEUE_FILE)


def _flush() -> bool:
    with _flush_lock:
        with _lock:
            md = dict(_minute_delta)
            _minute_delta.clear()
            ops = list(_ops)
            _ops.clear()
        if not _inited:
            if md or ops:
                _spill(md, ops)
            return False
        try:
            _replay()
            if md or ops:
                run(lambda c: _apply(c, md, ops), retries=1)
            return True
        except Exception as exc:
            if md or ops:
                _spill(md, ops)
            logger.warning("FOLLOWER db flush failed: %s", str(exc)[:160])
            return False


def flush_now(timeout: float = 8.0) -> bool:
    """Forced flush (shutdown / before a snapshot). Bounded wait."""
    if not enabled():
        return False
    result: List[bool] = []
    t = threading.Thread(target=lambda: result.append(_flush()), daemon=True)
    t.start()
    t.join(timeout)
    return bool(result and result[0])


# ── reads / writes used by the follower runtime and the web page ─────────────
def load_minute_rows() -> List[List[float]]:
    """60 rows: [trades, wins, stake_sum, payout_ratio_sum, profit_stake_units]."""
    def fn(conn):
        return conn.execute("SELECT minute,trades,wins,stake_sum,payout_ratio_sum,profit_stake_units "
                            "FROM minute_stats").fetchall()
    rows = [[0, 0, 0.0, 0.0, 0.0] for _ in range(60)]
    for m, t, w, ss, rs, psu in run(fn):
        if 0 <= int(m) < 60:
            rows[int(m)] = [int(t), int(w), float(ss), float(rs), float(psu)]
    return rows


def import_rows_once(rows: List[List[float]], flag_key: str) -> bool:
    """One-time import: only if minute_stats is empty and the flag is not set."""
    def fn(conn):
        flag = conn.execute("SELECT value FROM meta WHERE key=%s", (flag_key,)).fetchone()
        have = conn.execute("SELECT COALESCE(SUM(trades),0) FROM minute_stats").fetchone()[0]
        if flag or have > 0:
            return False
        with conn.cursor() as cur:
            cur.executemany(UPSERT_MIN, [(i, int(r[0]), int(r[1]), float(r[2]), float(r[3]), float(r[4]))
                                         for i, r in enumerate(rows)])
            cur.execute("INSERT INTO meta(key,value) VALUES (%s,'1') ON CONFLICT (key) DO NOTHING", (flag_key,))
        return True
    return bool(run(fn))


def meta_get(key: str) -> Optional[str]:
    r = run(lambda c: c.execute("SELECT value FROM meta WHERE key=%s", (key,)).fetchone())
    return r[0] if r else None


def meta_set(key: str, value: str) -> None:
    run(lambda c: c.execute("INSERT INTO meta(key,value) VALUES (%s,%s) ON CONFLICT (key) DO UPDATE "
                            "SET value=EXCLUDED.value", (key, value)))


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    r = run(lambda c: c.execute("SELECT value FROM settings WHERE key=%s", (key,)).fetchone())
    return r[0] if r else default


def set_setting(key: str, value: str) -> None:
    run(lambda c: c.execute("INSERT INTO settings(key,value) VALUES (%s,%s) ON CONFLICT (key) DO UPDATE "
                            "SET value=EXCLUDED.value", (key, value)))


def get_mode() -> str:
    m = (get_setting("mode", "demo") or "demo").lower()
    return m if m in ("demo", "real") else "demo"


def save_snapshot(window_id: str, zones: dict, minute_rows: List[List[float]], source: str) -> Optional[dict]:
    """Save (or fetch the already saved) snapshot for window_id. Returns {id, created_ts}."""
    def fn(conn):
        r = conn.execute(
            "INSERT INTO zone_snapshots(window_id,zones,minute_stats,source) VALUES (%s,%s,%s,%s) "
            "ON CONFLICT (window_id) DO NOTHING RETURNING id, EXTRACT(EPOCH FROM created_at)",
            (window_id, json.dumps(zones, separators=(",", ":")),
             json.dumps(minute_rows, separators=(",", ":")), source)).fetchone()
        if r is None:
            r = conn.execute("SELECT id, EXTRACT(EPOCH FROM created_at) FROM zone_snapshots WHERE window_id=%s",
                             (window_id,)).fetchone()
        return {"id": int(r[0]), "created_ts": float(r[1])} if r else None
    return run(fn)


def get_snapshot(window_id: str) -> Optional[dict]:
    r = run(lambda c: c.execute(
        "SELECT id, EXTRACT(EPOCH FROM created_at), zones, source FROM zone_snapshots WHERE window_id=%s",
        (window_id,)).fetchone())
    if not r:
        return None
    return {"id": int(r[0]), "created_ts": float(r[1]), "zones": json.loads(r[2]), "source": r[3]}


def latest_snapshot() -> Optional[dict]:
    r = run(lambda c: c.execute(
        "SELECT id, EXTRACT(EPOCH FROM created_at), zones, source, window_id FROM zone_snapshots "
        "ORDER BY id DESC LIMIT 1").fetchone())
    if not r:
        return None
    return {"id": int(r[0]), "created_ts": float(r[1]), "zones": json.loads(r[2]),
            "source": r[3], "window_id": r[4]}


_FCOLS = ("id,label,token_last4,app_id,enabled,p_base,peak,floor_amt,tier,last_reset_date,milestone_hit,"
          "last_balance,last_window_result")


def list_followers(with_token: bool = False, enabled_only: bool = False) -> List[dict]:
    cols = _FCOLS + (",token_enc" if with_token else "")
    q = f"SELECT {cols} FROM followers" + (" WHERE enabled" if enabled_only else "") + " ORDER BY id"
    names = [c.strip() for c in cols.split(",")]
    rows = run(lambda c: c.execute(q).fetchall())
    return [dict(zip(names, r)) for r in rows]


def count_followers() -> int:
    return int(run(lambda c: c.execute("SELECT COUNT(*) FROM followers").fetchone())[0])


def add_follower(label: str, token_enc: str, last4: str, app_id: str, enabled: bool) -> int:
    return int(run(lambda c: c.execute(
        "INSERT INTO followers(label,token_enc,token_last4,app_id,enabled) VALUES (%s,%s,%s,%s,%s) RETURNING id",
        (label, token_enc, last4, app_id, enabled)).fetchone())[0])


def update_follower(fid: int, label: str, app_id: str, enabled: bool,
                    token_enc: Optional[str] = None, last4: Optional[str] = None) -> None:
    def fn(conn):
        if token_enc:
            conn.execute("UPDATE followers SET label=%s,app_id=%s,enabled=%s,token_enc=%s,token_last4=%s,"
                         "updated_at=now() WHERE id=%s", (label, app_id, enabled, token_enc, last4, fid))
        else:
            conn.execute("UPDATE followers SET label=%s,app_id=%s,enabled=%s,updated_at=now() WHERE id=%s",
                         (label, app_id, enabled, fid))
    run(fn)


def delete_follower(fid: int) -> None:
    run(lambda c: c.execute("DELETE FROM followers WHERE id=%s", (fid,)))


def save_follower_state(fid: int, state: dict) -> bool:
    """Persist P/peak/floor/tier immediately. Falls back to the local queue if the DB is down
    (the op carries a timestamp so a late replay can never overwrite newer state)."""
    op = {"op": "fstate", "follower_id": fid, "ts": time.time(), **state}
    try:
        run(lambda c: _apply(c, {}, [op]), retries=2)
        return True
    except Exception as exc:
        logger.warning("FOLLOWER state save failed (%s) - queued locally", str(exc)[:120])
        _spill({}, [op])
        return False


def verdict_rows() -> List[Tuple[str, float, float]]:
    """One row per collector signal (not per follower): (source, profit/stake, payout_ratio).
    Only trades placed after a snapshot (snapshot_id set) and already settled."""
    def fn(conn):
        return conn.execute(
            """SELECT DISTINCT ON (signal_id) source, profit/NULLIF(stake,0), payout_ratio
               FROM follower_trades
               WHERE snapshot_id IS NOT NULL AND profit IS NOT NULL AND signal_id IS NOT NULL
                 AND payout_ratio IS NOT NULL AND stake > 0
               ORDER BY signal_id, opened_at""").fetchall()
    return [(r[0] or "?", float(r[1]), float(r[2])) for r in run(fn)]


def follower_pnl() -> List[dict]:
    def fn(conn):
        return conn.execute(
            """SELECT follower_id, COUNT(*), COALESCE(SUM(CASE WHEN profit>0 THEN 1 ELSE 0 END),0),
                      COALESCE(SUM(profit),0), COALESCE(SUM(stake),0)
               FROM follower_trades WHERE profit IS NOT NULL GROUP BY follower_id ORDER BY follower_id""").fetchall()
    return [{"follower_id": r[0], "trades": int(r[1]), "wins": int(r[2]), "pnl": float(r[3]),
             "staked": float(r[4])} for r in run(fn)]
