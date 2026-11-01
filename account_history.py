"""
account_history.py - 24 h balance line + 10-minute net-P&L buckets, rebuilt from Deriv's own statement
(server-side records) at every boot. Nothing is written anywhere: in-memory only.

If the statement call fails or has an unexpected shape the module degrades to the in-process series
(live balance updates / settled trades seen since this boot).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BUCKET = 600                 # seconds
DAY = 86400
MAX_POINTS = 300
ROW_CAP = 6000               # statement rows fetched at most (speed)
PAGE = 999                  # Deriv: limit must be 0..999
_lock = threading.Lock()
_srv_bal: List[Tuple[float, float]] = []      # from statement (oldest first)
_srv_buckets: Dict[int, float] = {}
_live_bal: List[Tuple[float, float]] = []     # since last refresh / since boot
_live_buckets: Dict[int, float] = {}          # provisional, cleared by every successful refresh
_state = {"ok": False, "truncated": False, "refreshed": 0.0, "logged_shape": False, "rows": 0}


# ── fetch ────────────────────────────────────────────────────────────────────
def _ts(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f > 1e9 else None
    except (TypeError, ValueError):
        return None


async def fetch_24h(client) -> bool:
    """Page Deriv's statement for the last 24 h; replace the server series. Returns True on success."""
    now = int(time.time())
    rows: List[dict] = []
    offset = 0
    truncated = False
    try:
        while True:
            resp = await client._send({"statement": 1, "description": 1, "limit": PAGE, "offset": offset,
                                       "date_from": now - DAY, "date_to": now}, timeout=30)
            st = resp.get("statement") if isinstance(resp, dict) else None
            tx = st.get("transactions") if isinstance(st, dict) else None
            if not isinstance(tx, list):
                if not _state["logged_shape"]:
                    _state["logged_shape"] = True
                    logger.warning("ACCOUNT HISTORY: unexpected statement shape, keys=%s statement keys=%s",
                                   sorted(resp.keys()) if isinstance(resp, dict) else type(resp),
                                   sorted(st.keys()) if isinstance(st, dict) else type(st))
                return False
            rows += tx
            if len(tx) < PAGE:
                break
            offset += PAGE
            if len(rows) >= ROW_CAP:
                truncated = True
                break
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("ACCOUNT HISTORY: statement request failed: %s", exc)
        return False
    if rows and not _state["logged_shape"]:
        _state["logged_shape"] = True
        logger.info("ACCOUNT HISTORY: %d statement rows, row keys=%s", len(rows), sorted(rows[0].keys()))
    bal: List[Tuple[float, float]] = []
    buckets: Dict[int, float] = {}
    for r in rows:
        t = _ts(r.get("transaction_time"))
        if t is None or t < now - DAY:
            continue
        b = r.get("balance_after")
        try:
            if b is not None:
                bal.append((t, float(b)))
        except (TypeError, ValueError):
            pass
        if str(r.get("action_type", "")).lower() in ("buy", "sell"):          # not deposits / resets
            try:
                k = int(t // BUCKET) * BUCKET
                buckets[k] = buckets.get(k, 0.0) + float(r.get("amount") or 0.0)
            except (TypeError, ValueError):
                pass
    bal.sort()
    with _lock:
        _srv_bal[:] = bal
        _srv_buckets.clear()
        _srv_buckets.update(buckets)
        last = bal[-1][0] if bal else 0.0
        _live_bal[:] = [p for p in _live_bal if p[0] > last]
        _live_buckets.clear()
        _state.update(ok=True, truncated=truncated, refreshed=time.time(), rows=len(rows))
    return True


async def run(client, period: float = 60.0) -> None:
    """Background task: fetch as soon as the client is connected, then refresh every `period` seconds.

    It only talks to Deriv while client.is_connected is True: between sessions the bot is disconnected on purpose,
    so the task just idles (short sleeps, no requests, no spinning) and fetches again right after the next connect.
    Cancel-safe: the engine cancels it with its other Deriv tasks at disconnect."""
    while True:
        wait = min(period, 5.0)                          # not connected: look again soon
        try:
            if getattr(client, "is_connected", False):
                ok = await fetch_24h(client)
                wait = period if ok else min(period, 15.0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("ACCOUNT HISTORY refresh error: %s", exc)
            wait = min(period, 15.0)
        await asyncio.sleep(wait)


# ── live appends ─────────────────────────────────────────────────────────────
def note_balance(bal: float, ts: Optional[float] = None) -> None:
    ts = ts or time.time()
    with _lock:
        if _live_bal and ts - _live_bal[-1][0] < 1.0:
            _live_bal[-1] = (ts, float(bal))
        else:
            _live_bal.append((ts, float(bal)))
        cut = ts - DAY
        while _live_bal and _live_bal[0][0] < cut:
            _live_bal.pop(0)


def note_pnl(profit: float, ts: Optional[float] = None) -> None:
    ts = ts or time.time()
    k = int(ts // BUCKET) * BUCKET
    with _lock:
        _live_buckets[k] = _live_buckets.get(k, 0.0) + float(profit)


# ── series ───────────────────────────────────────────────────────────────────
def _downsample(pts: List[Tuple[float, float]], n: int = MAX_POINTS) -> List[Tuple[float, float]]:
    if len(pts) <= n:
        return pts
    step = (len(pts) - 1) / (n - 1)
    return [pts[int(round(i * step))] for i in range(n)]


def get_series() -> dict:
    now = time.time()
    with _lock:
        bal = sorted(_srv_bal + _live_bal)
        bk: Dict[int, float] = dict(_srv_buckets)
        for k, v in _live_buckets.items():
            bk[k] = bk.get(k, 0.0) + v
        st = dict(_state)
    bal = [p for p in bal if p[0] >= now - DAY]
    first = int((now - DAY) // BUCKET) * BUCKET
    last = int(now // BUCKET) * BUCKET
    buckets = [(k, round(bk.get(k, 0.0), 4)) for k in range(first, last + BUCKET, BUCKET)]
    return {"balance": [(round(t, 1), round(b, 4)) for t, b in _downsample(bal)], "buckets": buckets,
            "source": "account statement" if st["ok"] else "in-process",
            "truncated": st["truncated"], "refreshed": st["refreshed"], "rows": st["rows"]}


# ── SVG (same look as the existing dashboard charts) ─────────────────────────
_FONT = 'font-family="Segoe UI,sans-serif"'


def _hhmm(ts: float) -> str:
    return time.strftime("%H:%M", time.gmtime(ts))


def _empty(w: int, h: int, msg: str) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:{h}px;">'
            f'<text x="{w // 2}" y="{h // 2}" text-anchor="middle" fill="#484f58" font-size="13" {_FONT}>{msg}</text></svg>')


def balance_svg() -> Optional[str]:
    ser = get_series()
    pts = ser["balance"]
    if len(pts) < 2:
        return None
    W, H, PL, PR, PT, PB = 900, 180, 62, 16, 14, 32
    t0, t1 = pts[0][0], pts[-1][0]
    span_t = max(t1 - t0, 1.0)
    bals = [b for _, b in pts]
    lo, hi = min(bals), max(bals)
    span = (hi - lo) or 1.0
    pw, ph = W - PL - PR, H - PT - PB
    px = lambda t: PL + (t - t0) / span_t * pw
    py = lambda b: PT + ph - (b - lo) / span * ph
    line = " ".join(f"{px(t):.1f},{py(b):.1f}" for t, b in pts)
    grid = ""
    for f in (0, 0.5, 1.0):
        y = PT + ph - f * ph
        grid += (f'<line x1="{PL}" y1="{y:.1f}" x2="{W - PR}" y2="{y:.1f}" stroke="#21262d"/>'
                 f'<text x="{PL - 4}" y="{y + 4:.1f}" text-anchor="end" fill="#8b949e" font-size="10" {_FONT}>${lo + f * span:.2f}</text>')
    xl = ""
    for i in range(0, 7):
        t = t0 + span_t * i / 6
        xl += f'<text x="{px(t):.1f}" y="{H - 4}" text-anchor="middle" fill="#8b949e" font-size="10" {_FONT}>{_hhmm(t)}</text>'
    note = "" if ser["source"] == "account statement" else \
        f'<text x="{W - PR}" y="{PT + 10}" text-anchor="end" fill="#d29922" font-size="10" {_FONT}>in-process series</text>'
    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:{H}px;">'
            f'{grid}<polyline points="{line}" fill="none" stroke="#58a6ff" stroke-width="2" stroke-linejoin="round"/>{xl}{note}</svg>')


def bucket_svg() -> Optional[str]:
    ser = get_series()
    bars = ser["buckets"]
    if not any(v != 0 for _, v in bars):
        return None
    W, H, PL, PR, PT, PB = 900, 160, 62, 16, 14, 32
    pw, ph = W - PL - PR, H - PT - PB
    mx = max(abs(v) for _, v in bars) or 1.0
    mid = PT + ph / 2
    bw = max(1.0, pw / len(bars) - 1)
    out = f'<line x1="{PL}" y1="{mid:.1f}" x2="{W - PR}" y2="{mid:.1f}" stroke="#30363d"/>'
    for i, (k, v) in enumerate(bars):
        x = PL + i * pw / len(bars)
        fr = max(abs(v) / mx * ph / 2, 1.0 if v else 0.0)
        y = mid - fr if v >= 0 else mid
        out += (f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{fr:.1f}" fill="{"#3fb950" if v >= 0 else "#f85149"}">'
                f'<title>{_hhmm(k)} UTC  {v:+.4f}</title></rect>')
    for i in range(0, len(bars), max(1, len(bars) // 6)):
        out += (f'<text x="{PL + i * pw / len(bars):.1f}" y="{H - 4}" text-anchor="middle" fill="#8b949e" font-size="10" {_FONT}>'
                f'{_hhmm(bars[i][0])}</text>')
    for tk in (-1, 0, 1):
        out += (f'<text x="{PL - 4}" y="{mid - tk * ph / 2 + 4:.1f}" text-anchor="end" fill="#8b949e" font-size="10" {_FONT}>'
                f'${tk * mx:+.2f}</text>')
    return f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:{H}px;">{out}</svg>'
