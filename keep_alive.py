"""
keep_alive.py – Flask web server + self-ping thread + HTML dashboard.

Routes:
  /          → Live HTML dashboard (auto-refreshes every 10 s)
  /health    → Plain JSON health check
  /stats     → Detailed JSON stats
  /trades    → Recent trade list as JSON
  /symbols   → Symbol leaderboard as JSON
  /debug     → Raw _state dump as JSON

v11 changes:
  - Dashboard fully rebuilt: all 10 sections from spec.
  - P&L Summary row: Daily/Weekly/Monthly/All-time + 15% loss limit bar.
  - Performance Stats: win rate, total/wins/losses, profit factor, avg PnL,
    best/worst trade, current streak.
  - Balance curve SVG — all trades, green=win, red=loss, yellow=timeout,
    start-balance reference line.
  - Hourly P&L bar chart SVG — one bar per hour of today.
  - Recent trades table — last 100, columns: Time UTC, Symbol, Direction,
    Stake, PnL, Balance After, Result, Strategy.
  - Open contracts — symbol, direction, stake, seconds open.
  - Suspended symbols — symbol + minutes remaining.
  - Symbol leaderboard — trades, win%, total PnL, best, worst.
  - Signal log — last 20 signals with symbol, direction, score, strategy, ts.
  - record_failure() added for failed placements.
  - Dashboard push interval exposed as DASHBOARD_PUSH_EVERY (used by bot_engine).
"""

import os
import threading
import time
import datetime
import logging
import requests
from flask import Flask, jsonify, Response, request
import config

try:
    from strategy_stats import stats as _strategy_stats
    _STRATEGY_STATS_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import guard
    _strategy_stats = None
    _STRATEGY_STATS_AVAILABLE = False

logger = logging.getLogger(__name__)
app = Flask(__name__)

_state: dict = {
    "running":               False,
    "balance":               0.0,
    "day_start_balance":     0.0,
    "daily_pnl":             0.0,
    "daily_pnl_pct":         0.0,
    "win_rate":              0.0,
    "wins":                  0,
    "losses":                0,
    "total_trades":          0,
    "streak":                0,
    "streak_label":          "—",
    "session":               "Starting",
    "tradeable_count":       0,
    "last_signal":           "No signal yet",
    "recent_trades":         [],
    "best_symbols":          [],
    "balance_history":       [],
    "suspended_symbols":     [],
    "paused_for_loss_limit": False,
    "current_symbol":        "—",
    "uptime_seconds":        0,
    "start_time":            time.time(),
    "gross_profit":          0.0,
    "gross_loss":            0.0,
    "profit_factor":         0.0,
    "avg_rr":                0.0,
    "best_trade":            0.0,
    "worst_trade":           0.0,
    "active_trades":         0,
    "all_trades":            [],
    "hourly_pnl":            {},
    "daily_pnl_history":     [],
    "weekly_pnl":            0.0,
    "weekly_pnl_pct":        0.0,
    "monthly_pnl":           0.0,
    "monthly_pnl_pct":       0.0,
    "week_start_balance":    0.0,
    "month_start_balance":   0.0,
    "current_day":           "",
    "current_week":          "",
    "current_month":         "",
    "avg_multiplier":        0.0,
    "multiplier_count":      0,
    "funding_fees_total":    0.0,
    "open_contracts":        [],
    "open_contracts_count":  0,
    # v11 additions
    "signal_log":            [],   # last 20 emitted signals
    "failure_log":           [],   # last 50 failed placements
    "session_start_balance": 0.0,  # set once on first trade or bot start
    # Task 5/6 fix (Aug 2026): "suspended_symbols" now holds per-
    # (indicator, symbol) pair suspensions, sourced correctly from
    # pair_suspension.snapshot() via update_suspended_symbols() below —
    # see that function's docstring for the full root-cause explanation.
    # Whole-symbol buy-failure circuit-breaker suspensions (the only
    # remaining caller of SymbolManager.suspend()) get their own,
    # separately-labeled key so the two are never conflated again.
    "circuit_breaker_symbols": [],
    # Session / schedule status (contract section 8). Written by the engine, the
    # lead code and redeploy_guard through update_status(**kw); read by the
    # dashboard. The free-text "session" key above stays as the human-readable line.
    "session_state":         "IDLE",   # IDLE | CONNECTING | WARMING | LIVE | DRAINING | REDEPLOYING
    "session_name":          "",       # "A" | "B" | "ALL" | ""  (active or upcoming)
    "session_open_ts":       0.0,      # epoch seconds (Deriv-synced), 0 = unknown
    "session_close_ts":      0.0,
    "session_connect_ts":    0.0,
    "session_lead_stage":    "",       # "" | connect | clock sync | adopt | refit | gap resync | quotes | ready
    "session_lead_seconds":  0.0,      # last measured lead length
    "clock_offset":          0.0,
    "clock_error":           0.0,
    "redeploy_state":        "",       # "" | scheduled | fired | skipped | failed
    "redeploy_fire_ts":      0.0,      # planned or actual fire time, epoch seconds
    "redeploy_note":         "",       # short text: why skipped, expected duration
}

_status = _state   # alias for bot_engine imports


# ── Public state helpers ────────────────────────────────────────────────────────

def update_status(**kwargs) -> None:
    _state.update(kwargs)
    _state["uptime_seconds"] = int(time.time() - _state["start_time"])


def get_state(key, default=None):
    """Read access to _state (e.g. redeploy_guard reads session_state)."""
    return _state.get(key, default)


def set_active_trades(count: int) -> None:
    _state["active_trades"] = max(0, int(count))


def get_active_trades() -> int:
    return int(_state.get("active_trades", 0))


# ── Period reset helper ─────────────────────────────────────────────────────────

def _check_period_resets(now: datetime.datetime, balance_after: float) -> None:
    today_str = now.strftime("%Y-%m-%d")
    iso_year, iso_week, _ = now.isocalendar()
    week_str  = f"{iso_year}-W{iso_week:02d}"
    month_str = now.strftime("%Y-%m")

    current_balance = _status.get("balance", balance_after)

    if not _status.get("current_day"):
        _status["current_day"]        = today_str
        _status["day_start_balance"]  = balance_after
    if not _status.get("current_week"):
        _status["current_week"]       = week_str
        _status["week_start_balance"] = balance_after
    if not _status.get("current_month"):
        _status["current_month"]       = month_str
        _status["month_start_balance"] = balance_after
    if not _status.get("session_start_balance"):
        _status["session_start_balance"] = balance_after

    if _status["current_day"] != today_str:
        prev_day_start = _status.get("day_start_balance", current_balance)
        _status["daily_pnl_history"].append({
            "date":    _status["current_day"],
            "pnl":     round(current_balance - prev_day_start, 4),
            "balance": round(current_balance, 4),
        })
        _status["daily_pnl_history"] = _status["daily_pnl_history"][-90:]
        _status["current_day"]       = today_str
        _status["day_start_balance"] = current_balance
        _status["wins"]              = 0
        _status["losses"]            = 0
        _status["total_trades"]      = 0
        _status["win_rate"]          = 0.0

    if _status["current_week"] != week_str:
        _status["current_week"]       = week_str
        _status["week_start_balance"] = current_balance

    if _status["current_month"] != month_str:
        _status["current_month"]       = month_str
        _status["month_start_balance"] = current_balance


# ── Trade / signal recording ────────────────────────────────────────────────────

def record_trade(symbol, direction, stake, pnl,
                 balance_after, won, strategy="",
                 multiplier=None, close_reason="normal", **_):
    from datetime import datetime, timezone
    now       = datetime.now(timezone.utc)
    is_timeout = (close_reason == "timeout")

    trade = {
        "time":         now.strftime("%H:%M:%S"),
        "date":         now.strftime("%Y-%m-%d"),
        "symbol":       symbol,
        "direction":    direction,
        "stake":        round(float(stake),         4),
        "pnl":          round(float(pnl),           4),
        "balance_after":round(float(balance_after), 4),
        "won":          bool(won),
        "strategy":     strategy,
        "multiplier":   multiplier,
        "close_reason": close_reason,
    }

    _check_period_resets(now, balance_after)

    _status["recent_trades"].insert(0, trade)
    _status["recent_trades"] = _status["recent_trades"][:200]
    _status["all_trades"].append(trade)

    _status["balance_history"].append({
        "time":         trade["time"],
        "balance":      balance_after,
        "won":          bool(won),
        "close_reason": close_reason,
    })
    _status["balance_history"] = _status["balance_history"][-500:]

    hour_key = now.strftime("%Y-%m-%d %H:00")
    _status["hourly_pnl"][hour_key] = round(
        _status["hourly_pnl"].get(hour_key, 0.0) + float(pnl), 4)

    if won:
        _status["wins"]   += 1
    else:
        _status["losses"] += 1
    _status["total_trades"] = _status["wins"] + _status["losses"]

    if _status["total_trades"] > 0:
        _status["win_rate"] = round(
            _status["wins"] / _status["total_trades"] * 100, 1)

    day_start = _status.get("day_start_balance") or balance_after
    _status["daily_pnl"]     = round(float(balance_after) - float(day_start), 4)
    _status["daily_pnl_pct"] = round(
        (_status["daily_pnl"] / float(day_start)) * 100, 2) if day_start else 0.0

    week_start = _status.get("week_start_balance") or balance_after
    _status["weekly_pnl"]     = round(float(balance_after) - float(week_start), 4)
    _status["weekly_pnl_pct"] = round(
        (_status["weekly_pnl"] / float(week_start)) * 100, 2) if week_start else 0.0

    month_start = _status.get("month_start_balance") or balance_after
    _status["monthly_pnl"]     = round(float(balance_after) - float(month_start), 4)
    _status["monthly_pnl_pct"] = round(
        (_status["monthly_pnl"] / float(month_start)) * 100, 2) if month_start else 0.0

    if multiplier is not None:
        try:
            mult = float(multiplier)
            n    = int(_status.get("multiplier_count", 0))
            avg  = float(_status.get("avg_multiplier", 0.0))
            _status["avg_multiplier"]    = round((avg * n + mult) / (n + 1), 2)
            _status["multiplier_count"]  = n + 1
        except (TypeError, ValueError):
            pass

    pnl_f = float(pnl)
    if pnl_f > float(_status.get("best_trade", 0.0)):
        _status["best_trade"]  = round(pnl_f, 4)
    if pnl_f < float(_status.get("worst_trade", 0.0)):
        _status["worst_trade"] = round(pnl_f, 4)

    # gross profit / loss → profit factor
    if pnl_f > 0:
        _status["gross_profit"] = round(
            float(_status.get("gross_profit", 0.0)) + pnl_f, 4)
    else:
        _status["gross_loss"] = round(
            float(_status.get("gross_loss", 0.0)) + abs(pnl_f), 4)
    gp = float(_status.get("gross_profit", 0.0))
    gl = float(_status.get("gross_loss",   0.0))
    _status["profit_factor"] = round(gp / gl, 3) if gl > 0 else (gp if gp > 0 else 0.0)

    if is_timeout and pnl_f < 0:
        _status["funding_fees_total"] = round(
            float(_status.get("funding_fees_total", 0.0)) + abs(pnl_f), 4)

    _status["balance"] = float(balance_after)


def record_signal(symbol, direction, strategy, score, timestamp=None):
    from datetime import datetime, timezone
    ts = timestamp or datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = {
        "symbol":    symbol,
        "direction": direction,
        "strategy":  strategy,
        "score":     round(float(score), 3),
        "ts":        ts,
    }
    _status["last_signal"] = entry
    _status["signal_log"].insert(0, entry)
    _status["signal_log"] = _status["signal_log"][:20]


def record_failure(symbol, direction, stake, strategy, reason=""):
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    entry = {
        "ts":       ts,
        "symbol":   symbol,
        "direction":direction,
        "stake":    round(float(stake), 4),
        "strategy": strategy,
        "reason":   reason,
    }
    _status["failure_log"].insert(0, entry)
    _status["failure_log"] = _status["failure_log"][:50]
    logger.warning(
        f"PLACEMENT FAILED: {symbol} {direction} stake=${stake:.2f} "
        f"strategy={strategy} reason={reason}")


def update_open_contracts(contracts: list) -> None:
    normalized = []
    for c in contracts or []:
        if isinstance(c, dict):
            normalized.append(c)
        else:
            normalized.append({"symbol": str(c)})
    _status["open_contracts"]       = normalized
    _status["open_contracts_count"] = len(normalized)


def update_suspended_symbols(suspended: list) -> None:
    """
    Task 5 fix: this was previously an orphaned function — nothing called
    it, and _push_dashboard() in bot_engine.py wrote _state["suspended_
    symbols"] itself via update_status(), sourced from SymbolManager's
    whole-symbol suspension state (the wrong data — see bot_engine.py's
    _push_dashboard() for the full root-cause explanation). Repurposed as
    the ONE function that owns this key: bot_engine.py's _push_dashboard()
    now calls this with data built from pair_suspension.snapshot() (a list
    of {"pair": "<strategy>/<symbol>", "seconds_remaining": <float>}
    dicts) — the correct, per-(indicator, symbol) suspension source.
    """
    _state["suspended_symbols"] = list(suspended)


# ── Time helpers ────────────────────────────────────────────────────────────────

def _exit_time_to_dt(raw) -> datetime.datetime | None:
    if raw is None:
        return None
    try:
        f = float(raw)
        if f > 1_000_000_000:
            return datetime.datetime.utcfromtimestamp(f)
    except (TypeError, ValueError):
        pass
    s = str(raw).replace("T", " ")[:19]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _parse_hms(raw) -> str:
    if raw is None:
        return "?"
    s = str(raw).strip()
    if len(s) >= 5 and s[2] == ":":
        return s[:5]
    try:
        f = float(s)
        if f > 1_000_000_000:
            return datetime.datetime.utcfromtimestamp(f).strftime("%H:%M")
    except (TypeError, ValueError):
        pass
    dt = _exit_time_to_dt(raw)
    if dt:
        return dt.strftime("%H:%M")
    return "?"


# ── SVG Balance Curve ───────────────────────────────────────────────────────────

def _build_svg_chart(balance_history: list, start_balance: float | None = None) -> str:
    W, H             = 900, 180
    PAD_L, PAD_R     = 62, 16
    PAD_T, PAD_B     = 14, 32

    points = []
    for entry in balance_history:
        try:
            bal  = float(entry.get("balance") or entry.get("balance_after") or 0)
            if bal == 0:
                continue
            won          = bool(entry.get("won", False))
            close_reason = str(entry.get("close_reason", "normal"))
            label        = _parse_hms(entry.get("time") or entry.get("exit_time"))
            points.append({"t": label, "bal": bal, "won": won, "cr": close_reason})
        except Exception:
            continue

    if not points:
        return (
            f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;height:{H}px;">'
            f'<text x="{W // 2}" y="{H // 2}" text-anchor="middle" '
            f'fill="#484f58" font-size="13" font-family="Segoe UI,sans-serif">'
            f'No trades yet — chart populates after first closed trade'
            f'</text></svg>'
        )

    bals       = [p["bal"] for p in points]
    range_vals = list(bals) + ([float(start_balance)] if start_balance else [])
    min_b      = min(range_vals)
    max_b      = max(range_vals)
    span       = max_b - min_b if max_b != min_b else 1.0
    plot_w     = W - PAD_L - PAD_R
    plot_h     = H - PAD_T - PAD_B
    n          = len(points)

    def px(i):   return PAD_L + (i / max(n - 1, 1)) * plot_w
    def py(b):   return PAD_T + plot_h - ((float(b) - min_b) / span) * plot_h

    coords = " ".join(f"{px(i):.1f},{py(p['bal']):.1f}" for i, p in enumerate(points))
    line   = (
        f'<polyline points="{coords}" fill="none" stroke="#58a6ff" '
        f'stroke-width="2" stroke-linejoin="round"/>'
    )

    dots = ""
    for i, p in enumerate(points):
        fill = "#d29922" if p["cr"] == "timeout" else ("#3fb950" if p["won"] else "#f85149")
        dots += (
            f'<circle cx="{px(i):.1f}" cy="{py(p["bal"]):.1f}" r="4" '
            f'fill="{fill}" stroke="#0d1117" stroke-width="1.5">'
            f'<title>{p["t"]}  ${p["bal"]:.4f}</title></circle>'
        )

    y_labels = ""
    for tick in [0, 0.5, 1.0]:
        val = min_b + tick * span
        y   = PAD_T + plot_h - tick * plot_h
        y_labels += (
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}" '
            f'stroke="#21262d" stroke-width="1"/>'
            f'<text x="{PAD_L - 4}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#8b949e" font-size="10" font-family="Segoe UI,sans-serif">'
            f'${val:.2f}</text>'
        )

    x_labels = ""
    step = max(1, n // 8)
    for i in range(0, n, step):
        x = px(i)
        x_labels += (
            f'<text x="{x:.1f}" y="{H - 4}" text-anchor="middle" '
            f'fill="#8b949e" font-size="10" font-family="Segoe UI,sans-serif">'
            f'{points[i]["t"]}</text>'
        )

    ref_line = ""
    if start_balance:
        y_ref = py(float(start_balance))
        ref_line = (
            f'<line x1="{PAD_L}" y1="{y_ref:.1f}" x2="{W - PAD_R}" y2="{y_ref:.1f}" '
            f'stroke="#8b949e" stroke-width="1" stroke-dasharray="4,3"/>'
            f'<text x="{W - PAD_R}" y="{y_ref - 4:.1f}" text-anchor="end" '
            f'fill="#8b949e" font-size="10" font-family="Segoe UI,sans-serif">'
            f'Start: ${float(start_balance):.2f}</text>'
        )

    return (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:{H}px;">'
        f'{y_labels}{ref_line}{line}{dots}{x_labels}'
        f'</svg>'
    )


# ── SVG Hourly P&L Bar Chart ────────────────────────────────────────────────────

def _build_hourly_svg(hourly_pnl: dict) -> str:
    W, H             = 900, 160
    PAD_L, PAD_R     = 62, 16
    PAD_T, PAD_B     = 14, 32

    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    bars  = []
    for h in range(24):
        key = f"{today} {h:02d}:00"
        val = float(hourly_pnl.get(key, 0.0))
        bars.append({"h": h, "pnl": val})

    non_zero = [b for b in bars if b["pnl"] != 0]
    if not non_zero:
        return (
            f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;height:{H}px;">'
            f'<text x="{W // 2}" y="{H // 2}" text-anchor="middle" '
            f'fill="#484f58" font-size="13" font-family="Segoe UI,sans-serif">'
            f'No hourly data yet'
            f'</text></svg>'
        )

    vals     = [b["pnl"] for b in bars]
    max_abs  = max(abs(v) for v in vals) or 1.0
    plot_w   = W - PAD_L - PAD_R
    plot_h   = H - PAD_T - PAD_B
    bar_w    = max(2, plot_w / 24 - 2)
    mid_y    = PAD_T + plot_h / 2

    # zero line
    zero_line = (
        f'<line x1="{PAD_L}" y1="{mid_y:.1f}" x2="{W - PAD_R}" y2="{mid_y:.1f}" '
        f'stroke="#30363d" stroke-width="1"/>'
    )

    rects = ""
    for b in bars:
        pnl  = b["pnl"]
        x    = PAD_L + b["h"] * (plot_w / 24) + 1
        frac = abs(pnl) / max_abs * (plot_h / 2)
        fill = "#3fb950" if pnl >= 0 else "#f85149"
        if pnl >= 0:
            rect_y = mid_y - frac
            rect_h = frac
        else:
            rect_y = mid_y
            rect_h = frac
        if rect_h < 1:
            rect_h = 1
        rects += (
            f'<rect x="{x:.1f}" y="{rect_y:.1f}" width="{bar_w:.1f}" height="{rect_h:.1f}" '
            f'fill="{fill}" rx="1">'
            f'<title>{b["h"]:02d}:00  {pnl:+.4f}</title></rect>'
        )

    # x labels every 4 hours
    x_labels = ""
    for h in range(0, 24, 4):
        x = PAD_L + h * (plot_w / 24) + bar_w / 2
        x_labels += (
            f'<text x="{x:.1f}" y="{H - 4}" text-anchor="middle" '
            f'fill="#8b949e" font-size="10" font-family="Segoe UI,sans-serif">'
            f'{h:02d}h</text>'
        )

    # y labels
    y_labels = ""
    for tick in [-1, 0, 1]:
        val = tick * max_abs
        y   = mid_y - tick * (plot_h / 2)
        y_labels += (
            f'<text x="{PAD_L - 4}" y="{y + 4:.1f}" text-anchor="end" '
            f'fill="#8b949e" font-size="10" font-family="Segoe UI,sans-serif">'
            f'${val:+.2f}</text>'
        )

    return (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;height:{H}px;">'
        f'{y_labels}{zero_line}{rects}{x_labels}'
        f'</svg>'
    )


# ── Dashboard renderer ──────────────────────────────────────────────────────────

# ── Session / schedule status panel (contract section 8) ────────────────────────
# Server-rendered (works with JavaScript off); a tiny inline script ticks the
# countdowns between the 10 s page refreshes. All timestamps are Deriv-synced
# epoch seconds, 0 = unknown. "Now" for the countdowns is the bot host's
# time.time() + clock_offset at render time, then advanced with the browser's
# monotonic clock, so the viewer's own (possibly wrong) clock never matters.

import html as _html
import math as _math

_SESSION_STATES = ("IDLE", "CONNECTING", "WARMING", "LIVE", "DRAINING", "REDEPLOYING")

# which countdown is meaningful in which state
_CD_WHEN = {
    "connect": ("IDLE", "REDEPLOYING"),
    "open":    ("IDLE", "CONNECTING", "WARMING", "REDEPLOYING"),
    "close":   ("LIVE",),
}

_SESSION_CSS = """
  .sx-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;margin-bottom:16px;}
  .sx-head{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:10px;}
  .sx-badge{display:inline-block;padding:4px 10px;border-radius:6px;font-size:13px;font-weight:700;letter-spacing:.03em;}
  .sx-idle{background:#21262d;color:var(--muted);}
  .sx-warn{background:#3a2a1a;color:var(--yellow);}
  .sx-info{background:#1a2a3a;color:var(--accent);}
  .sx-live{background:#1a3a1f;color:var(--green);}
  .sx-redeploy{background:#2a1a3a;color:#c084fc;}
  .sx-sub{font-size:12px;color:var(--muted);}
  .sx-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;}
  .sx-item{background:#0d1117;border:1px solid var(--border);border-radius:6px;padding:8px 10px;}
  .sx-wide{grid-column:1/-1;}
  .sx-val{font-size:16px;font-weight:700;font-variant-numeric:tabular-nums;}
  .sx-dim{color:var(--muted);font-weight:400;}
  .sx-rd-scheduled{color:var(--accent);} .sx-rd-fired{color:#c084fc;}
  .sx-rd-skipped{color:var(--yellow);}   .sx-rd-failed{color:var(--red);}
"""

_SESSION_JS = """
(function () {
  var p = document.getElementById('sess-panel');
  if (!p || !window.performance || !performance.now) return;
  var base = parseFloat(p.getAttribute('data-now'));
  if (!isFinite(base)) return;
  var t0 = performance.now();
  function two(n) { return (n < 10 ? '0' : '') + n; }
  function fmt(s) {
    s = Math.floor(s);
    if (!(s > 0)) return '0s';
    var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
    return h ? h + 'h ' + two(m) + 'm ' + two(x) + 's' : m ? m + 'm ' + two(x) + 's' : x + 's';
  }
  function tick() {
    var now = base + (performance.now() - t0) / 1000;
    var els = p.querySelectorAll('[data-cd]');
    for (var i = 0; i < els.length; i++) {
      var ts = parseFloat(els[i].getAttribute('data-cd'));
      if (isFinite(ts)) els[i].textContent = fmt(ts - now);
    }
  }
  tick();
  setInterval(tick, 250);
})();
"""


def _num(v, default: float = 0.0) -> float:
    """float(v), or default for None / text / NaN / inf."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if _math.isfinite(f) else default


def _fmt_countdown(sec: float) -> str:
    """'1h 02m 03s' | '4m 05s' | '7s'. Never negative (clamped to '0s'). Same format as _SESSION_JS."""
    s = int(_math.floor(_num(sec)))
    if s <= 0:
        return "0s"
    h, m, x = s // 3600, (s % 3600) // 60, s % 60
    if h:
        return f"{h}h {m:02d}m {x:02d}s"
    if m:
        return f"{m}m {x:02d}s"
    return f"{x}s"


def _fmt_utc(ts) -> str:
    """'HH:MM:SS UTC' for an epoch timestamp, or a dash when unknown (<= 0)."""
    t = _num(ts)
    if t <= 0:
        return "—"
    try:
        return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%H:%M:%S") + " UTC"
    except (OverflowError, OSError, ValueError):
        return "—"


def _session_view(s: dict) -> dict:
    """Pure: turn the section-8 keys into what the panel shows. Tolerates missing / odd values."""
    state = str(s.get("session_state") or "").strip().upper()
    name  = str(s.get("session_name") or "").strip().upper()
    ts    = {k: _num(s.get(f"session_{k}_ts")) for k in ("connect", "open", "close")}
    # No usable schedule data (old engine writing only the free-text "session", or a
    # default-IDLE state nobody has published to yet): fall back to the old text.
    legacy = (state not in _SESSION_STATES) or (
        state == "IDLE" and not name and not any(v > 0 for v in ts.values()))

    if state == "LIVE":
        label = ("LIVE - always on" if name == "ALL"
                 else f"SESSION {name} LIVE" if name else "LIVE")
        cls = "sx-live"
    elif state == "CONNECTING":
        label, cls = "CONNECTING", "sx-warn"
    elif state == "WARMING":
        label, cls = "WARMING UP", "sx-info"
    elif state == "DRAINING":
        label, cls = "DRAINING", "sx-warn"
    elif state == "REDEPLOYING":
        label, cls = "REDEPLOYING", "sx-redeploy"
    else:
        label, cls = "IDLE - disconnected", "sx-idle"

    cd = {}
    for kind in ("connect", "open", "close"):
        if name == "ALL":
            cd[kind] = ("always", 0.0)                       # schedule disabled: no countdown
        elif ts[kind] <= 0 or state not in _CD_WHEN[kind]:
            cd[kind] = ("dash", ts[kind])
        else:
            cd[kind] = ("cd", ts[kind])

    return {
        "legacy": legacy, "state": state, "name": name, "label": label, "cls": cls, "cd": cd,
        "ts": ts,
        "stage": str(s.get("session_lead_stage") or "").strip(),
        "lead_s": _num(s.get("session_lead_seconds")),
        "offset": _num(s.get("clock_offset")),
        "error": _num(s.get("clock_error")),
        "rd_state": str(s.get("redeploy_state") or "").strip().lower(),
        "rd_ts": _num(s.get("redeploy_fire_ts")),
        "rd_note": str(s.get("redeploy_note") or "").strip()[:200],
        "old_text": str(s.get("session", "") or ""),
    }


def _render_session_panel(s: dict) -> str:
    """HTML (+ its script) for the session status card. Returns '' on any error: the page must still render."""
    try:
        v = _session_view(s)
        esc = _html.escape
        if v["legacy"]:
            return (
                '<div class="sx-card" id="sess-panel">'
                f'<div class="sx-head"><span class="sx-badge sx-idle">{esc(v["old_text"][:80] or "—")}</span>'
                '<span class="sx-sub">no schedule data from the engine</span></div></div>'
            )
        server_now = time.time() + v["offset"]

        def cd_span(ts: float) -> str:
            return f'<span data-cd="{ts:.3f}">{_fmt_countdown(ts - server_now)}</span>'

        def item(title: str, value: str, sub: str = "", wide: bool = False) -> str:
            return (f'<div class="sx-item{" sx-wide" if wide else ""}"><div class="card-title">{title}</div>'
                    f'<div class="sx-val">{value}</div>'
                    + (f'<div class="card-sub">{sub}</div>' if sub else "") + '</div>')

        def cd_item(title: str, kind: str) -> str:
            mode, t = v["cd"][kind]
            if mode == "always":
                return item(title, '<span class="sx-dim">always on</span>')
            sub = _fmt_utc(t) if t > 0 else ""
            if mode == "cd":
                return item(title, cd_span(t), sub)
            return item(title, '<span class="sx-dim">—</span>', sub)

        stage_shown = v["state"] in ("CONNECTING", "WARMING") and v["stage"]
        sub_txt = f'stage: {esc(v["stage"])}' if stage_shown else ""
        name_txt = ("ALL <span class=\"sx-dim\">(always on)</span>" if v["name"] == "ALL"
                    else esc(v["name"]) if v["name"] else '<span class="sx-dim">—</span>')

        # redeploy
        rs, rd_ts, note = v["rd_state"], v["rd_ts"], v["rd_note"]
        if not rs:
            rd_val = '<span class="sx-dim">none planned</span>'
        elif rs == "scheduled":
            rd_val = ('<span class="sx-rd-scheduled">scheduled</span>'
                      + (f' <span class="sx-dim">fires {_fmt_utc(rd_ts)} · in </span>{cd_span(rd_ts)}' if rd_ts > 0 else ""))
        elif rs == "fired":
            rd_val = ('<span class="sx-rd-fired">fired</span>'
                      + (f' <span class="sx-dim">at {_fmt_utc(rd_ts)}</span>' if rd_ts > 0 else ""))
        elif rs in ("skipped", "failed"):
            rd_val = (f'<span class="sx-rd-{rs}">{rs}</span>'
                      + (f' <span class="sx-dim">planned {_fmt_utc(rd_ts)}</span>' if rd_ts > 0 else ""))
        else:
            rd_val = esc(rs)
        rd_item = item("Redeploy", rd_val, esc(note), wide=True)

        offset_txt = f'{v["offset"]:+.3f} s' if (v["offset"] or v["error"]) else "—"
        error_txt  = f'±{v["error"]:.3f} s' if v["error"] > 0 else "—"
        lead_txt   = f'{v["lead_s"]:.1f} s' if v["lead_s"] > 0 else "—"

        return (
            f'<div class="sx-card" id="sess-panel" data-now="{server_now:.3f}">'
            f'<div class="sx-head"><span class="sx-badge {v["cls"]}">{esc(v["label"])}</span>'
            f'<span class="sx-sub">{sub_txt}</span></div>'
            '<div class="sx-grid">'
            + item("Session", name_txt)
            + cd_item("Connects in", "connect")
            + cd_item("Opens in", "open")
            + cd_item("Closes in", "close")
            + item("Lead stage", esc(v["stage"]) if stage_shown else '<span class="sx-dim">—</span>')
            + item("Last lead", lead_txt)
            + item("Clock offset", offset_txt)
            + item("Clock error", error_txt)
            + rd_item
            + '</div></div>'
            f'<script>{_SESSION_JS}</script>'
        )
    except Exception as exc:                      # never take the dashboard down for a status card
        logger.exception(f"_render_session_panel error: {exc}")
        return ""


def _render_dashboard() -> str:
    try:
        s = _state

        # ── Core values ────────────────────────────────────────────────────────
        balance      = float(s.get("balance",           0.0))
        day_start    = float(s.get("day_start_balance", 0.0))
        sess_start   = float(s.get("session_start_balance", day_start or balance))
        daily_pnl    = balance - day_start
        daily_pnl_pct= (daily_pnl / day_start * 100) if day_start else 0.0
        weekly_pnl   = float(s.get("weekly_pnl",       0.0))
        weekly_pnl_pct = float(s.get("weekly_pnl_pct", 0.0))
        monthly_pnl  = float(s.get("monthly_pnl",      0.0))
        monthly_pnl_pct = float(s.get("monthly_pnl_pct", 0.0))
        session_pnl  = balance - sess_start

        wins         = int(s.get("wins",           0))
        losses       = int(s.get("losses",         0))
        trades       = int(s.get("total_trades",   wins + losses))
        win_rate     = round(wins / trades * 100, 1) if trades else 0.0
        streak       = int(s.get("streak",         0))
        streak_lbl   = str(s.get("streak_label",   "—"))
        pf           = float(s.get("profit_factor", 0.0))
        gp           = float(s.get("gross_profit",  0.0))
        gl           = float(s.get("gross_loss",    0.0))
        avg_pnl      = round((gp - gl) / trades, 4) if trades else 0.0
        best_trade   = float(s.get("best_trade",    0.0))
        worst_trade  = float(s.get("worst_trade",   0.0))
        avg_mult     = float(s.get("avg_multiplier",0.0))
        mult_count   = int(s.get("multiplier_count",0))
        funding_fees = float(s.get("funding_fees_total", 0.0))

        open_list    = s.get("open_contracts", [])
        open_count   = int(s.get("open_contracts_count", len(open_list)))

        loss_pct     = max(-daily_pnl_pct, 0.0)
        loss_bar     = min(loss_pct / 15.0 * 100, 100)
        danger_class = "danger" if loss_bar > 70 else ""

        up     = int(s.get("uptime_seconds", 0))
        uptime = f"{up // 3600}h {(up % 3600) // 60}m"
        now_utc= datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        session      = str(s.get("session", "Starting"))
        session_panel = _render_session_panel(s)
        queue_count  = int(s.get("tradeable_count", 0))
        current_sym  = str(s.get("current_symbol", "—"))

        dot_class    = ("dot-green"  if s.get("running") and not s.get("paused_for_loss_limit")
                        else "dot-yellow" if s.get("paused_for_loss_limit")
                        else "dot-red")

        def c(v, pos_cls="green", neg_cls="red", zero_cls="yellow"):
            return pos_cls if v > 0 else (neg_cls if v < 0 else zero_cls)

        balance_color = "green" if balance >= day_start else "red"
        wr_color      = "green" if win_rate >= 55 else "yellow" if win_rate >= 45 else "red"
        pf_color      = "green" if pf >= 1.2 else "yellow" if pf >= 1.0 else "red"
        streak_color  = "green" if streak > 0 else "red" if streak < 0 else "yellow"
        streak_disp   = f"+{streak}" if streak > 0 else str(streak) if streak < 0 else "0"

        paused_banner = ""
        if s.get("paused_for_loss_limit"):
            paused_banner = (
                '<div class="paused-banner">'
                '⛔ Daily loss limit (15%) reached. Trading paused until UTC midnight.'
                '</div>'
            )

        # ── Link to the predictor minute-of-hour panel (replaces the old widgets) ──
        ks_html = ('<div style="margin:0 0 16px 0"><a href="/predictor" style="color:var(--blue,#58a6ff);'
                   'font-weight:700">Predictor · minute-of-hour stats &rarr;</a></div>')

        # ── SVG charts ────────────────────────────────────────────────────────
        svg_balance = _build_svg_chart(
            s.get("balance_history", []),
            start_balance=day_start or None,
        )
        svg_hourly  = _build_hourly_svg(s.get("hourly_pnl", {}))
        chart_scope, bar_scope = "Today", "Hourly P&amp;L — Today (UTC)"
        try:                                   # 24 h account history (MH mode) replaces both charts when available
            import account_history as _ah
            _b, _k = _ah.balance_svg(), _ah.bucket_svg()
            if _b:
                svg_balance, chart_scope = _b, "Last 24 h"
            if _k:
                svg_hourly, bar_scope = _k, "P&amp;L per 10 min — Last 24 h (UTC)"
        except Exception:
            pass

        # ── Recent trades table (last 100) ────────────────────────────────────
        recent_trades_list = s.get("recent_trades", [])
        trade_rows = ""
        bad_rows   = 0
        for t in recent_trades_list[:100]:
            try:
                pnl_raw = t.get("pnl") if t.get("pnl") is not None else t.get("profit", 0)
                pnl_v   = float(pnl_raw or 0)
                won_v   = bool(t.get("won", pnl_v > 0))
                cr      = str(t.get("close_reason", "normal"))
                if cr == "timeout":
                    badge = '<span class="badge badge-timeout">TIMEOUT</span>'
                elif won_v:
                    badge = '<span class="badge badge-win">WIN</span>'
                else:
                    badge = '<span class="badge badge-loss">LOSS</span>'
                raw_dir = str(t.get("direction", "") or "").upper()
                if "LONG" in raw_dir or "CALL" in raw_dir:
                    dir_b = '<span class="badge badge-long">LONG</span>'
                elif "SHORT" in raw_dir or "PUT" in raw_dir:
                    dir_b = '<span class="badge badge-short">SHORT</span>'
                else:
                    dir_b = f'<span class="ticker">{raw_dir or "—"}</span>'
                pnl_c   = "green" if pnl_v >= 0 else "red"
                ts_raw  = t.get("time") or t.get("exit_time") or t.get("close_time")
                ts_str  = _parse_hms(ts_raw) if ts_raw else "—"
                stake_v = float(t.get("stake", t.get("amount", 0)) or 0)
                bal_a   = float(t.get("balance_after", t.get("balance", 0)) or 0)
                sym_v   = str(t.get("symbol", ""))
                strat_v = str(t.get("strategy", ""))
                trade_rows += (
                    f"<tr>"
                    f"<td class='ticker'>{ts_str}</td>"
                    f"<td><b>{sym_v}</b></td>"
                    f"<td>{dir_b}</td>"
                    f"<td>${stake_v:.2f}</td>"
                    f"<td class='{pnl_c}'>{'+' if pnl_v >= 0 else ''}{pnl_v:.4f}</td>"
                    f"<td>${bal_a:.4f}</td>"
                    f"<td>{badge}</td>"
                    f"<td class='ticker'>{strat_v}</td>"
                    f"</tr>"
                )
            except Exception as row_err:
                bad_rows += 1
                logger.warning(f"trade row skipped — {row_err}")
                continue
        if not trade_rows:
            detail = (f" ({len(recent_trades_list)} in state, {bad_rows} errors)"
                      if recent_trades_list else "")
            trade_rows = (f"<tr><td colspan='8' style='text-align:center;color:#484f58'>"
                          f"No trades yet{detail}</td></tr>")

        # ── Open contracts ────────────────────────────────────────────────────
        open_rows = ""
        now_ts    = time.time()
        for c_entry in open_list:
            if isinstance(c_entry, dict):
                sym_oc   = str(c_entry.get("symbol",    "—"))
                dir_oc   = str(c_entry.get("direction", "—")).upper()
                stk_oc   = float(c_entry.get("stake",   0))
                ot       = float(c_entry.get("opened_at", now_ts))
                secs_open= int(now_ts - ot)
            else:
                sym_oc   = str(c_entry)
                dir_oc   = "—"
                stk_oc   = 0.0
                secs_open= 0
            if "LONG" in dir_oc or "CALL" in dir_oc:
                dir_badge = '<span class="badge badge-long">LONG</span>'
            elif "SHORT" in dir_oc or "PUT" in dir_oc:
                dir_badge = '<span class="badge badge-short">SHORT</span>'
            else:
                dir_badge = f'<span class="ticker">{dir_oc}</span>'
            open_rows += (
                f"<tr>"
                f"<td><b>{sym_oc}</b></td>"
                f"<td>{dir_badge}</td>"
                f"<td>${stk_oc:.2f}</td>"
                f"<td class='ticker'>{secs_open}s</td>"
                f"</tr>"
            )
        if not open_rows:
            open_rows = ("<tr><td colspan='4' style='text-align:center;color:#484f58'>"
                         "No open contracts</td></tr>")

        # ── Suspended (indicator, symbol) pairs — Task 5/6 fix ──────────────────
        # Sourced from pair_suspension.snapshot() via bot_engine._push_dashboard()
        # -> update_suspended_symbols() -> _state["suspended_symbols"]. Each entry
        # is {"pair": "<strategy>/<symbol>", "seconds_remaining": <float>} — NOT
        # the old whole-symbol {"symbol", "suspended_until"} shape (that data now
        # lives separately under "circuit_breaker_symbols", rendered below).
        susp_rows = ""
        for x in sorted(
            [z for z in s.get("suspended_symbols", [])
             if float(z.get("seconds_remaining", 0)) > 0],
            key=lambda z: float(z.get("seconds_remaining", 0)),
        ):
            secs_left = float(x.get("seconds_remaining", 0))
            mins_left = max(0, int(secs_left / 60))
            secs_disp = f"{mins_left} min" if mins_left > 0 else f"{int(secs_left)}s"
            susp_rows += (
                f"<tr>"
                f"<td><b>{x.get('pair', '—')}</b></td>"
                f"<td><span class='susp-badge'>⏸ {secs_disp} remaining</span></td>"
                f"</tr>"
            )
        if not susp_rows:
            susp_rows = ("<tr><td colspan='2' style='text-align:center;color:#484f58'>"
                         "None suspended</td></tr>")

        # ── Circuit-breaker suspended symbols (whole-symbol, buy-failure) ───────
        # This is what "Suspended Symbols" used to show before the Task 5 fix —
        # kept, but now its own clearly-labeled section rather than being
        # conflated with per-pair performance suspensions above. Only
        # SymbolManager's buy-failure circuit breaker writes to this state.
        cb_rows = ""
        for x in sorted(
            [z for z in s.get("circuit_breaker_symbols", [])
             if float(z.get("suspended_until", 0)) > now_ts],
            key=lambda z: float(z.get("suspended_until", 0)),
        ):
            mins_left = max(0, int((float(x["suspended_until"]) - now_ts) / 60))
            cb_rows += (
                f"<tr>"
                f"<td><b>{x.get('symbol', '—')}</b></td>"
                f"<td><span class='susp-badge'>⏸ {mins_left} min remaining</span></td>"
                f"</tr>"
            )
        if not cb_rows:
            cb_rows = ("<tr><td colspan='2' style='text-align:center;color:#484f58'>"
                       "None suspended</td></tr>")

        # ── Symbol leaderboard (all symbols, full stats) ──────────────────────
        sym_rows = ""
        for sym in s.get("best_symbols", []):
            try:
                pnl_s   = float(sym.get("pnl", 0))
                pnl_sc  = "green" if pnl_s >= 0 else "red"
                wr_s    = float(sym.get("win_rate", 0))
                wr_sc   = "green" if wr_s >= 55 else "yellow" if wr_s >= 45 else "red"
                best_s  = float(sym.get("best_trade",  0))
                worst_s = float(sym.get("worst_trade", 0))
                sym_rows += (
                    f"<tr>"
                    f"<td><b>{sym.get('symbol','')}</b></td>"
                    f"<td>{sym.get('trades', 0)}</td>"
                    f"<td class='{wr_sc}'>{wr_s:.1f}%</td>"
                    f"<td class='{pnl_sc}'>${pnl_s:+.4f}</td>"
                    f"<td class='green'>${best_s:+.4f}</td>"
                    f"<td class='red'>${worst_s:+.4f}</td>"
                    f"</tr>"
                )
            except Exception:
                continue
        if not sym_rows:
            sym_rows = ("<tr><td colspan='6' style='text-align:center;color:#484f58'>"
                        "No data yet</td></tr>")

        # ── Signal log (last 20) ──────────────────────────────────────────────
        sig_rows = ""
        for sig in s.get("signal_log", []):
            try:
                sig_dir  = str(sig.get("direction", "—")).upper()
                if "LONG" in sig_dir or "CALL" in sig_dir:
                    sig_db = '<span class="badge badge-long">LONG</span>'
                elif "SHORT" in sig_dir or "PUT" in sig_dir:
                    sig_db = '<span class="badge badge-short">SHORT</span>'
                else:
                    sig_db = f'<span class="ticker">{sig_dir}</span>'
                score_v  = float(sig.get("score", 0))
                score_c  = "green" if score_v >= 0.7 else "yellow" if score_v >= 0.5 else "red"
                sig_rows += (
                    f"<tr>"
                    f"<td class='ticker'>{sig.get('ts','—')}</td>"
                    f"<td><b>{sig.get('symbol','—')}</b></td>"
                    f"<td>{sig_db}</td>"
                    f"<td class='{score_c}'>{score_v:.3f}</td>"
                    f"<td class='ticker'>{sig.get('strategy','—')}</td>"
                    f"</tr>"
                )
            except Exception:
                continue
        if not sig_rows:
            sig_rows = ("<tr><td colspan='5' style='text-align:center;color:#484f58'>"
                        "No signals yet</td></tr>")

        # ── Helper: coloured PnL cell ─────────────────────────────────────────
        def pnl_span(v, fmt="+.4f"):
            cls = "green" if v >= 0 else "red"
            return f'<span class="{cls}">${v:{fmt}}</span>'

        # ── Render ────────────────────────────────────────────────────────────
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="10">
<title>Deriv Bot Dashboard</title>
<style>
  :root {{
    --bg:#0d1117;--card:#161b22;--border:#30363d;
    --text:#c9d1d9;--muted:#8b949e;--accent:#58a6ff;
    --green:#3fb950;--red:#f85149;--yellow:#d29922;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;font-size:14px;padding:16px;}}
  h1{{font-size:20px;font-weight:700;color:var(--accent);margin-bottom:4px;}}
  .subtitle{{color:var(--muted);font-size:12px;margin-bottom:16px;}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin-bottom:16px;}}
  .grid-wide{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:16px;}}
  .pnl-row{{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px;}}
  .pnl-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;flex:1;min-width:160px;}}
  .card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px;}}
  .card-title{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin-bottom:6px;}}
  .card-value{{font-size:20px;font-weight:700;}}
  .card-sub{{font-size:11px;color:var(--muted);margin-top:4px;}}
  .green{{color:var(--green);}} .red{{color:var(--red);}} .yellow{{color:var(--yellow);}}
  .section-title{{font-size:13px;font-weight:600;color:var(--muted);text-transform:uppercase;
                  letter-spacing:.05em;margin:20px 0 8px;}}
  table{{width:100%;border-collapse:collapse;background:var(--card);
         border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:16px;}}
  th{{background:#21262d;color:var(--muted);font-size:11px;text-transform:uppercase;
      padding:8px 10px;text-align:left;}}
  td{{padding:7px 10px;border-top:1px solid var(--border);font-size:12px;}}
  .ticker{{color:var(--muted);font-size:11px;}}
  .badge{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;}}
  .badge-win     {{background:#1a3a1f;color:var(--green);}}
  .badge-loss    {{background:#3a1a1a;color:var(--red);}}
  .badge-timeout {{background:#3a2a1a;color:var(--yellow);}}
  .badge-long    {{background:#1a2a3a;color:var(--accent);}}
  .badge-short   {{background:#2a1a3a;color:#c084fc;}}
  .status-row{{display:flex;align-items:center;gap:8px;margin-bottom:16px;}}
  .dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0;}}
  .dot-green {{background:var(--green);box-shadow:0 0 6px var(--green);}}
  .dot-yellow{{background:var(--yellow);box-shadow:0 0 6px var(--yellow);}}
  .dot-red   {{background:var(--red);   box-shadow:0 0 6px var(--red);}}
  .status-text{{font-size:12px;color:var(--muted);}}
  .ks-wrap{{margin:0 0 16px 0;}}
  .ks-btn{{width:100%;padding:18px 14px;font-size:17px;font-weight:700;border:0;border-radius:10px;
           cursor:pointer;letter-spacing:.02em;-webkit-tap-highlight-color:transparent;}}
  .ks-pause{{background:#da3633;color:#fff;}}
  .ks-resume{{background:#238636;color:#fff;}}
  .ks-btn:active{{transform:scale(.98);}}
  .ks-note{{font-size:11px;color:var(--yellow);margin-top:6px;}}
  .paused-banner{{background:#3a1a1a;border:1px solid var(--red);border-radius:6px;
                  padding:10px 14px;margin-bottom:14px;color:var(--red);font-size:13px;}}
  .bar-wrap{{background:#21262d;border-radius:4px;height:8px;margin-top:6px;overflow:hidden;}}
  .bar-fill{{height:100%;border-radius:4px;background:var(--green);transition:width .4s;}}
  .bar-fill.danger{{background:var(--red);}}
  .chart-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;
               padding:16px;margin-bottom:16px;}}
  .chart-title{{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;
                letter-spacing:.05em;margin-bottom:10px;}}
  .susp-badge{{display:inline-block;padding:2px 7px;border-radius:4px;
               font-size:10px;font-weight:700;background:#3a2a1a;color:var(--yellow);}}
  .footer{{margin-top:20px;font-size:11px;color:var(--muted);text-align:right;}}
{_SESSION_CSS}
</style>
</head>
<body>

<h1>⚡ Deriv Bot</h1>
<p class="subtitle">Auto-refreshes every 10 s &nbsp;|&nbsp; UTC {now_utc} &nbsp;|&nbsp; Queue: {queue_count}</p>

{ks_html}
{paused_banner}

<div class="status-row">
  <div class="dot {dot_class}"></div>
  <span class="status-text">
    <b>{session}</b> &nbsp;·&nbsp; Up {uptime}
    &nbsp;·&nbsp; Open: <b>{open_count}</b>
    &nbsp;·&nbsp; Balance: <b class="{balance_color}">${balance:.4f}</b>
  </span>
</div>

{session_panel}

<!-- ── 1. Header card ── -->
<div class="grid">
  <div class="card">
    <div class="card-title">Balance</div>
    <div class="card-value {balance_color}">${balance:.4f}</div>
    <div class="card-sub">Day start: ${day_start:.4f}</div>
  </div>
  <div class="card">
    <div class="card-title">Uptime</div>
    <div class="card-value">{uptime}</div>
    <div class="card-sub">Queue: {queue_count} &nbsp;·&nbsp; {session}</div>
  </div>
</div>

<!-- ── 2. P&L Summary row ── -->
<div class="section-title">P&amp;L Summary</div>
<div class="pnl-row">

  <div class="pnl-card">
    <div class="card-title">Daily P&amp;L</div>
    <div class="card-value {c(daily_pnl)}">${daily_pnl:+.4f}</div>
    <div class="card-sub">{daily_pnl_pct:+.2f}%</div>
    <div class="bar-wrap">
      <div class="bar-fill {danger_class}" style="width:{loss_bar:.0f}%"></div>
    </div>
    <div class="card-sub" style="margin-top:4px">
      Loss limit 15% &nbsp;·&nbsp; used {loss_pct:.1f}%
    </div>
  </div>

  <div class="pnl-card">
    <div class="card-title">Weekly P&amp;L</div>
    <div class="card-value {c(weekly_pnl)}">${weekly_pnl:+.4f}</div>
    <div class="card-sub">{weekly_pnl_pct:+.2f}%</div>
  </div>

  <div class="pnl-card">
    <div class="card-title">Monthly P&amp;L</div>
    <div class="card-value {c(monthly_pnl)}">${monthly_pnl:+.4f}</div>
    <div class="card-sub">{monthly_pnl_pct:+.2f}%</div>
  </div>

  <div class="pnl-card">
    <div class="card-title">Session P&amp;L</div>
    <div class="card-value {c(session_pnl)}">${session_pnl:+.4f}</div>
    <div class="card-sub">Since bot start (${sess_start:.4f})</div>
  </div>

</div>

<!-- ── 3. Performance Stats ── -->
<div class="section-title">Performance Stats</div>
<div class="grid">

  <div class="card">
    <div class="card-title">Win Rate</div>
    <div class="card-value {wr_color}">{win_rate:.1f}%</div>
    <div class="card-sub">{wins}W / {losses}L / {trades} trades today</div>
  </div>

  <div class="card">
    <div class="card-title">Profit Factor</div>
    <div class="card-value {pf_color}">{pf:.3f}</div>
    <div class="card-sub">GP ${gp:.4f} / GL ${gl:.4f}</div>
  </div>

  <div class="card">
    <div class="card-title">Avg P&amp;L / Trade</div>
    <div class="card-value {c(avg_pnl)}">${avg_pnl:+.4f}</div>
    <div class="card-sub">Across {trades} trade(s)</div>
  </div>

  <div class="card">
    <div class="card-title">Best Trade</div>
    <div class="card-value green">${best_trade:+.4f}</div>
    <div class="card-sub red">Worst: ${worst_trade:+.4f}</div>
  </div>

  <div class="card">
    <div class="card-title">Streak</div>
    <div class="card-value {streak_color}">{streak_disp}</div>
    <div class="card-sub">{streak_lbl}</div>
  </div>

  <div class="card">
    <div class="card-title">Avg Multiplier</div>
    <div class="card-value">{avg_mult:.1f}x</div>
    <div class="card-sub">Across {mult_count} trades</div>
  </div>

  <div class="card">
    <div class="card-title">Funding Fees</div>
    <div class="card-value red">${funding_fees:.4f}</div>
    <div class="card-sub">Neg PnL on timeout closes</div>
  </div>

  <div class="card">
    <div class="card-title">Open Contracts</div>
    <div class="card-value">{open_count}</div>
    <div class="card-sub">Active right now</div>
  </div>

</div>

<!-- ── 4. Balance Curve ── -->
<div class="chart-card">
  <div class="chart-title">Balance Curve — {chart_scope}
    <span style="float:right;font-weight:400;color:#3fb950">● win</span>
    <span style="float:right;font-weight:400;color:#f85149;margin-right:10px">● loss</span>
    <span style="float:right;font-weight:400;color:#d29922;margin-right:10px">● timeout</span>
  </div>
  {svg_balance}
</div>

<!-- ── 5. Hourly P&L ── -->
<div class="chart-card">
  <div class="chart-title">{bar_scope}</div>
  {svg_hourly}
</div>

<!-- ── 6. Recent Trades (last 100) ── -->
<div class="section-title">Recent Trades (last 100)</div>
<table>
  <thead>
    <tr>
      <th>Time (UTC)</th><th>Symbol</th><th>Direction</th>
      <th>Stake</th><th>P&amp;L</th><th>Balance After</th>
      <th>Result</th><th>Strategy</th>
    </tr>
  </thead>
  <tbody>{trade_rows}</tbody>
</table>

<!-- ── 7. Open Contracts ── -->
<div class="section-title">Open Contracts ({open_count})</div>
<table>
  <thead>
    <tr><th>Symbol</th><th>Direction</th><th>Stake</th><th>Open For</th></tr>
  </thead>
  <tbody>{open_rows}</tbody>
</table>

<!-- ── 8. Suspended Indicator/Symbol Pairs (Task 5/6 fix) ── -->
<div class="section-title">Suspended Indicator/Symbol Pairs</div>
<table>
  <thead><tr><th>Strategy/Symbol</th><th>Status</th></tr></thead>
  <tbody>{susp_rows}</tbody>
</table>

<!-- ── 8b. Circuit-Breaker Suspended Symbols ── -->
<div class="section-title">Circuit-Breaker Suspended Symbols</div>
<table>
  <thead><tr><th>Symbol</th><th>Status</th></tr></thead>
  <tbody>{cb_rows}</tbody>
</table>

<!-- ── 9. Symbol Leaderboard ── -->
<div class="section-title">Symbol Leaderboard</div>
<table>
  <thead>
    <tr><th>Symbol</th><th>Trades</th><th>Win %</th><th>Total P&amp;L</th><th>Best</th><th>Worst</th></tr>
  </thead>
  <tbody>{sym_rows}</tbody>
</table>

<!-- ── 10. Signal Log (last 20) ── -->
<div class="section-title">Signal Log (last 20)</div>
<table>
  <thead>
    <tr><th>Time</th><th>Symbol</th><th>Direction</th><th>Score</th><th>Strategy</th></tr>
  </thead>
  <tbody>{sig_rows}</tbody>
</table>

<div class="footer">Deriv Bot &middot; Render deployment &middot; {now_utc} UTC</div>
</body>
</html>"""

    except Exception as exc:
        logger.exception(f"_render_dashboard error: {exc}")
        return (
            "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
            "<meta http-equiv='refresh' content='10'>"
            "<title>Deriv Bot</title></head>"
            "<body style='background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:30px'>"
            f"<h2 style='color:#f85149'>Dashboard render error</h2>"
            f"<pre style='color:#8b949e'>{exc}</pre>"
            "<p>Bot may still be running. Check logs. Page reloads in 10 s.</p>"
            "</body></html>"
        )


# ── Strategy/symbol stats page renderer ─────────────────────────────────────────

def _render_strategy_stats_page() -> str:
    """
    Standalone HTML page listing every (strategy, symbol) pair from
    strategy_stats.py — win rate, 95% Wilson confidence interval, and
    trade count. Sorted by trade count (all-time) descending. Reuses the
    same dark-theme CSS variables/classes as the main dashboard so it
    matches visually, but is rendered as its own page (not injected into
    _render_dashboard) to avoid touching that function.
    """
    now_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    try:
        if not _STRATEGY_STATS_AVAILABLE:
            raise RuntimeError("strategy_stats module not available")

        rows = _strategy_stats.all_stats()
        rows.sort(key=lambda d: d.get("total_trades", 0), reverse=True)

        body_rows = ""
        for r in rows:
            strategy = str(r.get("strategy", "—"))
            symbol   = str(r.get("symbol", "—"))
            win_rate = float(r.get("win_rate", 0.0)) * 100
            ci_low   = float(r.get("ci_low", 0.0)) * 100
            ci_high  = float(r.get("ci_high", 0.0)) * 100
            n        = int(r.get("n", 0))
            total    = int(r.get("total_trades", 0))
            underperf= bool(r.get("underperforming", False))

            wr_color = "green" if win_rate >= 55 else "yellow" if win_rate >= 45 else "red"
            flag = ('<span class="badge badge-loss">UNDERPERFORMING</span>'
                    if underperf else
                    '<span class="ticker">—</span>')

            body_rows += (
                f"<tr>"
                f"<td><b>{strategy}</b></td>"
                f"<td>{symbol}</td>"
                f"<td class='{wr_color}'>{win_rate:.1f}%</td>"
                f"<td class='ticker'>{ci_low:.1f}% – {ci_high:.1f}%</td>"
                f"<td>{n}</td>"
                f"<td>{total}</td>"
                f"<td>{flag}</td>"
                f"</tr>"
            )

        if not body_rows:
            body_rows = (
                "<tr><td colspan='7' style='text-align:center;color:#484f58'>"
                "No strategy/symbol trades recorded yet</td></tr>"
            )

        backend_name = _strategy_stats.backend_name

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta http-equiv="refresh" content="30">
<title>Strategy Stats — Deriv Bot</title>
<style>
  :root {{
    --bg:#0d1117;--card:#161b22;--border:#30363d;
    --text:#c9d1d9;--muted:#8b949e;--accent:#58a6ff;
    --green:#3fb950;--red:#f85149;--yellow:#d29922;
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;font-size:14px;padding:16px;}}
  h1{{font-size:20px;font-weight:700;color:var(--accent);margin-bottom:4px;}}
  .subtitle{{color:var(--muted);font-size:12px;margin-bottom:16px;}}
  .section-title{{font-size:13px;font-weight:600;color:var(--muted);text-transform:uppercase;
                  letter-spacing:.05em;margin:20px 0 8px;}}
  table{{width:100%;border-collapse:collapse;background:var(--card);
         border:1px solid var(--border);border-radius:8px;overflow:hidden;margin-bottom:16px;}}
  th{{background:#21262d;color:var(--muted);font-size:11px;text-transform:uppercase;
      padding:8px 10px;text-align:left;}}
  td{{padding:7px 10px;border-top:1px solid var(--border);font-size:12px;}}
  .ticker{{color:var(--muted);font-size:11px;}}
  .badge{{display:inline-block;padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;}}
  .badge-loss{{background:#3a1a1a;color:var(--red);}}
  .green{{color:var(--green);}} .red{{color:var(--red);}} .yellow{{color:var(--yellow);}}
  .footer{{margin-top:20px;font-size:11px;color:var(--muted);text-align:right;}}
</style>
</head>
<body>

<h1>📊 Strategy / Symbol Stats</h1>
<p class="subtitle">Auto-refreshes every 30 s &nbsp;|&nbsp; UTC {now_utc} &nbsp;|&nbsp; Backend: {backend_name} &nbsp;|&nbsp; <a href="/" style="color:var(--accent)">← Dashboard</a></p>

<div class="section-title">All Strategy / Symbol Pairs (sorted by total trades)</div>
<table>
  <thead>
    <tr>
      <th>Strategy</th><th>Symbol</th><th>Win Rate</th>
      <th>95% CI</th><th>N (window)</th><th>Total Trades</th><th>Flag</th>
    </tr>
  </thead>
  <tbody>{body_rows}</tbody>
</table>

<div class="footer">Deriv Bot &middot; Strategy Stats &middot; {now_utc} UTC</div>
</body>
</html>"""

    except Exception as exc:
        logger.exception(f"_render_strategy_stats_page error: {exc}")
        return (
            "<!DOCTYPE html><html><head><meta charset='UTF-8'>"
            "<meta http-equiv='refresh' content='30'>"
            "<title>Strategy Stats</title></head>"
            "<body style='background:#0d1117;color:#c9d1d9;font-family:sans-serif;padding:30px'>"
            f"<h2 style='color:#f85149'>Strategy stats render error</h2>"
            f"<pre style='color:#8b949e'>{exc}</pre>"
            "<p>Bot may still be running. Check logs. Page reloads in 30 s.</p>"
            "</body></html>"
        )


# ── Flask routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return Response(_render_dashboard(), mimetype="text/html")


import hmac as _hmac

_pin_fail: dict = {}   # ip -> (count, window_start) — tiny PIN brute-force limiter


def _expected_pin() -> str:
    pin = (os.environ.get("DASHBOARD_PIN") or "").strip()
    if pin:
        return pin
    tok = (getattr(config, "DERIV_API_TOKEN", "") or "").strip()
    return tok[-6:] if tok else ""


def _check_pin(supplied: str) -> bool:
    want = _expected_pin()
    if not want:
        return True
    return _hmac.compare_digest(str(supplied or "").strip().encode(), want.encode())


@app.route("/predictor")
def predictor_page():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
    with open(path, encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/html")


@app.route("/api/predictor_stats")
def predictor_stats_api():
    ip  = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()
    now = time.time()
    cnt, since = _pin_fail.get(ip, (0, now))
    if now - since > 60:
        cnt, since = 0, now
    if cnt >= 5:
        return jsonify({"error": "too many attempts"}), 429
    pin = request.headers.get("X-Pin") or request.args.get("pin", "")
    if not _check_pin(pin):
        _pin_fail[ip] = (cnt + 1, since)
        return jsonify({"error": "bad pin"}), 403
    _pin_fail.pop(ip, None)
    import predictor_stats
    return jsonify(predictor_stats.analyze(predictor_stats.get_stats().snapshot()))


@app.route("/api/exp_report")
def exp_report_api():
    """Pre-registered test report: verdict, balance vs expected path, three numbers, diagnostics (PIN protected)."""
    ip  = request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip()
    now = time.time()
    cnt, since = _pin_fail.get(ip, (0, now))
    if now - since > 60:
        cnt, since = 0, now
    if cnt >= 5:
        return jsonify({"error": "too many attempts"}), 429
    pin = request.headers.get("X-Pin") or request.args.get("pin", "")
    if not _check_pin(pin):
        _pin_fail[ip] = (cnt + 1, since)
        return jsonify({"error": "bad pin"}), 403
    _pin_fail.pop(ip, None)
    import exp_stats
    return jsonify(exp_stats.EXP.latest())


@app.route("/api/history24")
def history24_api():
    import account_history
    return jsonify(account_history.get_series())


@app.route("/health")
def health():
    return jsonify({"status": "ok", "ts": time.time()}), 200


@app.route("/stats")
def stats_route():
    s         = _state
    balance   = float(s.get("balance", 0.0))
    day_start = float(s.get("day_start_balance", 0.0))
    wins      = int(s.get("wins", 0))
    losses    = int(s.get("losses", 0))
    trades    = int(s.get("total_trades", wins + losses))
    return jsonify({
        "balance":            round(balance, 4),
        "day_start_balance":  round(day_start, 4),
        "daily_pnl":          round(balance - day_start, 4),
        "daily_pnl_pct":      round((balance - day_start) / day_start * 100, 2) if day_start else 0,
        "weekly_pnl":         round(float(s.get("weekly_pnl", 0.0)), 4),
        "weekly_pnl_pct":     round(float(s.get("weekly_pnl_pct", 0.0)), 2),
        "monthly_pnl":        round(float(s.get("monthly_pnl", 0.0)), 4),
        "monthly_pnl_pct":    round(float(s.get("monthly_pnl_pct", 0.0)), 2),
        "trades":             trades,
        "wins":               wins,
        "losses":             losses,
        "win_rate":           round(wins / max(trades, 1) * 100, 1),
        "profit_factor":      round(float(s.get("profit_factor", 0)), 3),
        "avg_rr":             round(float(s.get("avg_rr", 0)), 2),
        "avg_multiplier":     round(float(s.get("avg_multiplier", 0.0)), 2),
        "multiplier_count":   int(s.get("multiplier_count", 0)),
        "best_trade":         round(float(s.get("best_trade", 0.0)), 4),
        "worst_trade":        round(float(s.get("worst_trade", 0.0)), 4),
        "funding_fees_total": round(float(s.get("funding_fees_total", 0.0)), 4),
        "open_contracts":     s.get("open_contracts", []),
        "open_contracts_count": int(s.get("open_contracts_count", 0)),
        "streak":             int(s.get("streak", 0)),
        "streak_label":       str(s.get("streak_label", "—")),
        "paused":             bool(s.get("paused_for_loss_limit", False)),
        "target_balance":     s.get("target_balance"),
        "money":              s.get("money"),   # main-account profit floor / stake sizing
        "target_reached":     bool(s.get("target_reached", False)),
        "target_reached_balance": s.get("target_reached_balance"),
        "current_symbol":     str(s.get("current_symbol", "—")),
        "session":            str(s.get("session", "—")),
        "session_state":      str(s.get("session_state", "IDLE")),
        "session_name":       str(s.get("session_name", "")),
        "session_open_ts":    _num(s.get("session_open_ts")),
        "session_close_ts":   _num(s.get("session_close_ts")),
        "session_connect_ts": _num(s.get("session_connect_ts")),
        "session_lead_stage": str(s.get("session_lead_stage", "")),
        "session_lead_seconds": _num(s.get("session_lead_seconds")),
        "clock_offset":       _num(s.get("clock_offset")),
        "clock_error":        _num(s.get("clock_error")),
        "redeploy_state":     str(s.get("redeploy_state", "")),
        "redeploy_fire_ts":   _num(s.get("redeploy_fire_ts")),
        "redeploy_note":      str(s.get("redeploy_note", "")),
        "server_time":        time.time(),   # bot-host clock now; the page adds clock_offset for countdowns
        "uptime_seconds":     int(s.get("uptime_seconds", 0)),
        "last_signal":        s.get("last_signal", "—"),
        "tradeable_count":    int(s.get("tradeable_count", 0)),
        "active_trades":      int(s.get("active_trades", 0)),
        "suspended_symbols":  s.get("suspended_symbols", []),  # Task 5 fix: now per-(indicator,symbol) pair data — see update_suspended_symbols() docstring
        "circuit_breaker_symbols": s.get("circuit_breaker_symbols", []),  # whole-symbol buy-failure suspensions, Task 5/6
        "hourly_pnl":         s.get("hourly_pnl", {}),
        "daily_pnl_history":  s.get("daily_pnl_history", []),
        "all_trades_count":   len(s.get("all_trades", [])),
        "signal_log":         s.get("signal_log", []),
        "failure_log":        s.get("failure_log", []),
    })


@app.route("/strategy-stats")
def strategy_stats_route():
    return Response(_render_strategy_stats_page(), mimetype="text/html")


@app.route("/trades")
def trades_route():
    return jsonify({
        "recent_trades":    _state.get("recent_trades", []),
        "all_trades_count": len(_state.get("all_trades", [])),
    })


@app.route("/symbols")
def symbols_route():
    return jsonify({"symbols": _state.get("best_symbols", [])})


@app.route("/signals")
def signals_route():
    return jsonify({
        "signal_log":  _state.get("signal_log",  []),
        "failure_log": _state.get("failure_log", []),
    })


@app.route("/debug")
def debug_route():
    safe = {}
    for k, v in _state.items():
        if isinstance(v, list):
            safe[k] = {"type": "list", "length": len(v),
                       "first_item": v[0] if v else None}
        elif isinstance(v, dict):
            safe[k] = {"type": "dict", "keys": list(v.keys())}
        else:
            safe[k] = v
    return jsonify(safe), 200


# ── Keep-alive ping loop ────────────────────────────────────────────────────────

def _ping_loop():
    time.sleep(20)
    while True:
        try:
            r = requests.get(f"{config.SELF_URL}/health", timeout=10)
            logger.debug(f"Keep-alive ping → {r.status_code}")
        except Exception as exc:
            logger.warning(f"Keep-alive ping failed: {exc}")
        time.sleep(config.KEEP_ALIVE_INTERVAL)


def start_keep_alive():
    t = threading.Thread(target=_ping_loop, name="keep-alive", daemon=True)
    t.start()
    logger.info(
        f"Keep-alive pinger started "
        f"(interval={config.KEEP_ALIVE_INTERVAL}s, url={config.SELF_URL})"
    )
