"""
NOTE: symbol_audit.py is a STANDALONE diagnostic (run it by hand: python symbol_audit.py). The bot never imports
or starts it - there is no startup audit, and it must never run from the engine, keep_alive or the session clock.

symbol_audit.py — One-time diagnostic: map every live Deriv symbol to its
supported contract types, and cross-reference against what deriv_client.py
can actually execute today.

AUTH NOTE (read before running):
  deriv_client.py's connect() currently authenticates via a REST OTP flow
  against https://api.derivws.com/trading/v1/options/... . That endpoint
  does not appear in any official Deriv SDK (JS, Python, Rust, PHP) — every
  official client connects directly to a WebSocket endpoint
  (wss://ws.derivws.com/websockets/v3?app_id=...) and sends
  {"authorize": API_TOKEN} as the first message. deriv_client.py already
  contains that exact correct pattern in its unused _authorize() method.
  This script uses the documented direct-WS + authorize flow instead of
  the OTP flow, so it will work regardless of whether the OTP endpoint is
  real, fake, or half-broken.

Requires the same env vars the bot already uses:
  DERIV_API_TOKEN, DERIV_APP_ID, and optionally DERIV_WS_URL
  (defaults to wss://ws.derivws.com/websockets/v3 if unset).

Usage:
  python symbol_audit.py
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("symbol_audit")

# ── Config (env-first, falls back to config.py if present) ────────────────
try:
    import config as _config
except Exception:
    _config = None


def _cfg(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name) or getattr(_config, name, None) or default


DERIV_API_TOKEN = _cfg("DERIV_API_TOKEN")
DERIV_APP_ID = _cfg("DERIV_APP_ID")
DERIV_WS_URL = _cfg("DERIV_WS_URL", "wss://ws.derivws.com/websockets/v3")

# NOTE: env vars are validated lazily inside connect_and_authorize() rather
# than at import time. This module is now imported by bot_engine.py and
# keep_alive.py (not just run standalone via `python symbol_audit.py`), so
# raising at import time would crash unrelated startup/routes if this
# module happened to load before config finished populating. Standalone
# CLI usage (`python symbol_audit.py`) still gets a clear error — it just
# surfaces at connect_and_authorize() instead of at import.

# ── What deriv_client.py's existing buy methods can actually send ─────────
# buy_contract()        -> CALL / PUT               (Rise/Fall; barrier kwarg
#                           exists but no call site uses it for Higher/Lower)
# buy_accumulator()     -> ACCU
# buy_digit_contract()  -> DIGITMATCH / DIGITDIFF ONLY (Matches/Differs).
#                           NOT Over/Under, NOT Even/Odd.
BOT_SUPPORTED_CONTRACT_TYPES = {"CALL", "PUT", "ACCU", "DIGITMATCH", "DIGITDIFF"}

# contract_type -> friendly label, for the printed summary only
CONTRACT_TYPE_LABELS = {
    "CALL": "Rise/Fall or Higher/Lower (CALL)",
    "PUT": "Rise/Fall or Higher/Lower (PUT)",
    "ACCU": "Accumulator",
    "DIGITMATCH": "Digit Matches",
    "DIGITDIFF": "Digit Differs",
    "DIGITOVER": "Digit Over",
    "DIGITUNDER": "Digit Under",
    "DIGITEVEN": "Digit Even",
    "DIGITODD": "Digit Odd",
    "MULTUP": "Multiplier Up",
    "MULTDOWN": "Multiplier Down",
    "ONETOUCH": "Touch",
    "NOTOUCH": "No Touch",
    "CALLE": "Higher (legacy euro_non_atm)",
    "PUTE": "Lower (legacy euro_non_atm)",
    "RESETCALL": "Reset Call",
    "RESETPUT": "Reset Put",
    "TICKHIGH": "High Tick",
    "TICKLOW": "Low Tick",
}

REQUEST_DELAY = 0.35  # seconds between contracts_for calls (rate-limit safety)
CONTRACTS_FOR_TIMEOUT = 15.0

# ── One-time-per-deploy guard (used by run_audit_once(), called from
#    bot_engine.py's startup) ────────────────────────────────────────────
DEFAULT_GUARD_PATH = "symbol_audit.done"
DEFAULT_JSON_PATH = "symbol_contract_map.json"


class MiniDerivClient:
    """
    Minimal, self-contained client for this diagnostic only. Deliberately
    does NOT import DerivClient from deriv_client.py, since that class's
    connect() drives the broken OTP flow — reimplementing the documented
    direct-WS + authorize pattern here keeps this script runnable
    independent of that bug.
    """

    def __init__(self):
        self._ws = None
        self._req_id = 1
        self._pending: Dict[int, asyncio.Future] = {}
        self._loop = None
        self._listen_task = None

    async def connect_and_authorize(self):
        if not DERIV_API_TOKEN:
            raise RuntimeError("DERIV_API_TOKEN is not set (env var or config.py).")
        if not DERIV_APP_ID:
            raise RuntimeError("DERIV_APP_ID is not set (env var or config.py).")

        self._loop = asyncio.get_event_loop()
        ws_url = f"{DERIV_WS_URL}?app_id={DERIV_APP_ID}"
        logger.info(f"Connecting to {DERIV_WS_URL}?app_id=*** …")
        self._ws = await websockets.connect(ws_url, ping_interval=20, ping_timeout=20)
        self._listen_task = asyncio.create_task(self._listen())

        resp = await self._send({"authorize": DERIV_API_TOKEN})
        if resp.get("error"):
            raise RuntimeError(f"Authorize failed: {resp['error']}")
        account = resp.get("authorize", {})
        logger.info(
            f"Authorized OK | loginid={account.get('loginid')} "
            f"| is_virtual={account.get('is_virtual')} "
            f"| balance={account.get('balance')} {account.get('currency')}"
        )
        return account

    async def _listen(self):
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                req_id = msg.get("req_id")
                if req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.done():
                        fut.set_result(msg)
        except websockets.exceptions.ConnectionClosed:
            pass

    async def _send(self, payload: dict, timeout: float = 20.0) -> dict:
        req_id = self._req_id
        self._req_id += 1
        payload = dict(payload)
        payload["req_id"] = req_id
        fut = self._loop.create_future()
        self._pending[req_id] = fut
        await self._ws.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            return {"error": {"code": "LocalTimeout", "message": "request timed out"}}

    async def active_symbols(self) -> List[dict]:
        resp = await self._send({"active_symbols": "brief", "product_type": "basic"}, timeout=20)
        if resp.get("error"):
            raise RuntimeError(f"active_symbols failed: {resp['error']}")
        return resp.get("active_symbols", [])

    async def contracts_for(self, symbol: str) -> Optional[dict]:
        resp = await self._send(
            {"contracts_for": symbol, "currency": "USD"}, timeout=CONTRACTS_FOR_TIMEOUT
        )
        if resp.get("error"):
            logger.warning(f"contracts_for({symbol}) error: {resp['error']}")
            return None
        return resp.get("contracts_for")

    async def close(self):
        if self._listen_task:
            self._listen_task.cancel()
        if self._ws:
            await self._ws.close()


# ── Category classification (content-based, not hardcoded prefixes) ───────
# We trust the API's own market/submarket fields rather than assuming
# symbol-prefix conventions, since Deriv has changed these before.

SUBMARKET_LABELS = {
    "random_index": "Volatility Indices",  # split into standard/1s below
    "crash_index": "Boom/Crash",
    "jump_index": "Jump Indices",
    "step_index": "Step Index",
}


def classify_symbol(sym: dict) -> str:
    market = (sym.get("market") or "").lower()
    submarket = (sym.get("submarket") or "").lower()
    display = (sym.get("display_name") or "")
    code = (sym.get("symbol") or "").upper()

    if submarket in SUBMARKET_LABELS:
        base = SUBMARKET_LABELS[submarket]
        if submarket == "random_index":
            return "Volatility Indices (1s)" if code.startswith("1HZ") else "Volatility Indices (standard)"
        return base

    # Generic fallback for anything we don't have a hardcoded label for —
    # this is what lets Bear/Bull Market, Range Break, and DEX symbols
    # surface correctly even if Deriv has changed/added submarket codes,
    # instead of silently mis-bucketing or dropping them.
    if "bear" in submarket or "bull" in submarket or "bear" in display.lower() or "bull" in display.lower():
        return "Bear/Bull Market"
    if "range" in submarket or "range" in display.lower():
        return "Range Break"
    if "dex" in submarket or "dex" in display.lower() or code.startswith("DEX"):
        return "DEX"

    label = sym.get("submarket_display_name") or sym.get("market_display_name") or submarket or market
    return label.title() if label else "Uncategorized"


async def run_audit(json_path: str = DEFAULT_JSON_PATH) -> dict:
    """
    Runs the full live audit exactly once: connects (direct WS + authorize,
    bypassing DerivClient.connect()'s OTP flow entirely), pulls
    active_symbols + contracts_for for every symbol, classifies, cross-
    references against BOT_SUPPORTED_CONTRACT_TYPES, saves the JSON to
    disk, and returns the result dict. Raises on unrecoverable failure
    (e.g. auth failure, active_symbols failure) — callers are expected to
    catch and log rather than let this crash the bot.
    """
    client = MiniDerivClient()
    await client.connect_and_authorize()

    try:
        symbols = await client.active_symbols()
    except Exception as exc:
        await client.close()
        raise RuntimeError(f"Could not fetch active_symbols: {exc}") from exc

    logger.info(f"active_symbols returned {len(symbols)} symbols")

    categories: Dict[str, List[dict]] = defaultdict(list)
    for sym in symbols:
        categories[classify_symbol(sym)].append(sym)

    results: Dict[str, List[dict]] = defaultdict(list)
    total = len(symbols)
    for i, sym in enumerate(symbols, 1):
        code = sym.get("symbol")
        cat = classify_symbol(sym)
        logger.info(f"[{i}/{total}] contracts_for {code} ({cat}) …")

        cf = await client.contracts_for(code)
        await asyncio.sleep(REQUEST_DELAY)

        entry = {
            "symbol": code,
            "display_name": sym.get("display_name"),
            "market": sym.get("market"),
            "submarket": sym.get("submarket"),
            "exchange_is_open": sym.get("exchange_is_open"),
        }

        if cf is None:
            entry["contract_types"] = []
            entry["barrier_categories"] = []
            entry["error"] = "contracts_for call failed"
        else:
            available = cf.get("available", [])
            contract_types = sorted({c.get("contract_type") for c in available if c.get("contract_type")})
            barrier_cats = sorted({c.get("barrier_category") for c in available if c.get("barrier_category")})
            entry["contract_types"] = contract_types
            entry["barrier_categories"] = barrier_cats

            api_set = set(contract_types)
            entry["bot_ready_types"] = sorted(api_set & BOT_SUPPORTED_CONTRACT_TYPES)
            entry["api_supports_but_bot_has_no_method"] = sorted(api_set - BOT_SUPPORTED_CONTRACT_TYPES)
            entry["bot_has_method_but_api_does_not_support"] = sorted(BOT_SUPPORTED_CONTRACT_TYPES - api_set)

        results[cat].append(entry)

    await client.close()

    # ── Explicit flags requested ───────────────────────────────────────────
    flags = {}

    for special in ("Range Break", "DEX"):
        entries = results.get(special, [])
        supports_any_bot_relevant = any(
            e.get("bot_ready_types") or (set(e.get("contract_types", [])) & BOT_SUPPORTED_CONTRACT_TYPES)
            for e in entries
        )
        flags[special] = {
            "symbols_found": len(entries),
            "symbol_list": [e["symbol"] for e in entries],
            "supports_rise_fall_digits_or_accumulator": supports_any_bot_relevant,
            "all_contract_types_seen": sorted({ct for e in entries for ct in e.get("contract_types", [])}),
        }

    bear_bull_entries = results.get("Bear/Bull Market", [])
    flags["Bear/Bull Market"] = {
        "present_in_live_list": len(bear_bull_entries) > 0,
        "symbol_list": [e["symbol"] for e in bear_bull_entries],
    }

    # ── Save full JSON ──────────────────────────────────────────────────────
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_symbols": total,
        "bot_supported_contract_types": sorted(BOT_SUPPORTED_CONTRACT_TYPES),
        "categories": results,
        "flags": flags,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved full result to {json_path}")

    return output


# ── Shared report-building helpers (used by both text and HTML renderers,
#    and by both a fresh run and a cached/loaded-from-disk result) ─────────

def _category_rows(output: dict) -> List[dict]:
    rows = []
    for cat, entries in sorted(output.get("categories", {}).items()):
        fully_ready = sum(
            1 for e in entries
            if e.get("contract_types") and set(e["contract_types"]) <= BOT_SUPPORTED_CONTRACT_TYPES
        )
        needs_new = sum(1 for e in entries if e.get("api_supports_but_bot_has_no_method"))
        rows.append({
            "category":         cat,
            "symbols":          len(entries),
            "fully_ready":      fully_ready,
            "needs_new_method": needs_new,
        })
    return rows


def render_text_summary(output: dict) -> str:
    """
    The console summary table — used both for the standalone CLI run and
    for bot_engine's startup log (same text either way).
    """
    if not output:
        return "SYMBOL AUDIT: no result available."

    total = output.get("total_symbols", 0)
    rows = _category_rows(output)
    flags = output.get("flags", {})

    lines = []
    lines.append("=" * 78)
    lines.append("SYMBOL / CONTRACT-TYPE AUDIT SUMMARY")
    lines.append(f"(generated_at={output.get('generated_at', '—')})")
    lines.append("=" * 78)
    lines.append(f"{'Category':<28}{'Symbols':>8}{'Fully bot-ready':>18}{'Needs new method':>19}")
    lines.append("-" * 78)
    for r in rows:
        lines.append(
            f"{r['category']:<28}{r['symbols']:>8}{r['fully_ready']:>18}{r['needs_new_method']:>19}"
        )
    lines.append("-" * 78)
    lines.append(f"{'TOTAL':<28}{total:>8}")
    lines.append("=" * 78)

    lines.append("")
    lines.append("EXPLICIT FLAGS")
    lines.append("-" * 78)
    for special in ("Range Break", "DEX"):
        f_ = flags.get(special, {})
        lines.append(
            f"{special}: {f_.get('symbols_found', 0)} symbol(s) found | "
            f"supports Rise/Fall, Digits, or Accumulator? "
            f"{'YES' if f_.get('supports_rise_fall_digits_or_accumulator') else 'NO'}"
        )
        if f_.get("all_contract_types_seen"):
            lines.append(f"   contract types actually offered: {f_['all_contract_types_seen']}")

    bb = flags.get("Bear/Bull Market", {})
    lines.append(
        "Bear/Bull Market: "
        + (f"PRESENT — {bb.get('symbol_list')}" if bb.get("present_in_live_list")
           else "NOT present in current active_symbols list")
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def render_html_fragment(output: Optional[dict]) -> str:
    """
    HTML fragment only (a heading, a table, and a flags block) — no
    <html>/<head>/<body>/<style>. Meant to be embedded inside a page shell
    that already defines the dashboard's CSS variables (--bg, --card,
    --border, --text, --muted, --accent, --green, --red, --yellow), the
    same ones keep_alive.py's other pages use, so it inherits the existing
    look with zero extra CSS.
    """
    if not output:
        return (
            "<p style='color:var(--muted)'>No symbol audit result yet — "
            "it runs once automatically at bot startup (or on the next "
            "deploy if it hasn't run before).</p>"
        )

    rows = _category_rows(output)
    flags = output.get("flags", {})
    total = output.get("total_symbols", 0)

    body_rows = ""
    for r in rows:
        body_rows += (
            "<tr>"
            f"<td><b>{r['category']}</b></td>"
            f"<td>{r['symbols']}</td>"
            f"<td class='green'>{r['fully_ready']}</td>"
            f"<td class='{'yellow' if r['needs_new_method'] else ''}'>{r['needs_new_method']}</td>"
            "</tr>"
        )
    if not body_rows:
        body_rows = (
            "<tr><td colspan='4' style='text-align:center;color:var(--muted)'>"
            "No categories found</td></tr>"
        )

    flag_rows = ""
    for special in ("Range Break", "DEX"):
        f_ = flags.get(special, {})
        supported = f_.get("supports_rise_fall_digits_or_accumulator")
        badge = (
            "<span class='badge badge-loss'>YES</span>" if supported
            else "<span class='ticker'>NO</span>"
        )
        types_seen = ", ".join(f_.get("all_contract_types_seen", [])) or "—"
        flag_rows += (
            "<tr>"
            f"<td><b>{special}</b></td>"
            f"<td>{f_.get('symbols_found', 0)}</td>"
            f"<td>{badge}</td>"
            f"<td class='ticker'>{types_seen}</td>"
            "</tr>"
        )

    bb = flags.get("Bear/Bull Market", {})
    bb_present = bb.get("present_in_live_list")
    bb_html = (
        f"<span class='green'>PRESENT</span> — {', '.join(bb.get('symbol_list', []))}"
        if bb_present else
        "<span class='ticker'>NOT present in current active_symbols list</span>"
    )

    return f"""
<div class="section-title">Symbols by Category ({total} total)</div>
<table>
  <thead>
    <tr><th>Category</th><th>Symbols</th><th>Fully Bot-Ready</th><th>Needs New Method</th></tr>
  </thead>
  <tbody>{body_rows}</tbody>
</table>

<div class="section-title">Explicit Flags — Range Break / DEX</div>
<table>
  <thead>
    <tr><th>Category</th><th>Symbols Found</th><th>Supports Rise/Fall, Digits, or ACCU?</th><th>Contract Types Actually Offered</th></tr>
  </thead>
  <tbody>{flag_rows}</tbody>
</table>

<div class="section-title">Explicit Flag — Bear/Bull Market</div>
<table>
  <tbody><tr><td>{bb_html}</td></tr></tbody>
</table>
"""


async def run_audit_once(
    guard_path: str = DEFAULT_GUARD_PATH,
    json_path: str = DEFAULT_JSON_PATH,
    force: bool = False,
):
    """
    Runs run_audit() only once per deploy.

    Guard logic:
      - If FORCE_SYMBOL_AUDIT env var is set (1/true/yes) or force=True,
        always runs fresh regardless of the guard file.
      - Else, if guard_path already exists, skips the network run and
        instead loads json_path from disk (if present) so callers can
        still render the last known result.
      - On a successful fresh run, writes guard_path so future startups
        skip it. On failure, the guard file is NOT written, so the next
        deploy will retry automatically.

    To force a re-run manually: delete the guard file, or set
    FORCE_SYMBOL_AUDIT=1 in the environment for one deploy.

    Returns (output, ran, error):
      output: dict result (fresh or cached from disk), or None if neither
              is available
      ran:    True if a fresh network audit was actually executed
      error:  the exception from a failed fresh run, else None
    """
    force = force or os.environ.get("FORCE_SYMBOL_AUDIT", "").strip().lower() in ("1", "true", "yes")

    if not force and os.path.exists(guard_path):
        cached = None
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    cached = json.load(f)
            except Exception as exc:
                logger.warning(f"Symbol audit guard present but couldn't load cached {json_path}: {exc}")
        else:
            logger.info(
                f"Symbol audit guard present ({guard_path}) but no cached "
                f"{json_path} found — nothing to show until it's forced to re-run."
            )
        return cached, False, None

    try:
        output = await run_audit(json_path=json_path)
    except Exception as exc:
        logger.error(f"Symbol audit run failed (guard file not written, will retry next deploy): {exc}")
        return None, True, exc

    try:
        with open(guard_path, "w") as f:
            f.write(datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        logger.warning(f"Symbol audit succeeded but couldn't write guard file {guard_path}: {exc}")

    return output, True, None


async def main():
    """Standalone CLI entry point — unchanged behavior from before."""
    output = await run_audit()
    print("\n" + render_text_summary(output) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
