"""
target_guard.py – the cycle's stop rules: a manual PROFIT target and a LOW-BALANCE exit, as pure decision
functions plus ONE process-wide, once-only "trading stopped" claim.

Set on Render (Environment tab), BY HAND - the bot only ever READS this variable and never writes it:

    TARGET_PROFIT = 50      # absolute profit per cycle, in account currency ("stop the cycle at +50")
                            # empty / 0  ->  profit target OFF (still records a starting balance and trades,
                            #                never redeploys on profit; the low-balance suspend still applies)
                            # not a number / negative / rounds to 0.00  ->  ERROR logged, profit target OFF

The cycle (profit_cycle.py persists its state, bot_engine.py runs it):

  start   : connected + flat portfolio (checked against Deriv's own portfolio) -> the live balance is recorded as
            the cycle STARTING BALANCE (Render env var CYCLE_STARTING_BALANCE).
  trade   : net profit = live balance - starting balance.  Target balance (display) = start + TARGET_PROFIT.
  target  : net profit >= TARGET_PROFIT -> claim("target") (synchronous, once-only: no new proposal/buy can be sent
            after that) -> drain open contracts -> record the cycle -> CLEAR the starting balance (verified) ->
            redeploy -> the fresh process starts the next cycle.
  low bal.: live balance < the minimum stake while NO contract is open -> claim("low_balance") -> clear the starting
            balance (verified) -> suspend this Render service.  Top up, then Resume the service in Render.

claim() is shared by both terminal actions: whichever fires first wins, the other is inert (redeploy vs suspend can
never both happen).  is_reached() = "a terminal claim exists" and is what deriv_client's send gate and the engines'
_blocked() read, so every engine variant stops buying the instant the claim is made.

TARGET_BALANCE is DEPRECATED and ignored (a WARNING is logged at boot when it is still set).
"""

import logging
import math
import os
import threading
import time
from typing import Optional

import requests

import render_api

logger = logging.getLogger(__name__)

ACCOUNT_DECIMALS = 2          # the bot trades USD (contract currency is "USD"): cents
EPS = 1e-9

# boot_decision() results
RECORD = "record"             # no saved cycle: record the live balance as the starting balance and trade
RESUME = "resume"             # saved cycle below target: keep the saved starting balance and trade
TARGET_HIT = "target_hit"     # saved cycle already at/above target: drain -> clear -> redeploy, do not trade
LOW_BALANCE = "low_balance"   # balance below the minimum stake and the portfolio is flat: clear -> suspend
WAIT_FLAT = "wait_flat"       # a starting balance can only be recorded with NO contract open
WAIT_BALANCE = "wait_balance" # no usable live balance (None / NaN / negative / unconfirmed zero): never act on it

# claim() kinds
TARGET = "target"
LOW = "low_balance"

_lock = threading.Lock()
_terminal: Optional[str] = None
_terminal_at: float = 0.0
_terminal_balance: float = 0.0
_suspend_result: str = "not attempted"
_invalid_logged: set = set()


# ── environment (read on every call, never cached at import) ──────────────────
def _clean(raw) -> str:
    return (raw or "").strip().replace(",", "").lstrip("$").strip()


def money(x: float, decimals: int = ACCOUNT_DECIMALS) -> float:
    """Round a money amount to the account's decimals (cents)."""
    return round(float(x), decimals)


def parse_target_profit(raw, log: bool = True) -> Optional[float]:
    """TARGET_PROFIT text -> amount, or None (= profit target OFF) when unset / 0 / invalid.
    Tolerant like the old TARGET_BALANCE parser: whitespace, commas and a leading '$' are stripped.
    Invalid (not a number, negative, NaN/inf, or below one cent) is logged as an ERROR once per distinct value; the
    target is then OFF - never a guessed value, never a crash."""
    txt = _clean(raw)
    if not txt:
        return None
    try:
        val = float(txt)
    except ValueError:
        val = float("nan")
    if val == 0:
        return None
    if not math.isfinite(val) or val < 0 or money(val) < 10 ** -ACCOUNT_DECIMALS - EPS:
        if log and txt not in _invalid_logged:
            _invalid_logged.add(txt)
            logger.error("TARGET_PROFIT=%r is not a positive amount (>= 0.01) - the profit target is OFF "
                         "(fix the variable on Render; nothing was guessed)", txt)
        return None
    return money(val)


def target_profit() -> Optional[float]:
    """Current TARGET_PROFIT (read from the environment now), or None when the profit target is OFF."""
    return parse_target_profit(os.environ.get("TARGET_PROFIT"))


def deprecated_target_balance_set() -> bool:
    return bool(_clean(os.environ.get("TARGET_BALANCE")))


def _env_float(name: str, default: float, lo: float) -> float:
    try:
        v = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return max(lo, v) if math.isfinite(v) else default


def check_seconds() -> float:
    return _env_float("TARGET_CHECK_SECONDS", 3.0, 1.0)


def drain_timeout_seconds() -> float:
    return _env_float("CYCLE_DRAIN_TIMEOUT_SECONDS", 900.0, 1.0)


def cooldown_seconds() -> float:
    return _env_float("CYCLE_COOLDOWN_SECONDS", 0.0, 0.0)


# ── pure money decisions ──────────────────────────────────────────────────────
def _num(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def valid_balance(balance, zero_confirmed: bool = False) -> bool:
    """A balance the bot may act on: a finite number >= 0, where an exact 0 counts only when the caller confirmed it
    with repeated live reads (an unset/default 0.0 must never trigger a terminal action)."""
    b = _num(balance)
    if b is None or b < 0:
        return False
    return b > 0 or zero_confirmed


def net_profit(balance, start) -> Optional[float]:
    """Live balance - starting balance, rounded to cents; None when either side is unusable (no start recorded)."""
    b, s = _num(balance), _num(start)
    if b is None or s is None or s <= 0:
        return None
    return money(b - s)


def target_balance_for(start, tp: Optional[float]) -> Optional[float]:
    s = _num(start)
    if s is None or s <= 0 or tp is None:
        return None
    return money(s + tp)


def target_hit(balance, start, tp: Optional[float] = None) -> bool:
    """True when the profit target is ON and net profit >= TARGET_PROFIT (cent-rounded, tiny tolerance)."""
    tp = target_profit() if tp is None else tp
    net = net_profit(balance, start)
    return tp is not None and net is not None and net >= money(tp) - EPS


def low_balance(balance, min_stake, flat, zero_confirmed: bool = False) -> bool:
    """Low-balance exit rule: the balance is below the minimum stake AND the portfolio is verified flat (pending
    payouts of open contracts can make the balance look temporarily low, so it is never judged while one is open)."""
    ms = _num(min_stake)
    if flat is not True or ms is None or ms <= 0 or not valid_balance(balance, zero_confirmed):
        return False
    return money(balance) < money(ms) - EPS


def boot_decision(start, balance, tp: Optional[float], min_stake, flat, zero_confirmed: bool = False) -> str:
    """What to do right after connecting (pure).  `start`: saved CYCLE_STARTING_BALANCE (None/0 = no cycle);
    `tp`: TARGET_PROFIT or None (off); `flat`: Deriv's own portfolio is verified empty (True) / not (False or None)."""
    if not valid_balance(balance, zero_confirmed):
        return WAIT_BALANCE
    if low_balance(balance, min_stake, flat, zero_confirmed):
        return LOW_BALANCE
    s = _num(start)
    if s is not None and s > 0:
        return TARGET_HIT if (tp is not None and target_hit(balance, s, tp)) else RESUME
    return RECORD if flat is True else WAIT_FLAT


# ── once-only terminal claim (thread-safe) ────────────────────────────────────
def is_reached() -> bool:
    """True once ANY terminal action (profit target or low balance) has been claimed: trading is stopped."""
    return _terminal is not None


def terminal_kind() -> Optional[str]:
    return _terminal


def claim(kind: str, balance: float = 0.0) -> bool:
    """Claim the terminal action `kind` (TARGET or LOW). True only for the FIRST caller of either kind, so the drain /
    clear / redeploy-or-suspend sequence runs exactly once and the other kind is inert. Safe from sync code inside the
    event loop (a plain lock, no await)."""
    global _terminal, _terminal_at, _terminal_balance
    with _lock:
        if _terminal is not None:
            return False
        _terminal, _terminal_at, _terminal_balance = kind, time.time(), float(balance or 0.0)
    logger.warning("CYCLE STOP (%s): live balance %.2f - no new proposal or buy can be sent from now on",
                   kind, float(balance or 0.0))
    return True


def mark_reached(balance: float) -> bool:
    """Profit target hit (see claim)."""
    return claim(TARGET, balance)


def mark_low_balance(balance: float) -> bool:
    """Low-balance exit (see claim)."""
    return claim(LOW, balance)


def release(kind: str) -> bool:
    """Undo a LOW claim made a moment ago when the post-claim re-check shows the balance recovered (a straggler win
    landed). Only the low-balance claim can be released, and only by its owner before anything irreversible ran."""
    global _terminal
    with _lock:
        if _terminal != kind or kind != LOW:
            return False
        _terminal = None
    logger.warning("CYCLE STOP (%s) RELEASED: balance recovered above the minimum stake - trading resumes", kind)
    return True


def snapshot() -> dict:
    return {
        "target_profit":   target_profit(),
        "reached":         is_reached(),
        "kind":            _terminal,
        "reached_at":      _terminal_at,
        "reached_balance": _terminal_balance,
        "suspend_result":  _suspend_result,
    }


# ── Render suspend ────────────────────────────────────────────────────────────
def suspend_service() -> bool:
    """Suspend THIS Render service (POST /v1/services/{id}/suspend). Blocking - call from a thread
    (asyncio.to_thread). Returns True if Render accepted it. Must only be called AFTER the cycle's starting balance
    was cleared and verified (suspension kills the process)."""
    global _suspend_result
    if not render_api.api_configured():
        _suspend_result = "no RENDER_API_KEY / service id - service left running (idle)"
        logger.error(f"CYCLE: cannot suspend Render service: {_suspend_result}")
        return False
    url = f"{render_api.API_BASE}/services/{render_api.service_id()}/suspend"
    headers = {"Authorization": f"Bearer {render_api.api_key()}", "Accept": "application/json"}
    try:
        r = requests.post(url, headers=headers, timeout=20)
        if r.status_code in (200, 201, 202, 204):
            _suspend_result = "suspended"
            logger.warning("CYCLE: Render service SUSPEND accepted - process will be stopped by Render")
            return True
        _suspend_result = f"Render refused: HTTP {r.status_code} {r.text[:150]}"
        logger.error(f"CYCLE: {_suspend_result}")
    except Exception as exc:
        _suspend_result = f"suspend request failed: {exc}"
        logger.error(f"CYCLE: {_suspend_result}")
    return False
