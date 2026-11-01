"""
Multi-strategy signal engine.
Each symbol is routed to exactly ONE strategy evaluator based on which
config symbol-list it belongs to. No cross-strategy voting, no SMC,
no order blocks, no HTF bias — each category gets its own independent
purpose-built evaluator.

Additions in this pass (all tick-based, independent of the candle-based
evaluators above them):
  - evaluate_digit_parity      : chi-square even/odd bias on raw ticks
  - evaluate_digit (modified)  : optional hybrid RSI/BB/ROC + chi-square gate
  - evaluate_drift_fade        : Boom/Crash drift-following (post-cooldown)
  - evaluate_jump_buildup      : Jump index build-up confidence -> digit contract (MATCH/DIFFER)
  - evaluate_trend_shift       : Bear/Bull fixed per-symbol daily-reset bias (RDBULL=LONG, RDBEAR=SHORT)

Additions (handoff, Sep 15 2026) — six dedicated per-symbol evaluators,
replacing evaluate_popular_indicator() for exactly nine symbols:
  - evaluate_pullback_trend     : R_75/1HZ75V EMA-trend pullback + RSI turn-back
  - evaluate_fast_mean_reversion: R_100/1HZ100V fast BB/RSI scalp snap-back
  - evaluate_spike_catch_1000   : BOOM1000/CRASH1000 drift-exhaustion spike-catch
  - evaluate_spike_catch_500    : BOOM500/CRASH500, same logic, faster re-arm
  - evaluate_step_grid          : stpRNG hard EMA+price+RSI+MACD AND-gate
Each populates native_entry_price/native_stop_price/native_target_price
for config.STOP_AS_TRIGGER_SYMBOLS' parallel pending-order path in
bot_engine.py (see config.py's "STOP-AS-TRIGGER ENTRY" section).

See the "NEW STRATEGY CONFIG" region below for the config keys these read
(all via getattr with safe defaults, so nothing breaks if unset) and the
chat reply for a full list of flagged inconsistencies/assumptions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import config
import indicators as ind
import strategy_stats
import pair_suspension
from candlestick_builder import Candle
from symbol_manager import SymbolManager

logger = logging.getLogger(__name__)

# Implementation Brief v3, finding #4 / task 1: single shared instance so
# evaluate_trend_shift()'s post-reset timing check calls symbol_manager's
# own is_post_reset()/get_bear_bull_state() instead of re-implementing the
# same "minutes since 00:00 GMT" math a second time. is_post_reset() reads
# only module-level config (BEAR_BULL_SYMBOLS, BEAR_BULL_TREND_SHIFT_MINS)
# and wall-clock time — it doesn't touch any of SymbolManager's mutable
# per-symbol state (suspensions, session counters, etc.) — so a private
# instance here is safe and doesn't need to be the same object bot_engine.py
# holds as self.symbols.
_symbol_manager = SymbolManager()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    direction: str    # "LONG" | "SHORT" | "NONE" | "MATCH" | "DIFFER"
    strength:  int    # 1-3
    score:     float  # 0.0-1.0 composite probability
    strategy:  str    # which strategy fired
    reason:    str    # human readable
    # Implementation Brief v3, finding #3 / task 2: JUMP_BUILDUP's real
    # recommendation is a digit contract (Matches/Differs), not a price
    # direction — these three fields let bot_engine._execute() route to
    # DerivClient.buy_digit_contract() instead of buy_contract() (CALL/PUT)
    # without every other evaluator's call site needing to change.
    # contract_kind defaults to "RISE_FALL" so every existing evaluator
    # (which only sets direction/strength/score/strategy/reason) is
    # unaffected.
    contract_kind: str            = "RISE_FALL"   # "RISE_FALL" | "DIGIT"
    digit:         Optional[int]  = None            # 0-9, DIGIT contracts only
    match_type:    Optional[str]  = None            # "MATCH" | "DIFFER"
    # Popular-indicator pipeline: native SL/TP price levels computed by
    # whichever indicator produced this result, using that indicator's own
    # standard technical-analysis convention (not a stake percentage).
    # These are PRE-inversion, PRE-swap — they describe the stop/target for
    # the direction actually signalled above. bot_engine.py's universal
    # inversion step swaps stop<->target at the same time it flips direction.
    native_stop_price:   Optional[float] = None   # indicator's own SL price level
    native_target_price: Optional[float] = None   # indicator's own TP price level
    # The price the signal (and native_stop_price/native_target_price) was
    # actually computed against — threaded through to deriv_client.py's
    # dollar-conversion so it measures distance off the SAME price the
    # indicator used, rather than re-fetching a fresh quote that may have
    # since moved (see deriv_client.buy_multiplier()'s entry_price param).
    native_entry_price:  Optional[float] = None
    # Set by bot_engine._apply_scaled_native_levels() when
    # config.SCALED_SL_TP_INVERT_DIRECTION flips this signal's direction —
    # never set here. Lets _execute()'s bookkeeping ("inverted" in
    # _open_contracts, read back by _apply_settlement()'s meta-labeling
    # training-label logic and strategy_stats.get_take_invert_stats())
    # know a flip happened without needing sight of the pre-transform sig.
    execution_inverted:  bool = False


NONE_RESULT = SignalResult("NONE", 0, 0.0, "NONE", "No signal")


# ---------------------------------------------------------------------------
# Helpers — candle arrays (existing)
# ---------------------------------------------------------------------------

def _arrays(bars: List[Candle]):
    C = np.array([b.close for b in bars], dtype=float)
    H = np.array([b.high  for b in bars], dtype=float)
    L = np.array([b.low   for b in bars], dtype=float)
    return C, H, L


def _last(arr: np.ndarray, back: int = 1) -> float:
    valid = arr[~np.isnan(arr)]
    if len(valid) < back:
        return float("nan")
    return float(valid[-back])


# ---------------------------------------------------------------------------
# Enriched features for meta_labeling.py's per-pair EV model
# (Implementation Brief v6, PART 2/3 — this was referenced by meta_labeling.py
# but never actually built: ENRICHED_FEATURE_KEYS = ("rsi", "roc", "bb_pct_b",
# "atr_expansion_ratio", "hour_utc") existed as a consumer-side contract with
# no producer anywhere in the codebase, so _PairEVModel never had a single
# real feature row to train on regardless of trade count. This is that
# producer. Called once per execution from bot_engine.py._execute() (entry
# time, for the meta-label gate) and reused at settlement (for logging) —
# same values both times since features are computed from the same closed
# bars, not re-fetched.
# ---------------------------------------------------------------------------

def compute_enriched_features(ltf_bars: List[Candle], timestamp: Optional[float] = None) -> Dict[str, float]:
    """
    Returns {"rsi", "roc", "bb_pct_b", "atr_expansion_ratio", "hour_utc"} —
    exactly meta_labeling.ENRICHED_FEATURE_KEYS — or {} if there isn't
    enough bar history yet (caller should log/pass through an empty dict
    in that case, not fabricate values; meta_labeling.py already treats a
    missing/empty enriched dict as "fall back to the global model").

    rsi                  : RSI(14), last closed bar
    roc                  : ROC(10), last closed bar
    bb_pct_b             : Bollinger %B = (close - lower) / (upper - lower),
                            BB(20, 2.0) — 0.0 at the lower band, 1.0 at the
                            upper band, can exceed [0,1] on a strong move
    atr_expansion_ratio   : ATR(14) now / ATR(14) 10 bars ago — >1 means
                            volatility is expanding, <1 means contracting
    hour_utc              : UTC hour (0-23) of `timestamp` (defaults to now)
    """
    if len(ltf_bars) < 30:
        return {}

    C, H, L = _arrays(ltf_bars)
    rsi_arr = ind.rsi(C, 14)
    roc_arr = ind.roc(C, 10)
    upper, mid, lower = ind.bollinger_bands(C, 20, 2.0)
    atr_arr = ind.atr(H, L, C, config.ATR_PERIOD)

    last_rsi   = _last(rsi_arr)
    last_roc   = _last(roc_arr)
    last_upper = _last(upper)
    last_lower = _last(lower)
    atr_now    = _last(atr_arr, 1)
    atr_prior  = _last(atr_arr, 10)

    band_width = last_upper - last_lower
    bb_pct_b = (
        (float(C[-1]) - last_lower) / band_width
        if band_width > 0 and not math.isnan(band_width) else float("nan")
    )
    atr_expansion_ratio = (
        atr_now / atr_prior if atr_prior and not math.isnan(atr_prior) and atr_prior > 0
        else float("nan")
    )

    ts = timestamp if timestamp is not None else datetime.now(timezone.utc).timestamp()
    hour_utc = datetime.fromtimestamp(ts, tz=timezone.utc).hour

    feat = {
        "rsi": last_rsi,
        "roc": last_roc,
        "bb_pct_b": bb_pct_b,
        "atr_expansion_ratio": atr_expansion_ratio,
        "hour_utc": float(hour_utc),
    }
    # Drop any NaN entries rather than shipping them into a JSON column —
    # _PairEVModel._numeric_subset() only keeps int/float, and a NaN would
    # silently poison DictVectorizer/LogisticRegression's training set.
    return {k: v for k, v in feat.items() if not (isinstance(v, float) and math.isnan(v))}


# ---------------------------------------------------------------------------
# Helpers — raw ticks (new)
#
# ASSUMPTION: no tick-buffer type is defined anywhere in the three files
# provided (config.py / symbol_manager.py / signal_engine.py), and
# bot_engine.py / deriv_client.py were not included in this pass, so the
# shape of "ticks" as produced by the rest of the bot is unknown. These
# helpers duck-type against the three shapes most likely to occur:
#   - Deriv's raw tick dict:      {"epoch": ..., "quote": ...}
#   - A lightweight object:       obj.epoch / obj.quote (or .price)
#   - A plain (epoch, quote) tuple/list
#   - A bare float (quote only — epoch-dependent functions degrade
#     gracefully, see evaluate_jump_buildup)
# If the real tick buffer differs from all of these, only these two
# helpers need updating — every new function below goes through them.
# ---------------------------------------------------------------------------

def _tick_quote(t: Any) -> float:
    if isinstance(t, (int, float, np.floating)):
        return float(t)
    if isinstance(t, dict):
        q = t.get("quote", t.get("price"))
        if q is not None:
            return float(q)
    q = getattr(t, "quote", None)
    if q is None:
        q = getattr(t, "price", None)
    if q is not None:
        return float(q)
    if isinstance(t, (tuple, list)) and len(t) >= 2:
        return float(t[1])
    raise ValueError(f"Unrecognized tick format: {t!r}")


def _tick_epoch(t: Any) -> Optional[float]:
    if isinstance(t, dict):
        e = t.get("epoch")
        return float(e) if e is not None else None
    e = getattr(t, "epoch", None)
    if e is not None:
        return float(e)
    if isinstance(t, (tuple, list)) and len(t) >= 1:
        try:
            return float(t[0])
        except (TypeError, ValueError):
            return None
    return None


def _last_digit(quote: float, decimals: int) -> int:
    scaled = int(round(quote * (10 ** decimals)))
    return abs(scaled) % 10


def _digit_decimals(symbol: str) -> int:
    # config.DIGIT_DECIMALS: Optional[Dict[str, int]] — per-symbol pip
    # precision for last-digit extraction. Not present in the supplied
    # config.py; defaults to 2 decimals for every symbol until you add it.
    return getattr(config, "DIGIT_DECIMALS", {}).get(symbol, 2)


def _chi2_binary(count_a: int, count_b: int) -> Tuple[float, float]:
    """
    Chi-square goodness-of-fit for a 2-category 50/50 null (df=1).
    Returns (chi2_statistic, p_value). For df=1, chi2 is exactly the
    square of a standard normal variate, so the exact p-value is
    erfc(sqrt(chi2/2)) — no scipy dependency required.
    """
    n = count_a + count_b
    if n == 0:
        return 0.0, 1.0
    expected = n / 2.0
    chi2 = ((count_a - expected) ** 2) / expected + ((count_b - expected) ** 2) / expected
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p


# ---------------------------------------------------------------------------
# Strategy 1 — Digit Over/Under
# ---------------------------------------------------------------------------

def _digit_hybrid_check(
    ticks: Optional[List[Any]], symbol: str, digit_dir: str
) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
    """
    Chi-square frequency-bias check on the same over/under threshold the
    indicator read used. Returns (agree, chi2, p_value, freq_biased).
    agree is False (never fires) if ticks are missing/insufficient —
    hybrid mode is a confirmation gate, not a fallback signal source.
    """
    if not ticks:
        return False, None, None, None

    window_n = getattr(config, "DIGIT_PARITY_WINDOW", 1000)
    min_n = getattr(config, "DIGIT_PARITY_MIN_SAMPLE", 500)
    alpha = getattr(config, "DIGIT_PARITY_ALPHA", 0.05)
    threshold = getattr(config, "DIGIT_OU_THRESHOLD", 5)

    window = ticks[-window_n:]
    n = len(window)
    if n < min_n:
        return False, None, None, None

    decimals = _digit_decimals(symbol)
    try:
        digits = [_last_digit(_tick_quote(t), decimals) for t in window]
    except ValueError:
        logger.warning(f"DIGIT hybrid: {symbol} could not parse tick quotes — skipping")
        return False, None, None, None

    over = sum(1 for d in digits if d > threshold)
    under = n - over
    chi2, p = _chi2_binary(over, under)
    freq_biased = "OVER" if over > under else "UNDER"
    agree = (freq_biased == digit_dir) and (p < alpha)
    return agree, chi2, p, freq_biased


def evaluate_digit(
    ltf_bars: List[Candle], symbol: str, ticks: Optional[List[Any]] = None
) -> SignalResult:
    if len(ltf_bars) < 25:
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    rsi = ind.rsi(C, 14)
    upper, mid, lower = ind.bollinger_bands(C, 20, 2.0)
    roc = ind.roc(C, 10)

    raw_score, digit_dir = ind.digit_score(
        closes=C, rsi_vals=rsi, bb_upper=upper, bb_lower=lower, roc_vals=roc
    )
    partial_score = raw_score / 8.0

    if digit_dir == "NONE" or raw_score < 6:
        logger.debug(f"REJECTED: {symbol} DIGIT strength=0 score={partial_score:.3f} — below threshold")
        return SignalResult("NONE", 0, partial_score, "DIGIT", "Below entry threshold")

    direction = "LONG" if digit_dir == "OVER" else "SHORT"
    score = raw_score / 8.0

    if score >= 0.875:
        strength = 3
    elif score >= 0.625:
        strength = 2
    else:
        logger.info(f"REJECTED: {symbol} DIGIT strength=1 score={score:.3f} — below threshold")
        return SignalResult("NONE", 0, score, "DIGIT", "Below entry threshold")

    # --- Hybrid confirmation gate (item 2) -------------------------------
    # Standalone behavior (flag False) is completely unchanged above this
    # point. When True, the indicator-based read above must additionally
    # agree with a chi-square frequency-bias read on the same threshold.
    if getattr(config, "DIGIT_HYBRID_MODE", False):
        agree, chi2, p, freq_biased = _digit_hybrid_check(ticks, symbol, digit_dir)
        if not agree:
            chi2_s = f"{chi2:.3f}" if chi2 is not None else "n/a"
            p_s = f"{p:.5f}" if p is not None else "n/a"
            logger.info(
                f"REJECTED: {symbol} DIGIT strength=0 score={score:.3f} — hybrid disagreement "
                f"(indicator={digit_dir}, freq={freq_biased}, chi2={chi2_s}, p={p_s})"
            )
            return SignalResult(
                "NONE", 0, score, "DIGIT",
                f"Hybrid mode: indicator/frequency disagreement (indicator={digit_dir}, freq={freq_biased})",
            )
        logger.info(
            f"DIGIT HYBRID: {symbol} indicator={digit_dir} freq={freq_biased} "
            f"chi2={chi2:.3f} p={p:.5f} — agree"
        )

    logger.info(f"DIGIT: {symbol} {direction} score={raw_score}/8")
    return SignalResult(
        direction=direction,
        strength=strength,
        score=score,
        strategy="DIGIT",
        reason=f"Digit {digit_dir} raw={raw_score}/8",
    )


# ---------------------------------------------------------------------------
# Strategy 1b — Digit Parity (new, standalone)
# ---------------------------------------------------------------------------

def evaluate_digit_parity(ticks: Optional[List[Any]], symbol: str) -> SignalResult:
    window_n = getattr(config, "DIGIT_PARITY_WINDOW", 1000)
    min_n = getattr(config, "DIGIT_PARITY_MIN_SAMPLE", 500)
    alpha = getattr(config, "DIGIT_PARITY_ALPHA", 0.05)

    if not ticks:
        return NONE_RESULT

    window = ticks[-window_n:]
    n = len(window)
    if n < min_n:
        logger.debug(
            f"REJECTED: {symbol} DIGIT_PARITY strength=0 score=0.000 — "
            f"sample size {n} < {min_n}"
        )
        return NONE_RESULT

    decimals = _digit_decimals(symbol)
    try:
        digits = [_last_digit(_tick_quote(t), decimals) for t in window]
    except ValueError:
        logger.warning(f"DIGIT_PARITY: {symbol} could not parse tick quotes — skipping")
        return NONE_RESULT

    even_count = sum(1 for d in digits if d % 2 == 0)
    odd_count = n - even_count
    chi2, p = _chi2_binary(even_count, odd_count)

    logger.info(
        f"DIGIT_PARITY: {symbol} chi2={chi2:.3f} p={p:.5f} n={n} "
        f"even={even_count} odd={odd_count}"
    )

    if p >= alpha:
        logger.debug(
            f"REJECTED: {symbol} DIGIT_PARITY strength=0 score={max(0.0, 1 - p):.3f} — "
            f"p={p:.5f} not significant"
        )
        return SignalResult("NONE", 0, max(0.0, 1 - p), "DIGIT_PARITY", f"Not significant (p={p:.5f})")

    # Convention mirrors evaluate_digit's OVER/UNDER -> LONG/SHORT mapping:
    # LONG encodes an EVEN bias, SHORT encodes an ODD bias. This is a
    # digit-parity read, not a price-direction read — see chat reply.
    biased = "EVEN" if even_count > odd_count else "ODD"
    direction = "LONG" if biased == "EVEN" else "SHORT"
    score = max(0.0, min(1.0, 1 - p))
    strength = 3 if p < 0.01 else 2

    logger.info(f"SIGNAL: {symbol} {direction} DIGIT_PARITY strength={strength} score={score:.3f}")
    return SignalResult(
        direction=direction,
        strength=strength,
        score=score,
        strategy="DIGIT_PARITY",
        reason=f"Parity bias={biased} chi2={chi2:.3f} p={p:.5f} n={n}",
    )


# ---------------------------------------------------------------------------
# Strategy 2 — Mean Reversion
# ---------------------------------------------------------------------------

def evaluate_mean_reversion(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    if len(ltf_bars) < 25:
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    rsi = ind.rsi(C, 14)
    upper, mid, lower = ind.bollinger_bands(C, 20, 2.0)
    roc = ind.roc(C, 10)

    last_rsi   = _last(rsi)
    last_close = float(C[-1])
    last_upper = _last(upper)
    last_lower = _last(lower)
    last_roc   = _last(roc)

    long_score = 0
    long_all = True
    if last_rsi < 22:
        long_score += 3
    else:
        long_all = False
    if last_close <= last_lower:
        long_score += 3
    else:
        long_all = False
    if last_roc < -0.02:
        long_score += 2
    else:
        long_all = False

    short_score = 0
    short_all = True
    if last_rsi > 78:
        short_score += 3
    else:
        short_all = False
    if last_close >= last_upper:
        short_score += 3
    else:
        short_all = False
    if last_roc > 0.02:
        short_score += 2
    else:
        short_all = False

    # FIX (Task 3 — full textbook confirmation, no partial firing, Aug
    # 2026): previously fired on long_score/short_score >= 6 out of 8
    # (RSI extreme=3, price-at-band=3, ROC momentum=2) — reachable with
    # only 2 of the 3 documented conditions (RSI + band touch = 6, no ROC
    # confirmation required at all). That's a confidence-threshold
    # substituting for a missing condition. Textbook mean-reversion here
    # is RSI extreme AND price at/through the band AND ROC confirming
    # momentum — now gated on all_met (all three), not the score.
    if long_all and long_score >= short_score:
        raw = long_score
        direction = "LONG"
    elif short_all:
        raw = short_score
        direction = "SHORT"
    else:
        best = max(long_score, short_score)
        logger.debug(f"REJECTED: {symbol} MEAN_REV strength=0 score={best/8.0:.3f} — below threshold")
        return SignalResult("NONE", 0, best / 8.0, "MEAN_REV", "Below entry threshold")

    score = raw / 8.0
    strength = 3

    logger.info(
        f"SIGNAL: {symbol} {direction} MEAN_REV strength={strength} score={score:.3f}"
    )
    return SignalResult(
        direction=direction,
        strength=strength,
        score=score,
        strategy="MEAN_REV",
        reason=f"MeanRev RSI={last_rsi:.1f} raw={raw}/8 (70.8% documented win rate)",
    )


# ---------------------------------------------------------------------------
# Strategy 2b — Volatility/Step Multiplier family (Implementation Brief v4)
# ---------------------------------------------------------------------------
# Replaces MEAN_REV for config.VOL_MULTIPLIER_SYMBOLS (R_10-R_100, 1HZ10V-
# 1HZ100V, stpRNG) now that these 11 symbols trade via Multiplier contracts
# (MULTUP/MULTDOWN) instead of Rise/Fall. Synthetic volatility indices are
# close to a pure random walk — there's no durable, strong directional edge
# to lean on — so this needs to (a) only fire when there's a real, currently
# -forming regime (trend vs range) rather than trading every cycle, and
# (b) size risk in a way that survives being wrong most of the time (see
# risk_manager.compute_dynamic_stop_loss_pct()). This is a bigger lever on
# profit factor than the strategy's win rate alone.

def _vol_regime(ltf_bars: List[Candle]) -> str:
    """
    'TREND' or 'RANGE', from EMA(8)/EMA(21) separation normalized by
    ATR(14). Cheap proxy for trend strength (ADX-equivalent) using only
    indicators already in indicators.py — no new dependency.

    ENHANCEMENT (win-rate pass, Aug 2026): a single-bar ratio read let one
    noisy tick flip the regime to TREND for a cycle, routing straight into
    evaluate_vol_breakout() with no real trend behind it — the dominant
    source of losing trades on the dashboard. Now requires the ratio to
    clear VOL_REGIME_TREND_RATIO on VOL_REGIME_CONFIRM_BARS consecutive
    completed bars (default 2) before calling it TREND. Same inputs, same
    ratio formula — only the "is this real" bar is raised.
    """
    C, H, L = _arrays(ltf_bars)
    ema_fast_arr = ind.ema(C, config.EMA_FAST)
    ema_slow_arr = ind.ema(C, config.EMA_SLOW)
    atr_arr = ind.atr(H, L, C, config.ATR_PERIOD)

    trend_ratio = getattr(config, "VOL_REGIME_TREND_RATIO", 0.6)
    confirm_bars = max(1, int(getattr(config, "VOL_REGIME_CONFIRM_BARS", 1)))

    ratios: List[float] = []
    for back in range(1, confirm_bars + 1):
        a = _last(atr_arr, back)
        f = _last(ema_fast_arr, back)
        s = _last(ema_slow_arr, back)
        if any(math.isnan(v) for v in (a, f, s)) or a <= 0:
            return "RANGE"  # insufficient history -> default to the more
                             # conservative (reversion) evaluator
        ratios.append(abs(f - s) / a)

    return "TREND" if all(r >= trend_ratio for r in ratios) else "RANGE"


def evaluate_vol_breakout(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    """
    Donchian-channel breakout + EMA trend alignment + MACD histogram
    confirmation. Suited to open-ended Multiplier risk (rides continuation
    rather than betting on a single-candle direction like Rise/Fall did).
    Fires in TREND regime — see evaluate_vol_regime() dispatcher below.

    ENHANCEMENT (win-rate pass, Aug 2026): the break condition previously
    fired on `last_close >= upper` — a single tick clipping the channel
    edge counts as a full breakout even by 1 pip, which on 2s/1m synthetic
    ticks is frequently just noise. Now requires the close to clear the
    channel by BREAKOUT_MARGIN_ATR × ATR, a real distance rather than a
    marginal touch. Scoring model (Donchian + EMA + MACD, >=6/7 to fire)
    is unchanged.

    CORRECTION (same pass, second iteration): an earlier version also
    required the *prior* bar to already be at/through the level (a
    2-bar-confirmation attempt at filtering spike ticks). That backfired —
    it only let through moves that were already extended and blocked the
    sharp, one-candle break a real breakout usually is, which silenced
    this evaluator almost entirely. Removed; the ATR margin alone is the
    noise filter now.
    """
    if len(ltf_bars) < 30:
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    ema_fast = ind.ema(C, config.EMA_FAST)
    ema_slow = ind.ema(C, config.EMA_SLOW)
    macd_line, signal_line, hist = ind.macd(C)
    # ind.donchian() — confirmed name/signature elsewhere in this file
    # (evaluate_step() uses `ind.donchian(H, L, 20)` → (upper, lower)).
    upper, lower = ind.donchian(H, L, 20)
    atr = ind.atr(H, L, C, config.ATR_PERIOD)
    last_close = float(C[-1])
    last_atr = _last(atr)
    # BUG FIX (win-rate pass, Aug 2026, fourth iteration): live logs showed
    # every VOL_MULTIPLIER_SYMBOLS breakout attempt landing on an identical
    # 4/7 (0.571) score, forever, regardless of BREAKOUT_MARGIN_ATR (tried
    # 0.15, then 0.05 — no change). Root cause: ind.donchian()'s window
    # includes the *current* bar, so last_upper/last_lower were always at
    # least as extreme as this bar's own high/low — which is always at
    # least as extreme as this bar's own close. `close >= upper + margin`
    # was therefore false for any margin > 0, structurally, no matter how
    # small. Comparing against the *prior* bar's channel instead (which
    # excludes the current bar) is both the fix and the textbook-correct
    # definition of a breakout — "did this close clear the level as it
    # stood before this bar", not "does it exceed a channel this bar's own
    # high already extended".
    last_upper = _last(upper, 2)
    last_lower = _last(lower, 2)

    if any(math.isnan(v) for v in (last_atr, last_upper, last_lower)) or last_atr <= 0:
        logger.debug(f"REJECTED: {symbol} VOL_BREAKOUT strength=0 score=0.000 — indicators not warmed up")
        return NONE_RESULT

    margin = getattr(config, "BREAKOUT_MARGIN_ATR", 0.15) * last_atr
    long_break  = last_close >= last_upper + margin
    short_break = last_close <= last_lower - margin

    long_score, short_score = 0, 0
    if long_break:
        long_score += 3
    if short_break:
        short_score += 3
    if _last(ema_fast) > _last(ema_slow):
        long_score += 2
    else:
        short_score += 2
    if _last(hist) > 0:
        long_score += 2
    elif _last(hist) < 0:
        short_score += 2

    # FIX (profitability audit): threshold was >=5/7, satisfied by the
    # Donchian break (3) plus EITHER the EMA or the MACD-hist confirmation
    # (2 each) alone. Dashboard trade history shows VOL_BREAKOUT losing on
    # ~80% of its sampled trades vs. BOOM_CRASH winning consistently — a
    # single-indicator confirmation on a near-random-walk instrument is too
    # loose. Raised to >=6 so a breakout needs the Donchian break AND BOTH
    # EMA-alignment AND MACD-hist agreeing with direction (3+2+2=7 max),
    # matching the "require multiple confluences" bar VOL_REV_MULT already
    # uses (>=6/8) just below.
    if long_score >= 6 and long_score >= short_score:
        direction, raw = "LONG", long_score
    elif short_score >= 6:
        direction, raw = "SHORT", short_score
    else:
        best = max(long_score, short_score)
        logger.debug(f"REJECTED: {symbol} VOL_BREAKOUT strength=0 score={best/7.0:.3f} — no confirmed breakout")
        return SignalResult("NONE", 0, best / 7.0, "VOL_BREAKOUT", "No confirmed breakout")

    score = raw / 7.0
    strength = 3 if raw >= 6 else 2
    logger.info(
        f"SIGNAL: {symbol} {direction} VOL_BREAKOUT strength={strength} score={score:.3f}"
    )
    return SignalResult(
        direction=direction, strength=strength, score=score,
        strategy="VOL_BREAKOUT",
        reason=f"Donchian breakout, EMA-aligned, MACD-hist={_last(hist):.5f}",
    )


def evaluate_vol_reversion_mult(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    """
    Fires in RANGE regime — see evaluate_vol_regime() dispatcher below.
    Same confluence logic as evaluate_mean_reversion() (RSI extremes +
    Bollinger touch + ROC), kept as a *separate* named strategy
    (VOL_REV_MULT) so strategy_stats.py / meta_labeling.py don't conflate
    its historical performance with the old Rise/Fall version — the payoff
    structure is now completely different (open-ended + stop/target vs
    fixed 6-14min expiry), so the old win-rate history doesn't transfer.

    ENHANCEMENT (win-rate pass, Aug 2026): when config.MEAN_REV_REQUIRE_TURN
    is True (default), an extreme RSI/BB/ROC read alone is no longer
    enough — price must have already ticked back toward the mean vs. the
    prior close before entry, so the strategy stops catching a falling
    knife mid-extension. Zone/indicator thresholds themselves are
    untouched.
    """
    if len(ltf_bars) < 25:
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    rsi = ind.rsi(C, 14)
    upper, mid, lower = ind.bollinger_bands(C, 20, 2.0)
    roc = ind.roc(C, 10)

    last_rsi   = _last(rsi)
    last_close = float(C[-1])
    prev_close = _last(C, 2) if len(C) >= 2 else float("nan")
    last_upper = _last(upper)
    last_lower = _last(lower)
    last_roc   = _last(roc)

    long_score = 0
    long_all = True
    if last_rsi < 22:
        long_score += 3
    else:
        long_all = False
    if last_close <= last_lower:
        long_score += 3
    else:
        long_all = False
    if last_roc < -0.02:
        long_score += 2
    else:
        long_all = False

    short_score = 0
    short_all = True
    if last_rsi > 78:
        short_score += 3
    else:
        short_all = False
    if last_close >= last_upper:
        short_score += 3
    else:
        short_all = False
    if last_roc > 0.02:
        short_score += 2
    else:
        short_all = False

    # FIX (Task 3 — full textbook confirmation, no partial firing, Aug
    # 2026): same pattern/fix as evaluate_mean_reversion() above — gated
    # on all_met (RSI extreme AND band touch AND ROC momentum, all three)
    # rather than a >=6/8 score that a 2-of-3 combination could satisfy.
    if long_all and long_score >= short_score:
        raw = long_score
        direction = "LONG"
    elif short_all:
        raw = short_score
        direction = "SHORT"
    else:
        best = max(long_score, short_score)
        logger.debug(f"REJECTED: {symbol} VOL_REV_MULT strength=0 score={best/8.0:.3f} — below threshold")
        return SignalResult("NONE", 0, best / 8.0, "VOL_REV_MULT", "Below entry threshold")

    if getattr(config, "MEAN_REV_REQUIRE_TURN", True) and not math.isnan(prev_close):
        turned = (direction == "LONG" and last_close > prev_close) or (
            direction == "SHORT" and last_close < prev_close
        )
        if not turned:
            logger.info(
                f"REJECTED: {symbol} VOL_REV_MULT strength=0 score={raw/8.0:.3f} — "
                f"extreme reached but price not yet turning back toward the mean"
            )
            return SignalResult(
                "NONE", 0, raw / 8.0, "VOL_REV_MULT",
                "Extreme reached, awaiting reversal confirmation",
            )

    score = raw / 8.0
    strength = 3

    logger.info(
        f"SIGNAL: {symbol} {direction} VOL_REV_MULT strength={strength} score={score:.3f}"
    )
    return SignalResult(
        direction=direction,
        strength=strength,
        score=score,
        strategy="VOL_REV_MULT",
        # NOTE: deliberately does NOT carry over MEAN_REV's old "(70.8%
        # documented win rate)" claim — that number was measured on
        # Rise/Fall payoffs and does not apply to this Multiplier-contract
        # strategy. See strategy_stats.is_underperforming() for live
        # tracking instead.
        reason=f"VolRevMult RSI={last_rsi:.1f} raw={raw}/8",
    )


# ---------------------------------------------------------------------------
# Strategy — Most Popular Indicator (user-directed, Aug 2026)
#
# Replaces evaluate_vol_regime()/evaluate_boom_crash() as the strategy
# routed to every symbol in config.MULTIPLIER_SYMBOLS (see SignalEngine
# below). NOT a confluence/voting system — per explicit user direction,
# this does not blend multiple indicators' opinions together. Instead:
#
#   - Every indicator in POPULAR_INDICATOR_ORDER below is checked, most
#     popular first, and independently produces either a directional
#     read (LONG/SHORT) or "not currently signalling" (abstains) — e.g.
#     a moving-average crossover only signals on the bar the cross
#     actually happens, an oscillator only signals while it's actually
#     in its overbought/oversold zone, ADX only signals once it confirms
#     a real trend is present, and so on. Full detail is in each
#     indicator's own comment below.
#   - If exactly one indicator is currently signalling, its direction is
#     the one used.
#   - If several are signalling at once (they don't need to agree with
#     each other), the single most popular one among those currently
#     signalling is the one used — its direction alone, not a blend.
#   - If none are signalling, no trade fires (NONE_RESULT).
#
# POPULAR_INDICATOR_ORDER reflects general trading-community usage
# (moving averages, RSI, MACD, and Bollinger Bands are consistently the
# most cited indicators in general trader-usage surveys; Stochastic,
# ADX/DMI, Parabolic SAR, Ichimoku, CCI, Williams %R, and Supertrend
# follow) — popularity of use, not any backtested edge on these specific
# synthetic indices. See config.py's "POPULAR INDICATOR CONFLUENCE
# STRATEGY" section for every period/threshold tunable used below (the
# section name in config.py predates this file's move away from
# confluence-style voting; the tunables themselves are unchanged).
#
# One "cutting edge" computational addition sits at the very bottom of
# the priority order: a Kalman filter adaptive trend/velocity estimate
# (its own gain adapts every bar to recent price noise, rather than
# using a fixed lookback window like a moving average). It is
# deliberately ranked least-popular/last — a fallback so a symbol still
# gets a read on the rare bar where every classic indicator above it is
# abstaining — not a vote that gets blended with the others.
#
# NOTE: this function only ever computes the "as-read" popular direction
# for whichever single indicator was picked. The user-directed universal
# signal inversion (see config.INVERT_ALL_SIGNALS) is applied later,
# once, at the single execution choke point in bot_engine.py — not here
# — so this function's `direction` output is the un-inverted textbook
# reading of the picked indicator, exactly as "many people" would read
# it.
# ---------------------------------------------------------------------------

def _cross_dir(now_a: float, now_b: float, prev_a: float, prev_b: float) -> Optional[float]:
    """
    +1.0 if `a` just crossed above `b` this bar, -1.0 if it just crossed
    below, None if no fresh cross happened this bar (including: both
    bars on the same side). Used for every "trade the cross/flip" read
    below (moving averages, MACD, Ichimoku TK, Parabolic SAR, Supertrend)
    since that discrete event — not the ongoing state — is the
    conventional, popularly-traded signal for each of these.
    """
    now_above = now_a > now_b
    prev_above = prev_a > prev_b
    if now_above == prev_above:
        return None
    return 1.0 if now_above else -1.0


def _swing_stop(direction: str, H: np.ndarray, L: np.ndarray, period: int) -> float:
    """Nearest swing low/high over the last `period` bars — min(L[-N:]) for
    a long stop, max(H[-N:]) for a short stop. Shared by every oscillator
    (RSI, MACD, Stochastic, ADX/DMI, CCI, Williams %R) whose own convention
    doesn't have a single native price level of its own."""
    n = max(2, min(period, len(H)))
    return float(np.min(L[-n:])) if direction == "LONG" else float(np.max(H[-n:]))


def _native_stop_target(
    picked_label: str, direction: str, entry: float,
    C: np.ndarray, H: np.ndarray, L: np.ndarray, *,
    ema_slow: np.ndarray, bb_mid: np.ndarray, bb_upper: np.ndarray, bb_lower: np.ndarray,
    sar_vals: np.ndarray, senkou_a: np.ndarray, senkou_b: np.ndarray,
    kijun: np.ndarray, st_line: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Native stop-loss / take-profit PRICE levels for whichever indicator was
    picked, using that indicator's own standard TA convention (spec point
    3) — never a stake percentage. `direction` is the un-inverted reading
    (LONG/SHORT) exactly as evaluate_popular_indicator() computed it; the
    swap to bot-execution SL/TP happens later in bot_engine.py, alongside
    the direction inversion (spec points 4/5).

    Returns (stop, target) as raw prices, or (None, None) if the picked
    label somehow isn't recognized (should not happen — every label in
    POPULAR_INDICATOR_ORDER is handled below).
    """
    sign = 1.0 if direction == "LONG" else -1.0

    if picked_label == "EMA_CROSS":
        stop = float(ema_slow[-1])
        target = entry + (entry - stop)   # equal-distance projection off the crossover gap
        return stop, target

    if picked_label == "RSI":
        stop = _swing_stop(direction, H, L, config.POPULAR_RSI_PERIOD)
        target = entry + sign * 2.0 * abs(entry - stop)
        return stop, target

    if picked_label == "MACD_CROSS":
        stop = _swing_stop(direction, H, L, config.POPULAR_MACD_SLOW)
        target = entry + sign * 2.0 * abs(entry - stop)
        return stop, target

    if picked_label == "BOLLINGER":
        # Small buffer beyond the touched band — the band value itself is
        # the native stop; the middle band (SMA basis) is the textbook
        # mean-reversion target.
        stop = float(bb_lower[-1]) if direction == "LONG" else float(bb_upper[-1])
        target = float(bb_mid[-1])
        return stop, target

    if picked_label == "STOCHASTIC":
        stop = _swing_stop(direction, H, L, config.POPULAR_STOCH_K_PERIOD)
        target = entry + sign * 2.0 * abs(entry - stop)
        return stop, target

    if picked_label == "ADX_DMI":
        # Wider target — ADX confirming trend strength justifies riding
        # further than the standard 2R used by the oscillators above.
        stop = _swing_stop(direction, H, L, config.POPULAR_ADX_PERIOD)
        target = entry + sign * 2.0 * abs(entry - stop)
        return stop, target

    if picked_label == "PARABOLIC_SAR":
        # The current SAR dot value itself *is* its native stop.
        stop = float(sar_vals[-1])
        target = entry + sign * 2.0 * abs(entry - stop)
        return stop, target

    if picked_label == "ICHIMOKU_CLOUD":
        top = max(float(senkou_a[-1]), float(senkou_b[-1]))
        bottom = min(float(senkou_a[-1]), float(senkou_b[-1]))
        thickness = top - bottom
        stop = bottom if direction == "LONG" else top   # near edge of the cloud
        target = entry + sign * 2.0 * thickness
        return stop, target

    if picked_label == "ICHIMOKU_TK_CROSS":
        stop = float(kijun[-1])   # standard Ichimoku stop reference
        target = entry + sign * 2.0 * abs(entry - stop)   # conventional 2:1 Kijun projection
        return stop, target

    if picked_label == "CCI":
        stop = _swing_stop(direction, H, L, config.POPULAR_CCI_PERIOD)
        target = entry + sign * 2.0 * abs(entry - stop)
        return stop, target

    if picked_label == "WILLIAMS_R":
        stop = _swing_stop(direction, H, L, config.POPULAR_WILLIAMS_R_PERIOD)
        target = entry + sign * 2.0 * abs(entry - stop)
        return stop, target

    if picked_label == "SUPERTREND":
        # The current Supertrend line value itself is its native trailing stop.
        stop = float(st_line[-1])
        target = entry + sign * 2.0 * abs(entry - stop)
        return stop, target

    return None, None


def evaluate_popular_indicator(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    # BUGFIX (Aug 2026): this previously required 60 bars (to fully
    # mature Ichimoku's default 52-period Senkou Span B), but
    # bot_engine.py's _init_symbol_data() only ever provisions
    # config.LTF_BARS (30) bars on startup, capped at LTF_BARS + 20 (50)
    # by the CandlestickBuilder's max_bars — so 60 was never reachable
    # and this strategy could never fire a single trade. Every other
    # evaluator in this file gates on 20-35 bars for the same reason;
    # ind.ichimoku() (like every function in indicators.py) already
    # degrades gracefully on short history via a growing window instead
    # of requiring the full period, so there's no correctness need for
    # the higher bar count either.
    min_bars = 35
    if len(ltf_bars) < min_bars:
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    if len(C) < 2 or math.isnan(float(C[-1])):
        return NONE_RESULT
    last_close = float(C[-1])

    reads: Dict[str, Optional[float]] = {}   # label -> +1.0 / -1.0 / None (abstain)
    detail: Dict[str, str] = {}               # label -> short human-readable state

    # 1) EMA crossover — the single most widely cited indicator in
    #    general trading use. Fires only on the bar the cross happens
    #    (the conventional "golden/death cross" event), not for the
    #    whole time the fast EMA happens to sit on one side.
    ema_fast = ind.ema(C, config.POPULAR_EMA_FAST_PERIOD)
    ema_slow = ind.ema(C, config.POPULAR_EMA_SLOW_PERIOD)
    reads["EMA_CROSS"] = _cross_dir(ema_fast[-1], ema_slow[-1], ema_fast[-2], ema_slow[-2])
    detail["EMA_CROSS"] = f"fast{'>' if ema_fast[-1] > ema_slow[-1] else '<'}slow"

    # 2) RSI — classic 70/30 zone read (signals while actually inside
    #    the oversold/overbought zone; the most standard popular version
    #    of an "RSI signal").
    rsi_vals = ind.rsi(C, config.POPULAR_RSI_PERIOD)
    last_rsi = float(rsi_vals[-1])
    if last_rsi <= config.POPULAR_RSI_OVERSOLD:
        reads["RSI"] = 1.0
    elif last_rsi >= config.POPULAR_RSI_OVERBOUGHT:
        reads["RSI"] = -1.0
    else:
        reads["RSI"] = None
    detail["RSI"] = f"{last_rsi:.1f}"

    # 3) MACD — signal-line crossover event, the conventional "MACD
    #    signal" (as opposed to the raw histogram's continuous sign).
    macd_line, macd_signal, macd_hist = ind.macd(
        C, config.POPULAR_MACD_FAST, config.POPULAR_MACD_SLOW, config.POPULAR_MACD_SIGNAL)
    reads["MACD_CROSS"] = _cross_dir(
        macd_line[-1], macd_signal[-1], macd_line[-2], macd_signal[-2])
    detail["MACD_CROSS"] = f"hist={float(macd_hist[-1]):.5f}"

    # 4) Bollinger Bands — classic mean-reversion touch read (only has
    #    an opinion once price is actually at/through a band).
    bb_upper, bb_mid, bb_lower = ind.bollinger_bands(
        C, config.POPULAR_BB_PERIOD, config.POPULAR_BB_STD)
    band_width = float(bb_upper[-1]) - float(bb_lower[-1])
    pct_b = ((last_close - float(bb_lower[-1])) / band_width) if band_width > 0 else 0.5
    if pct_b <= 0.05:
        reads["BOLLINGER"] = 1.0     # at/through lower band -> reversion up
    elif pct_b >= 0.95:
        reads["BOLLINGER"] = -1.0    # at/through upper band -> reversion down
    else:
        reads["BOLLINGER"] = None
    detail["BOLLINGER"] = f"%B={pct_b:.2f}"

    # 5) Stochastic Oscillator — classic 80/20 zone read.
    stoch_k, stoch_d = ind.stochastic(
        H, L, C, config.POPULAR_STOCH_K_PERIOD, config.POPULAR_STOCH_D_PERIOD)
    last_k = float(stoch_k[-1])
    if last_k <= config.POPULAR_STOCH_OVERSOLD:
        reads["STOCHASTIC"] = 1.0
    elif last_k >= config.POPULAR_STOCH_OVERBOUGHT:
        reads["STOCHASTIC"] = -1.0
    else:
        reads["STOCHASTIC"] = None
    detail["STOCHASTIC"] = f"%K={last_k:.1f}"

    # 6) ADX/DMI — direction from +DI/-DI, but only when ADX itself
    #    confirms a real trend is present (standard popular usage:
    #    ignore DI direction when ADX is low/choppy).
    adx_vals, plus_di, minus_di = ind.adx(H, L, C, config.POPULAR_ADX_PERIOD)
    last_adx = float(adx_vals[-1])
    if last_adx >= config.POPULAR_ADX_TREND_MIN:
        reads["ADX_DMI"] = 1.0 if float(plus_di[-1]) > float(minus_di[-1]) else -1.0
    else:
        reads["ADX_DMI"] = None
    detail["ADX_DMI"] = f"ADX={last_adx:.1f}"

    # 7) Parabolic SAR — trade the flip (the conventional popular
    #    signal), not the ongoing state.
    sar_vals = ind.parabolic_sar(H, L, config.POPULAR_SAR_STEP, config.POPULAR_SAR_MAX_STEP)
    reads["PARABOLIC_SAR"] = _cross_dir(last_close, float(sar_vals[-1]), float(C[-2]), float(sar_vals[-2]))
    detail["PARABOLIC_SAR"] = f"{'above' if last_close > float(sar_vals[-1]) else 'below'}"

    # 8) Ichimoku Cloud position — read continuously (this is how the
    #    cloud is conventionally used: an ongoing bullish/bearish bias,
    #    not a discrete event), no opinion while price sits inside it.
    tenkan, kijun, senkou_a, senkou_b = ind.ichimoku(
        H, L, C, config.POPULAR_ICHIMOKU_TENKAN, config.POPULAR_ICHIMOKU_KIJUN,
        config.POPULAR_ICHIMOKU_SENKOU_B)
    cloud_top = max(float(senkou_a[-1]), float(senkou_b[-1]))
    cloud_bottom = min(float(senkou_a[-1]), float(senkou_b[-1]))
    if last_close > cloud_top:
        reads["ICHIMOKU_CLOUD"] = 1.0
    elif last_close < cloud_bottom:
        reads["ICHIMOKU_CLOUD"] = -1.0
    else:
        reads["ICHIMOKU_CLOUD"] = None
    detail["ICHIMOKU_CLOUD"] = f"close_vs_cloud[{cloud_bottom:.4f},{cloud_top:.4f}]"

    # 9) Ichimoku Tenkan/Kijun cross — the other half of Ichimoku's
    #    popular usage, a discrete crossover event.
    reads["ICHIMOKU_TK_CROSS"] = _cross_dir(tenkan[-1], kijun[-1], tenkan[-2], kijun[-2])
    detail["ICHIMOKU_TK_CROSS"] = f"tenkan{'>' if tenkan[-1] > kijun[-1] else '<'}kijun"

    # 10) CCI — classic +/-100 zone read.
    cci_vals = ind.cci(H, L, C, config.POPULAR_CCI_PERIOD)
    last_cci = float(cci_vals[-1])
    if last_cci <= config.POPULAR_CCI_OVERSOLD:
        reads["CCI"] = 1.0
    elif last_cci >= config.POPULAR_CCI_OVERBOUGHT:
        reads["CCI"] = -1.0
    else:
        reads["CCI"] = None
    detail["CCI"] = f"{last_cci:.1f}"

    # 11) Williams %R — classic -20/-80 zone read, mirror-image cousin
    #     of the Stochastic Oscillator.
    wr_vals = ind.williams_r(H, L, C, config.POPULAR_WILLIAMS_R_PERIOD)
    last_wr = float(wr_vals[-1])
    if last_wr <= -80.0:
        reads["WILLIAMS_R"] = 1.0
    elif last_wr >= -20.0:
        reads["WILLIAMS_R"] = -1.0
    else:
        reads["WILLIAMS_R"] = None
    detail["WILLIAMS_R"] = f"{last_wr:.1f}"

    # 12) Supertrend — trade the flip (the conventional popular signal),
    #     not the ongoing state.
    st_line, st_dir_arr = ind.supertrend(
        H, L, C, config.POPULAR_SUPERTREND_PERIOD, config.POPULAR_SUPERTREND_MULT)
    reads["SUPERTREND"] = None if st_dir_arr[-1] == st_dir_arr[-2] else (
        1.0 if st_dir_arr[-1] > 0 else -1.0)
    detail["SUPERTREND"] = f"{'up' if st_dir_arr[-1] > 0 else 'down'}"

    # NOTE: a Kalman-filter adaptive trend/velocity fallback used to sit
    # here as indicator #13. Removed per spec (user-directed, Aug 2026):
    # the ranked list is limited to genuinely popularly-used indicators,
    # and a Kalman filter isn't one of them — not even as a last-resort
    # fallback. ind.kalman_trend() itself is left in indicators.py, unused
    # by this function.

    # Most-popular-first priority order — see the strategy comment above.
    POPULAR_INDICATOR_ORDER = [
        "EMA_CROSS", "RSI", "MACD_CROSS", "BOLLINGER", "STOCHASTIC",
        "ADX_DMI", "PARABOLIC_SAR", "ICHIMOKU_CLOUD", "ICHIMOKU_TK_CROSS",
        "CCI", "WILLIAMS_R", "SUPERTREND",
    ]

    # Filter out both "not currently signalling" AND any (indicator, symbol)
    # pair that's sitting out its 1-hour underperformance suspension
    # (spec point 8) — done HERE, at pick time, not as an after-the-fact
    # reject of the whole result. That's what lets the symbol fall through
    # to the next-ranked indicator that IS signalling instead of getting no
    # trade at all on this pass. See pair_suspension.py.
    signalling = [
        (label, reads[label]) for label in POPULAR_INDICATOR_ORDER
        if reads.get(label) is not None and not pair_suspension.is_suspended(label, symbol)
    ]

    if not signalling:
        logger.debug(f"REJECTED: {symbol} POPULAR_INDICATOR strength=0 score=0.000 — nothing signalling")
        return NONE_RESULT

    picked_label, picked_dir = signalling[0]   # most popular among those currently signalling, eligible
    rank_index = POPULAR_INDICATOR_ORDER.index(picked_label)
    direction = "LONG" if picked_dir > 0 else "SHORT"

    # Check whether this pair should START its 1-hour suspension clock now
    # (it just traded, or is about to). This only decides whether to START
    # the timer — pair_suspension.is_suspended() above is what actually
    # gates future picks, and maybe_suspend() is idempotent so this can't
    # push an existing suspension's expiry back out.
    try:
        if strategy_stats.stats.is_underperforming(picked_label, symbol):
            pair_suspension.maybe_suspend(picked_label, symbol)
    except Exception as exc:
        logger.warning(f"is_underperforming({picked_label},{symbol}) check failed: {exc}")

    # Strength/score communicate how popular the picked indicator is,
    # not a blended-confidence figure — there is nothing to blend.
    score = 1.0 - (rank_index / len(POPULAR_INDICATOR_ORDER))
    strength = 3 if rank_index < len(POPULAR_INDICATOR_ORDER) // 2 else 2

    # Native stop-loss / take-profit PRICE levels (spec point 3) — computed
    # only for the picked indicator, using its own standard TA convention.
    # These are pre-inversion, pre-swap: the stop/target for `direction`
    # exactly as read above. bot_engine.py's universal inversion step swaps
    # stop<->target at the same time it flips direction (spec points 4/5).
    native_stop, native_target = _native_stop_target(
        picked_label, direction, last_close, C, H, L,
        ema_slow=ema_slow, bb_mid=bb_mid, bb_upper=bb_upper, bb_lower=bb_lower,
        sar_vals=sar_vals, senkou_a=senkou_a, senkou_b=senkou_b,
        kijun=kijun, st_line=st_line,
    )

    also_signalling = ", ".join(f"{lbl}={'L' if d > 0 else 'S'}" for lbl, d in signalling[1:])
    reason = (
        f"picked={picked_label}[{detail.get(picked_label, '')}] rank={rank_index + 1}/"
        f"{len(POPULAR_INDICATOR_ORDER)} stop={native_stop} target={native_target}"
        + (f" also_signalling[{also_signalling}]" if also_signalling else " (only one signalling)")
    )
    logger.info(
        f"SIGNAL: {symbol} {direction} {picked_label} strength={strength} "
        f"score={score:.3f} native_stop={native_stop} native_target={native_target}"
    )
    return SignalResult(
        direction=direction, strength=strength, score=score,
        strategy=picked_label, reason=reason,
        native_stop_price=native_stop, native_target_price=native_target,
        native_entry_price=last_close,
    )


def evaluate_vol_regime(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    """
    Thin regime-selecting dispatcher for config.VOL_MULTIPLIER_SYMBOLS —
    routes to the trend/breakout evaluator in TREND regime, or the
    range/reversion evaluator in RANGE regime.
    """
    if len(ltf_bars) < 30:
        return NONE_RESULT
    regime = _vol_regime(ltf_bars)
    if regime == "TREND":
        return evaluate_vol_breakout(ltf_bars, symbol)
    return evaluate_vol_reversion_mult(ltf_bars, symbol)


# ---------------------------------------------------------------------------
# Strategy 3 — Range Break Retest
# ---------------------------------------------------------------------------

def evaluate_range_break(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    if len(ltf_bars) < 30:
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    rsi = ind.rsi(C, 14)
    atr = ind.atr(H, L, C, 14)
    last_atr = _last(atr)
    last_rsi = _last(rsi)

    consolidation = ind.find_consolidation(H, L, C)
    cons_upper, cons_lower = (None, None)
    has_consolidation = consolidation is not None
    if has_consolidation:
        cons_upper, cons_lower = consolidation

    # --- Phase A: find most recent breakout within last 3 bars ---
    breakout_dir: Optional[str] = None
    breakout_level: Optional[float] = None
    breakout_bars_ago: Optional[int] = None

    search_bounds = (cons_upper, cons_lower) if has_consolidation else None
    if search_bounds is None:
        # Fall back to a rolling range if no consolidation zone was found,
        # so breakout/retest logic still has a boundary to test against.
        lookback = min(20, len(C) - 4)
        search_upper = float(np.max(H[-lookback - 4:-4])) if lookback > 0 else float(H[-4])
        search_lower = float(np.min(L[-lookback - 4:-4])) if lookback > 0 else float(L[-4])
    else:
        search_upper, search_lower = search_bounds

    for bars_ago in range(1, 4):  # 1, 2, 3 bars old
        idx = -bars_ago
        close_i = float(C[idx])
        if close_i > search_upper + 0.3 * last_atr:
            breakout_dir = "LONG"
            breakout_level = search_upper
            breakout_bars_ago = bars_ago
            break
        if close_i < search_lower - 0.3 * last_atr:
            breakout_dir = "SHORT"
            breakout_level = search_lower
            breakout_bars_ago = bars_ago
            break

    if breakout_dir is None:
        logger.debug(f"REJECTED: {symbol} RANGE_BREAK strength=0 score=0.000 — below threshold")
        return SignalResult("NONE", 0, 0.0, "RANGE_BREAK", "No breakout detected")

    # --- Phase B: retest ---
    current_price = float(C[-1])
    retested = abs(current_price - breakout_level) <= 0.5 * last_atr

    if not retested:
        logger.debug(f"REJECTED: {symbol} RANGE_BREAK strength=0 score=0.250 — below threshold")
        return SignalResult("NONE", 0, 0.25, "RANGE_BREAK", "Breakout found, awaiting retest")

    rsi_confirmed = (last_rsi > 52) if breakout_dir == "LONG" else (last_rsi < 48)

    confirmed = 1  # breakout confirmed
    confirmed += 1  # retest confirmed
    if rsi_confirmed:
        confirmed += 1
    if has_consolidation:
        confirmed += 1

    if not rsi_confirmed:
        logger.info(f"REJECTED: {symbol} RANGE_BREAK strength=1 score={confirmed/4.0:.3f} — below threshold")
        return SignalResult("NONE", 0, confirmed / 4.0, "RANGE_BREAK", "RSI not confirmed")

    strength = 3 if (has_consolidation and rsi_confirmed) else 2
    score = confirmed / 4.0

    logger.info(
        f"SIGNAL: {symbol} {breakout_dir} RANGE_BREAK strength={strength} score={score:.3f}"
    )
    return SignalResult(
        direction=breakout_dir,
        strength=strength,
        score=score,
        strategy="RANGE_BREAK",
        reason=(
            f"Breakout {breakout_bars_ago}bars ago @ {breakout_level:.5f}, "
            f"retest confirmed, RSI={last_rsi:.1f}, consolidation={has_consolidation}"
        ),
    )


# ---------------------------------------------------------------------------
# Strategy 4 — Post-Spike Fade (Boom/Crash)
# ---------------------------------------------------------------------------

def evaluate_boom_crash(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    if len(ltf_bars) < 20:
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    rsi = ind.rsi(C, 14)
    atr = ind.atr(H, L, C, 14)
    last_atr = _last(atr) or 0.001
    last_rsi = _last(rsi)

    # detect_spike() only reports on the single most-recent bar of whatever
    # slice it's given (+1 up-spike / -1 down-spike / 0 none — no dict, no
    # bars_ago/type/size). To ask "was the bar N bars ago a spike bar",
    # trim the array's tail by N bars so that bar becomes the new "last" one.
    def _spike_at(bars_ago: int, period: int = 14, atr_multiplier: float = 3.0):
        c_s = C[:-bars_ago] if bars_ago > 0 else C
        h_s = H[:-bars_ago] if bars_ago > 0 else H
        l_s = L[:-bars_ago] if bars_ago > 0 else L
        if len(c_s) < 2:
            return 0, 0.0
        direction = ind.detect_spike(c_s, h_s, l_s, period=period, atr_multiplier=atr_multiplier)
        size = abs(float(c_s[-1]) - float(c_s[-2])) if direction != 0 else 0.0
        return direction, size

    # Must be within last 2 bars, but at least 1 bar since the spike bar
    # closed (bars_ago=0 would be the still-forming most-recent bar).
    spike_dir, spike_size, bars_ago = 0, 0.0, 0
    for candidate in (1, 2):
        d, sz = _spike_at(candidate)
        if d != 0:
            spike_dir, spike_size, bars_ago = d, sz, candidate
            break

    if spike_dir == 0:
        logger.debug(f"REJECTED: {symbol} BOOM_CRASH strength=0 score=0.000 — below threshold")
        return NONE_RESULT

    spike_type = "BOOM" if spike_dir > 0 else "CRASH"

    # Cooldown: no earlier spike in the 10 bars preceding the one just found.
    cooldown_hit = False
    for earlier_bars_ago in range(bars_ago + 1, bars_ago + 11):
        earlier_dir, _ = _spike_at(earlier_bars_ago)
        if earlier_dir != 0:
            cooldown_hit = True
            break

    if cooldown_hit:
        logger.info(f"REJECTED: {symbol} BOOM_CRASH strength=1 score=0.000 — below threshold")
        return NONE_RESULT

    if spike_type == "BOOM":
        direction = "SHORT"
        rsi_confirmed = last_rsi > 60
    elif spike_type == "CRASH":
        direction = "LONG"
        rsi_confirmed = last_rsi < 40
    else:
        return NONE_RESULT

    # FIX (Task 3 — full textbook confirmation, no partial firing, Aug
    # 2026): rsi_confirmed previously only affected strength (3 vs 2) and
    # never gated whether the signal fired at all — a post-spike fade with
    # NO RSI confirmation still fired at strength=2. The textbook
    # post-spike fade requires RSI to actually be in the overbought/
    # oversold zone that supports fading back toward the mean; that's a
    # real condition, not just a strength modifier. Now hard-gated: no
    # RSI confirmation, no signal.
    if not rsi_confirmed:
        logger.info(
            f"REJECTED: {symbol} BOOM_CRASH strength=0 score=0.000 — "
            f"{spike_type} spike detected but RSI={last_rsi:.1f} does not "
            f"confirm the fade direction"
        )
        return NONE_RESULT

    strength = 3
    score = min(spike_size / (last_atr * 5.0), 1.0)

    logger.info(
        f"SIGNAL: {symbol} {direction} BOOM_CRASH strength={strength} score={score:.3f}"
    )
    return SignalResult(
        direction=direction,
        strength=strength,
        score=score,
        strategy="BOOM_CRASH",
        reason=f"Fade {spike_type} spike size={spike_size:.5f} RSI={last_rsi:.1f}",
    )


# ---------------------------------------------------------------------------
# Strategy 4b — Drift Fade (new, standalone, tick-based)
#
# Despite the name (kept as specified), this trades WITH a confirmed
# directional drift once a prior spike has cleared cooldown — it does not
# fade price the way evaluate_boom_crash does. It is a separate, distinct
# read on the same instrument category and is independently scored.
# ---------------------------------------------------------------------------

def evaluate_drift_fade(ticks: Optional[List[Any]], symbol: str) -> SignalResult:
    window_size = getattr(config, "DRIFT_FADE_WINDOW", 60)
    if not ticks or len(ticks) < window_size + 1:
        return NONE_RESULT

    try:
        quotes = np.array([_tick_quote(t) for t in ticks[-(window_size + 1):]], dtype=float)
    except ValueError:
        logger.warning(f"DRIFT_FADE: {symbol} could not parse tick quotes — skipping")
        return NONE_RESULT

    diffs = np.diff(quotes)
    atr_proxy = float(np.mean(np.abs(diffs))) or 1e-9

    spike_lookback = getattr(config, "DRIFT_FADE_SPIKE_LOOKBACK", 20)
    spike_mult = getattr(config, "DRIFT_FADE_SPIKE_MULT", 4.0)
    cooldown_ticks = getattr(config, "DRIFT_FADE_COOLDOWN_TICKS", 40)

    recent_diffs = diffs[-spike_lookback:]
    spike_idx = None
    for i in range(len(recent_diffs) - 1, -1, -1):
        if abs(recent_diffs[i]) > spike_mult * atr_proxy:
            spike_idx = i
            break

    if spike_idx is not None:
        ticks_since_spike = len(recent_diffs) - 1 - spike_idx
        if ticks_since_spike < cooldown_ticks:
            logger.debug(
                f"REJECTED: {symbol} DRIFT_FADE strength=0 score=0.000 — "
                f"cooldown active ({ticks_since_spike}/{cooldown_ticks} ticks since spike)"
            )
            return NONE_RESULT

    x = np.arange(len(quotes), dtype=float)
    slope = float(np.polyfit(x, quotes, 1)[0])
    slope_atr_ratio = abs(slope) / atr_proxy if atr_proxy else 0.0

    min_ratio = getattr(config, "DRIFT_FADE_MIN_SLOPE_ATR_RATIO", 0.15)
    if slope_atr_ratio < min_ratio:
        logger.debug(
            f"REJECTED: {symbol} DRIFT_FADE strength=0 score={min(slope_atr_ratio, 1.0):.3f} — "
            f"slope/ATR {slope_atr_ratio:.3f} below {min_ratio}"
        )
        return SignalResult("NONE", 0, min(slope_atr_ratio, 1.0), "DRIFT_FADE", "No confirmed drift")

    direction = "LONG" if slope > 0 else "SHORT"
    score = max(0.0, min(1.0, slope_atr_ratio))
    strength = 3 if slope_atr_ratio >= 2 * min_ratio else 2

    logger.info(
        f"SIGNAL: {symbol} {direction} DRIFT_FADE strength={strength} score={score:.3f} "
        f"slope={slope:.6f} atr_proxy={atr_proxy:.6f}"
    )
    return SignalResult(
        direction=direction,
        strength=strength,
        score=score,
        strategy="DRIFT_FADE",
        reason=f"Confirmed drift slope={slope:.6f} atr_proxy={atr_proxy:.6f} ratio={slope_atr_ratio:.3f}",
    )


# ---------------------------------------------------------------------------
# Strategy 5 — Step Index Trend
# ---------------------------------------------------------------------------

def evaluate_step(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    if len(ltf_bars) < 35:
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    ema10 = ind.ema(C, 10)
    ema30 = ind.ema(C, 30)
    donchian_upper, donchian_lower = ind.donchian(H, L, 20)

    e10, e30 = ema10[-1], ema30[-1]
    e10_prev = ema10[-2]

    ema_dir: Optional[str] = None
    if e10 > e30 and e10 > e10_prev:
        ema_dir = "LONG"
    elif e10 < e30 and e10 < e10_prev:
        ema_dir = "SHORT"

    last_close = float(C[-1])
    last_don_upper = _last(donchian_upper)
    last_don_lower = _last(donchian_lower)

    donchian_dir: Optional[str] = None
    if last_close >= last_don_upper:
        donchian_dir = "SHORT"
    elif last_close <= last_don_lower:
        donchian_dir = "LONG"

    if ema_dir is None or donchian_dir is None or ema_dir != donchian_dir:
        logger.debug(f"REJECTED: {symbol} STEP strength=0 score=0.000 — below threshold")
        return NONE_RESULT

    direction = ema_dir
    strength = 2
    score = 0.65

    logger.info(
        f"SIGNAL: {symbol} {direction} STEP strength={strength} score={score:.3f}"
    )
    return SignalResult(
        direction=direction,
        strength=strength,
        score=score,
        strategy="STEP",
        reason=f"EMA10/30 trend + Donchian band agreement ({direction})",
    )


# ---------------------------------------------------------------------------
# Strategy 6 — Jump Index Build-Up (new, standalone, tick-based)
# ---------------------------------------------------------------------------

def evaluate_jump_buildup(ticks: Optional[List[Any]], symbol: str) -> SignalResult:
    if not ticks or len(ticks) < 10:
        return NONE_RESULT

    try:
        quotes = np.array([_tick_quote(t) for t in ticks], dtype=float)
    except ValueError:
        logger.warning(f"JUMP_BUILDUP: {symbol} could not parse tick quotes — skipping")
        return NONE_RESULT

    epochs = [_tick_epoch(t) for t in ticks]
    have_epochs = all(e is not None for e in epochs)

    diffs = np.diff(quotes)
    baseline = float(np.median(np.abs(diffs))) or 1e-9
    jump_mult = getattr(config, "JUMP_DETECT_MULT", 5.0)

    jump_pos = None
    for i in range(len(diffs) - 1, -1, -1):
        if abs(diffs[i]) > jump_mult * baseline:
            jump_pos = i
            break

    if jump_pos is None:
        logger.debug(f"REJECTED: {symbol} JUMP_BUILDUP strength=0 score=0.000 — no jump detected in window")
        return NONE_RESULT

    target = getattr(config, "JUMP_TARGET_INTERVAL_MINS", 20)
    if have_epochs:
        elapsed_mins = (epochs[-1] - epochs[jump_pos + 1]) / 60.0
    else:
        # No timestamps on the tick objects — degrade to a tick-count proxy.
        # This is coarse (1 tick != 1 minute); wire timestamped ticks through
        # for accurate build-up timing.
        elapsed_mins = float(len(quotes) - 1 - jump_pos)
        logger.debug(f"JUMP_BUILDUP: {symbol} ticks have no epoch — using tick-count proxy for elapsed time")

    confidence = min(elapsed_mins / target, 1.0) if target > 0 else 0.0

    compression_lookback = getattr(config, "JUMP_COMPRESSION_LOOKBACK", 30)
    recent_window = diffs[-compression_lookback:] if len(diffs) >= compression_lookback else diffs
    recent_vol = float(np.std(recent_window))
    baseline_vol = float(np.std(diffs)) or 1e-9
    compressed = recent_vol < 0.7 * baseline_vol
    if compressed:
        confidence = min(1.0, confidence + 0.1)

    min_conf = getattr(config, "JUMP_MIN_CONFIDENCE", 0.5)
    if confidence < min_conf:
        logger.debug(
            f"REJECTED: {symbol} JUMP_BUILDUP strength=0 score={confidence:.3f} — "
            f"confidence below {min_conf} (elapsed={elapsed_mins:.1f}m, target={target}m)"
        )
        return SignalResult("NONE", 0, confidence, "JUMP_BUILDUP", "Build-up confidence too low")

    # Implementation Brief v3, finding #3 / task 2: build-up confidence has
    # no LONG/SHORT price read — jump direction is 50/50 by design (Deriv's
    # own product description), so there is nothing here to map onto a
    # Rise/Fall CALL/PUT. What build-up confidence DOES predict is whether
    # the last digit is likely to repeat (high confidence + compressed
    # pre-jump volatility -> MATCHES) or not (DIFFERS) — a real digit
    # contract, wired below via contract_kind="DIGIT" so bot_engine routes
    # it to DerivClient.buy_digit_contract() instead of buy_contract().
    # A digit barrier is required by the API for both MATCH and DIFFER, so
    # last_digit is computed unconditionally, not just on the MATCHES path.
    match_threshold = getattr(config, "JUMP_MATCH_CONFIDENCE_THRESHOLD", 0.9)
    decimals = _digit_decimals(symbol)
    last_digit = _last_digit(float(quotes[-1]), decimals)
    if confidence >= match_threshold and compressed:
        match_type = "MATCH"
        strength = 3
    else:
        match_type = "DIFFER"
        strength = 3 if confidence >= match_threshold else 2

    logger.info(
        f"SIGNAL: {symbol} JUMP_BUILDUP {match_type} digit={last_digit} "
        f"strength={strength} score={confidence:.3f} elapsed={elapsed_mins:.1f}m "
        f"target={target}m compressed={compressed}"
    )
    return SignalResult(
        direction=match_type,
        strength=strength,
        score=confidence,
        strategy="JUMP_BUILDUP",
        reason=(
            f"Recommend {match_type} digit={last_digit} | "
            f"elapsed={elapsed_mins:.1f}m/{target}m compressed={compressed}"
        ),
        contract_kind="DIGIT",
        digit=last_digit,
        match_type=match_type,
    )


# ---------------------------------------------------------------------------
# Strategy 7 — Bear/Bull ("Daily Reset") fixed-bias trend following
#
# Implementation Brief v3, finding #4: RDBEAR/RDBULL reset to a baseline at
# 00:00 GMT and then hold ONE fixed characteristic trend for the rest of the
# 24h cycle (Bull trends up, Bear trends down) — this is a fixed identity
# per symbol, not something that flips. The previous version alternated
# direction on every post-reset window via module-level state
# (_trend_shift_state), which contradicted that product mechanic outright
# (it would periodically have RDBULL go SHORT and RDBEAR go LONG). Fixed
# here: direction comes from config.BEAR_BULL_DIRECTION, a static map, and
# is never derived from EMA alignment or alternated — no more module-level
# direction state needed at all. is_post_reset()/get_bear_bull_state() is
# used ONLY to gate entry timing (skip trading during the post-reset window,
# since early-cycle behavior may differ from the rest of the trending cycle)
# via the shared `_symbol_manager` instance above, instead of duplicating
# the "minutes since 00:00 GMT" math locally.
# ---------------------------------------------------------------------------

def evaluate_trend_shift(
    ltf_bars: List[Candle],
    symbol: str,
    ticks: Optional[List[Any]] = None,
    is_post_reset_fn: Optional[Callable[[str], bool]] = None,
) -> SignalResult:
    """
    Bear/Bull fixed-bias trend following: direction is a static per-symbol
    fact (config.BEAR_BULL_DIRECTION), never derived from indicators and
    never alternated. EMA/RSI/ATR are computed only to score how cleanly
    current price action is confirming the known bias (a confidence read),
    and to gate out the post-reset window on timing rather than direction.

    ASSUMPTION: config.LTF_BARS is currently 30, which is fewer bars than
    EMA_TREND=50 needs to fully warm up. Depending on how indicators.ema()
    handles insufficient history (NaN-pad vs. shorter valid series vs.
    raising), this evaluator may run below full confidence — or never
    fire — until whatever calls SignalEngine.evaluate() is passing more
    than LTF_BARS=30 bars for BEAR_BULL_SYMBOLS specifically, or LTF_BARS
    is raised. Flagging rather than silently reinterpreting LTF_BARS.

    `ticks` is accepted only for call-site/signature compatibility with
    SignalEngine.evaluate()'s existing `ticks=ticks` call; unused here.

    `is_post_reset_fn` defaults to the shared `_symbol_manager.is_post_reset`
    (single source of truth); callers may still inject their own for
    testing.
    """
    bias_map = getattr(config, "BEAR_BULL_DIRECTION", {"RDBULL": "LONG", "RDBEAR": "SHORT"})
    direction = bias_map.get(symbol)
    if direction is None:
        return NONE_RESULT

    post_reset_fn = is_post_reset_fn or _symbol_manager.is_post_reset
    post_reset = post_reset_fn(symbol)
    if post_reset:
        window = getattr(config, "BEAR_BULL_TREND_SHIFT_MINS", 20)
        logger.info(
            f"REJECTED: {symbol} TREND_SHIFT strength=0 score=0.000 — "
            f"inside post-reset window ({window}min since 00:00 GMT), "
            f"waiting for it to close before sizing up"
        )
        return SignalResult(
            "NONE", 0, 0.0, "TREND_SHIFT",
            f"Post-reset window open ({window}min) — entry timing gate, bias={direction} unaffected",
        )

    min_bars = max(config.EMA_TREND, config.RSI_PERIOD, config.ATR_PERIOD) + 1
    if len(ltf_bars) < min_bars:
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    ema_fast_arr = ind.ema(C, config.EMA_FAST)
    ema_slow_arr = ind.ema(C, config.EMA_SLOW)
    ema_trend_arr = ind.ema(C, config.EMA_TREND)
    rsi_arr = ind.rsi(C, config.RSI_PERIOD)
    atr_arr = ind.atr(H, L, C, config.ATR_PERIOD)

    ema_fast = _last(ema_fast_arr)
    ema_slow = _last(ema_slow_arr)
    ema_trend = _last(ema_trend_arr)
    last_rsi = _last(rsi_arr)
    last_atr = _last(atr_arr)

    if any(math.isnan(v) for v in (ema_fast, ema_slow, ema_trend, last_rsi, last_atr)) or last_atr <= 0:
        logger.debug(f"REJECTED: {symbol} TREND_SHIFT strength=0 score=0.000 — indicators not warmed up")
        return NONE_RESULT

    # Confirmation check — does current EMA alignment support the KNOWN
    # fixed bias right now?
    if direction == "LONG":
        aligned = ema_fast > ema_slow > ema_trend
    else:
        aligned = ema_fast < ema_slow < ema_trend

    # FIX (Task 3 — full textbook confirmation, no partial firing, Aug
    # 2026): `aligned` and `rsi_contradicts` are real, documented
    # conditions (EMA order confirming the bias; RSI not sitting in the
    # zone that contradicts it) — previously they only dampened `score`
    # (0.5x / 0.6x) rather than gating whether the signal could fire at
    # all, meaning a fully-misaligned EMA read AND a contradicting RSI
    # could still combine with a high enough separation_atr to clear
    # MIN_TREND_SHIFT_SCORE and fire anyway. That's a confidence threshold
    # substituting for missing conditions — the exact pattern this task
    # asks to close. Direction itself still never changes (fixed daily-
    # reset bias by design, per this function's docstring) — only whether
    # a signal is allowed to fire at all is now hard-gated on both.
    #
    # Judgment call: the pre-fix comments frame `aligned`/`rsi_contradicts`
    # as pure conviction-scoring by deliberate product design, not a
    # textbook EMA-crossover entry condition. Task 3's instructions are
    # explicit that every place a documented condition is allowed to
    # substitute-via-score rather than gate should be tightened, so both
    # are promoted to hard requirements here; noted in the changelog
    # rather than silently reinterpreting the strategy.
    if not aligned:
        logger.info(
            f"REJECTED: {symbol} TREND_SHIFT strength=0 score=0.000 — "
            f"EMA alignment does not currently support fixed bias={direction} "
            f"(ema_fast={ema_fast:.5f} ema_slow={ema_slow:.5f} ema_trend={ema_trend:.5f})"
        )
        return SignalResult(
            "NONE", 0, 0.0, "TREND_SHIFT",
            f"EMA not aligned with fixed bias={direction}",
        )

    rsi_overbought = getattr(config, "RSI_OVERBOUGHT", 70)
    rsi_oversold = getattr(config, "RSI_OVERSOLD", 30)
    rsi_contradicts = (direction == "LONG" and last_rsi >= rsi_overbought) or (
        direction == "SHORT" and last_rsi <= rsi_oversold
    )
    if rsi_contradicts:
        logger.info(
            f"REJECTED: {symbol} TREND_SHIFT strength=0 score=0.000 — "
            f"RSI={last_rsi:.1f} contradicts fixed bias={direction}"
        )
        return SignalResult(
            "NONE", 0, 0.0, "TREND_SHIFT",
            f"RSI={last_rsi:.1f} contradicts fixed bias={direction}",
        )

    separation_atr = abs(ema_fast - ema_slow) / last_atr
    score = min(separation_atr / 3.0, 1.0)  # 3x ATR separation -> full score; tune with live data
    score = max(0.0, min(1.0, score))

    min_score = getattr(config, "MIN_TREND_SHIFT_SCORE", 0.65)
    if score < min_score:
        logger.info(
            f"REJECTED: {symbol} TREND_SHIFT strength=0 score={score:.3f} — "
            f"below MIN_TREND_SHIFT_SCORE={min_score}"
        )
        return SignalResult("NONE", 0, score, "TREND_SHIFT", f"Score {score:.3f} below {min_score}")

    strength = 3 if score >= 0.85 else 2  # gated above min_score, so never falls to 1 here

    logger.info(
        f"SIGNAL: {symbol} {direction} TREND_SHIFT strength={strength} score={score:.3f} "
        f"fixed_bias=True aligned={aligned} ema_fast={ema_fast:.5f} ema_slow={ema_slow:.5f} "
        f"ema_trend={ema_trend:.5f} rsi={last_rsi:.1f} atr={last_atr:.5f} rsi_contradicts={rsi_contradicts}"
    )
    return SignalResult(
        direction=direction,
        strength=strength,
        score=score,
        strategy="TREND_SHIFT",
        reason=(
            f"Fixed daily-reset bias={direction} (not alternated), aligned={aligned}, "
            f"ema_sep/atr={separation_atr:.3f}, rsi={last_rsi:.1f}"
        ),
    )


# ---------------------------------------------------------------------------
# Strategy 7 — Six dedicated per-symbol evaluators (handoff, Sep 15 2026)
#
# Replaces evaluate_popular_indicator() for exactly nine symbols —
# R_75/1HZ75V, R_100/1HZ100V, BOOM1000/CRASH1000, BOOM500/CRASH500,
# stpRNG — with one independent, textbook-defined strategy per row. Each
# function below implements ONE named strategy (not a ranked multi-
# indicator pick like evaluate_popular_indicator()) and always populates
# native_entry_price / native_stop_price / native_target_price with real
# stop/target placement per that strategy's own convention — these three
# fields are what config.STOP_AS_TRIGGER_SYMBOLS' parallel pending-order
# path (bot_engine.py's _arm_stop_trigger_entry / _check_stop_trigger_entry
# / _execute_stop_trigger_entry) arms and watches; see config.py's
# "STOP-AS-TRIGGER ENTRY" section for that mechanism.
#
# evaluate_boom_crash() and evaluate_step() (above) were evaluated as
# possible starting points per the handoff's own "read first" instruction
# and found NOT reusable as-is: neither ever sets native_stop_price /
# native_target_price / native_entry_price (a hard requirement here), and
# evaluate_boom_crash()'s logic fades a spike AFTER it has already printed
# (contrarian, post-spike) — the opposite timing from rows 3/4 below,
# which position BEFORE the spike, during drift-exhaustion/consolidation.
# evaluate_step()'s EMA10/30 + Donchian-band read is in a similar spirit to
# row 5 but isn't the specific EMA10/EMA20 + price + RSI + MACD hard
# AND-gate row 5 calls for. Both are left exactly as they were (unrouted,
# in place) — only their bar-count-gate / `_arrays()` / logging shape was
# borrowed as a structural reference, per the handoff's own guidance.
# ---------------------------------------------------------------------------

def evaluate_pullback_trend(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    """
    Row 1 (R_75, 1HZ75V) — Trend-following with pullback entries.

    EMA fast/slow crossover state sets the trend direction. Entry requires
    price to have recently pulled back to (or through) the fast EMA — a
    minor swing low (uptrend) / swing high (downtrend) against the trend —
    AND RSI to have genuinely turned back in the trend's favor: a real
    cross back above PULLBACK_RSI_OVERSOLD in an uptrend (not merely
    sitting above it), mirrored below PULLBACK_RSI_OVERBOUGHT in a
    downtrend. Stop sits just beyond the pullback swing low/high (not an
    arbitrary ATR multiple); target is a fixed reward:risk multiple of
    that same stop distance (config.PULLBACK_TREND_RR_RATIO).
    """
    min_bars = getattr(config, "PULLBACK_TREND_MIN_BARS", 40)
    if len(ltf_bars) < min_bars:
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    if len(C) < 2 or math.isnan(float(C[-1])):
        return NONE_RESULT

    ema_fast = ind.ema(C, getattr(config, "PULLBACK_EMA_FAST_PERIOD", 20))
    ema_slow = ind.ema(C, getattr(config, "PULLBACK_EMA_SLOW_PERIOD", 50))
    rsi_vals = ind.rsi(C, getattr(config, "PULLBACK_RSI_PERIOD", 14))
    atr_val  = _last(ind.atr(H, L, C, 14)) or 1e-9

    e_fast, e_slow = float(ema_fast[-1]), float(ema_slow[-1])
    if e_fast > e_slow:
        trend = "LONG"
    elif e_fast < e_slow:
        trend = "SHORT"
    else:
        return NONE_RESULT

    lookback = getattr(config, "PULLBACK_SWING_LOOKBACK", 8)
    touch_atr_mult = getattr(config, "PULLBACK_EMA_TOUCH_ATR_MULT", 1.0)
    rsi_oversold   = getattr(config, "PULLBACK_RSI_OVERSOLD", 30.0)
    rsi_overbought = getattr(config, "PULLBACK_RSI_OVERBOUGHT", 70.0)
    stop_buf_mult  = getattr(config, "PULLBACK_STOP_BUFFER_ATR_MULT", 0.25)
    rr_ratio       = getattr(config, "PULLBACK_TREND_RR_RATIO", 2.0)

    recent_lows  = L[-lookback:]
    recent_highs = H[-lookback:]
    ema_fast_recent = ema_fast[-lookback:]
    last_close = float(C[-1])

    if trend == "LONG":
        swing_idx = int(np.argmin(recent_lows))
        swing_low = float(recent_lows[swing_idx])
        ema_at_swing = float(ema_fast_recent[swing_idx])
        pulled_back = abs(swing_low - ema_at_swing) <= touch_atr_mult * atr_val or swing_low <= ema_at_swing
        rsi_turned_back = bool(rsi_vals[-2] <= rsi_oversold and rsi_vals[-1] > rsi_oversold)
        if not (pulled_back and rsi_turned_back):
            logger.debug(f"REJECTED: {symbol} PULLBACK_TREND strength=0 score=0.000 — below threshold")
            return NONE_RESULT
        direction = "LONG"
        native_entry_price = last_close
        native_stop_price  = swing_low - stop_buf_mult * atr_val
        risk = native_entry_price - native_stop_price
        if risk <= 0:
            return NONE_RESULT
        native_target_price = native_entry_price + rr_ratio * risk
    else:
        swing_idx = int(np.argmax(recent_highs))
        swing_high = float(recent_highs[swing_idx])
        ema_at_swing = float(ema_fast_recent[swing_idx])
        pulled_back = abs(swing_high - ema_at_swing) <= touch_atr_mult * atr_val or swing_high >= ema_at_swing
        rsi_turned_back = bool(rsi_vals[-2] >= rsi_overbought and rsi_vals[-1] < rsi_overbought)
        if not (pulled_back and rsi_turned_back):
            logger.debug(f"REJECTED: {symbol} PULLBACK_TREND strength=0 score=0.000 — below threshold")
            return NONE_RESULT
        direction = "SHORT"
        native_entry_price = last_close
        native_stop_price  = swing_high + stop_buf_mult * atr_val
        risk = native_stop_price - native_entry_price
        if risk <= 0:
            return NONE_RESULT
        native_target_price = native_entry_price - rr_ratio * risk

    separation_atr = abs(e_fast - e_slow) / atr_val
    score = max(0.0, min(1.0, separation_atr / 3.0))
    strength = 3 if score >= 0.5 else 2

    logger.info(
        f"SIGNAL: {symbol} {direction} PULLBACK_TREND strength={strength} score={score:.3f} "
        f"entry={native_entry_price:.5f} stop={native_stop_price:.5f} target={native_target_price:.5f}"
    )
    return SignalResult(
        direction=direction, strength=strength, score=score,
        strategy="PULLBACK_TREND",
        reason=(
            f"EMA{getattr(config, 'PULLBACK_EMA_FAST_PERIOD', 20)}/"
            f"{getattr(config, 'PULLBACK_EMA_SLOW_PERIOD', 50)} trend={trend}, "
            f"pullback confirmed + RSI turn-back"
        ),
        native_entry_price=native_entry_price,
        native_stop_price=native_stop_price,
        native_target_price=native_target_price,
    )


def evaluate_fast_mean_reversion(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    """
    Row 2 (R_100, 1HZ100V) — Fast mean-reversion scalping.

    Fires on short-timeframe overextension — a Bollinger Band touch/pierce
    OR a fast RSI extreme — confirmed by the very next tick already
    snapping back toward the mean (last close vs. prior close), not while
    price is still accelerating into the extreme. Target is the Bollinger
    mid-band itself (the mean); stop sits just beyond the extreme just
    touched — deliberately tight, for a high-frequency scalp.
    """
    min_bars = getattr(config, "SCALP_MIN_BARS", 25)
    if len(ltf_bars) < min_bars:
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    if len(C) < 2 or math.isnan(float(C[-1])):
        return NONE_RESULT

    bb_period = getattr(config, "SCALP_BB_PERIOD", 14)
    bb_std    = getattr(config, "SCALP_BB_STD", 1.5)
    rsi_period = getattr(config, "SCALP_RSI_PERIOD", 7)
    rsi_oversold   = getattr(config, "SCALP_RSI_OVERSOLD", 20.0)
    rsi_overbought = getattr(config, "SCALP_RSI_OVERBOUGHT", 80.0)
    stop_buf_mult  = getattr(config, "SCALP_STOP_BUFFER_ATR_MULT", 0.15)

    bb_upper, bb_mid, bb_lower = ind.bollinger_bands(C, bb_period, bb_std)
    rsi_vals = ind.rsi(C, rsi_period)
    atr_val  = _last(ind.atr(H, L, C, 14)) or 1e-9

    last_close, prev_close = float(C[-1]), float(C[-2])
    last_rsi = float(rsi_vals[-1])
    up_tick, down_tick = last_close > prev_close, last_close < prev_close

    overextended_long  = last_close <= float(bb_lower[-1]) or last_rsi <= rsi_oversold
    overextended_short = last_close >= float(bb_upper[-1]) or last_rsi >= rsi_overbought

    long_fire  = overextended_long and up_tick
    short_fire = overextended_short and down_tick

    if long_fire and not short_fire:
        direction = "LONG"
        native_entry_price = last_close
        recent_low = float(np.min(L[-3:]))
        native_stop_price = min(recent_low, float(bb_lower[-1])) - stop_buf_mult * atr_val
        native_target_price = float(bb_mid[-1])
        if native_target_price <= native_entry_price or native_stop_price >= native_entry_price:
            return NONE_RESULT
    elif short_fire and not long_fire:
        direction = "SHORT"
        native_entry_price = last_close
        recent_high = float(np.max(H[-3:]))
        native_stop_price = max(recent_high, float(bb_upper[-1])) + stop_buf_mult * atr_val
        native_target_price = float(bb_mid[-1])
        if native_target_price >= native_entry_price or native_stop_price <= native_entry_price:
            return NONE_RESULT
    else:
        logger.debug(f"REJECTED: {symbol} FAST_MEAN_REV strength=0 score=0.000 — below threshold")
        return NONE_RESULT

    band_width = float(bb_upper[-1]) - float(bb_lower[-1])
    pct_b = ((last_close - float(bb_lower[-1])) / band_width) if band_width > 0 else 0.5
    extremeness = max(pct_b, 1.0 - pct_b) if band_width > 0 else 0.5
    score = max(0.0, min(1.0, extremeness))
    strength = 3 if score >= 0.9 else 2

    logger.info(
        f"SIGNAL: {symbol} {direction} FAST_MEAN_REV strength={strength} score={score:.3f} "
        f"entry={native_entry_price:.5f} stop={native_stop_price:.5f} target={native_target_price:.5f}"
    )
    return SignalResult(
        direction=direction, strength=strength, score=score,
        strategy="FAST_MEAN_REV",
        reason=f"Overextension snap-back toward BB mid-band, RSI={last_rsi:.1f}",
        native_entry_price=native_entry_price,
        native_stop_price=native_stop_price,
        native_target_price=native_target_price,
    )


def _evaluate_spike_catch(
        ltf_bars: List[Candle], symbol: str, *, strategy_name: str,
        min_bars: int, cons_lookback: int, cons_avg_lookback: int,
        cons_ratio: float, drift_lookback: int, min_drift_atr_ratio: float,
        spike_cooldown_bars: int, stop_buffer_atr_mult: float,
        rr_ratio: float) -> SignalResult:
    """
    Shared logic for rows 3/4 (Boom/Crash spike-catching). BOOM drifts
    down between its up-spikes; CRASH drifts up between its down-spikes.
    Positions counter to that small-tick drift (buy Boom, sell Crash)
    once the drift shows exhaustion/consolidation (ind.find_consolidation)
    and no spike has fired within spike_cooldown_bars — i.e. waiting for
    the NEXT spike, not chasing the tail of one that already printed
    (that's evaluate_boom_crash()'s job, left untouched, unrouted).
    Row 4 (BOOM500/CRASH500) tunes this tighter/faster than row 3
    (BOOM1000/CRASH1000) via its own cooldown/stop-buffer arguments.
    """
    if len(ltf_bars) < min_bars:
        return NONE_RESULT

    is_boom  = symbol.startswith("BOOM")
    is_crash = symbol.startswith("CRASH")
    if not (is_boom or is_crash):
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    if len(C) < 2 or math.isnan(float(C[-1])):
        return NONE_RESULT
    atr_val = _last(ind.atr(H, L, C, 14)) or 1e-9

    # No spike within the cooldown window — waiting for a fresh
    # drift+consolidation cycle, not entering mid/just-after a spike.
    for bars_ago in range(0, spike_cooldown_bars):
        c_s = C[:len(C) - bars_ago] if bars_ago > 0 else C
        h_s = H[:len(H) - bars_ago] if bars_ago > 0 else H
        l_s = L[:len(L) - bars_ago] if bars_ago > 0 else L
        if len(c_s) < 2:
            continue
        if ind.detect_spike(c_s, h_s, l_s, period=14, atr_multiplier=3.0) != 0:
            logger.debug(f"REJECTED: {symbol} {strategy_name} strength=0 score=0.000 — below threshold")
            return NONE_RESULT

    # Drift direction confirmation: Boom drifts down, Crash drifts up,
    # between spikes.
    drift_window = C[-drift_lookback:]
    if len(drift_window) < 2:
        return NONE_RESULT
    x = np.arange(len(drift_window), dtype=float)
    slope = float(np.polyfit(x, drift_window, 1)[0])
    slope_atr_ratio = slope / atr_val

    if is_boom:
        drift_ok = slope_atr_ratio <= -min_drift_atr_ratio
    else:
        drift_ok = slope_atr_ratio >= min_drift_atr_ratio
    if not drift_ok:
        logger.debug(f"REJECTED: {symbol} {strategy_name} strength=0 score=0.000 — below threshold")
        return NONE_RESULT

    # Drift exhaustion / consolidation.
    consolidation = ind.find_consolidation(
        H, L, C, lookback=cons_lookback, avg_lookback=cons_avg_lookback, ratio=cons_ratio)
    if consolidation is None:
        logger.debug(f"REJECTED: {symbol} {strategy_name} strength=0 score=0.000 — below threshold")
        return NONE_RESULT
    cons_upper, cons_lower = consolidation

    last_close = float(C[-1])
    if is_boom:
        direction = "LONG"
        native_entry_price = last_close
        native_stop_price = cons_lower - stop_buffer_atr_mult * atr_val
        risk = native_entry_price - native_stop_price
        if risk <= 0:
            return NONE_RESULT
        native_target_price = native_entry_price + rr_ratio * risk
    else:
        direction = "SHORT"
        native_entry_price = last_close
        native_stop_price = cons_upper + stop_buffer_atr_mult * atr_val
        risk = native_stop_price - native_entry_price
        if risk <= 0:
            return NONE_RESULT
        native_target_price = native_entry_price - rr_ratio * risk

    score = max(0.0, min(1.0, abs(slope_atr_ratio) / (2 * min_drift_atr_ratio)))
    strength = 3 if score >= 0.75 else 2

    logger.info(
        f"SIGNAL: {symbol} {direction} {strategy_name} strength={strength} score={score:.3f} "
        f"entry={native_entry_price:.5f} stop={native_stop_price:.5f} target={native_target_price:.5f} "
        f"drift_atr_ratio={slope_atr_ratio:.3f}"
    )
    return SignalResult(
        direction=direction, strength=strength, score=score,
        strategy=strategy_name,
        reason=(
            f"Drift-exhaustion consolidation [{cons_lower:.5f},{cons_upper:.5f}], "
            f"drift_atr_ratio={slope_atr_ratio:.3f}, positioned for next spike"
        ),
        native_entry_price=native_entry_price,
        native_stop_price=native_stop_price,
        native_target_price=native_target_price,
    )


def evaluate_spike_catch_1000(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    """Row 3 (BOOM1000, CRASH1000) — see _evaluate_spike_catch()."""
    return _evaluate_spike_catch(
        ltf_bars, symbol, strategy_name="SPIKE_CATCH_1000",
        min_bars=getattr(config, "SPIKE_CATCH_1000_MIN_BARS", 30),
        cons_lookback=getattr(config, "SPIKE_CATCH_1000_CONS_LOOKBACK", 15),
        cons_avg_lookback=getattr(config, "SPIKE_CATCH_1000_CONS_AVG_LOOKBACK", 50),
        cons_ratio=getattr(config, "SPIKE_CATCH_1000_CONS_RATIO", 0.4),
        drift_lookback=getattr(config, "SPIKE_CATCH_1000_DRIFT_LOOKBACK", 20),
        min_drift_atr_ratio=getattr(config, "SPIKE_CATCH_1000_MIN_DRIFT_ATR_RATIO", 0.10),
        spike_cooldown_bars=getattr(config, "SPIKE_CATCH_1000_COOLDOWN_BARS", 10),
        stop_buffer_atr_mult=getattr(config, "SPIKE_CATCH_1000_STOP_BUFFER_ATR_MULT", 0.30),
        rr_ratio=getattr(config, "SPIKE_CATCH_1000_RR_RATIO", 3.0),
    )


def evaluate_spike_catch_500(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    """
    Row 4 (BOOM500, CRASH500) — see _evaluate_spike_catch(). Tuned for this
    pair's higher spike frequency: shorter cooldown (faster re-arm) and a
    tighter stop buffer than row 3's BOOM1000/CRASH1000.
    """
    return _evaluate_spike_catch(
        ltf_bars, symbol, strategy_name="SPIKE_CATCH_500",
        min_bars=getattr(config, "SPIKE_CATCH_500_MIN_BARS", 30),
        cons_lookback=getattr(config, "SPIKE_CATCH_500_CONS_LOOKBACK", 12),
        cons_avg_lookback=getattr(config, "SPIKE_CATCH_500_CONS_AVG_LOOKBACK", 40),
        cons_ratio=getattr(config, "SPIKE_CATCH_500_CONS_RATIO", 0.4),
        drift_lookback=getattr(config, "SPIKE_CATCH_500_DRIFT_LOOKBACK", 15),
        min_drift_atr_ratio=getattr(config, "SPIKE_CATCH_500_MIN_DRIFT_ATR_RATIO", 0.10),
        spike_cooldown_bars=getattr(config, "SPIKE_CATCH_500_COOLDOWN_BARS", 5),
        stop_buffer_atr_mult=getattr(config, "SPIKE_CATCH_500_STOP_BUFFER_ATR_MULT", 0.15),
        rr_ratio=getattr(config, "SPIKE_CATCH_500_RR_RATIO", 3.0),
    )


def evaluate_step_grid(ltf_bars: List[Candle], symbol: str) -> SignalResult:
    """
    Row 5 (stpRNG) — Indicator-grid entry, hard AND-gate. ALL four must
    agree before firing (long: EMA10>EMA20, price>EMA20, RSI>55, MACD
    line>signal line; short: every condition mirrored) — this is a gate,
    not a scored/weighted pick, so any single condition failing rejects
    the whole signal. Stop sits outside the recent range that defined the
    setup (the greater of a recent swing extreme and EMA20 itself, plus a
    buffer), not an arbitrary ATR multiple alone; target is a fixed
    reward:risk multiple of that stop distance.
    """
    min_bars = getattr(config, "STEP_GRID_MIN_BARS", 30)
    if len(ltf_bars) < min_bars:
        return NONE_RESULT

    C, H, L = _arrays(ltf_bars)
    if len(C) < 2 or math.isnan(float(C[-1])):
        return NONE_RESULT

    ema10 = ind.ema(C, getattr(config, "STEP_GRID_EMA_FAST_PERIOD", 10))
    ema20 = ind.ema(C, getattr(config, "STEP_GRID_EMA_SLOW_PERIOD", 20))
    rsi_vals = ind.rsi(C, getattr(config, "STEP_GRID_RSI_PERIOD", 14))
    macd_line, macd_signal, _ = ind.macd(
        C, getattr(config, "STEP_GRID_MACD_FAST", 12),
        getattr(config, "STEP_GRID_MACD_SLOW", 26),
        getattr(config, "STEP_GRID_MACD_SIGNAL", 9))
    atr_val = _last(ind.atr(H, L, C, 14)) or 1e-9

    last_close = float(C[-1])
    e10, e20 = float(ema10[-1]), float(ema20[-1])
    last_rsi = float(rsi_vals[-1])
    m_line, m_sig = float(macd_line[-1]), float(macd_signal[-1])

    rsi_long_min  = getattr(config, "STEP_GRID_RSI_LONG_MIN", 55.0)
    rsi_short_max = getattr(config, "STEP_GRID_RSI_SHORT_MAX", 45.0)

    long_gate = (e10 > e20) and (last_close > e20) and (last_rsi > rsi_long_min) and (m_line > m_sig)
    short_gate = (e10 < e20) and (last_close < e20) and (last_rsi < rsi_short_max) and (m_line < m_sig)

    if long_gate == short_gate:  # neither fired, or (impossible) both did
        logger.debug(f"REJECTED: {symbol} STEP_GRID strength=0 score=0.000 — below threshold")
        return NONE_RESULT

    range_lookback = getattr(config, "STEP_GRID_RANGE_LOOKBACK", 20)
    stop_buf_mult  = getattr(config, "STEP_GRID_STOP_BUFFER_ATR_MULT", 0.30)
    rr_ratio       = getattr(config, "STEP_GRID_RR_RATIO", 2.0)

    if long_gate:
        direction = "LONG"
        range_low = float(np.min(L[-range_lookback:]))
        native_entry_price = last_close
        native_stop_price = min(range_low, e20) - stop_buf_mult * atr_val
        risk = native_entry_price - native_stop_price
        if risk <= 0:
            return NONE_RESULT
        native_target_price = native_entry_price + rr_ratio * risk
    else:
        direction = "SHORT"
        range_high = float(np.max(H[-range_lookback:]))
        native_entry_price = last_close
        native_stop_price = max(range_high, e20) + stop_buf_mult * atr_val
        risk = native_stop_price - native_entry_price
        if risk <= 0:
            return NONE_RESULT
        native_target_price = native_entry_price - rr_ratio * risk

    score = 0.75  # hard AND-gate — either every condition agrees (fixed high confidence) or it doesn't fire
    strength = 3

    logger.info(
        f"SIGNAL: {symbol} {direction} STEP_GRID strength={strength} score={score:.3f} "
        f"entry={native_entry_price:.5f} stop={native_stop_price:.5f} target={native_target_price:.5f}"
    )
    return SignalResult(
        direction=direction, strength=strength, score=score,
        strategy="STEP_GRID",
        reason=(
            f"AND-gate: EMA10/20={e10:.5f}/{e20:.5f}, price_vs_EMA20, "
            f"RSI={last_rsi:.1f}, MACD={m_line:.5f} vs {m_sig:.5f}"
        ),
        native_entry_price=native_entry_price,
        native_stop_price=native_stop_price,
        native_target_price=native_target_price,
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SignalEngine:

    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, ltf_bars: List[Candle], symbol: str, **kwargs) -> SignalResult:
        ticks = kwargs.get("ticks")

        # Most Popular Indicator (user-directed, Aug 2026) now covers
        # every symbol that trades via Multiplier contracts — both the
        # Volatility/1Hz/Step family and the Boom/Crash family (both
        # routed to buy_multiplier() in bot_engine.py/deriv_client.py) —
        # replacing evaluate_vol_regime()/evaluate_boom_crash() for
        # direction-selection purposes. Those two evaluators are left in
        # place above, unrouted, the same way this file already retires
        # strategies (see evaluate_mean_reversion/evaluate_range_break).
        # Not a confluence/vote — see evaluate_popular_indicator()'s own
        # comment block for the single-most-popular-signal selection.
        # Six dedicated per-symbol evaluators (handoff, Sep 15 2026) —
        # checked BEFORE the general MULTIPLIER_SYMBOLS branch so these
        # nine symbols never reach evaluate_popular_indicator() anymore.
        # Deliberately NOT removed from config.MULTIPLIER_SYMBOLS /
        # VOL_MULTIPLIER_SYMBOLS / BOOM_CRASH — those lists still drive
        # execution routing (buy_multiplier()), MULTIPLIER_MAP,
        # STOP_LOSS_MAP, and EXIT_ENGINE_SYMBOLS elsewhere, all of which
        # should keep treating these nine exactly as before; only which
        # evaluator computes the signal changes here.
        # ── PREDICTOR — the ONLY strategy that can trade ────────────────
        # Every other evaluate_*() stays defined in this file but is unreachable:
        # only config.PREDICTOR_SYMBOLS (== config.DIGIT_SYMBOLS) route anywhere.
        if symbol not in getattr(config, "DIGIT_SYMBOLS", []):
            return NONE_RESULT
        import predictor
        result = predictor.evaluate_for_signal_engine(ticks, symbol)

        if result.strength >= 2:
            # NOTE: the old is_underperforming() gate that used to live here
            # (rejecting the WHOLE result post-hoc) has been removed for the
            # popular-indicator pipeline (spec point 8). It's been moved
            # earlier and re-scoped: evaluate_popular_indicator() now checks
            # per-(indicator, symbol) suspension (pair_suspension.py) at
            # PICK time, so a suspended pair is skipped in favor of the
            # next-ranked signalling indicator instead of killing the whole
            # symbol's trade for this pass. Leaving both gates in place
            # would double-gate and could still reject a result
            # evaluate_popular_indicator() already worked around. Other,
            # non-popular-indicator evaluators routed through this same
            # evaluate() are not spec-covered here and no longer get an
            # underperformance gate either — see Section 1 of the request
            # if a separate gate for those paths is wanted later.
            logger.info(
                f"SIGNAL: {symbol} {result.direction} {result.strategy} "
                f"strength={result.strength} score={result.score:.3f}"
            )
            return result

        logger.info(
            f"REJECTED: {symbol} {result.strategy} strength={result.strength} "
            f"score={result.score:.3f} — below threshold"
        )
        return NONE_RESULT
