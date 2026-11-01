"""
indicators.py – Pure-numpy trading indicators for Deriv SIFM bot.

Architectural contract (enforced throughout):
  • Every function handles series as short as 15 bars without NaN.
  • Never raises exceptions — all internal errors are caught and a
    neutral/safe value is returned instead.
  • Never returns None.
  • Accepts plain Python lists or numpy arrays.
  • All outputs are numpy arrays (or scalars/tuples where documented).
"""

import math
import numpy as np
from typing import List, Optional, Tuple, Union

ArrayLike = Union[List[float], np.ndarray]


# ─── Internal helpers ──────────────────────────────────────────────────────

def _to(data: ArrayLike) -> np.ndarray:
    """Convert any array-like to a float64 numpy array, silently."""
    try:
        arr = np.asarray(data, dtype=np.float64).flatten()
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr
    except Exception:
        return np.array([], dtype=np.float64)


def _safe_last(arr: np.ndarray, fallback: float = 0.0) -> float:
    try:
        return float(arr[-1]) if len(arr) > 0 else fallback
    except Exception:
        return fallback


def _fill(n: int, value: float) -> np.ndarray:
    try:
        return np.full(max(n, 0), value, dtype=np.float64)
    except Exception:
        return np.array([], dtype=np.float64)


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Growing-window SMA — no NaN, same length as input."""
    n = len(data)
    if n == 0:
        return np.array([], dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        out[i] = float(np.mean(data[start:i + 1]))
    return out


# ─── EMA ────────────────────────────────────────────────────────────────────

def ema(closes: ArrayLike, period: int) -> np.ndarray:
    """
    Standard EMA. Returns array same length as closes.
    For series shorter than period, fills with growing-window SMA.
    """
    try:
        p = _to(closes)
        n = len(p)
        if n == 0:
            return np.array([], dtype=np.float64)
        period = max(int(period), 1)

        if n < period:
            return _sma(p, period)

        out = np.empty(n, dtype=np.float64)
        k = 2.0 / (period + 1.0)
        # Back-fill pre-seed positions with growing-window SMA
        for i in range(period - 1):
            out[i] = float(np.mean(p[:i + 1]))
        out[period - 1] = float(np.mean(p[:period]))
        for i in range(period, n):
            out[i] = float(p[i]) * k + out[i - 1] * (1.0 - k)
        return out
    except Exception:
        p = _to(closes)
        return _fill(len(p), _safe_last(p))


# ─── RSI ────────────────────────────────────────────────────────────────────

def rsi(closes: ArrayLike, period: int = 14) -> np.ndarray:
    """
    Standard RSI 0-100. Returns array same length as closes.
    Minimum 15 bars required; returns array of 50s (neutral) otherwise.
    """
    try:
        p = _to(closes)
        n = len(p)
        if n == 0:
            return np.array([], dtype=np.float64)
        if n < 15:
            return _fill(n, 50.0)

        out = _fill(n, 50.0)
        delta = np.diff(p)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        avg_g = float(np.mean(gains[:period]))
        avg_l = float(np.mean(losses[:period]))

        def _val(ag: float, al: float) -> float:
            if al == 0.0:
                return 100.0 if ag > 0 else 50.0
            return 100.0 - 100.0 / (1.0 + ag / al)

        out[period] = _val(avg_g, avg_l)
        for i in range(period, len(delta)):
            avg_g = (avg_g * (period - 1) + gains[i]) / period
            avg_l = (avg_l * (period - 1) + losses[i]) / period
            out[i + 1] = _val(avg_g, avg_l)
        return out
    except Exception:
        p = _to(closes)
        return _fill(len(p), 50.0)


# ─── MACD ───────────────────────────────────────────────────────────────────

def macd(
    closes: ArrayLike,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (macd_line, signal_line, histogram) arrays, same length as input.
    """
    try:
        p = _to(closes)
        n = len(p)
        if n == 0:
            return (np.array([], dtype=np.float64),) * 3

        fast_ema = ema(p, fast)
        slow_ema = ema(p, slow)
        macd_line = fast_ema - slow_ema
        sig_line = ema(macd_line, signal)
        hist = macd_line - sig_line
        return macd_line, sig_line, hist
    except Exception:
        p = _to(closes)
        z = _fill(len(p), 0.0)
        return z.copy(), z.copy(), z.copy()


# ─── Bollinger Bands ────────────────────────────────────────────────────────

def bollinger_bands(
    closes: ArrayLike,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (upper, mid, lower) arrays, same length as closes.
    """
    try:
        p = _to(closes)
        n = len(p)
        if n == 0:
            return (np.array([], dtype=np.float64),) * 3

        mid = _sma(p, period)
        std = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - period + 1)
            std[i] = float(np.std(p[start:i + 1], ddof=0))

        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return upper, mid, lower
    except Exception:
        p = _to(closes)
        neutral = _safe_last(p)
        a = _fill(len(p), neutral)
        return a.copy(), a.copy(), a.copy()


# ─── ATR ────────────────────────────────────────────────────────────────────

def atr(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 14,
) -> np.ndarray:
    """
    Average True Range (Wilder smoothing). Returns array same length as closes.
    No NaN for series >= 15 bars.
    """
    try:
        H, L, C = _to(highs), _to(lows), _to(closes)
        n = len(C)
        if n == 0:
            return np.array([], dtype=np.float64)
        if len(H) != n:
            H = C.copy()
        if len(L) != n:
            L = C.copy()

        tr = np.empty(n, dtype=np.float64)
        tr[0] = float(H[0]) - float(L[0])
        for i in range(1, n):
            tr[i] = max(
                float(H[i]) - float(L[i]),
                abs(float(H[i]) - float(C[i - 1])),
                abs(float(L[i]) - float(C[i - 1])),
            )

        out = np.empty(n, dtype=np.float64)
        if n < period:
            out = _sma(tr, period)
            return out

        out[period - 1] = float(np.mean(tr[:period]))
        for i in range(period - 1):
            out[i] = float(np.mean(tr[:i + 1]))
        for i in range(period, n):
            out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
        return out
    except Exception:
        C = _to(closes)
        return _fill(len(C), 0.0)


# ─── Stochastic Oscillator ──────────────────────────────────────────────────

def stochastic(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (%K, %D) arrays, 0-100, same length as closes.
    %K crossing up from below 25 = bullish.
    """
    try:
        H, L, C = _to(highs), _to(lows), _to(closes)
        n = len(C)
        if n == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        if len(H) != n:
            H = C.copy()
        if len(L) != n:
            L = C.copy()

        k_raw = _fill(n, 50.0)
        for i in range(n):
            start = max(0, i - k_period + 1)
            hh = float(np.max(H[start:i + 1]))
            ll = float(np.min(L[start:i + 1]))
            if hh == ll:
                k_raw[i] = 50.0
            else:
                k_raw[i] = 100.0 * (float(C[i]) - ll) / (hh - ll)

        d = _sma(k_raw, d_period)
        return k_raw, d
    except Exception:
        C = _to(closes)
        n = len(C)
        return _fill(n, 50.0), _fill(n, 50.0)


# ─── Rate of Change ─────────────────────────────────────────────────────────

def rate_of_change(closes: ArrayLike, period: int = 10) -> np.ndarray:
    """
    ROC = (close - close[n periods ago]) / close[n periods ago]
    Returns array same length as closes. Early bars use largest available lag.
    """
    try:
        p = _to(closes)
        n = len(p)
        if n == 0:
            return np.array([], dtype=np.float64)

        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            lag = min(period, i)
            ref = float(p[i - lag]) if lag > 0 else float(p[i])
            if ref == 0.0:
                out[i] = 0.0
            else:
                out[i] = (float(p[i]) - ref) / ref
        return out
    except Exception:
        p = _to(closes)
        return _fill(len(p), 0.0)


# ─── Donchian Channel ───────────────────────────────────────────────────────

def donchian_channel(
    highs: ArrayLike,
    lows: ArrayLike,
    period: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (upper, lower, mid) arrays, same length as highs.
    Upper = highest high, Lower = lowest low over period.
    """
    try:
        H, L = _to(highs), _to(lows)
        n = len(H)
        if n == 0:
            return (np.array([], dtype=np.float64),) * 3
        if len(L) != n:
            L = H.copy()

        upper = np.empty(n, dtype=np.float64)
        lower = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - period + 1)
            upper[i] = float(np.max(H[start:i + 1]))
            lower[i] = float(np.min(L[start:i + 1]))
        mid = (upper + lower) / 2.0
        return upper, lower, mid
    except Exception:
        H = _to(highs)
        neutral = _safe_last(H)
        a = _fill(len(H), neutral)
        return a.copy(), a.copy(), a.copy()


# ─── Consolidation Detector ─────────────────────────────────────────────────

def find_consolidation(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    lookback: int = 15,
    avg_lookback: int = 50,
    ratio: float = 0.4,
) -> Optional[Tuple[float, float]]:
    """
    Returns (upper, lower) bounds of the current consolidation zone if the
    recent lookback range is tight (< ratio * avg_lookback range), else None.

    Used for Range Break strategy. signal_engine.py calls this as
    find_consolidation(H, L, C) and does:
        consolidation = ind.find_consolidation(H, L, C)
        has_consolidation = consolidation is not None
        if has_consolidation:
            cons_upper, cons_lower = consolidation
    — so the argument order and None/tuple return here must match that.
    """
    try:
        H, L, C = _to(highs), _to(lows), _to(closes)
        n = len(C)
        if len(H) != n:
            H = C.copy()
        if len(L) != n:
            L = C.copy()
        if n < lookback:
            return None

        recent_start = max(0, n - lookback)
        recent_high  = float(np.max(H[recent_start:n]))
        recent_low   = float(np.min(L[recent_start:n]))
        recent_range = recent_high - recent_low

        avg_start    = max(0, n - avg_lookback)
        avg_window_h = H[avg_start:n]
        avg_window_l = L[avg_start:n]
        if len(avg_window_h) == 0:
            return None
        avg_range = float(np.max(avg_window_h) - np.min(avg_window_l))

        if avg_range == 0.0:
            return None
        if recent_range < ratio * avg_range:
            return (recent_high, recent_low)
        return None
    except Exception:
        return None


# ─── Spike Detector ─────────────────────────────────────────────────────────

def detect_spike(
    closes: ArrayLike,
    highs: ArrayLike,
    lows: ArrayLike,
    period: int = 14,
    atr_multiplier: float = 3.0,
) -> int:
    """
    Returns +1 (upward spike), -1 (downward spike), 0 (none).
    Spike = single bar (last bar) moves more than atr_multiplier * ATR(period).
    """
    try:
        C = _to(closes)
        H = _to(highs)
        L = _to(lows)
        n = len(C)
        if n < 2:
            return 0
        if len(H) != n:
            H = C.copy()
        if len(L) != n:
            L = C.copy()

        atr_vals = atr(H, L, C, period)
        current_atr = _safe_last(atr_vals, 0.0)
        if current_atr <= 0.0:
            return 0

        move = float(C[-1]) - float(C[-2])
        threshold = atr_multiplier * current_atr
        if abs(move) > threshold:
            return 1 if move > 0 else -1
        return 0
    except Exception:
        return 0


# ─── Digit Over/Under Score ─────────────────────────────────────────────────

def digit_score(
    closes: ArrayLike,
    rsi_vals: ArrayLike,
    bb_upper: ArrayLike,
    bb_lower: ArrayLike,
    roc_vals: ArrayLike,
) -> Tuple[int, str]:
    """
    Scoring system for digit over/under strategy, using the latest values
    of each input series. Returns (score, direction) where
    direction = "OVER" | "UNDER" | "NONE".

    FIX (Task 3 — full textbook confirmation, no partial firing, Aug 2026):
    previously fired on over_score/under_score >= 6 out of a possible 8
    (RSI extreme=3, BB touch=3, ROC momentum=2) — reachable with only 2 of
    the 3 documented conditions true (RSI extreme + BB touch = 6, with no
    ROC confirmation at all). That's a confidence-threshold standing in
    for a missing condition, exactly the pattern this task asks to close.
    Now requires ALL THREE conditions (RSI extreme AND price at/through
    the matching Bollinger Band AND ROC momentum agreeing) before firing
    either direction — the score is still returned (now always 8 when it
    fires, since all three components are mandatory) purely for the
    caller's existing score/strength scaling, not as a substitute gate.
    """
    try:
        C = _to(closes)
        R = _to(rsi_vals)
        BU = _to(bb_upper)
        BL = _to(bb_lower)
        ROC = _to(roc_vals)

        if len(C) == 0 or len(R) == 0 or len(BU) == 0 or len(BL) == 0 or len(ROC) == 0:
            return 0, "NONE"

        close_v = _safe_last(C)
        rsi_v = _safe_last(R, 50.0)
        bbu_v = _safe_last(BU, close_v)
        bbl_v = _safe_last(BL, close_v)
        roc_v = _safe_last(ROC, 0.0)

        over_rsi   = rsi_v < 22
        over_bb    = close_v <= bbl_v
        over_roc   = roc_v < -0.02
        over_score = (3 if over_rsi else 0) + (3 if over_bb else 0) + (2 if over_roc else 0)

        under_rsi   = rsi_v > 78
        under_bb    = close_v >= bbu_v
        under_roc   = roc_v > 0.02
        under_score = (3 if under_rsi else 0) + (3 if under_bb else 0) + (2 if under_roc else 0)

        if over_rsi and over_bb and over_roc:
            return over_score, "OVER"
        if under_rsi and under_bb and under_roc:
            return under_score, "UNDER"
        return max(over_score, under_score), "NONE"
    except Exception:
        return 0, "NONE"


# ─── RSI Divergence ─────────────────────────────────────────────────────────

def find_rsi_divergence(
    closes: ArrayLike,
    rsi_vals: ArrayLike,
    lookback: int = 20,
) -> int:
    """
    Bullish divergence (price lower low, RSI higher low): returns +1
    Bearish divergence (price higher high, RSI lower high): returns -1
    None: returns 0. Works on a minimum of 20 bars.
    """
    try:
        C = _to(closes)
        R = _to(rsi_vals)
        n = min(len(C), len(R))
        if n < lookback or n < 20:
            return 0

        C = C[-lookback:]
        R = R[-lookback:]
        m = len(C)
        half = max(1, m // 2)

        first_c = C[:half]
        second_c = C[half:]
        first_r = R[:half]
        second_r = R[half:]

        if len(second_c) == 0 or len(first_c) == 0:
            return 0

        low1_idx = int(np.argmin(first_c))
        low2_idx = int(np.argmin(second_c))
        price_low1 = float(first_c[low1_idx])
        price_low2 = float(second_c[low2_idx])
        rsi_low1 = float(first_r[low1_idx])
        rsi_low2 = float(second_r[low2_idx])

        if price_low2 < price_low1 and rsi_low2 > rsi_low1:
            return 1

        high1_idx = int(np.argmax(first_c))
        high2_idx = int(np.argmax(second_c))
        price_high1 = float(first_c[high1_idx])
        price_high2 = float(second_c[high2_idx])
        rsi_high1 = float(first_r[high1_idx])
        rsi_high2 = float(second_r[high2_idx])

        if price_high2 > price_high1 and rsi_high2 < rsi_high1:
            return -1

        return 0
    except Exception:
        return 0


# ─── SMA ────────────────────────────────────────────────────────────────────

def sma(closes: ArrayLike, period: int) -> np.ndarray:
    """
    Simple Moving Average — one of the single most widely used technical
    indicators. Growing-window for series shorter than period (no NaN).
    """
    try:
        p = _to(closes)
        return _sma(p, max(int(period), 1))
    except Exception:
        p = _to(closes)
        return _fill(len(p), _safe_last(p))


# ─── ADX / +DI / -DI (Wilder's Directional Movement System) ─────────────────

def adx(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (adx, plus_di, minus_di) — Welles Wilder's Average Directional
    Index plus its two directional components. One of the most cited
    trend-strength/trend-direction indicators in general use. Same
    never-raises / no-NaN / same-length-as-input contract as the rest of
    this module.
    """
    try:
        H, L, C = _to(highs), _to(lows), _to(closes)
        n = len(C)
        if n == 0:
            return (np.array([], dtype=np.float64),) * 3
        if len(H) != n:
            H = C.copy()
        if len(L) != n:
            L = C.copy()
        if n < period + 1:
            z = _fill(n, 0.0)
            return z.copy(), z.copy(), z.copy()

        up_move = np.zeros(n, dtype=np.float64)
        down_move = np.zeros(n, dtype=np.float64)
        tr = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            um = float(H[i]) - float(H[i - 1])
            dm = float(L[i - 1]) - float(L[i])
            up_move[i] = um if (um > dm and um > 0) else 0.0
            down_move[i] = dm if (dm > um and dm > 0) else 0.0
            tr[i] = max(
                float(H[i]) - float(L[i]),
                abs(float(H[i]) - float(C[i - 1])),
                abs(float(L[i]) - float(C[i - 1])),
            )
        tr[0] = float(H[0]) - float(L[0])

        atr_w = _fill(n, 0.0)
        plus_dm_w = _fill(n, 0.0)
        minus_dm_w = _fill(n, 0.0)
        atr_w[period] = float(np.sum(tr[1:period + 1]))
        plus_dm_w[period] = float(np.sum(up_move[1:period + 1]))
        minus_dm_w[period] = float(np.sum(down_move[1:period + 1]))
        for i in range(period + 1, n):
            atr_w[i] = atr_w[i - 1] - (atr_w[i - 1] / period) + tr[i]
            plus_dm_w[i] = plus_dm_w[i - 1] - (plus_dm_w[i - 1] / period) + up_move[i]
            minus_dm_w[i] = minus_dm_w[i - 1] - (minus_dm_w[i - 1] / period) + down_move[i]

        plus_di = _fill(n, 0.0)
        minus_di = _fill(n, 0.0)
        for i in range(period, n):
            if atr_w[i] > 0:
                plus_di[i] = 100.0 * plus_dm_w[i] / atr_w[i]
                minus_di[i] = 100.0 * minus_dm_w[i] / atr_w[i]

        dx = _fill(n, 0.0)
        for i in range(period, n):
            s = plus_di[i] + minus_di[i]
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / s if s > 0 else 0.0

        adx_out = _fill(n, 0.0)
        first_adx_idx = min(2 * period, n - 1)
        if first_adx_idx > period:
            adx_out[first_adx_idx] = float(np.mean(dx[period:first_adx_idx + 1]))
            for i in range(first_adx_idx + 1, n):
                adx_out[i] = (adx_out[i - 1] * (period - 1) + dx[i]) / period
        return adx_out, plus_di, minus_di
    except Exception:
        C = _to(closes)
        z = _fill(len(C), 0.0)
        return z.copy(), z.copy(), z.copy()


# ─── Parabolic SAR ────────────────────────────────────────────────────────

def parabolic_sar(
    highs: ArrayLike,
    lows: ArrayLike,
    step: float = 0.02,
    max_step: float = 0.20,
) -> np.ndarray:
    """
    Wilder's Parabolic SAR — classic trend-following stop-and-reverse dot
    series, extremely widely used for trailing stops and trend direction
    (price above SAR = uptrend, below = downtrend). Returns array same
    length as highs.
    """
    try:
        H, L = _to(highs), _to(lows)
        n = len(H)
        if len(L) != n:
            L = H.copy()
        if n < 3:
            return H.copy()

        sar = np.empty(n, dtype=np.float64)
        uptrend = H[1] >= H[0]
        af = step
        ep = float(H[0]) if uptrend else float(L[0])
        sar[0] = float(L[0]) if uptrend else float(H[0])

        for i in range(1, n):
            prev_sar = sar[i - 1]
            new_sar = prev_sar + af * (ep - prev_sar)

            if uptrend:
                new_sar = min(new_sar, float(L[i - 1]), float(L[max(0, i - 2)]))
                if float(L[i]) < new_sar:
                    uptrend = False
                    new_sar = ep
                    ep = float(L[i])
                    af = step
                else:
                    if float(H[i]) > ep:
                        ep = float(H[i])
                        af = min(af + step, max_step)
            else:
                new_sar = max(new_sar, float(H[i - 1]), float(H[max(0, i - 2)]))
                if float(H[i]) > new_sar:
                    uptrend = True
                    new_sar = ep
                    ep = float(H[i])
                    af = step
                else:
                    if float(L[i]) < ep:
                        ep = float(L[i])
                        af = min(af + step, max_step)

            sar[i] = new_sar
        return sar
    except Exception:
        H = _to(highs)
        return _fill(len(H), _safe_last(H))


# ─── Ichimoku Cloud ─────────────────────────────────────────────────────────

def ichimoku(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b), each
    same length as closes and NOT shifted forward (i.e. these are the
    cloud values as of the current bar, for "is price above/below the
    cloud right now" reads rather than plotting the traditional
    26-bar-forward-shifted cloud). One of the most widely used
    all-in-one trend systems, especially in FX and index trading.
    """
    try:
        H, L, C = _to(highs), _to(lows), _to(closes)
        n = len(C)
        if len(H) != n:
            H = C.copy()
        if len(L) != n:
            L = C.copy()
        if n == 0:
            return (np.array([], dtype=np.float64),) * 4

        def _mid(period: int) -> np.ndarray:
            out = np.empty(n, dtype=np.float64)
            for i in range(n):
                start = max(0, i - period + 1)
                out[i] = (float(np.max(H[start:i + 1])) + float(np.min(L[start:i + 1]))) / 2.0
            return out

        tenkan = _mid(tenkan_period)
        kijun = _mid(kijun_period)
        senkou_a = (tenkan + kijun) / 2.0
        senkou_b = _mid(senkou_b_period)
        return tenkan, kijun, senkou_a, senkou_b
    except Exception:
        C = _to(closes)
        neutral = _safe_last(C)
        a = _fill(len(C), neutral)
        return a.copy(), a.copy(), a.copy(), a.copy()


# ─── CCI (Commodity Channel Index) ───────────────────────────────────────────

def cci(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 20,
) -> np.ndarray:
    """
    Commodity Channel Index. >100 = overbought, <-100 = oversold. Same
    never-raises contract as the rest of this module.
    """
    try:
        H, L, C = _to(highs), _to(lows), _to(closes)
        n = len(C)
        if len(H) != n:
            H = C.copy()
        if len(L) != n:
            L = C.copy()
        if n == 0:
            return np.array([], dtype=np.float64)

        tp = (H + L + C) / 3.0
        sma_tp = _sma(tp, period)
        mean_dev = np.empty(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - period + 1)
            window = tp[start:i + 1]
            mean_dev[i] = float(np.mean(np.abs(window - sma_tp[i])))

        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if mean_dev[i] > 0:
                out[i] = (tp[i] - sma_tp[i]) / (0.015 * mean_dev[i])
        return out
    except Exception:
        C = _to(closes)
        return _fill(len(C), 0.0)


# ─── Williams %R ──────────────────────────────────────────────────────────

def williams_r(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 14,
) -> np.ndarray:
    """
    Williams %R, -100..0. Above -20 = overbought, below -80 = oversold.
    Mirror-image cousin of the Stochastic Oscillator and just as widely
    used.
    """
    try:
        H, L, C = _to(highs), _to(lows), _to(closes)
        n = len(C)
        if len(H) != n:
            H = C.copy()
        if len(L) != n:
            L = C.copy()
        if n == 0:
            return np.array([], dtype=np.float64)

        out = _fill(n, -50.0)
        for i in range(n):
            start = max(0, i - period + 1)
            hh = float(np.max(H[start:i + 1]))
            ll = float(np.min(L[start:i + 1]))
            if hh == ll:
                out[i] = -50.0
            else:
                out[i] = -100.0 * (hh - float(C[i])) / (hh - ll)
        return out
    except Exception:
        C = _to(closes)
        return _fill(len(C), -50.0)


# ─── Supertrend ───────────────────────────────────────────────────────────

def supertrend(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 10,
    multiplier: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (supertrend_line, direction) where direction[i] is +1.0
    (uptrend, line sits below price) or -1.0 (downtrend, line sits above
    price). One of the most popular single-line ATR-based trend-following
    overlays on modern charting platforms.
    """
    try:
        H, L, C = _to(highs), _to(lows), _to(closes)
        n = len(C)
        if len(H) != n:
            H = C.copy()
        if len(L) != n:
            L = C.copy()
        if n == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        atr_vals = atr(H, L, C, period)
        hl2 = (H + L) / 2.0
        upper_basic = hl2 + multiplier * atr_vals
        lower_basic = hl2 - multiplier * atr_vals

        upper_final = np.empty(n, dtype=np.float64)
        lower_final = np.empty(n, dtype=np.float64)
        st = np.empty(n, dtype=np.float64)
        direction = np.empty(n, dtype=np.float64)

        upper_final[0] = upper_basic[0]
        lower_final[0] = lower_basic[0]
        direction[0] = 1.0
        st[0] = lower_final[0]

        for i in range(1, n):
            upper_final[i] = (upper_basic[i] if (upper_basic[i] < upper_final[i - 1]
                               or float(C[i - 1]) > upper_final[i - 1]) else upper_final[i - 1])
            lower_final[i] = (lower_basic[i] if (lower_basic[i] > lower_final[i - 1]
                               or float(C[i - 1]) < lower_final[i - 1]) else lower_final[i - 1])

            if direction[i - 1] == 1.0:
                direction[i] = -1.0 if float(C[i]) < lower_final[i] else 1.0
            else:
                direction[i] = 1.0 if float(C[i]) > upper_final[i] else -1.0

            st[i] = lower_final[i] if direction[i] == 1.0 else upper_final[i]

        return st, direction
    except Exception:
        C = _to(closes)
        z = _fill(len(C), 0.0)
        return z.copy(), _fill(len(C), 1.0)


# ─── Pivot Points (classic/floor) ────────────────────────────────────────

def pivot_points(
    prior_high: float,
    prior_low: float,
    prior_close: float,
) -> Tuple[float, float, float, float, float]:
    """
    Classic floor-trader pivot points from the prior period's H/L/C.
    Returns (pivot, r1, s1, r2, s2). One of the most widely used
    support/resistance frameworks, especially intraday.
    """
    try:
        p = (float(prior_high) + float(prior_low) + float(prior_close)) / 3.0
        r1 = 2.0 * p - float(prior_low)
        s1 = 2.0 * p - float(prior_high)
        r2 = p + (float(prior_high) - float(prior_low))
        s2 = p - (float(prior_high) - float(prior_low))
        return p, r1, s1, r2, s2
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0


# ─── Kalman Filter adaptive trend estimate ("cutting edge" addition) ────────

def kalman_trend(
    closes: ArrayLike,
    process_var: float = 1e-5,
    measure_var: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1-D constant-velocity Kalman filter over price. Returns (level,
    velocity) arrays, same length as closes. Unlike a fixed-lookback
    moving average, the filter's own gain adapts every bar to how noisy
    recent price has actually been, so it tracks genuine trend changes
    faster in choppy/high-volatility conditions and smooths harder in
    calm ones. velocity[i] > 0 reads as an up-trending state estimate,
    velocity[i] < 0 as down-trending — used here as one confluence vote
    plus (via its magnitude) a confidence weight.
    """
    try:
        p = _to(closes)
        n = len(p)
        if n == 0:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        # State: [level, velocity]. Constant-velocity motion model.
        x = np.array([p[0], 0.0], dtype=np.float64)
        P = np.eye(2, dtype=np.float64) * 1.0
        F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        Q = np.array([[process_var, 0.0], [0.0, process_var]], dtype=np.float64)
        H_mat = np.array([[1.0, 0.0]], dtype=np.float64)
        R = np.array([[measure_var]], dtype=np.float64)

        levels = np.empty(n, dtype=np.float64)
        velocities = np.empty(n, dtype=np.float64)
        levels[0] = x[0]
        velocities[0] = x[1]

        for i in range(1, n):
            # Predict
            x = F @ x
            P = F @ P @ F.T + Q
            # Update
            z = float(p[i])
            y = z - float(H_mat @ x)
            S = float(H_mat @ P @ H_mat.T) + float(R[0, 0])
            if S <= 0:
                S = 1e-9
            K = (P @ H_mat.T) / S  # 2x1 Kalman gain
            x = x + (K.flatten() * y)
            P = (np.eye(2) - K @ H_mat) @ P

            levels[i] = x[0]
            velocities[i] = x[1]

        return levels, velocities
    except Exception:
        p = _to(closes)
        z = _fill(len(p), _safe_last(p))
        return z, _fill(len(p), 0.0)


# ─── Hurst Exponent regime detector ("cutting edge" addition) ───────────────

def hurst_exponent(closes: ArrayLike, min_lag: int = 2, max_lag: int = 20) -> float:
    """
    Rescaled-range-style Hurst exponent estimate over the full series
    passed in (caller controls the lookback by slicing). H > 0.5 implies
    a trending/persistent series (momentum/trend-following indicators
    should be trusted more), H < 0.5 implies mean-reverting/anti-
    persistent behaviour (oscillator/mean-reversion indicators should be
    trusted more), H ~= 0.5 implies a random walk (no regime edge either
    way). Returns 0.5 (neutral) on any error or insufficient data —
    never raises, never returns None.
    """
    try:
        p = _to(closes)
        n = len(p)
        if n < max(20, max_lag * 2):
            return 0.5

        lags = range(min_lag, min(max_lag, n // 2))
        tau = []
        used_lags = []
        for lag in lags:
            diffs = p[lag:] - p[:-lag]
            if len(diffs) < 2:
                continue
            std = float(np.std(diffs))
            if std > 0:
                tau.append(std)
                used_lags.append(lag)

        if len(tau) < 2:
            return 0.5

        log_lags = np.log(np.array(used_lags, dtype=np.float64))
        log_tau = np.log(np.array(tau, dtype=np.float64))
        # Slope of log(tau) vs log(lag) ≈ Hurst exponent (standard
        # variance-scaling estimator: Var(lag) ~ lag^(2H)).
        slope, _intercept = np.polyfit(log_lags, log_tau, 1)
        h = float(slope)
        if math.isnan(h) or math.isinf(h):
            return 0.5
        return max(0.0, min(1.0, h))
    except Exception:
        return 0.5


# ─── Compatibility aliases ───────────────────────────────────────────────────
# signal_engine.py calls these under the shorter names below. Kept as thin
# wrappers so the original functions/docstrings above stay untouched.

def roc(closes: ArrayLike, period: int = 10) -> np.ndarray:
    """Alias for rate_of_change(), matching the name signal_engine.py calls."""
    return rate_of_change(closes, period)


def donchian(
    highs: ArrayLike,
    lows: ArrayLike,
    period: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alias for donchian_channel(), but returns only (upper, lower) — matching
    the 2-value unpack signal_engine.py does (it doesn't use the mid band).
    """
    upper, lower, _mid = donchian_channel(highs, lows, period)
    return upper, lower
