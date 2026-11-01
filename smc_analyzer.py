"""
volatility_analyzer.py (replaces smc_analyzer.py)
Regime detection for Deriv synthetic indices.
Determines if current conditions favour momentum
trading and which direction has the edge.
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import indicators as ind
from candlestick_builder import Candle

logger = logging.getLogger(__name__)


@dataclass
class SMCContext:
    """
    Renamed fields kept for compatibility
    with bot_engine.py imports.
    bias              = overall momentum direction
    structure         = regime type
    confluence_score  = signal quality 0-1
    zone_freshness    = momentum freshness 0-1
    """
    bias:             str    = "NEUTRAL"
    structure:        str    = "NEUTRAL"
    confluence_score: float  = 0.0
    zone_freshness:   float  = 1.0
    regime:           str    = "RANGING"
    momentum_score:   float  = 0.0
    direction:        int    = 0
    order_blocks:     list   = field(default_factory=list)
    fvgs:             list   = field(default_factory=list)
    breakers:         list   = field(default_factory=list)
    sweep_detected:   int    = 0
    mss_detected:     int    = 0
    in_ob:            bool   = False
    in_fvg:           bool   = False
    in_breaker:       bool   = False
    in_premium:       str    = "EQUILIBRIUM"
    nearest_ob:       object = None
    nearest_fvg:      object = None
    # Legacy compatibility aliases (read-only via properties)
    current_atr:      float  = 0.0
    swing_highs:      list   = field(default_factory=list)
    swing_lows:       list   = field(default_factory=list)
    bullish_obs:      list   = field(default_factory=list)
    bearish_obs:      list   = field(default_factory=list)
    bullish_fvgs:     list   = field(default_factory=list)
    bearish_fvgs:     list   = field(default_factory=list)
    liquidity_highs:  list   = field(default_factory=list)
    liquidity_lows:   list   = field(default_factory=list)


class SMCAnalyzer:
    """
    Volatility / momentum regime analyzer for Deriv synthetic indices.
    Public API is fully backward-compatible with the old SMCAnalyzer
    so bot_engine.py requires zero changes.
    """

    def __init__(self, ob_expiry_bars: int = 50, **kwargs):
        # ob_expiry_bars kept for signature compatibility; unused
        self.ob_expiry_bars = ob_expiry_bars

    # ── Primary entry point ───────────────────────────────────────────────────

    def analyse(
        self,
        htf_bars: List[Candle],
        mtf_bars: List[Candle] = None,
        current_price: float = 0.0,
        # Legacy keyword args accepted and silently ignored
        atr: float = 0.0,
        symbol: str = "",
        **kwargs,
    ) -> SMCContext:
        """
        Analyse momentum regime on synthetic index.
        Returns SMCContext compatible with existing bot_engine.py code.

        Parameters
        ----------
        htf_bars      : higher-timeframe candles (primary analysis)
        mtf_bars      : optional mid-timeframe candles (unused, kept for API compat)
        current_price : current market price (fallback if bars are empty)
        atr           : legacy param, ignored
        symbol        : legacy param, ignored
        """
        # Minimum bar requirement
        if len(htf_bars) < 15:
            logger.debug("analyse: insufficient bars (%d < 15) → NEUTRAL", len(htf_bars))
            return SMCContext(bias="NEUTRAL", structure="NEUTRAL")

        H = np.array([b.high  for b in htf_bars], dtype=float)
        L = np.array([b.low   for b in htf_bars], dtype=float)
        C = np.array([b.close for b in htf_bars], dtype=float)

        # ── Core indicators ───────────────────────────────────────────────────

        # Composite momentum score [0.0–1.0] and direction {-1, 0, 1}
        score, direction = ind.momentum_score(C, H, L)

        # Volatility regime: "RANGING", "TRENDING", "EXPLOSIVE"
        regime = ind.volatility_regime(C, H, L)

        # Trend strength scalar [0.0–1.0]
        strength = ind.detect_trend_strength(C)

        # Momentum shift: positive = bullish shift, negative = bearish, 0 = none
        shift = ind.detect_momentum_shift(C, H, L)

        # ── Bias ─────────────────────────────────────────────────────────────
        if direction == 0 or score < 0.35:
            bias = "NEUTRAL"
        elif direction > 0:
            bias = "LONG"
        else:
            bias = "SHORT"

        logger.debug(
            "analyse: score=%.3f dir=%d regime=%s strength=%.3f shift=%d bias=%s",
            score, direction, regime, strength, shift, bias,
        )

        # ── Confluence score ──────────────────────────────────────────────────
        conf = float(score)

        # Reward aligned momentum shift
        if shift != 0 and (
            (shift > 0 and bias == "LONG") or
            (shift < 0 and bias == "SHORT")
        ):
            conf = min(conf + 0.15, 1.0)

        # Reward explosive regime (strong directional energy)
        if regime == "EXPLOSIVE":
            conf = min(conf + 0.10, 1.0)

        # ── Zone freshness — how fresh is current momentum ────────────────────
        freshness = min(float(strength) + 0.2, 1.0)

        return SMCContext(
            bias             = bias,
            structure        = regime,
            confluence_score = conf,
            zone_freshness   = freshness,
            regime           = regime,
            momentum_score   = score,
            direction        = direction,
            # Map momentum state onto legacy SMC boolean flags so any
            # bot_engine.py code that reads these still gets a sensible value
            in_ob            = score >= 0.50,
            in_fvg           = score >= 0.60,
            sweep_detected   = int(shift),
            mss_detected     = int(shift),
        )

    # ── Legacy wrappers ───────────────────────────────────────────────────────

    def price_in_smc_zone(
        self,
        price: float,
        bias: str,
        ctx: SMCContext,
    ) -> bool:
        """
        Legacy name kept for bot_engine.py compatibility.
        Delegates to price_in_zone().
        """
        return self.price_in_zone(price, ctx)

    def price_in_zone(
        self,
        price: float,
        ctx: SMCContext,
    ) -> bool:
        """Returns True when momentum confluence is strong enough to trade."""
        return ctx.confluence_score >= 0.40

    def get_sl_tp(
        self,
        price: float,
        bias: str,
        ctx: SMCContext,
    ) -> Tuple[float, float]:
        """
        ATR-based SL/TP calculation.
        Uses momentum_score to scale risk distance:
          stronger momentum → tighter SL, farther TP.
        """
        # Derive a pseudo-ATR from recent price if ctx.current_atr is zero
        atr = ctx.current_atr if ctx.current_atr > 0 else price * 0.002

        # Scale risk by inverse of score: higher conviction → tighter stop
        score_factor = max(1.0 - ctx.momentum_score * 0.3, 0.6)
        min_sl_dist  = atr * 1.5 * score_factor

        if bias == "LONG":
            sl   = price - min_sl_dist
            risk = price - sl
            tp   = price + risk * 2.0
        else:
            sl   = price + min_sl_dist
            risk = sl - price
            tp   = price - risk * 2.0

        return round(sl, 5), round(tp, 5)
