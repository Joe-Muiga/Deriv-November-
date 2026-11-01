"""
candlestick_builder.py – Assembles OHLCV bars from a stream of raw ticks.

Tick format from Deriv: {"epoch": int, "quote": float}

Two modes:
  • Time-based  – fixed N-second windows (default, used for 5-min LTF bars)
  • Volume-based – N-tick windows (alternative)

The builder keeps a rolling buffer of completed bars plus the current
in-progress bar.  Call `add_tick()` on every incoming tick and check
`new_bar_ready` for a completed bar.
"""

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Deque

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    timestamp: int    # epoch of bar open (seconds)
    open:  float = 0.0
    high:  float = 0.0
    low:   float = 0.0
    close: float = 0.0
    volume: int  = 0   # tick count used as proxy volume

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "open":  self.open,
            "high":  self.high,
            "low":   self.low,
            "close": self.close,
            "volume": self.volume,
        }


class CandlestickBuilder:
    """
    Builds time-based OHLCV candles from a tick stream.

    Parameters
    ----------
    granularity : int
        Bar width in seconds (e.g. 300 for 5-min, 3600 for 1-hour).
    max_bars : int
        Maximum number of completed bars to keep in memory.
    """

    def __init__(self, granularity: int = 300, max_bars: int = 500):
        self.granularity  = granularity
        self.max_bars     = max_bars
        self._bars: Deque[Candle] = deque(maxlen=max_bars)
        self._current:   Optional[Candle] = None
        self.new_bar_ready: bool = False
        self._last_completed: Optional[Candle] = None

    # ── Seed with historical candles from Deriv ticks_history ────────────────

    def seed(self, historical: List[dict]):
        """
        Load historical OHLCV bars returned by Deriv's ticks_history API
        (style='candles').  Each dict has keys: epoch, open, high, low, close.
        """
        for c in historical:
            candle = Candle(
                timestamp = int(c["epoch"]),
                open      = float(c["open"]),
                high      = float(c["high"]),
                low       = float(c["low"]),
                close     = float(c["close"]),
                volume    = int(c.get("volume", 1)),
            )
            self._bars.append(candle)
        logger.info(f"Seeded {len(historical)} historical bars "
                    f"(granularity={self.granularity}s)")

    # ── Live tick ingestion ───────────────────────────────────────────────────

    def add_tick(self, epoch: int, price: float) -> bool:
        """
        Ingest one tick.  Returns True if a new bar was just completed.
        """
        self.new_bar_ready = False
        bar_start = (epoch // self.granularity) * self.granularity

        if self._current is None:
            # First tick ever
            self._current = Candle(timestamp=bar_start, open=price,
                                   high=price, low=price, close=price, volume=1)
            return False

        if bar_start == self._current.timestamp:
            # Still inside the same bar
            self._current.high   = max(self._current.high,  price)
            self._current.low    = min(self._current.low,   price)
            self._current.close  = price
            self._current.volume += 1
            return False

        if bar_start > self._current.timestamp:
            # A new bar has started → complete the current one
            self._bars.append(self._current)
            self._last_completed = self._current
            self.new_bar_ready   = True

            # Fill any missing intermediate bars with close price (gap handling)
            expected = self._current.timestamp + self.granularity
            while expected < bar_start:
                filler = Candle(timestamp=expected,
                                open=self._current.close, high=self._current.close,
                                low=self._current.close, close=self._current.close,
                                volume=0)
                self._bars.append(filler)
                expected += self.granularity

            # Start the new bar
            self._current = Candle(timestamp=bar_start, open=price,
                                   high=price, low=price, close=price, volume=1)
            return True

        # tick is older than current bar (out-of-order) – ignore
        return False

    # ── Data accessors ───────────────────────────────────────────────────────

    @property
    def completed_bars(self) -> List[Candle]:
        """All completed bars as a list (oldest first)."""
        return list(self._bars)

    @property
    def last_completed(self) -> Optional[Candle]:
        return self._last_completed

    @property
    def opens(self)   -> List[float]: return [b.open  for b in self._bars]
    @property
    def highs(self)   -> List[float]: return [b.high  for b in self._bars]
    @property
    def lows(self)    -> List[float]: return [b.low   for b in self._bars]
    @property
    def closes(self)  -> List[float]: return [b.close for b in self._bars]
    @property
    def volumes(self) -> List[int]:   return [b.volume for b in self._bars]

    @property
    def count(self) -> int:
        return len(self._bars)

    def last_n_closes(self, n: int) -> List[float]:
        bars = self.completed_bars
        return [b.close for b in bars[-n:]]

    def last_n_candles(self, n: int) -> List[Candle]:
        return self.completed_bars[-n:]
