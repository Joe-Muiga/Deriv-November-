"""
news_filter.py — high-impact economic-news blackout window for Forex/gold/
commodity trading.

This file did not exist anywhere in the uploaded project even though
bot_engine.py has always imported it (`from news_filter import
NewsFilter`) and constructed one in BotEngine.__init__
(`self.news = NewsFilter(block_minutes=config.NEWS_BLOCK_MINUTES)`) — so
the bot could not start at all before this file was added, independent of
the SMC/ICT rewrite.

News-driven volatility (NFP, CPI, FOMC/central-bank rate decisions, etc.)
is a materially bigger real risk for Forex/gold/commodity trading than it
ever was for synthetic indices (which have no macroeconomic calendar at
all) — a structurally perfect ICT setup can still get stopped out by a
2-minute spike around a release that has nothing to do with market
structure. That's what this filter exists to avoid.

IMPORTANT — this ships with NO economic calendar data pre-loaded. There is
no free, reliable, machine-readable economic-calendar API reachable from
this environment's network allowlist, and hand-typing plausible-looking
release times/dates into this file would be fabricating data that could
silently fail to protect you (worse than no filter at all, because it
looks like protection). Instead:
  - `is_blocked()` fails OPEN (never blocks) until you load real events.
  - `load_events_from_json()` reads a simple JSON file you maintain
    yourself (by hand, from a connector, or from any calendar source you
    trust) — see the format below.
  - `add_event()` lets you (or a future integration) add events
    programmatically at runtime.

JSON format for load_events_from_json() — a list of objects:
    [
      {"time_utc": "2026-09-19T12:30:00", "impact": "high",
       "currency": "USD", "label": "NFP"},
      {"time_utc": "2026-10-01T18:00:00", "impact": "high",
       "currency": "USD", "label": "FOMC rate decision"}
    ]
`currency` is matched against the two 3-letter legs of a Forex symbol
(e.g. "frxEURUSD" -> {"EUR", "USD"}) or, for gold/commodities, the quote
currency (USD for all of gold.py/MAJOR_COMMODITIES here) plus the special
value "ALL" always applies to every symbol.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    time_utc: datetime
    impact:   str        # "high" | "medium" | "low"
    currency: str        # "USD", "EUR", ... or "ALL"
    label:    str = ""


def _currencies_for_symbol(symbol: str) -> set:
    """
    "frxEURUSD" -> {"EUR", "USD"}. "frxXAUUSD"/"frxXAGUSD" -> {"USD"}
    (gold/silver are priced in USD and dominated by USD-driver events).
    "frxUSOIL"/"frxUKOIL" -> {"USD"}. Falls back to {} (no currency match,
    so only "ALL"-tagged events apply) for anything unrecognised.
    """
    s = symbol.replace("frx", "")
    if len(s) == 6 and s.isalpha():
        return {s[:3].upper(), s[3:].upper()}
    if s.upper() in ("XAUUSD", "XAGUSD", "USOIL", "UKOIL"):
        return {"USD"}
    return set()


class NewsFilter:
    """
    block_minutes: how many minutes BEFORE and AFTER a matching event to
    treat as blocked. Symmetric window (before: avoid entering right into
    a spike; after: avoid entering while the post-release whipsaw is still
    settling).
    """

    def __init__(self, block_minutes: int = 30):
        self.block_minutes = block_minutes
        self._events: List[NewsEvent] = []

    # ── Loading events ───────────────────────────────────────────────────

    def load_events_from_json(self, path: str) -> int:
        """
        Loads/replaces the event list from a JSON file (see module
        docstring for format). Returns the number of events loaded. Safe
        to call repeatedly (e.g. on a daily timer) to refresh — each call
        replaces the previous list rather than appending, so stale events
        from a file you've since updated don't linger.
        """
        p = Path(path)
        if not p.exists():
            logger.warning(f"NewsFilter: {path} does not exist — no events loaded (fail-open)")
            return 0
        try:
            raw = json.loads(p.read_text())
        except Exception as exc:
            logger.error(f"NewsFilter: failed to parse {path}: {exc} — no events loaded (fail-open)")
            return 0

        events = []
        for item in raw:
            try:
                events.append(NewsEvent(
                    time_utc=datetime.fromisoformat(item["time_utc"]).replace(tzinfo=timezone.utc),
                    impact=item.get("impact", "high"),
                    currency=item.get("currency", "ALL").upper(),
                    label=item.get("label", ""),
                ))
            except Exception as exc:
                logger.warning(f"NewsFilter: skipping malformed event {item!r}: {exc}")

        self._events = events
        logger.info(f"NewsFilter: loaded {len(events)} event(s) from {path}")
        return len(events)

    def add_event(self, time_utc: datetime, impact: str = "high",
                  currency: str = "ALL", label: str = "") -> None:
        if time_utc.tzinfo is None:
            time_utc = time_utc.replace(tzinfo=timezone.utc)
        self._events.append(NewsEvent(time_utc, impact, currency.upper(), label))

    def clear_events(self) -> None:
        self._events = []

    # ── Checking ─────────────────────────────────────────────────────────

    def is_blocked(self, symbol: str, at_time: Optional[datetime] = None,
                    min_impact: str = "high") -> bool:
        """
        True if `at_time` (defaults to now) falls within block_minutes of
        a loaded event whose currency matches this symbol (or is "ALL")
        and whose impact is >= min_impact. Always False if no events have
        been loaded — this filter never fabricates a blackout window.
        """
        if not self._events:
            return False

        now = at_time or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        impact_rank = {"low": 0, "medium": 1, "high": 2}
        min_rank = impact_rank.get(min_impact, 2)

        symbol_ccys = _currencies_for_symbol(symbol)
        window = timedelta(minutes=self.block_minutes)

        for ev in self._events:
            if impact_rank.get(ev.impact, 2) < min_rank:
                continue
            if ev.currency != "ALL" and ev.currency not in symbol_ccys:
                continue
            if abs((now - ev.time_utc).total_seconds()) <= window.total_seconds():
                logger.info(
                    f"NewsFilter: {symbol} BLOCKED — {ev.label or ev.currency} "
                    f"at {ev.time_utc.isoformat()} (within {self.block_minutes}min)")
                return True

        return False

    def next_event_for(self, symbol: str, min_impact: str = "high") -> Optional[NewsEvent]:
        """Nearest upcoming matching event for `symbol`, or None. Useful
        for dashboard display ("next blackout: NFP in 2h14m")."""
        now = datetime.now(timezone.utc)
        impact_rank = {"low": 0, "medium": 1, "high": 2}
        min_rank = impact_rank.get(min_impact, 2)
        symbol_ccys = _currencies_for_symbol(symbol)

        upcoming = [
            ev for ev in self._events
            if ev.time_utc >= now
            and impact_rank.get(ev.impact, 2) >= min_rank
            and (ev.currency == "ALL" or ev.currency in symbol_ccys)
        ]
        return min(upcoming, key=lambda e: e.time_utc) if upcoming else None
