"""
main_money.py - uniform (flat) staking for the MAIN (collector) account.

No money rules: no floor, no tiers, no daily reset, no milestone, no persistence.
Every buy uses the same stake = the PREDICTOR_STAKE env var (default 0.35).

The engines call the same small API as before, so nothing else had to change:

    start(client)   once at boot: reads the live balance (never raises)
    on_balance(b)   every live balance update from Deriv
    next_stake()    the flat stake (never None)
    reserve(stake)  True unless the KNOWN balance cannot pay for the stake (then Deriv would reject it)
    note_buy(buy)   feed the buy response so the cached balance drops immediately
    release(stake)  drop the hold taken by reserve()
    snapshot()      dict for the dashboard (keep_alive "money")
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

EPS = 1e-9


def _flat_stake(default: float = 0.35) -> float:
    """Flat stake = the PREDICTOR_STAKE environment variable."""
    try:
        v = float((os.environ.get("PREDICTOR_STAKE") or "").strip())
        return round(v, 2) if v > 0 else default
    except ValueError:
        return default


class MainMoney:
    def __init__(self):
        self.stake = _flat_stake()
        self.balance: Optional[float] = None
        self.ready = False
        self._pending = 0.0           # stakes reserved but not yet reflected in the balance
        self._credit = 0.0            # part of the holds already reflected via note_buy()

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def start(self, client) -> None:
        try:
            bal = await client.fetch_balance_live()
        except Exception as exc:
            logger.warning("STAKE: live balance fetch failed (%s) - using cached value", exc)
            bal = getattr(client, "balance", None)
        try:
            self.balance = float(bal) if bal is not None else None
        except (TypeError, ValueError):
            self.balance = None
        self.ready = True                                 # flat staking never blocks on state
        logger.info("STAKE ready: flat stake=%.2f balance=%s", self.stake,
                    "-" if self.balance is None else f"{self.balance:.2f}")

    # ── balance feed ─────────────────────────────────────────────────────────
    def on_balance(self, bal) -> None:
        try:
            self.balance = float(bal)
        except (TypeError, ValueError):
            return

    # ── stake / buy ──────────────────────────────────────────────────────────
    def next_stake(self) -> Optional[float]:
        return self.stake

    def reserve(self, stake: float) -> bool:
        """Only refuse when the known balance (minus stakes in flight) cannot pay the stake."""
        if self.balance is not None and self.balance - self._pending < stake - EPS:
            return False
        self._pending += stake
        return True

    def note_buy(self, buy: dict) -> None:
        """Deriv deducts the buy price at once; use balance_after so the next reserve() is exact."""
        try:
            price = float((buy or {}).get("buy_price") or 0.0)
            after = (buy or {}).get("balance_after")
            if after is not None:
                self.on_balance(float(after))
            elif self.balance is not None and price:
                self.on_balance(self.balance - price)
            moved = min(self._pending, price)             # that money is now out of the balance itself
            self._pending -= moved
            self._credit += moved
        except Exception as exc:
            logger.debug("STAKE note_buy ignored: %s", exc)

    def release(self, stake: float) -> None:
        if self._credit >= stake - EPS:
            self._credit -= stake
        else:
            self._pending -= stake
        self._pending = max(self._pending, 0.0)
        self._credit = max(self._credit, 0.0)

    # ── dashboard ────────────────────────────────────────────────────────────
    def snapshot(self) -> dict:
        return {"ready": self.ready, "mode": "flat", "stake": self.stake, "next_stake": self.stake,
                "balance": self.balance}
