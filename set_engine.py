"""
set_engine.py - SET MODE engine (subclass of bot_engine.BotEngine, so the Deriv client, reconnects,
watchdog, periodic refit, settlement polling, stats and dashboard are the same proven code).

What is different from the original engine
  * Predictor: set_predictor.SetPredictor - the SET_SIZE (default 3) most likely next digits per symbol.
  * Contracts: ONLY Over/Under, and only ones that win on every digit of the predicted set. A spread-out
    set (cover wider than SET_MAX_COVER_WIDTH digits) produces no signal. No Matches/Differs/Even/Odd.
  * Scan: every symbol keeps one fresh signal (valid until that symbol's next tick). A single scanner
    ranks ALL fresh signals at the same instant, prices the best SET_PRICE_TOP of them IN PARALLEL with
    live proposals, ranks them by  P(win) x payout/stake  and buys the best SET_MAX_BUYS_PER_SCAN IN
    PARALLEL. No payout floor/ceiling, no edge gate (SET_MIN_SCORE / SET_MIN_SET_PROB
    are optional and off by default).
  * Money: stakes come from the profit-securing money rules (main_money.py): stake = max(PREDICTOR_STAKE,
    17.5% of balance-above-floor); never a buy that would take the balance under the floor.
  * Window: buys only in minutes 29-36 of the hour (config.TRADE_WINDOW_*).
  * Safety: DEMO ONLY unless SET_ALLOW_REAL=1; SET_MAX_OPEN caps simultaneous contracts; trades are NOT
    mirrored to follower accounts and are not fed into the follower zone statistics.
  * Reporting: every SET_REPORT_SECONDS the log shows live top-1 / top-N prediction accuracy next to
    chance level, and the trade win rate next to break-even, so the demo result is read off directly.
"""
import asyncio
import collections
import logging
import statistics
import time
from typing import Dict, List, Optional, Tuple

import config
import predictor as pred
import set_predictor as sp
import target_guard
from bot_engine import BotEngine, _cfg

logger = logging.getLogger(__name__)


def account_allowed() -> Tuple[bool, str]:
    """Set mode places many fast contracts; it only runs on a demo account unless explicitly overridden."""
    mode = str(getattr(config, "DERIV_ACCOUNT_MODE", "demo")).strip().lower()
    if mode == "demo":
        return True, "demo account"
    if bool(getattr(config, "TEST_MODE", False)):      # test mode: demo only, SET_ALLOW_REAL is never honoured
        return False, f"TEST_MODE is demo-only (DERIV_ACCOUNT_MODE={mode!r}); SET_ALLOW_REAL is ignored in test mode"
    if bool(getattr(config, "SET_ALLOW_REAL", False)):
        return True, f"account mode {mode!r} - explicitly allowed by SET_ALLOW_REAL=1"
    return False, (f"DERIV_ACCOUNT_MODE={mode!r}: SET MODE is demo-only. Use DERIV_ACCOUNT_MODE=demo, "
                   f"or set SET_ALLOW_REAL=1 if you really mean to run it on a real account.")


class SetEngine(BotEngine):
    def __init__(self):
        super().__init__()
        self.stake = float(_cfg("PREDICTOR_STAKE", 0.35))  # MINIMUM stake (env) - probe stake; real stakes: money rules
        self._feed_followers = False                       # no follower stats / mirroring in set mode
        self._signals: Dict[str, sp.SetSignal] = {}
        self._scan_event = asyncio.Event()
        self._used: set = set()                            # (symbol, tick epoch) already bought
        self._inflight = 0
        self._set_trades = 0
        self._set_hits = 0                                 # trades whose settling digit was in the predicted set
        self._set_known = 0                                # trades where the settling digit could be read
        self._scans = 0

    # the strict trade window (minutes 27-36 of the hour) is inherited from BotEngine._in_trade_window

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def run(self):
        ok, why = account_allowed()
        if not ok:
            logger.critical("SET MODE REFUSED TO START: %s", why)
            return
        logger.info("SET MODE: %s | set size %d, cover width <= %d, min stake %.2f, max open %d",
                    why, int(_cfg("SET_SIZE", 3)), int(_cfg("SET_MAX_COVER_WIDTH", 5)),
                    self.stake, int(_cfg("SET_MAX_OPEN", 6)))
        await super().run()

    def _extra_tasks(self) -> list:
        return [self._scanner_loop(), self._report_loop()]

    def _pr_desc(self, pr) -> str:
        return pr.describe()

    # ── warm-up / refit (replaces the single-contract fit) ───────────────────
    async def _fit_symbol(self, sym: str, initial: bool = False) -> bool:
        n = int(_cfg("WARMUP_TICKS", 5000))
        ticks = await self.client.get_tick_history(sym, n)
        if len(ticks) < 500:
            logger.warning("WARM-UP %s: only %d ticks returned", sym, len(ticks))
        if not initial and len(ticks) < 300:
            return False
        quotes = [t["quote"] for t in ticks]
        if sym not in self._decimals:
            dec_map = _cfg("DIGIT_DECIMALS", {}) or {}
            dec = dec_map.get(sym)
            if dec is None:
                dec = pred.infer_decimals(quotes) if quotes else 2
            self._decimals[sym] = int(dec)
        digits = pred.digits_from_quotes(quotes, self._decimals[sym])
        last_ep = ticks[-1]["epoch"] if ticks else None
        self._recent_epochs[sym] = [t["epoch"] for t in ticks[-50:]]
        pr = self.predictors.get(sym)
        if pr is None:
            self._ratios.setdefault(sym, {})
            pr = sp.SetPredictor(sym)
        if initial or not self._ratios.get(sym):
            await self._probe_one(sym)
        pr.fit(digits, last_epoch=last_ep)
        if last_ep is not None:                       # ticks that arrived while we were downloading
            for ep, d in list(self._live.get(sym, ())):
                if ep > last_ep and pr.on_tick(d, ep):
                    self._recent_epochs[sym].append(ep)
        self.predictors[sym] = pr
        return True

    async def _probe_one(self, sym: str) -> None:
        """One priced Over/Under contract proves the symbol offers digit contracts on this account."""
        cand = ("OVER", 4)
        prop = await self._propose(sym, cand, self.stake)
        if prop and prop.get("payout"):
            self._ratios.setdefault(sym, {})[cand] = float(prop["payout"]) / self.stake

    # ── per-tick: refresh this symbol's signal, wake the scanner ─────────────
    async def _worker(self, sym: str):
        ev = self._events[sym]
        while True:
            await ev.wait()
            ev.clear()
            try:
                pr = self.predictors[sym]
                sig = pr.signal(epoch=self._last_epoch.get(sym), wall=time.time())
                if sig is None:
                    self._signals.pop(sym, None)       # spread out / not ready: nothing to trade on this tick
                else:
                    self._signals[sym] = sig
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("signal error %s: %s", sym, exc, exc_info=True)
                self._signals.pop(sym, None)
            self._scan_event.set()

    def _tick_interval(self, sym: str) -> float:
        eps = self._recent_epochs.get(sym, [])
        gaps = [b - a for a, b in zip(eps, eps[1:]) if b > a]
        return float(statistics.median(gaps)) if gaps else 2.0

    def _max_age(self, sym: str) -> float:
        """A signal is for the NEXT tick of its symbol; after about a tick interval it is useless."""
        return float(_cfg("SET_SIGNAL_MAX_AGE_FRAC", 0.6)) * self._tick_interval(sym)

    def _est_score(self, sym: str, c: Tuple[pred.Candidate, int, float]) -> float:
        cand, _w, p_win = c
        ratio = self._ratios.get(sym, {}).get(cand) or pred.estimated_payout_ratio(cand)
        return p_win * ratio

    # ── the scanner: all symbols, all covering contracts, at the same instant ─
    async def _scanner_loop(self):
        coalesce = float(_cfg("SET_COALESCE_SECONDS", 0.05))
        gap = float(_cfg("SET_MIN_SCAN_GAP", 0.25))
        while True:
            await self._scan_event.wait()
            if coalesce > 0:
                await asyncio.sleep(coalesce)          # ticks of several symbols often land together
            self._scan_event.clear()
            if self._blocked():
                if not self._in_trade_window(time.time()):
                    self._last_decision = (f"outside trade window (minutes "
                                           f"{int(_cfg('TRADE_WINDOW_START_MINUTE', 0))}-"
                                           f"{int(_cfg('TRADE_WINDOW_END_MINUTE', 59))})")
                continue
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("scan error: %s", exc, exc_info=True)
            if gap > 0:
                await asyncio.sleep(gap)

    async def _scan_once(self):
        self._scans += 1
        now = time.time()
        slots = int(_cfg("SET_MAX_OPEN", 6)) - len(self._open) - self._inflight
        if slots <= 0:
            self._last_decision = f"max open contracts ({len(self._open)}) reached"
            return
        stake = self._next_stake()                      # money rules; None = stopped (floor reached)
        if stake is None:
            self._last_decision = "stopped by money rules (floor reached) - resumes at next reset"
            return
        fresh = []
        for sym, sig in list(self._signals.items()):
            key = (sym, sig.epoch if sig.epoch is not None else sig.wall)
            if key in self._used or now - sig.wall > self._max_age(sym):
                continue
            fresh.append(sig)
        if not fresh:
            return
        # best covering contract of every symbol by estimated score, best symbols first
        ranked = []
        for sig in fresh:
            best = max(sig.contracts, key=lambda c: self._est_score(sig.symbol, c))
            ranked.append((self._est_score(sig.symbol, best), sig, best))
        ranked.sort(key=lambda x: x[0], reverse=True)
        top = ranked[: max(1, int(_cfg("SET_PRICE_TOP", 4)))]

        # live proposals for all of them at once
        props = await asyncio.gather(*(self._propose(sig.symbol, c[0], stake) for _, sig, c in top),
                                     return_exceptions=True)
        min_score = float(_cfg("SET_MIN_SCORE", 0.0))
        scored = []
        for (_est, sig, c), prop in zip(top, props):
            if isinstance(prop, BaseException) or not prop:
                continue
            ratio = float(prop.get("payout") or 0.0) / stake
            if ratio <= 0:
                continue
            self._ratios.setdefault(sig.symbol, {})[c[0]] = ratio
            score = c[2] * ratio
            if score < min_score:
                continue
            scored.append((score, sig, c, ratio, prop))
        if not scored:
            self._last_decision = f"scan {self._scans}: {len(fresh)} signal(s), none passed"
            return
        scored.sort(key=lambda x: x[0], reverse=True)
        picks = scored[: max(1, min(slots, int(_cfg("SET_MAX_BUYS_PER_SCAN", 2))))]
        if self._blocked():
            return
        now = time.time()
        picks = [x for x in picks if now - x[1].wall <= self._max_age(x[1].symbol)]   # pricing took too long
        if not picks:
            self._last_decision = "signals expired while pricing"
            return
        affordable = []
        for x in picks:                               # hard rule: every buy must leave balance >= floor
            if self.money.reserve(stake):
                affordable.append(x)
        picks = affordable
        if not picks:
            self._last_decision = "skip: buying would breach the profit floor"
            return
        for x in picks:                               # claim them before any await
            sig = x[1]
            self._used.add((sig.symbol, sig.epoch if sig.epoch is not None else sig.wall))
        if len(self._used) > 5000:
            self._used.clear()
        self._inflight += len(picks)
        try:
            await asyncio.gather(*(self._buy_signal(*x, stake) for x in picks), return_exceptions=True)
        finally:
            self._inflight -= len(picks)
            for _ in picks:
                self.money.release(stake)

    async def _buy_signal(self, score, sig: sp.SetSignal, c, ratio: float, prop: dict, stake: float):
        sym, cand = sig.symbol, c[0]
        try:
            ask = float(prop.get("ask_price") or stake)
            try:
                if self._direct_buy:
                    resp = await self._buy_direct(sym, cand, ask, stake)
                else:
                    try:
                        resp = await self.client._send({"buy": prop["id"], "price": ask}, timeout=10)
                    except Exception as exc:
                        if "InvalidContractProposal" not in str(exc) and "Unknown contract proposal" not in str(exc):
                            raise
                        logger.warning("Buy by proposal id was refused (%s) - retrying with a direct buy",
                                       str(exc)[:120])
                        resp = await self._buy_direct(sym, cand, ask, stake)
                        self._direct_buy = True
            except Exception as exc:
                await self._on_api_error("buy", exc)
                return None
            buy = resp.get("buy") or {}
            self.money.note_buy(buy)
            cid = str(buy.get("contract_id", ""))
            if not cid:
                logger.warning("BUY returned no contract_id: %s", resp)
                return None
            opened = float(buy.get("purchase_time") or time.time())
            tg = sp.tag(cand)
            self._open[cid] = {"sym": sym, "cand": cand, "stake": stake, "ratio": ratio,
                               "opened_at": opened, "decision_epoch": sig.epoch, "t0": time.time(),
                               "ctx": sig.ranked, "p_lower": c[2], "pred_digits": sig.digits,
                               "pred_ranked": sig.ranked, "p_win": c[2], "score": score, "width": c[1]}
            self._last_decision = (f"BUY {sym} {tg} set={list(sig.digits)} p_win={c[2]:.3f} "
                                   f"ratio={ratio:.3f} score={score:.3f}")
            logger.info("SET TRADE %s %s %s predicted=%s probs=%s set_p=%.3f width=%d p_win=%.3f "
                        "ratio=%.3f score=%.3f shortcode=%s",
                        cid, sym, tg, list(sig.ranked), [round(x, 3) for x in sig.probs], sig.set_prob,
                        c[1], c[2], ratio, score, buy.get("shortcode", "?"))
            self.client._contract_symbol_map[cid] = sym
            try:
                await self.client.subscribe_contract(
                    cid, lambda msg, _c=cid: asyncio.create_task(self._on_contract_result(_c, msg)),
                    symbol=sym)
            except Exception as exc:
                logger.warning("subscribe_contract(%s) failed (settle loop will poll): %s", cid, exc)
            return cid
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("buy flow error %s: %s", sym, exc, exc_info=True)
            return None

    # ── settlement: read what Deriv actually settled, compare with what we meant ─
    async def _on_contract_result(self, cid: str, msg: dict):
        poc = msg.get("proposal_open_contract", msg)
        settled = poc.get("is_sold") or poc.get("is_expired") or poc.get("status") in ("sold", "won", "lost")
        info = self._open.get(cid) if settled else None
        if info is not None:
            try:
                self._inspect_settlement(cid, info, poc)
            except Exception as exc:
                logger.debug("settlement inspection failed: %s", exc)
        await super()._on_contract_result(cid, msg)

    def _inspect_settlement(self, cid: str, info: dict, poc: dict) -> None:
        cand = info["cand"]
        want_type, want_bar = pred.contract_type(cand), cand[1]
        got_type, got_bar = poc.get("contract_type"), poc.get("barrier")
        if (got_type and got_type != want_type) or \
           (want_bar is not None and got_bar is not None and str(got_bar) != str(want_bar)):
            logger.error("CONTRACT MISMATCH %s: bought %s barrier=%s but Deriv settled %s barrier=%s",
                         cid, want_type, want_bar, got_type, got_bar)
        self._set_trades += 1
        raw = poc.get("exit_tick")
        if raw is None:
            raw = poc.get("exit_tick_display_value")
        actual: Optional[int] = None
        try:
            actual = pred.last_digit(float(raw), self._decimals[info["sym"]])
        except Exception:
            pass
        if actual is not None:
            self._set_known += 1
            in_set = actual in info["pred_digits"]
            self._set_hits += int(in_set)
            logger.info("SET RESULT %s %s %s predicted=%s actual=%d in_set=%s",
                        cid, info["sym"], sp.tag(cand), list(info["pred_ranked"]), actual, in_set)

    # ── reporting ────────────────────────────────────────────────────────────
    async def _report_loop(self):
        period = float(_cfg("SET_REPORT_SECONDS", 300))
        if period <= 0:
            return
        while True:
            await asyncio.sleep(period)
            try:
                logger.info(self.report_text())
            except Exception as exc:
                logger.error("report error: %s", exc)

    def report_text(self) -> str:
        n = t1 = tn = 0
        for pr in self.predictors.values():
            n += pr.score.n
            t1 += pr.score.top1
            tn += pr.score.topn
        size = int(_cfg("SET_SIZE", 3))
        pred_part = (f"live prediction over {n} ticks / {len(self.predictors)} symbols: "
                     f"top-1 {100.0 * t1 / n:.1f}% (chance 10.0%), top-{size} {100.0 * tn / n:.1f}% "
                     f"(chance {10.0 * size:.1f}%)") if n else "live prediction: no scored ticks yet"
        T = self._n_trades
        if T:
            trade_part = (f"trades {T}: win {100.0 * self._n_wins / T:.1f}% vs break-even "
                          f"{100.0 * self._be_sum / T:.1f}% | P&L {self._total_pnl:+.2f}")
            if self._set_known:
                trade_part += (f" | settling digit in predicted set {100.0 * self._set_hits / self._set_known:.1f}% "
                               f"({self._set_known} trades)")
        else:
            trade_part = "trades 0"
        return f"SET REPORT | {pred_part} | {trade_part} | signals now {len(self._signals)} | scans {self._scans}"

    async def _dashboard_loop(self):
        while True:
            await asyncio.sleep(5)
            try:
                import keep_alive
                keep_alive.update_status(
                    balance=round(self.client.balance, 2), session="SET MODE (demo)",
                    current_symbol=",".join(self.symbols), last_signal=self._last_decision,
                    tradeable_count=len(self._signals), open_contracts_count=len(self._open),
                    target_balance=target_guard.target(), money=self.money.snapshot())
                keep_alive.set_active_trades(len(self._open))
            except Exception:
                pass
