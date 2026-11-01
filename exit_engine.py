"""
exit_engine.py
===============

Live exit management for Multiplier contracts (BOOM500/BOOM1000/CRASH500/
CRASH1000) that have no fixed expiry. Rise/Fall (CALL/PUT) contracts are
outside this module's scope entirely.

Implements a rolling, live version of the triple-barrier / meta-labeling
method (Lopez de Prado, *Advances in Financial Machine Learning*):

  1. Rule-based trailing layer (always active, needs no training data):
     - Early loss cap (Aug 2026): independent of arm state — once past a
       short grace window, forces CLOSE_NOW if the loss reaches
       EXIT_LOSS_CAP_FRACTION * static_sl_amount, realizing a small
       fraction of the worst-case static loss instead of riding it out
       in full. Checked first, every tick.
     - Arms once profit crosses EXIT_ARM_PROFIT_FRACTION * static_tp_amount.
     - Once armed, proposes tighter stop-loss updates that lock in
       EXIT_TRAIL_LOCK_FRACTION * peak_profit (never loosens risk).
     - Once armed, forces CLOSE_NOW if profit decays to
       EXIT_DECAY_CLOSE_FRACTION * peak_profit.
     - Before arming and above the loss cap, this layer is a no-op; the
       static SL/TP set at buy time are the only other bounds in effect.

  2. Lightweight ML layer (inert until META_LABEL_MIN_TRADES closed
     contracts have been logged by this module):
     - A small sklearn classifier (LogisticRegression or
       GradientBoostingClassifier) predicts, from a profit-series snapshot,
       whether the trade should have already been closed (label=0) versus
       held (label=1), where the label is assigned retrospectively in
       record_closed() by comparing each snapshot's profit to the
       contract's eventual peak profit.
     - Can only pull the exit earlier/safer than the rule layer: it may
       force CLOSE_NOW over a rule-layer HOLD/TRAIL_UPDATE, but can never
       override a rule-layer CLOSE_NOW and never loosens a stop.
     - Retrains in a background asyncio task every
       META_LABEL_RETRAIN_EVERY_N newly closed contracts; retraining never
       blocks the decision path.

Persistence tradeoff: snapshots and labeled training examples are appended
to local JSONL files (EXIT snapshot/training logs) and the trained model is
persisted via joblib to config.EXIT_ML_MODEL_PATH. On Render's free tier
this disk is ephemeral across redeploys, so training data and the model
reset on every redeploy. That's an accepted limitation here -- solving
persistent storage across redeploys is out of scope for this module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import config

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Optional ML dependencies
# --------------------------------------------------------------------------

_SKLEARN_OK = True
try:
    import joblib
    from sklearn.linear_model import LogisticRegression
except ImportError:
    _SKLEARN_OK = False
    joblib = None  # type: ignore
    LogisticRegression = None  # type: ignore

_ml_warned_once = False


def _ml_available() -> bool:
    """True only if sklearn/joblib import cleanly AND config enables ML."""
    global _ml_warned_once
    if not _SKLEARN_OK:
        if not _ml_warned_once:
            logger.warning(
                "EXIT ML DISABLED: scikit-learn/joblib not installed — "
                "add to requirements.txt"
            )
            _ml_warned_once = True
        return False
    return bool(getattr(config, "EXIT_ML_ENABLED", False))


# --------------------------------------------------------------------------
# Public types
# --------------------------------------------------------------------------

ExitAction = Literal["HOLD", "TRAIL_UPDATE", "CLOSE_NOW"]


@dataclass
class ExitDecision:
    action: ExitAction
    new_stop_loss: Optional[float]
    reason: str


@dataclass
class _Snapshot:
    ts: float
    elapsed_secs: float
    profit: float


@dataclass
class _ContractState:
    symbol: str
    stake: float
    multiplier: int
    static_sl_amount: float
    static_tp_amount: float
    armed: bool = False
    peak_profit: float = float("-inf")
    last_proposed_floor: Optional[float] = None
    history: List[_Snapshot] = field(default_factory=list)


# --------------------------------------------------------------------------
# Module state
# --------------------------------------------------------------------------

_contract_states: Dict[str, _ContractState] = {}

_model: Any = None
_model_lock = asyncio.Lock()
_closed_since_retrain = 0
_total_closed_logged = 0
_retrain_in_progress = False

SNAPSHOT_LOG_PATH = "exit_snapshots.jsonl"
TRAINING_LOG_PATH = "exit_training_labels.jsonl"

FEATURE_NAMES = [
    "elapsed_secs",
    "profit_pct_of_stake",
    "profit_velocity",
    "distance_to_static_sl_pct",
    "distance_to_static_tp_pct",
    "multiplier",
    "hour_of_day_utc",
]


# --------------------------------------------------------------------------
# record_snapshot
# --------------------------------------------------------------------------

def record_snapshot(
    contract_id: str,
    symbol: str,
    elapsed_secs: float,
    stake: float,
    current_profit: float,
    static_sl_amount: float,
    static_tp_amount: float,
    multiplier: int,
) -> None:
    """Update in-memory per-contract history and append to the JSONL log."""
    try:
        state = _contract_states.get(contract_id)
        if state is None:
            state = _ContractState(
                symbol=symbol,
                stake=stake,
                multiplier=multiplier,
                static_sl_amount=static_sl_amount,
                static_tp_amount=static_tp_amount,
            )
            _contract_states[contract_id] = state
        else:
            # static bounds can legitimately change if this module (or
            # something else) revises them via contract_update
            state.static_sl_amount = static_sl_amount
            state.static_tp_amount = static_tp_amount

        snap = _Snapshot(ts=time.time(), elapsed_secs=elapsed_secs, profit=current_profit)
        state.history.append(snap)
        if len(state.history) > 500:
            # bound memory for very long-lived contracts
            state.history = state.history[-500:]

        if current_profit > state.peak_profit:
            state.peak_profit = current_profit

        _append_jsonl(
            SNAPSHOT_LOG_PATH,
            {
                "contract_id": contract_id,
                "symbol": symbol,
                "ts": snap.ts,
                "elapsed_secs": elapsed_secs,
                "stake": stake,
                "current_profit": current_profit,
                "static_sl_amount": static_sl_amount,
                "static_tp_amount": static_tp_amount,
                "multiplier": multiplier,
            },
        )
    except Exception as exc:
        logger.warning("EXIT ENGINE: record_snapshot failed for %s: %s", contract_id, exc)


# --------------------------------------------------------------------------
# decide_exit
# --------------------------------------------------------------------------

def decide_exit(
    contract_id: str,
    symbol: str,
    elapsed_secs: float,
    stake: float,
    current_profit: float,
    static_sl_amount: float,
    static_tp_amount: float,
    multiplier: int,
) -> ExitDecision:
    """
    Runs the rule layer first, then the ML layer (if active).
    Never raises: any internal error degrades to a safe HOLD.
    """
    try:
        state = _contract_states.get(contract_id)
        if state is None:
            # decide_exit called without a prior record_snapshot -- treat
            # this call's data as the first snapshot so the rule layer has
            # something to work with.
            record_snapshot(
                contract_id, symbol, elapsed_secs, stake, current_profit,
                static_sl_amount, static_tp_amount, multiplier,
            )
            state = _contract_states[contract_id]

        rule_decision = _rule_layer_decide(state, current_profit, elapsed_secs)

        final_decision = rule_decision
        if rule_decision.action != "CLOSE_NOW":
            ml_decision = _ml_layer_decide(contract_id, state, elapsed_secs, current_profit)
            if ml_decision is not None:
                final_decision = ml_decision

        logger.info(
            "EXIT ENGINE: %s %s -> %s (%s)",
            contract_id, state.symbol, final_decision.action, final_decision.reason,
        )
        return final_decision

    except Exception as exc:
        logger.warning("EXIT ENGINE: decide_exit error for %s: %s", contract_id, exc)
        return ExitDecision("HOLD", None, f"error_fallback: {exc}")


def _rule_layer_decide(state: _ContractState, current_profit: float, elapsed_secs: float) -> ExitDecision:
    arm_fraction = config.EXIT_ARM_PROFIT_FRACTION
    lock_fraction = config.EXIT_TRAIL_LOCK_FRACTION
    decay_fraction = config.EXIT_DECAY_CLOSE_FRACTION

    # ── Early loss cap (user-directed, Aug 2026: "take low losses, keep
    # the magnitude minimal"). Runs BEFORE the arm gate below and applies
    # whether or not the trade ever arms — previously a losing trade that
    # never reached profit had zero active management here and could ride
    # all the way out to the full static_sl_amount (STOP_LOSS_MAP's
    # 30-70% of stake). This closes it early once the loss reaches a
    # small fraction of that static budget instead. Never loosens or
    # overrides the static stop-loss itself — only ever exits SOONER.
    if getattr(config, "EXIT_LOSS_CAP_ENABLED", False) and current_profit < 0:
        grace_secs = getattr(config, "EXIT_LOSS_CAP_GRACE_SECS", 20)
        if elapsed_secs >= grace_secs and state.static_sl_amount > 0:
            loss_cap_fraction = getattr(config, "EXIT_LOSS_CAP_FRACTION", 0.25)
            loss_cap_amount = loss_cap_fraction * state.static_sl_amount
            if current_profit <= -loss_cap_amount:
                return ExitDecision(
                    "CLOSE_NOW", None,
                    f"loss_cap: profit={current_profit:.2f} <= "
                    f"-{loss_cap_fraction:.2f}*static_sl({state.static_sl_amount:.2f})"
                    f"={-loss_cap_amount:.2f}",
                )

    arm_threshold = arm_fraction * state.static_tp_amount

    if not state.armed:
        if current_profit >= arm_threshold:
            state.armed = True
        else:
            return ExitDecision("HOLD", None, "not_armed")

    # armed from here on
    if current_profit > state.peak_profit:
        state.peak_profit = current_profit

    peak = state.peak_profit

    # decay-close check first: bail out regardless of trailing proposals
    if peak > 0 and current_profit <= decay_fraction * peak:
        return ExitDecision(
            "CLOSE_NOW", None,
            f"decay_close: profit={current_profit:.2f} <= "
            f"{decay_fraction} * peak={peak:.2f}",
        )

    locked_floor = lock_fraction * peak
    current_floor = -state.static_sl_amount

    # never loosen: only propose a tighter floor than both the static SL
    # and any previously proposed floor
    best_prior_floor = state.last_proposed_floor if state.last_proposed_floor is not None else current_floor
    if locked_floor > best_prior_floor:
        state.last_proposed_floor = locked_floor
        return ExitDecision(
            "TRAIL_UPDATE", locked_floor,
            f"trail_lock: peak={peak:.2f} -> floor={locked_floor:.2f}",
        )

    return ExitDecision("HOLD", None, f"armed_holding: peak={peak:.2f}")


# --------------------------------------------------------------------------
# ML layer
# --------------------------------------------------------------------------

def _ml_layer_decide(
    contract_id: str, state: _ContractState, elapsed_secs: float, current_profit: float
) -> Optional[ExitDecision]:
    if not _ml_available():
        return None
    if _total_closed_logged < getattr(config, "META_LABEL_MIN_TRADES", 200):
        return None
    if _model is None and not _load_model():
        return None

    try:
        feats = _build_features(state, elapsed_secs, current_profit)
        proba0 = _model.predict_proba([feats])[0][0]  # P(label=0) = "should have closed"
        min_conf = config.EXIT_ML_MIN_CONFIDENCE
        if proba0 >= min_conf:
            return ExitDecision(
                "CLOSE_NOW", None,
                f"ml_close: P(should_have_closed)={proba0:.3f} >= {min_conf}",
            )
        return None
    except Exception as exc:
        logger.warning("EXIT ENGINE: ML inference failed for %s: %s", contract_id, exc)
        return None


def _build_features(state: _ContractState, elapsed_secs: float, current_profit: float) -> List[float]:
    stake = state.stake or 1.0
    window = max(1, int(config.EXIT_ML_FEATURE_WINDOW))

    hist = state.history
    if len(hist) > window:
        past = hist[-window - 1].profit
    elif hist:
        past = hist[0].profit
    else:
        past = current_profit
    # Implementation Brief v4 §5.2 — normalize by (stake * multiplier)
    # rather than raw profit delta. A x4000 R_10 position's profit numbers
    # are on a wildly different scale from a x100 Boom/Crash position;
    # EXIT_ML_MIN_CONFIDENCE was tuned against the latter's distribution
    # only, so leaving this unnormalized would make the new Vol/1Hz/Step
    # Multiplier family's velocity feature incomparable to what the model
    # already knows.
    leverage_basis = stake * (state.multiplier or 1)
    profit_velocity = (current_profit - past) / leverage_basis if leverage_basis else 0.0

    distance_to_static_sl_pct = (current_profit - (-state.static_sl_amount)) / stake
    distance_to_static_tp_pct = (state.static_tp_amount - current_profit) / stake

    return [
        float(elapsed_secs),
        float(current_profit / stake),
        float(profit_velocity),
        float(distance_to_static_sl_pct),
        float(distance_to_static_tp_pct),
        float(state.multiplier),
        float(datetime.now(timezone.utc).hour),
    ]


def _load_model() -> bool:
    global _model
    path = config.EXIT_ML_MODEL_PATH
    if not path or not os.path.exists(path):
        return False
    try:
        _model = joblib.load(path)
        return True
    except Exception as exc:
        logger.warning("EXIT ENGINE: corrupt model file at %s (%s) — ML layer stays inert", path, exc)
        _model = None
        return False


# --------------------------------------------------------------------------
# record_closed
# --------------------------------------------------------------------------

def record_closed(contract_id: str, final_profit: float) -> None:
    """
    Called once a contract actually closes. Retroactively labels this
    contract's logged snapshots and appends them to the training set;
    triggers a background retrain every META_LABEL_RETRAIN_EVERY_N closures.
    """
    global _closed_since_retrain, _total_closed_logged
    try:
        state = _contract_states.pop(contract_id, None)
        if state is None or not state.history:
            logger.warning("EXIT ENGINE: record_closed called with no history for %s", contract_id)
            return

        decay_fraction = config.EXIT_DECAY_CLOSE_FRACTION
        eventual_peak = max(state.peak_profit, final_profit)
        if eventual_peak <= 0:
            eventual_peak = max(final_profit, 1e-9)

        examples = []
        for i, snap in enumerate(state.history):
            label = 1 if snap.profit >= decay_fraction * eventual_peak else 0
            elapsed_secs = snap.elapsed_secs
            # rebuild the feature vector as it would have looked at that
            # point in time, using only history up to and including i
            partial_state = _ContractState(
                symbol=state.symbol, stake=state.stake, multiplier=state.multiplier,
                static_sl_amount=state.static_sl_amount, static_tp_amount=state.static_tp_amount,
                history=state.history[: i + 1],
            )
            feats = _build_features(partial_state, elapsed_secs, snap.profit)
            example = {name: val for name, val in zip(FEATURE_NAMES, feats)}
            example["label"] = label
            example["contract_id"] = contract_id
            examples.append(example)
            _append_jsonl(TRAINING_LOG_PATH, example)

        _total_closed_logged += 1
        _closed_since_retrain += 1

        retrain_every = getattr(config, "META_LABEL_RETRAIN_EVERY_N", 100)
        min_trades = getattr(config, "META_LABEL_MIN_TRADES", 200)
        if (
            _ml_available()
            and _total_closed_logged >= min_trades
            and _closed_since_retrain >= retrain_every
        ):
            _closed_since_retrain = 0
            _schedule_retrain()

    except Exception as exc:
        logger.warning("EXIT ENGINE: record_closed failed for %s: %s", contract_id, exc)


def _schedule_retrain() -> None:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_retrain_background())
    except RuntimeError:
        # no running loop (e.g. called from sync test code) -- train inline
        _retrain_sync()


async def _retrain_background() -> None:
    global _retrain_in_progress
    if _retrain_in_progress:
        return
    _retrain_in_progress = True
    try:
        async with _model_lock:
            await asyncio.to_thread(_retrain_sync)
    except Exception as exc:
        logger.warning("EXIT ENGINE: background retrain failed: %s", exc)
    finally:
        _retrain_in_progress = False


def _retrain_sync() -> None:
    global _model
    if not _ml_available():
        return
    try:
        rows = _load_training_rows(TRAINING_LOG_PATH)
        if len(rows) < getattr(config, "META_LABEL_MIN_TRADES", 200):
            return

        X = [[row[name] for name in FEATURE_NAMES] for row in rows]
        y = [row["label"] for row in rows]

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)

        path = config.EXIT_ML_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        joblib.dump(clf, path)
        _model = clf
        logger.info("EXIT ENGINE: retrained model on %d examples -> %s", len(rows), path)
    except Exception as exc:
        logger.warning("EXIT ENGINE: retrain_sync failed: %s", exc)


def _load_training_rows(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    if not os.path.exists(path):
        return rows
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.warning("EXIT ENGINE: failed reading %s: %s", path, exc)
    return rows


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    try:
        with open(path, "a") as f:
            f.write(json.dumps(obj) + "\n")
    except Exception as exc:
        logger.warning("EXIT ENGINE: failed writing %s: %s", path, exc)
