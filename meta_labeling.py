"""
meta_labeling.py

Binary "take-this-signal-or-not" meta-filter sitting on top of the
rule-based strategies. Trains on closed-trade history already logged by
strategy_stats.py (same SQLite/JSON storage — this module reads it
directly via the public STRATEGY_STATS_DIR-derived paths, without
touching strategy_stats' private backend objects).

Model:
    - scikit-learn LogisticRegression + DictVectorizer if sklearn is
      importable
    - otherwise a hand-rolled, numpy-only logistic regression with the
      same DictVectorizer-style one-hot/passthrough encoding

Features per trade:
    strategy (one-hot), symbol (one-hot), entry_score, hour-of-day
    (encoded as sin/cos so 23:00 and 00:00 are close), rolling win rate
    for that (strategy, symbol) pair over the preceding trades, and the
    current win/loss streak for that pair going into this trade.

Gating: below config.META_LABEL_MIN_TRADES total logged trades, every
signal is passed through unfiltered (take=True) and a message is
logged noting the filter is inactive.

Walk-forward: features for a given historical row are built only from
trades that closed strictly *before* that row (per (strategy, symbol)
pair), so the training set itself never leaks future information into
a row's own features. The model is retrained from scratch on all
trades available at that moment every META_LABEL_RETRAIN_EVERY_N newly
closed trades; between retrains it only ever scores trades that close
*after* its training cutoff.

Every call to predict_take_trade() is logged (features, take,
confidence) so predictions can later be joined against actual outcomes
via record_outcome() to validate whether the filter is actually
helping.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
import strategy_stats

try:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


logger = logging.getLogger("meta_labeling")
try:
    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))
except Exception:
    logger.setLevel(logging.INFO)


# ── STORAGE (reuses strategy_stats' public path/flag constants) ──────
DATA_DIR = strategy_stats.DATA_DIR
SQLITE_AVAILABLE = strategy_stats.SQLITE_AVAILABLE
TRADES_SQLITE_PATH = strategy_stats.SQLITE_PATH
TRADES_JSON_PATH = strategy_stats.JSON_PATH

PRED_SQLITE_PATH = os.path.join(DATA_DIR, "meta_label_predictions.db")
PRED_JSON_PATH = os.path.join(DATA_DIR, "meta_label_predictions.json")

ROLLING_WINDOW = strategy_stats.DEFAULT_WINDOW  # same horizon as the dashboard's rolling win rate
# FIX (profitability audit): was 0.5 — a bare 50% cutoff has zero margin
# against estimation noise in a logistic model trained on limited, noisy
# trade data, so it approves roughly half of borderline signals purely by
# chance. 0.55 requires a real, if modest, edge over coin-flip. Decision
# cutoff on predicted P(win) — used only by the global entry_score-based
# model, i.e. when the EV gate below isn't active yet for this pair.
TAKE_THRESHOLD = 0.55

# ── ENRICHED-FEATURE EV GATE (Implementation Brief v6, PART 5) ───────────
# Feature keys evaluate_mean_reversion() now attaches to SignalResult.features
# (Implementation Brief v6, PART 2) and bot_engine._execute() folds into the
# dict passed to predict_take_trade() (PART 4). Only numeric keys from this
# set are ever used for the EV gate below — anything else on the incoming
# dict is ignored, so passing extra keys is always safe.
ENRICHED_FEATURE_KEYS = ("rsi", "roc", "bb_pct_b", "atr_expansion_ratio", "hour_utc")
# Minimum logged feature rows a (strategy, symbol) pair needs before the EV
# gate below is trusted over the existing global entry_score-based model.
# Deliberately separate from config.META_LABEL_MIN_TRADES (a global,
# all-pairs gate) since a pair can clear that floor in aggregate while still
# having very few rows with an actual feature vector attached (features are
# only populated for MEAN_REV signals as of PART 2/3).
EV_MIN_FEATURE_ROWS = getattr(config, "META_LABEL_EV_MIN_FEATURE_ROWS", 30)


# ── HISTORICAL DATA ACCESS (direct, read-only, no private access) ────

def _decode_ml_fields(features_raw) -> Tuple[float, float, str]:
    """
    strategy_stats.py's `trades` table already has a `features TEXT`
    column (JSON-encoded dict, see TradeRecord.features / record_trade()'s
    `features=` kwarg) — Implementation Brief v4 §5.1 rides on that
    existing column instead of adding new ones. Decodes multiplier/
    atr_pct/regime out of it, defaulting to (0.0, 0.0, "NONE") for rows
    with no features blob (old Rise/Fall trades, or any pair recorded
    before this pass) or a blob that doesn't carry these keys.
    """
    if not features_raw:
        return 0.0, 0.0, "NONE"
    try:
        decoded = json.loads(features_raw) if isinstance(features_raw, str) else features_raw
    except (TypeError, ValueError):
        return 0.0, 0.0, "NONE"
    if not isinstance(decoded, dict):
        return 0.0, 0.0, "NONE"
    return (
        float(decoded.get("multiplier", 0.0) or 0.0),
        float(decoded.get("atr_pct", 0.0) or 0.0),
        str(decoded.get("regime", "NONE") or "NONE"),
    )


def _load_all_trades() -> List[Tuple[str, str, float, bool, float, float, float, float, float, str]]:
    """
    All logged trades across every (strategy, symbol) pair, sorted
    ascending by timestamp. Each row:
        (strategy, symbol, entry_score, won, stake, payout, timestamp,
         multiplier, atr_pct, regime)

    multiplier/atr_pct/regime (Implementation Brief v4 §5.1) are decoded
    out of the existing `features` JSON blob (see _decode_ml_fields()) —
    no schema migration needed, since strategy_stats.py's trades table
    already carries a `features` column and bot_engine.py's
    record_trade() call now populates it for VOL_MULTIPLIER_SYMBOLS
    trades. Rows with no features blob default to (0.0, 0.0, "NONE").
    """
    rows: List[Tuple[str, str, float, bool, float, float, float, float, float, str]] = []
    if SQLITE_AVAILABLE and os.path.exists(TRADES_SQLITE_PATH):
        import sqlite3
        conn = sqlite3.connect(TRADES_SQLITE_PATH, check_same_thread=False)
        try:
            try:
                cur = conn.execute(
                    "SELECT strategy, symbol, entry_score, won, stake, payout, timestamp, "
                    "features FROM trades ORDER BY timestamp ASC"
                )
                fetched = cur.fetchall()
                has_features_col = True
            except sqlite3.OperationalError:
                # Pre-Brief-v6 database, before `features` existed at all
                # (strategy_stats.py's defensive ALTER TABLE normally
                # prevents this, but stay resilient against an even older
                # DB file that was never opened through that code path).
                cur = conn.execute(
                    "SELECT strategy, symbol, entry_score, won, stake, payout, timestamp "
                    "FROM trades ORDER BY timestamp ASC"
                )
                fetched = cur.fetchall()
                has_features_col = False

            for row in fetched:
                if has_features_col:
                    strategy, symbol, entry_score, won, stake, payout, ts, features_raw = row
                else:
                    strategy, symbol, entry_score, won, stake, payout, ts = row
                    features_raw = None
                multiplier, atr_pct, regime = _decode_ml_fields(features_raw)
                rows.append((strategy, symbol, float(entry_score or 0.0), bool(won),
                             float(stake or 0.0), float(payout or 0.0), float(ts),
                             multiplier, atr_pct, regime))
        finally:
            conn.close()
    elif os.path.exists(TRADES_JSON_PATH):
        with open(TRADES_JSON_PATH, "r") as f:
            data = json.load(f)
        for _key, records in data.items():
            for r in records:
                multiplier, atr_pct, regime = _decode_ml_fields(r.get("features"))
                rows.append((r["strategy"], r["symbol"], float(r.get("entry_score") or 0.0),
                             bool(r["won"]), float(r.get("stake") or 0.0),
                             float(r.get("payout") or 0.0), float(r["timestamp"]),
                             multiplier, atr_pct, regime))
        rows.sort(key=lambda r: r[6])
    return rows


def _count_all_trades() -> int:
    if SQLITE_AVAILABLE and os.path.exists(TRADES_SQLITE_PATH):
        import sqlite3
        conn = sqlite3.connect(TRADES_SQLITE_PATH, check_same_thread=False)
        try:
            cur = conn.execute("SELECT COUNT(*) FROM trades")
            return int(cur.fetchone()[0])
        finally:
            conn.close()
    elif os.path.exists(TRADES_JSON_PATH):
        with open(TRADES_JSON_PATH, "r") as f:
            data = json.load(f)
        return sum(len(v) for v in data.values())
    return 0


def _load_recent_results_for_pair(strategy: str, symbol: str, limit: int) -> List[bool]:
    """Most-recent-first list of win/loss bools for one (strategy, symbol) pair."""
    if SQLITE_AVAILABLE and os.path.exists(TRADES_SQLITE_PATH):
        import sqlite3
        conn = sqlite3.connect(TRADES_SQLITE_PATH, check_same_thread=False)
        try:
            cur = conn.execute(
                "SELECT won FROM trades WHERE strategy=? AND symbol=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (strategy, symbol, limit),
            )
            return [bool(row[0]) for row in cur.fetchall()]
        finally:
            conn.close()
    elif os.path.exists(TRADES_JSON_PATH):
        with open(TRADES_JSON_PATH, "r") as f:
            data = json.load(f)
        key = f"{strategy}::{symbol}"
        records = sorted(data.get(key, []), key=lambda r: r["timestamp"], reverse=True)
        return [bool(r["won"]) for r in records[:limit]]
    return []


# ── FEATURE ENGINEERING ───────────────────────────────────────────────

def _hour_sin_cos(timestamp: float) -> Tuple[float, float]:
    tm = time.gmtime(timestamp)
    hour = tm.tm_hour + tm.tm_min / 60.0
    angle = 2.0 * math.pi * hour / 24.0
    return math.sin(angle), math.cos(angle)


def _win_rate_and_streak(prior_results_ascending: List[bool], window: int) -> Tuple[float, float]:
    """
    prior_results_ascending: outcomes strictly BEFORE the trade in
    question, oldest first. Returns (rolling_win_rate, streak) where
    streak is positive for a win streak, negative for a loss streak,
    0.0 if there is no prior history for this pair yet.
    """
    if not prior_results_ascending:
        return 0.5, 0.0  # uninformative prior — no history yet
    recent = prior_results_ascending[-window:]
    win_rate = sum(recent) / len(recent)
    last = prior_results_ascending[-1]
    streak = 0
    for r in reversed(prior_results_ascending):
        if r == last:
            streak += 1
        else:
            break
    return win_rate, float(streak if last else -streak)


def _feature_dict(strategy: str, symbol: str, entry_score: float, timestamp: float,
                   recent_win_rate: float, streak: float,
                   multiplier: float = 0.0, atr_pct: float = 0.0,
                   regime: str = "NONE") -> Dict[str, object]:
    """
    Implementation Brief v4 §5.1 — leverage-aware entry filter. multiplier/
    atr_pct/regime give the model visibility into which instrument-family
    a trade is on and how leveraged/volatile it was — all now materially
    different across e.g. R_10 (x400 floor) vs R_100 (x40 floor), where
    previously the feature dict had no way to distinguish them. Defaults
    (0.0, 0.0, "NONE") apply to old Rise/Fall rows and any non-
    VOL_MULTIPLIER_SYMBOLS trade, so the model can tell old-regime rows
    apart from new ones rather than crashing on missing fields.
    """
    hour_sin, hour_cos = _hour_sin_cos(timestamp)
    return {
        "strategy": strategy,          # string value -> one-hot encoded
        "symbol": symbol,               # string value -> one-hot encoded
        "entry_score": float(entry_score),
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "recent_win_rate": float(recent_win_rate),
        "streak": float(streak),
        "multiplier": float(multiplier),
        "atr_pct": float(atr_pct),
        "regime": regime,              # one-hot via _ManualVectorizer, same as strategy/symbol
    }


def _build_training_set(rows: List[Tuple[str, str, float, bool, float, float, float, float, float, str]]
                         ) -> Tuple[List[Dict[str, object]], List[int]]:
    """
    rows must already be sorted ascending by timestamp. Each row's
    features are built only from that (strategy, symbol) pair's prior
    rows — walk-forward safe by construction.
    """
    history: Dict[Tuple[str, str], List[bool]] = {}
    X: List[Dict[str, object]] = []
    y: List[int] = []
    for strategy, symbol, entry_score, won, _stake, _payout, ts, multiplier, atr_pct, regime in rows:
        key = (strategy, symbol)
        prior = history.setdefault(key, [])
        win_rate, streak = _win_rate_and_streak(prior, ROLLING_WINDOW)
        X.append(_feature_dict(strategy, symbol, entry_score, ts, win_rate, streak,
                                multiplier, atr_pct, regime))
        y.append(1 if won else 0)
        prior.append(bool(won))
    return X, y


# ── MANUAL (NUMPY-ONLY) FALLBACK MODEL ────────────────────────────────

class _ManualVectorizer:
    """DictVectorizer-equivalent: string values -> one-hot, numeric values
    -> passthrough. Unseen categorical values at transform time are
    ignored (encoded as all-zero), matching sklearn's default DictVectorizer
    behavior for unseen keys."""

    def __init__(self):
        self._cat_index: Dict[str, int] = {}
        self._numeric_keys: List[str] = []
        self._numeric_index: Dict[str, int] = {}
        self.n_features_: int = 0

    def fit(self, dict_list: List[Dict[str, object]]) -> "_ManualVectorizer":
        cat_names = set()
        numeric_keys = set()
        for d in dict_list:
            for k, v in d.items():
                if isinstance(v, str):
                    cat_names.add(f"{k}={v}")
                else:
                    numeric_keys.add(k)
        self._cat_index = {name: i for i, name in enumerate(sorted(cat_names))}
        self._numeric_keys = sorted(numeric_keys)
        self._numeric_index = {k: i for i, k in enumerate(self._numeric_keys)}
        self.n_features_ = len(self._cat_index) + len(self._numeric_keys)
        return self

    def transform(self, dict_list: List[Dict[str, object]]) -> np.ndarray:
        n_cat = len(self._cat_index)
        X = np.zeros((len(dict_list), self.n_features_), dtype=float)
        for i, d in enumerate(dict_list):
            for k, v in d.items():
                if isinstance(v, str):
                    idx = self._cat_index.get(f"{k}={v}")
                    if idx is not None:
                        X[i, idx] = 1.0
                else:
                    j = self._numeric_index.get(k)
                    if j is not None:
                        X[i, n_cat + j] = float(v)
        return X

    def fit_transform(self, dict_list: List[Dict[str, object]]) -> np.ndarray:
        self.fit(dict_list)
        return self.transform(dict_list)


class _ManualLogisticRegression:
    """Plain batch-gradient-descent logistic regression with L2 regularization
    and feature standardization, implemented with numpy only."""

    def __init__(self, lr: float = 0.2, epochs: int = 800, l2: float = 1e-3):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_ManualLogisticRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        Xs = (X - self._mean) / self._std
        n, d = Xs.shape
        self.weights = np.zeros(d)
        self.bias = 0.0
        for _ in range(self.epochs):
            z = Xs @ self.weights + self.bias
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = Xs.T @ (p - y) / n + self.l2 * self.weights
            grad_b = float(np.mean(p - y))
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xs = (X - self._mean) / self._std
        z = Xs @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-z))


# ── MODEL WRAPPER (trains + retrains + predicts) ──────────────────────

class MetaLabelingModel:
    def __init__(self):
        self._lock = threading.RLock()
        self.backend_name = "sklearn" if SKLEARN_AVAILABLE else "manual"
        self._vectorizer = None
        self._clf = None
        self._fitted = False
        self._trained_on_n_trades = 0

    @property
    def fitted(self) -> bool:
        return self._fitted

    def maybe_retrain(self, force: bool = False) -> bool:
        """Retrains if never fitted, or if >= META_LABEL_RETRAIN_EVERY_N new
        trades have closed since the last training run. Returns True if a
        retrain happened."""
        with self._lock:
            total = _count_all_trades()
            if total < config.META_LABEL_MIN_TRADES:
                self._fitted = False
                return False

            due = force or (not self._fitted) or (
                total - self._trained_on_n_trades >= config.META_LABEL_RETRAIN_EVERY_N
            )
            if not due:
                return False

            rows = _load_all_trades()  # all trades closed so far — the "past" relative to now
            X_dicts, y = _build_training_set(rows)

            if SKLEARN_AVAILABLE:
                vectorizer = DictVectorizer(sparse=True)
                Xv = vectorizer.fit_transform(X_dicts)
                # BUG FIX (win-rate/drawdown pass, Aug 2026): class_weight=
                # "balanced" reweights the loss function so both classes
                # contribute equally regardless of their true frequency —
                # appropriate when you want a rare class not to be ignored,
                # wrong here, where predict_proba()'s output feeds directly
                # into an EV calculation (p*b - (1-p)) that needs p to be a
                # genuine calibrated probability, not a class-balance-
                # adjusted one. Verified directly: on a 35-trade sample
                # with a real 77% empirical win rate, "balanced" predicted
                # 0.512 (useless, ~coin-flip) for the exact same query that
                # the unweighted model correctly predicted 0.776 for. This
                # was silently defeating the whole EV gate — with p always
                # pulled toward 0.5, EV always landed near breakeven,
                # never confidently clearing META_LABEL_EV_MARGIN either
                # direction, regardless of how strong the real signal was.
                clf = LogisticRegression(max_iter=1000)
                clf.fit(Xv, y)
            else:
                vectorizer = _ManualVectorizer()
                Xv = vectorizer.fit_transform(X_dicts)
                clf = _ManualLogisticRegression()
                clf.fit(Xv, np.array(y, dtype=float))

            self._vectorizer = vectorizer
            self._clf = clf
            self._trained_on_n_trades = total
            self._fitted = True
            logger.info(
                "meta_labeling: retrained (%s backend) on %d trades",
                self.backend_name, total,
            )
            return True

    def predict_proba_one(self, feat: Dict[str, object]) -> float:
        with self._lock:
            if not self._fitted:
                return 0.5
            Xv = self._vectorizer.transform([feat])
            if SKLEARN_AVAILABLE:
                return float(self._clf.predict_proba(Xv)[0][1])
            return float(self._clf.predict_proba(Xv)[0])


# ── PER-PAIR ENRICHED-FEATURE EV MODEL (Implementation Brief v6, PART 5) ──

class _PairEVModel:
    """
    Lightweight per-(strategy, symbol) logistic model trained only on the
    enriched feature vectors logged via strategy_stats.stats.get_feature_rows()
    (rsi, roc, bb_pct_b, atr_expansion_ratio, hour_utc — see PART 2/3).
    Separate from MetaLabelingModel, which trains one global model across
    all pairs on entry_score/hour/recent_win_rate/streak only, since most
    pairs won't have enough feature rows to fit anything on for a long
    time. predict_proba() returns None (not a guess) whenever a pair
    doesn't clear EV_MIN_FEATURE_ROWS yet or nothing fittable exists —
    callers must treat None as "fall back to the global model", not 0.5.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._cache: Dict[Tuple[str, str], dict] = {}
        # cache[(strategy, symbol)] = {"vectorizer":..., "clf":..., "n": int}

    @staticmethod
    def _numeric_subset(d: Dict[str, object]) -> Dict[str, float]:
        return {
            k: float(v) for k, v in d.items()
            if k in ENRICHED_FEATURE_KEYS and isinstance(v, (int, float))
        }

    def _fit(self, rows: List[dict]) -> Optional[dict]:
        X_dicts: List[Dict[str, float]] = []
        y: List[int] = []
        for r in rows:
            feat = self._numeric_subset(r)
            if not feat:
                continue
            X_dicts.append(feat)
            # SIGNAL DIRECTION INVERSION (see config.py): this model's
            # features always describe the ORIGINAL signal direction, so
            # its label must too. For a row where bot_engine._execute()
            # inverted the executed trade, r["won"] reflects the INVERTED
            # trade's real outcome — flip it back here so the label stays
            # "did the original direction win", consistent with what the
            # features actually describe. Without this, INVERT trades
            # would silently corrupt the exact model that decided to
            # invert them in the first place.
            won_original = bool(r.get("won"))
            if r.get("inverted"):
                won_original = not won_original
            y.append(1 if won_original else 0)
        if len(X_dicts) < EV_MIN_FEATURE_ROWS or len(set(y)) < 2:
            return None  # not enough rows, or only one outcome class so far
        if SKLEARN_AVAILABLE:
            vectorizer = DictVectorizer(sparse=True)
            Xv = vectorizer.fit_transform(X_dicts)
            # BUG FIX — see the identical fix + full explanation on
            # MetaLabelingModel.maybe_retrain()'s LogisticRegression above.
            # This is the model predict_take_trade() actually consults, so
            # this instance is the one that mattered most.
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Xv, y)
        else:
            vectorizer = _ManualVectorizer()
            Xv = vectorizer.fit_transform(X_dicts)
            clf = _ManualLogisticRegression()
            clf.fit(Xv, np.array(y, dtype=float))
        return {"vectorizer": vectorizer, "clf": clf, "n": len(X_dicts)}

    def predict_proba(self, strategy: str, symbol: str,
                       feat: Dict[str, object]) -> Optional[float]:
        """
        Returns predicted P(win) from this pair's enriched-feature model,
        or None if the pair doesn't have enough logged feature rows yet
        (below EV_MIN_FEATURE_ROWS) or nothing fittable exists.
        """
        numeric_feat = self._numeric_subset(feat)
        if not numeric_feat:
            return None
        with self._lock:
            rows = strategy_stats.stats.get_feature_rows(strategy, symbol, window=500)
            if len(rows) < EV_MIN_FEATURE_ROWS:
                return None
            cached = self._cache.get((strategy, symbol))
            if cached is None or cached["n"] < len(rows):
                fitted = self._fit(rows)
                if fitted is None:
                    return None
                self._cache[(strategy, symbol)] = fitted
                cached = fitted
            Xv = cached["vectorizer"].transform([numeric_feat])
            if SKLEARN_AVAILABLE:
                return float(cached["clf"].predict_proba(Xv)[0][1])
            return float(cached["clf"].predict_proba(Xv)[0])


# ── PREDICTION LOG (for later validation of the filter) ───────────────

class _PredictionLogSqlite:
    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        self._init_db()

    def _conn(self):
        import sqlite3
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            self._local.conn = conn
        return conn

    def _init_db(self):
        import sqlite3
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                symbol TEXT NOT NULL,
                entry_score REAL,
                recent_win_rate REAL,
                streak REAL,
                take INTEGER NOT NULL,
                confidence REAL NOT NULL,
                bypassed INTEGER NOT NULL,
                pred_timestamp REAL NOT NULL,
                outcome INTEGER,
                outcome_timestamp REAL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_pending "
            "ON predictions (strategy, symbol, outcome, pred_timestamp)"
        )
        conn.commit()
        conn.close()

    def insert(self, strategy, symbol, entry_score, recent_win_rate, streak,
               take, confidence, bypassed, timestamp) -> int:
        conn = self._conn()
        cur = conn.execute(
            "INSERT INTO predictions (strategy, symbol, entry_score, recent_win_rate, "
            "streak, take, confidence, bypassed, pred_timestamp) VALUES (?,?,?,?,?,?,?,?,?)",
            (strategy, symbol, entry_score, recent_win_rate, streak,
             int(take), float(confidence), int(bypassed), timestamp),
        )
        conn.commit()
        return cur.lastrowid

    def record_outcome(self, strategy, symbol, won, timestamp) -> bool:
        conn = self._conn()
        cur = conn.execute(
            "SELECT id FROM predictions WHERE strategy=? AND symbol=? AND outcome IS NULL "
            "ORDER BY pred_timestamp ASC LIMIT 1",
            (strategy, symbol),
        )
        row = cur.fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE predictions SET outcome=?, outcome_timestamp=? WHERE id=?",
            (int(won), timestamp, row[0]),
        )
        conn.commit()
        return True


class _PredictionLogJson:
    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(self.path):
            self._write([])

    def _read(self) -> List[dict]:
        try:
            with open(self.path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write(self, data: List[dict]):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)

    def insert(self, strategy, symbol, entry_score, recent_win_rate, streak,
               take, confidence, bypassed, timestamp) -> int:
        data = self._read()
        new_id = (data[-1]["id"] + 1) if data else 1
        data.append({
            "id": new_id, "strategy": strategy, "symbol": symbol,
            "entry_score": entry_score, "recent_win_rate": recent_win_rate,
            "streak": streak, "take": bool(take), "confidence": confidence,
            "bypassed": bool(bypassed), "pred_timestamp": timestamp,
            "outcome": None, "outcome_timestamp": None,
        })
        self._write(data)
        return new_id

    def record_outcome(self, strategy, symbol, won, timestamp) -> bool:
        data = self._read()
        candidates = [
            r for r in data
            if r["strategy"] == strategy and r["symbol"] == symbol and r["outcome"] is None
        ]
        if not candidates:
            return False
        oldest = min(candidates, key=lambda r: r["pred_timestamp"])
        for r in data:
            if r["id"] == oldest["id"]:
                r["outcome"] = bool(won)
                r["outcome_timestamp"] = timestamp
                break
        self._write(data)
        return True


class PredictionLog:
    def __init__(self):
        self._lock = threading.RLock()
        if SQLITE_AVAILABLE:
            self._backend = _PredictionLogSqlite(PRED_SQLITE_PATH)
        else:
            self._backend = _PredictionLogJson(PRED_JSON_PATH)

    def insert(self, **kwargs) -> int:
        with self._lock:
            return self._backend.insert(**kwargs)

    def record_outcome(self, strategy: str, symbol: str, won: bool,
                        timestamp: Optional[float] = None) -> bool:
        with self._lock:
            return self._backend.record_outcome(
                strategy, symbol, won, timestamp if timestamp is not None else time.time()
            )


# ── BAYESIAN TAKE/INVERT BANDIT (win-rate pass, Aug 2026) ────────────────
# User-directed replacement for the enriched-feature EV-gate below: build
# purely from this account's own realized TAKE-vs-INVERT outcomes per
# symbol (strategy_stats.get_take_invert_stats()) — no other data, no
# generic priors borrowed from elsewhere. A Beta(1,1) prior is the
# standard maximally-uncertain starting point (uniform over [0,1]), not a
# borrowed opinion about what a "good" win rate looks like.
#
# Why this replaces the old 30-row logistic-regression EV-gate for this
# specific decision: that model needed EV_MIN_FEATURE_ROWS=30 rows before
# it would even attempt a fit, and evaluated a 5-feature (rsi/roc/etc.)
# regression — appropriate for a richer "should I take THIS specific
# setup" question, but overkill and slow for the simpler binary question
# "does TAKE or INVERT perform better on this symbol overall". A
# Beta-Bernoulli comparison needs far less data to say something useful
# (naturally humble — wide, low-confidence posteriors — with a handful of
# trades; naturally confident once enough accumulate), with no feature
# vector required at all, matching the user's repeated ask for faster
# reaction to real evidence. The old EV-gate/_PairEVModel infrastructure
# is left in place (unused by this decision now) rather than deleted, in
# case a richer model is worth revisiting once far more data exists.

def _beta_posterior_mean(wins: int, losses: int) -> float:
    """Beta(1,1) uniform prior -> posterior mean = (1+wins)/(2+wins+losses)."""
    alpha = 1 + wins
    beta = 1 + losses
    return alpha / (alpha + beta)


def _prob_a_beats_b(wins_a: int, losses_a: int, wins_b: int, losses_b: int,
                     n_samples: int = 4000) -> float:
    """
    Monte Carlo estimate of P(p_a > p_b), where p_a ~ Beta(1+wins_a,
    1+losses_a) and p_b ~ Beta(1+wins_b, 1+losses_b) are independent
    Beta(1,1)-prior posteriors over each mode's true win rate. This is
    "how confident can we be that mode A's real win rate exceeds mode
    B's, given only the trades observed so far" — wide posteriors (few
    trades) naturally pull this toward 0.5 regardless of the point
    estimate; narrow posteriors (many trades) let it approach 0 or 1.
    No hand-tuned sample-size cliff — the uncertainty is priced in
    directly by the shape of each Beta distribution.
    """
    alpha_a, beta_a = 1 + wins_a, 1 + losses_a
    alpha_b, beta_b = 1 + wins_b, 1 + losses_b
    a_wins = 0
    for _ in range(n_samples):
        if random.betavariate(alpha_a, beta_a) > random.betavariate(alpha_b, beta_b):
            a_wins += 1
    return a_wins / n_samples


# ── MODULE-LEVEL SINGLETONS ────────────────────────────────────────────
_model = MetaLabelingModel()
_prediction_log = PredictionLog()
_ev_model = _PairEVModel()  # Implementation Brief v6, PART 5


# ── PUBLIC API ──────────────────────────────────────────────────────────

def predict_take_trade(features: Dict[str, object]) -> Tuple[str, float]:
    """
    features must contain: "strategy", "symbol", "entry_score".
    Optional: "timestamp" (defaults to now), "recent_win_rate" / "streak"
    if the caller wants to override the live-computed values.

    Returns (action, confidence):
      action "TAKE"   — follow the signal's original direction
      action "INVERT" — take the OPPOSITE direction from the same entry
                         (see config.py's SIGNAL DIRECTION INVERSION section)

    Every SYMBOL has a DEFAULT action (config.META_LABEL_DEFAULT_ACTION_BY_SYMBOL,
    fallback config.META_LABEL_DEFAULT_ACTION_FALLBACK). A trade is never
    skipped outright; this only ever chooses between the symbol's default
    action and its opposite.

    REWRITTEN (win-rate pass, Aug 2026): per user request, this now uses a
    Bayesian Beta-Bernoulli bandit (see _beta_posterior_mean() /
    _prob_a_beats_b() above) built ONLY from this symbol's own realized
    TAKE-vs-INVERT outcomes (strategy_stats.get_take_invert_stats()) —
    replacing the old enriched-feature EV-gate (_ev_model /
    _PairEVModel), which needed 30 feature rows before attempting
    anything. The bandit needs far less data to say something useful,
    with uncertainty priced in automatically by each mode's Beta
    posterior width rather than a hand-tuned row-count cliff:
      1. Pull (wins_take, losses_take, wins_invert, losses_invert) for
         this symbol across all strategies that trade it.
      2. If the OVERRIDE mode (whichever ISN'T the default) has at least
         BAYESIAN_MIN_SAMPLES_FOR_OVERRIDE trades, compute
         P(override mode's true win rate > default mode's) via Monte
         Carlo sampling from both Beta posteriors.
      3. Switch to the override mode only if that probability clears
         BAYESIAN_OVERRIDE_CONFIDENCE. Otherwise stay on the default.
    Below the minimum sample count, always returns the default at
    confidence 1.0 — a pass-through/policy marker, not a calibrated
    probability. No enriched features, no external data — purely this
    symbol's own win/loss history.

    NOTE (user-directed, Aug 2026): bot_engine.py's execution path no
    longer uses this function's TAKE/INVERT output to decide trade
    direction — see config.INVERT_ALL_SIGNALS / bot_engine.py's
    "UNIVERSAL SIGNAL INVERSION" step, which unconditionally inverts
    every symbol's direction regardless of what this bandit would have
    picked. This function and its underlying win/loss bookkeeping are
    left in place and still callable (e.g. for the dashboard / smoke
    test / future use) but are no longer part of the live trade-decision
    path.
    """
    strategy = str(features["strategy"])
    symbol = str(features["symbol"])
    entry_score = float(features["entry_score"])
    timestamp = float(features.get("timestamp", time.time()))

    default_action = getattr(config, "META_LABEL_DEFAULT_ACTION_BY_SYMBOL", {}).get(
        symbol, getattr(config, "META_LABEL_DEFAULT_ACTION_FALLBACK", "TAKE"))
    override_action = "INVERT" if default_action == "TAKE" else "TAKE"

    if "recent_win_rate" in features and "streak" in features:
        recent_win_rate = float(features["recent_win_rate"])
        streak = float(features["streak"])
    else:
        results_desc = _load_recent_results_for_pair(strategy, symbol, ROLLING_WINDOW)
        recent_win_rate, streak = _win_rate_and_streak(list(reversed(results_desc)), ROLLING_WINDOW)

    action, confidence, bypassed = default_action, 1.0, True

    try:
        tis = strategy_stats.stats.get_take_invert_stats(symbol)
        wins_take, losses_take = tis["wins_take"], tis["losses_take"]
        wins_invert, losses_invert = tis["wins_invert"], tis["losses_invert"]
        n_override = (wins_invert + losses_invert) if override_action == "INVERT" \
            else (wins_take + losses_take)
        min_samples = getattr(config, "BAYESIAN_MIN_SAMPLES_FOR_OVERRIDE", 8)

        if n_override >= min_samples:
            if override_action == "INVERT":
                prob_override_better = _prob_a_beats_b(wins_invert, losses_invert, wins_take, losses_take)
            else:
                prob_override_better = _prob_a_beats_b(wins_take, losses_take, wins_invert, losses_invert)

            override_min_conf = getattr(config, "BAYESIAN_OVERRIDE_CONFIDENCE", 0.80)
            bypassed = False
            if prob_override_better >= override_min_conf:
                action, confidence = override_action, prob_override_better
            else:
                action = default_action
                confidence = _beta_posterior_mean(wins_take, losses_take) if default_action == "TAKE" \
                    else _beta_posterior_mean(wins_invert, losses_invert)

            logger.info(
                "BAYESIAN BANDIT: %s default=%s override=%s p(override_better)=%.3f "
                "(take:%dW/%dL invert:%dW/%dL) -> action=%s",
                symbol, default_action, override_action, prob_override_better,
                wins_take, losses_take, wins_invert, losses_invert, action,
            )
        # else: override mode doesn't have enough of its own trades yet —
        # stay on default, confidence stays the 1.0 pass-through marker.
    except Exception as exc:
        logger.warning(f"Bayesian TAKE/INVERT bandit failed for {symbol}: {exc}")
        action, confidence, bypassed = default_action, 1.0, True

    _prediction_log.insert(
        strategy=strategy, symbol=symbol, entry_score=entry_score,
        recent_win_rate=recent_win_rate, streak=streak,
        take=True, confidence=confidence, bypassed=bypassed, timestamp=timestamp,
    )
    return action, confidence


def record_outcome(strategy: str, symbol: str, won: bool,
                    timestamp: Optional[float] = None) -> bool:
    """
    Call this when a trade closes (alongside strategy_stats.stats.record_trade)
    to backfill the actual outcome onto the oldest still-pending logged
    prediction for that (strategy, symbol) pair, for later validation.
    Returns False if there was no pending prediction to match against.
    """
    return _prediction_log.record_outcome(strategy, symbol, won, timestamp)


def force_retrain() -> bool:
    """Manually trigger a retrain regardless of the retrain cadence."""
    return _model.maybe_retrain(force=True)


def status() -> dict:
    """Diagnostic snapshot, e.g. for a dashboard route."""
    total = _count_all_trades()
    return {
        "backend": _model.backend_name,
        "total_trades_logged": total,
        "min_trades_required": config.META_LABEL_MIN_TRADES,
        "filter_active": total >= config.META_LABEL_MIN_TRADES and _model.fitted,
        "trained_on_n_trades": _model._trained_on_n_trades,
        "retrain_every_n": config.META_LABEL_RETRAIN_EVERY_N,
        "rolling_window": ROLLING_WINDOW,
        "take_threshold": TAKE_THRESHOLD,
        "ev_gate_min_feature_rows": EV_MIN_FEATURE_ROWS,
        "ev_gate_pairs_active": len(_ev_model._cache),
    }


if __name__ == "__main__":
    # Quick smoke test against whatever strategy_stats data already exists
    # (or synthetic data if none does).
    n = _count_all_trades()
    print(f"Existing logged trades: {n}")
    if n < config.META_LABEL_MIN_TRADES:
        print("Generating synthetic trades for a smoke test...")
        import random
        random.seed(0)
        strategies = ["mean_reversion", "range_break", "step_drift"]
        symbols = ["R_75", "R_100", "1HZ75V"]
        for i in range(config.META_LABEL_MIN_TRADES + 50):
            strategy_stats.stats.record_trade(
                strategy=random.choice(strategies),
                symbol=random.choice(symbols),
                entry_score=round(random.uniform(0.6, 0.95), 3),
                won=random.random() < 0.53,
                stake=1.0,
                payout=1.9,
                timestamp=time.time() - (config.META_LABEL_MIN_TRADES + 50 - i) * 60,
            )

    take, conf = predict_take_trade({
        "strategy": "mean_reversion", "symbol": "R_75", "entry_score": 0.82,
    })
    print("predict_take_trade ->", take, conf)
    print("status ->", status())
    print("record_outcome ->", record_outcome("mean_reversion", "R_75", won=True))
