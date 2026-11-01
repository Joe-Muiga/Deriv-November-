"""
follower_config.py - every follower-system setting, read from the environment.

Required on Render:   DATABASE_URL (already set), ADMIN_PASSWORD, ENCRYPTION_KEY
Everything else has a default.  Nothing here touches config.py.

Generate ENCRYPTION_KEY once with:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""
import logging
import os

logger = logging.getLogger("follower.config")


def _s(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _i(name: str, default: int) -> int:
    try:
        return int(float(_s(name, str(default))))
    except Exception:
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(_s(name, str(default)))
    except Exception:
        return default


# ── secrets / connections ────────────────────────────────────────────────────
DATABASE_URL = _s("DATABASE_URL")
ADMIN_PASSWORD = _s("ADMIN_PASSWORD")
ENCRYPTION_KEY = _s("ENCRYPTION_KEY")

# ── zones ────────────────────────────────────────────────────────────────────
# Pre-registered HYPOTHESIS (not a finding): minutes 53..59 of every hour, inclusive.
# Start 53:00.0, last opening 59:59.9.
DEFAULT_ZONE_START = _i("DEFAULT_ZONE_START", 53)
DEFAULT_ZONE_END = _i("DEFAULT_ZONE_END", 59)
if not (0 <= DEFAULT_ZONE_START <= DEFAULT_ZONE_END <= 59):
    logger.warning("FOLLOWER bad DEFAULT_ZONE_START/END (%s,%s) - using 53..59",
                   DEFAULT_ZONE_START, DEFAULT_ZONE_END)
    DEFAULT_ZONE_START, DEFAULT_ZONE_END = 53, 59
ZONE_MIN_TRADES = _i("ZONE_MIN_TRADES", 150)
ZONE_MAX_PER_HOUR = _i("ZONE_MAX_PER_HOUR", 3)
ZONE_MIN_LEN = 3
ZONE_MAX_LEN = 8
SHRINK_K = 200

# ── verdict ──────────────────────────────────────────────────────────────────
VERDICT_MIN_TRADES = _i("VERDICT_MIN_TRADES", 300)

# ── stake ────────────────────────────────────────────────────────────────────
STAKE = _f("PREDICTOR_STAKE", 0.35)          # flat stake for every buy (same Render env var as the main account)
if STAKE <= 0:
    STAKE = 0.35
CURRENCY = _s("FOLLOWER_CURRENCY", "USD")

# ── runtime ──────────────────────────────────────────────────────────────────
PRE_CONNECT_SECONDS = 5.0       # connect followers this long before a session opens
SETTLE_GRACE_SECONDS = 30.0     # wait this long after a session for open contracts
SIGNAL_MAX_AGE = 1.5            # drop collector signals older than this (seconds)
FOLLOWER_CONCURRENCY = 3        # max buys in flight per follower
BALANCE_MAX_AGE = 5.0           # refresh the balance before a buy if older than this
RATE_LIMIT_BACKOFF = 2.0
MAX_CONSECUTIVE_FAILS = 5
SNAPSHOT_LEAD_SECONDS = 30.0    # hourly snapshot is taken this long before the hour starts

# ── database ─────────────────────────────────────────────────────────────────
DB_FLUSH_SECONDS = _f("FOLLOWER_DB_FLUSH_SECONDS", 30.0)
QUEUE_FILE = _s("FOLLOWER_QUEUE_FILE", "follower_db_queue.jsonl")
QUEUE_FILE_MAX_BYTES = 20 * 1024 * 1024

MAX_FOLLOWERS = 10


def db_enabled() -> bool:
    return bool(DATABASE_URL)


def web_enabled() -> bool:
    return bool(ADMIN_PASSWORD and ENCRYPTION_KEY)
