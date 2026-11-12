"""
env_defaults.py - the bot's core settings, applied automatically at start-up.

Imported FIRST by main.py (`import env_defaults`, above `import config`). Each line only fills a value in when
that variable is NOT already set, so anything you type in the Render Environment tab still wins.
Secrets (DERIV_API_TOKEN, DERIV_APP_ID, RENDER_API_KEY) are deliberately NOT here: you add those in Render.
To change a default for good, edit it here; to change it for one service, set it in Render.
"""
import os

CORE_DEFAULTS = {
    "DERIV_ACCOUNT_MODE": "real",
    "SET_ALLOW_REAL": "1",
    "PREDICTOR_STAKE": "0.35",
    "TARGET_PROFIT": "1",
    "SESSION_SCHEDULE_ENABLED": "0",
    "MH_GATE": "0",
    "MH_INVERT": "1",
    "MH_MIN_EDGE": "-1",
    "MH_MIN_PAYOUT_RATIO": "2",
    "MH_OOS_CHECK": "0",
    "MH_PRED_V2": "1",
    "MH_RANK_KEY": "edge",
    "V2_ACT_ON_NO_STRUCTURE": "1",
}

for _k, _v in CORE_DEFAULTS.items():
    os.environ.setdefault(_k, _v)
