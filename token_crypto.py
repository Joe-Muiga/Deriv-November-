"""
token_crypto.py - Fernet encryption for follower API tokens.

Tokens are encrypted with ENCRYPTION_KEY before they reach the database and are never
logged or sent to the browser.  (Small shared module: both the web page and the follower
runtime need it.)
"""
import base64
import hashlib
import logging
from typing import Optional

import follower_config as fc

logger = logging.getLogger("follower.crypto")

_fernet = None


def _get():
    global _fernet
    if _fernet is None:
        from cryptography.fernet import Fernet
        if not fc.ENCRYPTION_KEY:
            raise RuntimeError("ENCRYPTION_KEY is not set")
        _fernet = Fernet(fc.ENCRYPTION_KEY.encode())
    return _fernet


def available() -> bool:
    try:
        _get()
        return True
    except Exception:
        return False


def encrypt(token: str) -> str:
    return _get().encrypt(token.encode()).decode()


def decrypt(blob: str) -> Optional[str]:
    try:
        return _get().decrypt(blob.encode()).decode()
    except Exception:
        logger.error("FOLLOWER token could not be decrypted (wrong ENCRYPTION_KEY?)")
        return None


def session_secret() -> bytes:
    """Secret for signed web-session cookies, derived from ENCRYPTION_KEY."""
    return hashlib.sha256(b"follower-web-session|" + fc.ENCRYPTION_KEY.encode()).digest()


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")
