"""rng_eval.stubs - stand-ins for network-only third-party modules (aiohttp, websockets, ...) so the engine module can be IMPORTED in a
sandbox without them (tests only; never used by the bot)."""
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from unittest import mock

NAMES = ("aiohttp", "websockets", "psycopg", "psycopg_pool", "pandas_market_calendars", "pyarrow", "pytz", "gunicorn")


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, names):
        self.names = set(names)

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name)
        m.__path__ = []
        m.__spec__ = spec
        m.__name__ = spec.name
        return m

    def exec_module(self, module):
        pass


def install_missing_stubs() -> list:
    missing = [n for n in NAMES if importlib.util.find_spec(n) is None]
    if missing:
        sys.meta_path.append(_Stub(missing))
    return missing
