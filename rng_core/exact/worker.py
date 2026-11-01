"""rng_core.exact.worker - run_exact in a CHILD PROCESS with hard CPU-time and address-space limits.

    python -m rng_core.exact.worker           (reads one JSON request on stdin, writes one JSON reply on stdout)
request: {"digits": [...], "epoch0": float|null, "budget_s": float, "max_mb": int, "enable_brute": bool, "try_mt": bool, "progress": {}}
The parent (adapter.run_solver_async) enforces a wall-clock timeout and kills the child; this module enforces RLIMIT_CPU /
RLIMIT_AS inside it, so a runaway solver cannot starve the bot's event loop or exhaust the 512 MB container.
"""
import json
import sys


def _limits(budget_s: float, max_mb: int) -> None:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (int(budget_s) + 5, int(budget_s) + 10))
        if max_mb and max_mb > 0:
            lim = int(max_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except Exception:
        pass                                      # non-POSIX: the parent's wall-clock kill is then the only guard


def main() -> int:
    req = json.loads(sys.stdin.read())
    _limits(float(req.get("budget_s", 20)), int(req.get("max_mb", 400)))
    from rng_core.exact.runner import run_exact
    try:
        res = run_exact(req["digits"], req.get("epoch0"), float(req.get("budget_s", 20)), bool(req.get("enable_brute", False)),
                        bool(req.get("try_mt", False)), req.get("progress") or {})
        out = {"ok": True, **res}
    except MemoryError:
        out = {"ok": False, "error": "memory limit"}
    except Exception as exc:                      # report, never crash silently
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
