# donkey-finisher

MH (multi-horizon) Deriv digit bot. See `SET_MODE.md` for the SET engine it extends.

## MH tuning order (phases; compare results between phases)
1. **In-doubt buy handling** (bug fix, ON): a buy that times out / loses its socket is reconciled (late reply -> portfolio -> profit_table) instead of being dropped; its slot stays closed until the outcome is known.
2. **Env-only**: `MH_DEADLINE_RTT=p95`, `MH_HMAX_TRADE=2` (recommended, not defaults; see `render.yaml`). Effective config is logged at start (`MH CONFIG effective`). Payout-slip diagnostics.
3. **One conservatism knob**: `MH_WILSON_Z` *or* `MH_MIN_EDGE`, not both. Held-out (OOS) check of calibrated bins (log first, then `MH_OOS_CHECK=1`).
4. **Edge ranking**: `MH_RANK_KEY=edge` (shadow-logged as `rank_shadow` regardless).
5. **30 s windows / bar ranking**: `MH_RANK_WINDOW_SECONDS` (default 30; use 60 for the old per-minute basis while comparing phases 1-4), `MH_RANK_MODE=bar`.
6. **Optional** in-process refit: `MH_INPROC_REFIT_SECONDS`.

## New env vars
| var | default | effect |
|---|---|---|
| `MH_BUY_TIMEOUT_SECONDS` | 3.0 | MH buy reply timeout (other requests unchanged) |
| `MH_RECONCILE_WAIT_SECONDS` | 2.0 | wait for a late buy reply before asking Deriv |
| `MH_RECONCILE_RETRY_SECONDS` | 3.0 | retry the portfolio/profit_table query once after this |
| `MH_RECONCILE_SAFETY_SECONDS` | 30 | unresolved buy: hold the money reservation this much longer (slot never reopens) |
| `MH_ADOPT_ORPHANS_ON_BOOT` | 1 | adopt open digit contracts found on Deriv at boot (`arm=orphan`) |
| `MH_MAX_QUOTE_AGE_SECONDS` | 0 (off) | refetch admission quotes older than this |
| `MH_WILSON_Z` | 1.64 | Wilson z of the calibrated lower bound |
| `MH_OOS_CHECK` / `MH_OOS_MIN_N` / `MH_OOS_MARGIN` | 0 / 30 / 0.0 | held-out check: 0 = log only, 1 = enforce (held-out Wilson lb > 1/`MH_MIN_PAYOUT_RATIO` + margin, n >= min) |
| `MH_RANK_KEY` | mass | `mass` = raw top-cluster mass (old), `edge` = calibrated `lb*r-1` |
| `MH_RANK_MIN_EDGE` | `MH_MIN_EDGE` | edge floor under `MH_RANK_KEY=edge` (replaces the `SET_MIN_SET_PROB` mass floor) |
| `MH_RANK_WINDOW_SECONDS` | 30 | ranking window, 30 or 60 (60 = old minute) |
| `MH_WINDOW_TOP_N` | `MH_MINUTE_TOP_N` (1) | contracts bought per window |
| `MH_RANK_MODE` | snapshot | `bar` = rank the closed window, buy against the symbol's current state |
| `MH_INPROC_REFIT_SECONDS` | 0 (off) | background per-symbol model rebuild + atomic swap |

Expected trade-count change of 30 s windows: up to 2x the buys per minute (one per window), less where a window has no live admissible candidate. Compare with the per-window `WINDOW RANK ... pool snapshot=N bar=M` lines.

Tests: `python test_mh.py`, `test_mh_cluster.py`, `test_mh_phase1.py` ... `test_mh_phase5.py`, `test_mh_refit.py`, `smoke_test.py`.
