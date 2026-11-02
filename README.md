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

## Profit cycle (never-ending; the only reason this service redeploys)
There is **no time-based redeploy** any more. The bot runs one cycle at a time:

1. **Start** - connected, Deriv's own portfolio flat -> the live balance is saved as the cycle's **starting balance** (Render env var `CYCLE_STARTING_BALANCE`, written through the Render API). The bot connects only for sessions A/B, so this happens at the first connect after a boot.
2. **Trade** - net profit = live balance - starting balance. Target balance (display) = starting balance + `TARGET_PROFIT`.
3. **Target hit** (net profit >= `TARGET_PROFIT`) - no new proposal/buy is sent from that instant; the bot waits for every open contract to settle (polling Deriv's portfolio, max `CYCLE_DRAIN_TIMEOUT_SECONDS`), updates `CYCLE_COUNT` / `CYCLE_LIFETIME_PROFIT`, **clears** `CYCLE_STARTING_BALANCE` (verified by HTTP 200/201 + read-back), optionally waits `CYCLE_COOLDOWN_SECONDS`, then redeploys. The new process sees an empty starting balance and starts the next cycle. If the clear cannot be verified the bot does **not** redeploy (CRITICAL log, dashboard alert, retry with backoff).
4. **Low balance** - if the balance falls below the **minimum stake** and Deriv's portfolio is flat (never judged while a contract is open), the bot stops, clears the starting balance (verified) and **suspends** the Render service. The minimum stake is the engine's own flat stake (`PREDICTOR_STAKE`, the value every buy and `reserve()` use); the bot does not read Deriv's per-contract minimum, so keep `PREDICTOR_STAKE` at or above what Deriv accepts.
   **To resume:** top up the account, then **Resume** the service in Render. The starting balance is already cleared, so a fresh cycle begins. (Without `RENDER_API_KEY` the bot still stops and disconnects, but the service stays running idle - the dashboard says "IDLE and NOT suspended".)

### Env vars
| var | who sets it | meaning |
|---|---|---|
| `TARGET_PROFIT` | **you, by hand** (Render -> Environment) | absolute profit per cycle in account currency (`50` = stop at +50). The bot only **reads** it and never writes, clears or resets it. Unset / `0` = profit target OFF (it still records a starting balance and trades, never redeploys on profit; the low-balance suspend still applies). Not a number / negative / below 0.01 = ERROR logged, target OFF. `$` and commas are accepted. |
| `CYCLE_STARTING_BALANCE` | the bot | the cycle's starting balance. Never set it by hand, except `0` to force a fresh cycle. |
| `CYCLE_COUNT`, `CYCLE_LIFETIME_PROFIT` | the bot | completed cycles / sum of their net profit |
| `CYCLE_DRAIN_TIMEOUT_SECONDS` | you (optional) | default 900. On timeout: CRITICAL, dashboard flag, carry on (the next cycle's start still waits for a flat portfolio) |
| `CYCLE_COOLDOWN_SECONDS` | you (optional) | default 0. Pause between clear and redeploy |
| `RENDER_API_KEY` | you | **required**: without it nothing can be saved, cleared, redeployed or suspended, so no cycle starts and the bot does not trade (CRITICAL + dashboard alert). `RENDER_SERVICE_ID` is injected by Render. |
| `TARGET_CHECK_SECONDS` | you (optional) | default 3. How often the live balance is also polled |
| `PREDICTOR_STAKE` | you | the minimum stake (also the low-balance threshold) |

`TARGET_BALANCE` is **deprecated and ignored** (a WARNING is logged while it is set): delete it and use `TARGET_PROFIT`. The scheduler variables (`PRE_A_REDEPLOY_*`, `PRE_SESSION_REDEPLOY_CLEAR_CACHE`, `REDEPLOY_*`, `AUTO_REDEPLOY*`) and the old `PROFIT_CYCLE_*` percent settings are gone and ignored.

### Changing `TARGET_PROFIT` while a cycle runs
The saved starting balance stays; the new target applies to the existing cycle. A running process only sees an env-var change after it **restarts**: Render's dashboard asks, when you save, between "Save, rebuild, and deploy", "Save and deploy" and "Save only". Choose one of the deploy options (the process restarts, resumes the saved cycle and applies the new target) - with "Save only" the running process keeps the old value until the next restart/deploy. If the new target is already at or below the current net profit when the process boots, it takes the target-hit path (drain -> clear -> redeploy); the fresh process then records a new start with net profit 0, so this cannot loop.

### Known limitation
Net profit is a balance difference, so a deposit, withdrawal or demo-balance reset during a cycle shifts it. A single balance jump larger than any trade can explain is logged as `BALANCE JUMP` (WARNING); nothing is corrected.
