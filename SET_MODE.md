# SET MODE (demo)

Per symbol, per tick: predict the **3 most likely next last-digits**, then buy the **Over/Under** contract that wins on all 3.
All symbols are scanned at the same instant; the best signals are priced in parallel (live proposals) and ranked by
`P(win) x payout/stake`; the best few are bought in parallel. No payout floor/ceiling, no trade window, no Matches/Differs/Even/Odd.

* `{7,8,9}` -> Over 6   `{0,1,2}` -> Under 3   `{5,6,7}` -> Over 4   `{1,5,9}` -> skipped (spread out)
* Spread-out rule: the covering contract may win on at most `SET_MAX_COVER_WIDTH` digits (default 5).

**Demo only.** It refuses to start unless `DERIV_ACCOUNT_MODE=demo` (or `SET_ALLOW_REAL=1`).
Trades are not mirrored to follower accounts. `SET_MODE=0` runs the original engine.

| env var | default | meaning |
|---|---|---|
| `SET_SIZE` | 3 | digits predicted per tick |
| `SET_MAX_COVER_WIDTH` | 5 | "clustered" limit (Over 4 / Under 5 or tighter) |
| `SET_STAKE` | 1.0 | flat stake per contract |
| `SET_MAX_OPEN` | 6 | max contracts open at once |
| `SET_PRICE_TOP` | 4 | symbols priced in parallel per scan |
| `SET_MAX_BUYS_PER_SCAN` | 2 | contracts bought in parallel per scan |
| `SET_MIN_SCAN_GAP` | 0.25 | seconds between scans (API rate limits) |
| `SET_SIGNAL_MAX_AGE_FRAC` | 0.6 | a signal expires after this fraction of a tick interval |
| `SET_MIN_SCORE` | 0 | optional gate on P(win) x payout (off) |
| `SET_MIN_SET_PROB` | 0 | optional gate on P(next digit in set) (off) |
| `SET_REPORT_SECONDS` | 300 | accuracy/performance report period |

## Reading the demo result
Every `SET_REPORT_SECONDS` the log prints `SET REPORT`: live top-1 / top-3 hit rate (chance = 10% / 30%) and the trade
win rate next to break-even. Each trade logs `SET TRADE` (predicted digits, probabilities, contract, payout, Deriv
shortcode) and `SET RESULT` (settling digit, whether it was in the predicted set). `CONTRACT MISMATCH` = Deriv settled
something other than what was bought.

Tests: `python test_set_mode.py` (set mode) and `python test_predictor.py` (original engine).
