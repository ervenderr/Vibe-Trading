# Pre-registration: does strong Jev-judged news predict a 4-hour market-adjusted move?

Written 2026-10-13, committed BEFORE any confirmatory data was analysed.

## Why
The first event study (Feb to Sep 2026, 8 symbols, 7 horizons) found no horizon surviving
Bonferroni. The 4h horizon was closest (adjusted p=0.096, out of sample p=0.12). One look at
7 horizons can produce that by chance, so we test that single horizon on data the first study never saw.

## Hypothesis (the only confirmatory test)
For events with Jev-judged strong, relevant, confirmed news, the **side-signed, market-adjusted
(vs SPY) return over 4 hourly bars** has a mean above zero.

## Fixed in advance
- **Event definition:** unchanged `EventCriteria` defaults and `strong_events()`; Jev model `jev-1.13.0`.
- **Horizon:** `4h` from `DEFAULT_HORIZONS` (1h bars, same session, 4h minimum gap per symbol). No other horizon is confirmatory.
- **Market adjustment and cost:** SPY-adjusted; 6 bps round-trip cost.
- **Test:** two-sided sign-flip permutation test, seed 0.
- **Sample (none of it was in the first study):**
  - W1: 2025-10-15 to 2026-01-31, symbols NVDA TSLA AMD AAPL META MSFT AMZN GOOGL AVGO NFLX
  - W2: 2026-10-01 to 2026-10-13, same 10 symbols
  - W3: 2026-02-01 to 2026-09-30, AVGO and NFLX only (never studied)
  Pooled into one test. No window or symbol is dropped after seeing results.

## Success means ALL of
1. n >= 100 thinned events
2. p <= 0.05
3. mean market-adjusted return > 6 bps (positive after cost)
4. hit rate > 50%

## Outcomes
- **Success:** build a 4h-hold variant of the paper autopilot and judge it on 100+ paper trades. Still no live money.
- **Failure or n < 100:** report "no edge found", stop tuning news-based entries.

## Exploratory (reported, never used to claim success)
Splits by side (up vs down), by event type, and by symbol. Bonferroni-corrected across the
number of splits reported. These can only suggest the next pre-registered test.
