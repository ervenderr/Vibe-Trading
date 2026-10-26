# Pre-registration: does a 30-minute opening range breakout make money on SPY and QQQ, on a paper account, after real fills?

Written 2026-10-26 ~13:30 UTC (US open), BEFORE the script exists or any order is sent. Exploration of the idea, not a proof.

## Why
Stocks on Alpaca are commission-free, so the ~0.5% crypto round-trip fee that killed the crypto tests does not apply.
We have never run a stock day-trading rule with fixed, written-down rules. Published ORB results exist but are unverified here.

## Rules (fixed, never tuned during the test)
- Symbols: SPY and QQQ, each run separately. Regular session only.
- Opening range (OR): high and low of the first 30 minutes (13:30-14:00 UTC while US is on daylight time; 14:30-15:00 after Nov 1).
- Signal: after the OR, the first 5-minute bar that CLOSES above the OR high -> long; closes below the OR low -> short.
- Size: $5,000 notional per symbol, whole shares, market order at the signal.
- Stop: the opposite side of the OR (long stops at OR low; short stops at OR high), checked on 5-minute closes.
- Exit: the stop, or flat at 15:55 ET (5 minutes before the close). Never hold overnight.
- One trade per symbol per day. No trade if OR height < 0.15% or > 1.5% of price. No trade on a day the market closes early.
- Paper-only guard and kill switch (`src/live/halt.py`) as in the other bots. Stop the test if own net loss exceeds $1,000.

## Judging
Own fills only (not account equity). Judged no earlier than 20 trading days, and at least 25 trades.
PASS only if all hold: (1) net P&L > 0; (2) profit factor > 1.2; (3) the 95% bootstrap lower bound of mean P&L per trade is not below -$5 (otherwise "inconclusive").
Fewer than 25 trades -> "inconclusive". A fail or inconclusive result changes nothing in the repo.

## Outcomes
- Pass: run another 40 days on paper before any further step. Still no live money.
- Fail: report "no edge found" and stop.

## Not claimed
20 days is a short window; a pass is weak evidence. Paper fills are optimistic about slippage. No real money, ever.
