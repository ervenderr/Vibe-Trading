# Pre-registration: do two slow trend rules on BTC/USD and ETH/USD beat buy-and-hold on a paper account?

Written 2026-10-19, committed BEFORE the bot was built or run.

## Why
Intraday crypto loses to fees (~0.25%/side). Slow daily trend rules trade 9-28 times in 9 years, so fees
barely matter. An earlier look (8 combos tried, NOT pre-registered) showed promise, so it counts as
exploration only. This test is forward-only, on paper.

## Candidates (parameters fixed, never tuned)
- **A, breakout:** long when the daily close is above the highest close of the previous 50 days;
  exit (flat) when the close is below the lowest close of the previous 20 days.
- **B, cross:** long while the 50-day SMA of closes is above the 200-day SMA; flat otherwise.
- Assets: BTC/USD and ETH/USD, each run separately. Long/flat only, no leverage, no shorting.

## Execution
- One check per day, after the 00:00 UTC daily close. Signal uses the closed daily bar only.
- Alpaca PAPER market order at the check. Fixed size: $5,000 notional per asset per rule.
  Sell the quantity actually held (buys are net of the fee).
- Paper-only guard, kill switch (`src/live/halt.py`), journal line per check, heartbeat for the dashboard.

## Benchmark
Buy-and-hold of the same assets, same start date, same $5,000 per asset, same fee on entry.

## Judging
Judged no earlier than 90 days after the first check, on paper results net of actual fees.
A rule PASSES only if ALL hold, per rule (both assets pooled):
1. Net P&L > 0
2. Max drawdown no worse than the benchmark's
3. At least 1 completed round trip (otherwise "inconclusive", not a pass)

Note: 90 days holds very few trades, so a pass is weak evidence. A fail or inconclusive result changes nothing.
Trend rules lagging buy-and-hold in a rising market is expected and is not a failure by itself.

## Outcomes
- **Pass:** keep running paper for 6+ months. Still no live money.
- **Fail:** report "no edge found" and stop the bot.

## Not claimed
Passing means the rule did not fail on a short window. It does not mean future profit. No real money, ever.

## Outcome: ABANDONED early (2026-10-25)
The user ended the test before the 90-day judging date. Positions were closed on paper (BTC and ETH, roughly -$240 of
unrealized loss realized after a pullback). No pass or fail is claimed; the test is simply incomplete.
