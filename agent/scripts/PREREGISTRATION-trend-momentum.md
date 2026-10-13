# Pre-registration: do two classic slow trend rules beat buy-and-hold, net of costs?

Written 2026-10-13, committed BEFORE the backtest was run.

## Why
Short-horizon scalping and news entries failed our tests (replay, event study, 4h confirmation).
Slow, daily-bar trend and momentum rules have the longest published track record, and costs matter
least at a monthly rebalance. We test exactly two candidates, fixed here, with no tuning.

## Universe (fixed)
SPY QQQ IWM EFA EEM TLT IEF GLD DBC VNQ (daily, adjusted closes, yfinance). An asset joins the
test on the first date it has enough history for the rule.

## Candidates (parameters fixed, never tuned)
- **A, trend filter:** at each month-end hold an asset if its close is above its 200-day average,
  equal weight across the held assets, cash (0% return) when none qualifies.
- **B, momentum:** at each month-end rank assets by 12-month return skipping the latest month
  (the "12-1" rule); hold the top 3 equally weighted, only those with a positive 12-1 return.
- **Benchmarks:** SPY buy-and-hold; equal-weight buy-and-hold of the whole universe, rebalanced monthly.

## Execution assumptions
Signals use the month-end close and trades happen at the NEXT day's close (no look-ahead).
Cost: 5 bps per unit of turnover. Cash earns 0.

## Periods
- Development view (reported only): start of data to 2016-12-31.
- **Out of sample (the only confirmatory period): 2017-01-01 to latest data.**

## Success for a candidate means ALL of (out of sample, net of costs)
1. CAGR > 0
2. Sharpe ratio above the equal-weight benchmark's Sharpe
3. Maximum drawdown no worse than the equal-weight benchmark's
4. Lower bound of the 97.5% bootstrap interval for the candidate's Sharpe > 0
   (97.5% because two candidates are tested)

## Outcomes
- **A candidate passes:** it earns a paper-trading implementation with fixed rules, judged live
  for at least 3 months. Still no live money.
- **Neither passes:** report "no robust edge found in these rules" and stop.

## Not claimed
Passing means the rule did not fail on past data. It does not mean future profit.
