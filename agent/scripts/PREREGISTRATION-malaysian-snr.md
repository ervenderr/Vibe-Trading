# Pre-registration: does a coded Malaysian SnR rule beat buy-and-hold, net of costs?

Written 2026-10-19, committed BEFORE the backtest was run.

## The method, as coded (one documented interpretation of the commonly taught rules)
Levels come from closing prices on DAILY bars.
- **A-level (resistance):** a close higher than the closes on both sides (a peak on the line chart); price = that close.
- **V-level (support):** a close lower than the closes on both sides; price = that close.
- **Gap level:** two consecutive same-colour candles with a gap between the first close and the second open.
  Two up candles give a support level, two down candles a resistance level; price = the first candle's close.
- A level is usable from the second bar after the bar that confirms it, for at most 250 bars.
- **Freshness:** a level starts fresh. A bar whose WICK reaches the level while its BODY does not cross it makes
  it unfresh. A bar whose BODY crosses the level makes it fresh again.

## Trade rule (long only, daily bars, one position per asset)
Enter at the bar's close when ALL hold, using only data through that bar:
1. Close is above its 200-day average (trade with the higher-timeframe direction).
2. The nearest support level below the previous close (V-level or up-gap level) was FRESH at the start of the bar.
3. The bar's low reached within 0.25 ATR(14) above that level, the bar closed above the level, and it closed above its open.
Exit at a bar's close when the first of these holds: close below the level minus 0.5 ATR (stop); close at or above the
nearest resistance level above entry, or entry + 3 ATR if none (target); 40 bars held (timeout).
Fills happen at the NEXT bar's close and positions earn from the bar after (the lab's 2-bar lag). Costs: 5 bps fee + 5 bps slippage per side.

## Universe and periods (fixed)
SPY QQQ IWM EFA EEM TLT GLD DBC VNQ, daily, from the start of data. Each asset runs the rule on its own with capital
split equally across the nine (cash when flat). Benchmark: the same nine assets, equal-weight buy-and-hold, rebalanced monthly.
Out of sample (the only confirmatory period): 2017-01-01 to latest. Development view (reported only): earlier data.

## Parameters are fixed and never tuned
200-day filter, 0.25 ATR touch, 0.5 ATR stop buffer, 3 ATR fallback target, 40-bar timeout, 250-bar level life.

## Success means ALL of (out of sample, net of costs)
1. CAGR > 0.
2. Sharpe above the equal-weight benchmark's Sharpe.
3. Maximum drawdown no worse than the benchmark's.
4. Lower bound of the 95% bootstrap interval for the strategy's Sharpe > 0 (one candidate tested).
5. At least 100 trades across the nine assets out of sample.

## Outcomes
- **Pass:** the rule earns a chart overlay, a paper-trading implementation with fixed rules, and 3 months of live paper judgment. No live money.
- **Fail:** report "no robust edge in this coded version" and stop. The method may still be useful as a chart-reading aid.

## Exploratory (never claims success)
GLD alone; hourly bars; long and short; other parameter values. Reported only to suggest a next pre-registered test.

## Not claimed
The coded rule is an interpretation. A fail does not prove the discretionary method fails in skilled hands.
