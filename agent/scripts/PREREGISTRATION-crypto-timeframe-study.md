# Pre-registration: which timeframe and rule, if any, beats buy-and-hold on BTC and ETH, net of costs?

Written 2026-10-20, committed BEFORE the study code was run.

## Why
The paper bot uses two rules chosen after a non-pre-registered look. This study searches a FIXED list once,
selects on the tuning period only, and judges the winner once on a period it never saw.

## Data
yfinance BTC-USD and ETH-USD. Two timeframes: **1D** (from the first date both assets have data, about 2017-11-09)
and **4h** (resampled from 1h bars; yfinance only serves about 730 days of hourly data). Weekly is excluded:
too few observations for the longer lookbacks. 1h is excluded: fees (35 bps/side) swamp it, and 4h shows the trend.
- Split 1D: tune = start to 2022-12-31, **test = 2023-01-01 to latest**.
- Split 4h: tune = first half of the available bars, **test = second half**.

## Rule list (fixed; lookbacks in bars of the timeframe; long/flat; no other variants will be tried)
1. Donchian breakout, enter on close above the prior N-bar high, exit below the prior N/2-bar low, N in {20, 50, 100}
2. SMA cross, long while fast > slow, (fast, slow) in {(20,100), (50,200)}
3. Time-series momentum, long while the L-bar return > 0, L in {30, 90, 180}
4. Price above SMA(L), L in {50, 100, 200}
5. Vol-filtered trend: price > SMA(100) AND 20-bar realised vol below its trailing 365-bar median (1D) / 2190-bar median (4h)
6. Vote: long when at least 2 of {breakout N=50, SMA cross (50,200), momentum L=90} are long
Total 13 variants per timeframe.

## Method
- Each asset runs independently; the strategy return is the equal-weight average of BTC and ETH.
- Signal uses the closed bar; the position is held over the NEXT bar (no look-ahead).
- Cost: 25 bps fee + 10 bps slippage = 35 bps per side on every position change.
- Selection: per timeframe, the variant with the highest tuning-period Sharpe. Nothing else is chosen from the test period.
- Benchmark: equal-weight BTC and ETH buy-and-hold over the same test period, 35 bps on entry.
- Sharpe annualised with 365 days (1D) or 2190 bars (4h). Bootstrap: block bootstrap of test returns.

## A timeframe's selected variant PASSES only if ALL hold on its test period, net of costs
1. CAGR > 0
2. Sharpe above the benchmark's Sharpe
3. Max drawdown no worse than the benchmark's
4. Lower end of the 97.5% bootstrap interval of its Sharpe > 0 (two timeframes tested)
5. At least 8 completed round trips (pooled over both assets)
Also reported: all 13 variants' tuning and test numbers, for context only (not used to choose).

## Outcomes
- **Passes:** it becomes a candidate for a new forward paper test with its own pre-registration. Still no live money.
- **Nothing passes:** report "no robust edge found", keep the current paper bot unchanged, stop.

## Not claimed
Passing means the rule did not fail on past data. It does not mean future profit.
