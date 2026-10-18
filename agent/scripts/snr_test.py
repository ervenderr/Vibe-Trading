"""Pre-registered test of the coded Malaysian SnR rule. Rules: scripts/PREREGISTRATION-malaysian-snr.md.

    python scripts/snr_test.py --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.market_data import get_loader  # noqa: E402
from src.signals.snr import snr_signal  # noqa: E402
from src.signals.strategy_lab import Costs, backtest  # noqa: E402
from src.signals.trend_study import bootstrap_sharpe_lower, cagr, equal_weights, max_drawdown, run_backtest, sharpe  # noqa: E402

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "DBC", "VNQ"]
COSTS = Costs(fee_bps=5.0, slippage_bps=5.0)
OOS_START = pd.Timestamp("2017-01-01")
MIN_TRADES = 100
CONFIDENCE = 0.95


def stats(r: pd.Series) -> dict:
    return {"cagr": cagr(r), "sharpe": sharpe(r), "max_drawdown": max_drawdown(r), "days": len(r)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    loader, today = get_loader("yfinance")(), pd.Timestamp.today().date().isoformat()
    raw: dict = {}
    for _ in range(3):                                   # transient vendor failures happen; the universe is fixed in advance
        missing = [s for s in UNIVERSE if s not in raw or raw[s].empty]
        if not missing:
            break
        raw.update({k: v for k, v in loader.fetch(missing, "2005-01-01", today, interval="1D").items() if v is not None and not v.empty})
    missing = [s for s in UNIVERSE if s not in raw or raw[s].empty]
    if missing:
        print(f"refusing to run: no data for {missing}. The pre-registered universe must be complete.")
        return 1
    frames = {s: raw[s][["open", "high", "low", "close"]].dropna() for s in UNIVERSE}
    close = pd.DataFrame({s: f["close"] for s, f in frames.items()}).sort_index()
    print(f"data: {len(frames)} assets, {close.index[0].date()} to {close.index[-1].date()}", flush=True)

    per_asset, trades = {}, []
    for symbol, df in frames.items():
        result = backtest(df["close"], snr_signal(df), COSTS)
        per_asset[symbol] = result.returns
        trades += [{**t, "symbol": symbol} for t in result.trades]
    strategy = pd.DataFrame(per_asset).sort_index().mean(axis=1, skipna=True)     # capital split equally across assets with data
    bench = run_backtest(close, equal_weights(close), cost_bps=COSTS.fee_bps + COSTS.slippage_bps)

    oos_strategy, oos_bench = strategy.loc[OOS_START:], bench.loc[OOS_START:]
    oos_trades = [t for t in trades if t["entry_time"] >= int(OOS_START.timestamp())]
    s, b = stats(oos_strategy), stats(oos_bench)
    lower = bootstrap_sharpe_lower(oos_strategy, level=CONFIDENCE)
    checks = [
        (s["cagr"] > 0, f"CAGR {s['cagr']:+.2%} > 0"),
        (s["sharpe"] > b["sharpe"], f"Sharpe {s['sharpe']:.2f} > benchmark {b['sharpe']:.2f}"),
        (s["max_drawdown"] <= b["max_drawdown"], f"MaxDD {s['max_drawdown']:.1%} <= benchmark {b['max_drawdown']:.1%}"),
        (lower > 0, f"Sharpe lower bound (95%) {lower:+.2f} > 0"),
        (len(oos_trades) >= MIN_TRADES, f"{len(oos_trades)} out-of-sample trades (need >= {MIN_TRADES})"),
    ]
    passed = all(ok for ok, _ in checks)

    dev_end = OOS_START - pd.Timedelta(days=1)
    print(f"\nDEVELOPMENT (not confirmatory): rule {stats(strategy.loc[:dev_end])}")
    print(f"                                 hold {stats(bench.loc[:dev_end])}")
    print(f"\nOUT OF SAMPLE (confirmatory), net of costs\n  Rule              CAGR {s['cagr']:+7.2%}  Sharpe {s['sharpe']:5.2f}  MaxDD {s['max_drawdown']:6.1%}")
    print(f"  Equal-wt hold     CAGR {b['cagr']:+7.2%}  Sharpe {b['sharpe']:5.2f}  MaxDD {b['max_drawdown']:6.1%}")
    print("\nPRE-REGISTERED VERDICT: " + ("PASS" if passed else "FAIL"))
    for ok, text in checks:
        print(f"  [{'x' if ok else ' '}] {text}")
    wins = [t for t in oos_trades if t["return"] > 0]
    print(f"\nOOS trades {len(oos_trades)}, win rate {len(wins) / max(1, len(oos_trades)):.0%}, "
          f"avg trade {sum(t['return'] for t in oos_trades) / max(1, len(oos_trades)):+.2%}")
    print("\nEXPLORATORY per asset, out of sample (not used for the verdict):")
    for symbol, r in per_asset.items():
        x = stats(r.loc[OOS_START:])
        n = sum(1 for t in oos_trades if t["symbol"] == symbol)
        print(f"  {symbol:<4} trades {n:<3} CAGR {x['cagr']:+7.2%}  Sharpe {x['sharpe']:5.2f}  MaxDD {x['max_drawdown']:6.1%}")
    if args.out:
        args.out.write_text(json.dumps({"passed": passed, "strategy": s, "benchmark": b, "lower_bound": lower,
                                        "oos_trades": len(oos_trades)}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
