"""Pre-registered test of two slow trend rules. Rules: scripts/PREREGISTRATION-trend-momentum.md.

    python scripts/trend_test.py --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.market_data import get_loader  # noqa: E402
from src.signals.trend_study import (  # noqa: E402
    bootstrap_sharpe_lower, cagr, equal_weights, max_drawdown, momentum_weights, run_backtest, sharpe, trend_weights,
)

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
COST_BPS = 5.0
OOS_START = "2017-01-01"
CONFIDENCE = 0.975  # two candidates tested


def stats(r: pd.Series) -> dict:
    return {"cagr": cagr(r), "sharpe": sharpe(r), "max_drawdown": max_drawdown(r), "days": len(r)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    raw = get_loader("yfinance")().fetch(UNIVERSE, "2005-01-01", pd.Timestamp.today().date().isoformat(), interval="1D")
    close = pd.DataFrame({s: raw[s]["close"] for s in UNIVERSE if s in raw and not raw[s].empty}).sort_index()
    print(f"data: {close.shape[1]} assets, {close.index[0].date()} to {close.index[-1].date()}", flush=True)

    runs = {
        "A trend filter (200d)": run_backtest(close, trend_weights(close, 200), cost_bps=COST_BPS),
        "B momentum (12-1, top 3)": run_backtest(close, momentum_weights(close, 252, 21, 3), cost_bps=COST_BPS),
        "Benchmark: equal-weight": run_backtest(close, equal_weights(close), cost_bps=COST_BPS),
        "Benchmark: SPY": close["SPY"].pct_change().fillna(0.0),
    }
    first_signal = close.index[260]  # every rule has history from here
    result = {}
    for label, start, end in (("DEVELOPMENT (not confirmatory)", first_signal, "2016-12-31"), ("OUT OF SAMPLE (confirmatory)", OOS_START, None)):
        print(f"\n== {label}")
        result[label] = {}
        for name, r in runs.items():
            s = stats(r.loc[start:end])
            result[label][name] = s
            print(f"  {name:<26} CAGR {s['cagr']:+7.2%}  Sharpe {s['sharpe']:5.2f}  MaxDD {s['max_drawdown']:6.1%}")

    oos = {k: v.loc[OOS_START:] for k, v in runs.items()}
    bench = stats(oos["Benchmark: equal-weight"])
    print("\n== PRE-REGISTERED VERDICT (out of sample, net of 5 bps costs)")
    verdict = {}
    for name in ("A trend filter (200d)", "B momentum (12-1, top 3)"):
        s = stats(oos[name])
        lower = bootstrap_sharpe_lower(oos[name], level=CONFIDENCE)
        checks = [
            (s["cagr"] > 0, f"CAGR {s['cagr']:+.2%} > 0"),
            (s["sharpe"] > bench["sharpe"], f"Sharpe {s['sharpe']:.2f} > benchmark {bench['sharpe']:.2f}"),
            (s["max_drawdown"] <= bench["max_drawdown"], f"MaxDD {s['max_drawdown']:.1%} <= benchmark {bench['max_drawdown']:.1%}"),
            (lower > 0, f"Sharpe lower bound (97.5%) {lower:+.2f} > 0"),
        ]
        passed = all(ok for ok, _ in checks)
        verdict[name] = {"passed": passed, "lower_bound": lower}
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
        for ok, text in checks:
            print(f"    [{'x' if ok else ' '}] {text}")
    if args.out:
        args.out.write_text(json.dumps({"periods": result, "verdict": verdict}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
