"""Pre-registered crypto timeframe study. Rules: scripts/PREREGISTRATION-crypto-timeframe-study.md

    python scripts/crypto_study.py --out result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.signals import crypto_study as cs  # noqa: E402

ASSETS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
TEST_START_1D = "2023-01-01"
MIN_TRIPS = 8
CONFIDENCE = 0.975  # two timeframes tested


def fetch(interval: str) -> dict[str, pd.Series]:
    import yfinance as yf

    out = {}
    for name, ticker in ASSETS.items():
        kw = dict(period="730d", interval="1h") if interval == "4h" else dict(start="2014-01-01", interval="1d")
        df = yf.Ticker(ticker).history(auto_adjust=False, **kw)
        close = df["Close"].dropna()
        close.index = close.index.tz_convert("UTC").tz_localize(None) if close.index.tz else close.index
        if interval == "4h":
            close = close.resample("4h").last().dropna()
        elif close.index[-1].date() >= pd.Timestamp.now("UTC").date():
            close = close.iloc[:-1]  # drop the still-forming daily bar
        out[name] = close
    start = max(c.index[0] for c in out.values())
    end = min(c.index[-1] for c in out.values())
    return {k: c.loc[start:end] for k, c in out.items()}


def stats(r: pd.Series, per_year: float) -> dict:
    return {"cagr": cs.cagr(r, per_year), "sharpe": cs.sharpe(r, per_year), "max_dd": cs.max_drawdown(r), "bars": len(r)}


def run(interval: str) -> dict:
    per_year, median_window = (365.0, 365) if interval == "1D" else (2190.0, 2190)
    closes = fetch("4h" if interval == "4h" else "1d")
    index = next(iter(closes.values())).index
    split = pd.Timestamp(TEST_START_1D) if interval == "1D" else index[len(index) // 2]
    tune = {k: c[c.index < split] for k, c in closes.items()}
    test = {k: c[c.index >= split] for k, c in closes.items()}
    print(f"\n=== {interval}: {index[0]} to {index[-1]}; test from {split}")
    rules, rows = cs.variants(median_window), {}
    for name, rule in rules.items():
        # signals use the full history (indicators warm up), results are cut to each period
        full = {k: strategy_full(c, rule) for k, c in closes.items()}
        rows[name] = {"tune": stats(slice_period(full, tune), per_year), "test": stats(slice_period(full, test), per_year)}
        print(f"  {name:14s} tune Sharpe {rows[name]['tune']['sharpe']:+.2f} | test Sharpe {rows[name]['test']['sharpe']:+.2f} CAGR {rows[name]['test']['cagr']:+.1%}")
    best = max(rows, key=lambda n: rows[n]["tune"]["sharpe"])
    full = {k: strategy_full(c, rules[best]) for k, c in closes.items()}
    chosen = slice_period(full, test)
    bench = slice_period({k: cs.strategy_returns(c, pd.Series(True, index=c.index)) for k, c in closes.items()}, test)
    trips = int(sum(((rules[best](c)[c.index >= split]).astype(int).diff() == -1).sum() for c in closes.values()))
    lower = cs.sharpe_lower(chosen, per_year, level=CONFIDENCE)
    c, b = stats(chosen, per_year), stats(bench, per_year)
    checks = {"cagr>0": c["cagr"] > 0, "sharpe>bench": c["sharpe"] > b["sharpe"], "maxdd<=bench": c["max_dd"] <= b["max_dd"],
              "boot_lower>0": lower > 0, f"trips>={MIN_TRIPS}": trips >= MIN_TRIPS}
    print(f"  SELECTED on tuning: {best}\n  test {c}\n  bench {b}\n  bootstrap lower {lower:+.2f}, round trips {trips}")
    print(f"  checks {checks} -> {'PASS' if all(checks.values()) else 'FAIL'}")
    return {"selected": best, "test": c, "benchmark": b, "boot_lower": lower, "trips": trips, "checks": checks,
            "pass": all(checks.values()), "variants": rows}


def strategy_full(close: pd.Series, rule) -> pd.Series:
    return cs.strategy_returns(close, rule(close))


def slice_period(full: dict[str, pd.Series], period: dict[str, pd.Series]) -> pd.Series:
    return pd.concat([full[k].loc[period[k].index] for k in full], axis=1).mean(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = {tf: run(tf) for tf in ("1D", "4h")}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
