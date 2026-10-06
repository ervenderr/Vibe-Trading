"""Event study: at which horizon does Jev-judged news predict market-adjusted returns?

Research tool only; places no orders.

    python scripts/event_study.py --symbols NVDA TSLA --start 2026-02-01 --end 2026-09-30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from typesafe_sdk import TypeSafeClient  # noqa: E402

from src.market_data import get_loader  # noqa: E402
from src.signals.event_study import (  # noqa: E402
    DEFAULT_HORIZONS, Event, HorizonStats, localize_daily, mentions_company, run_study, split_by_time,
)
from src.signals.jev_analyzer import JevNewsAnalyzer, JudgmentCache  # noqa: E402
from src.signals.news_source import FinnhubNewsClient, NewsFetchError, NewsItem  # noqa: E402
from src.signals.scalp_eval import localize_bars, strong_events  # noqa: E402
from src.signals.verdict import headline_key  # noqa: E402

ALIASES: dict[str, tuple[str, ...]] = {
    "NVDA": ("nvda", "nvidia"), "TSLA": ("tsla", "tesla"), "AMD": ("amd", "advanced micro devices"),
    "AAPL": ("aapl", "apple"), "META": ("meta", "facebook", "instagram"), "MSFT": ("msft", "microsoft"),
    "AMZN": ("amzn", "amazon"), "GOOGL": ("googl", "goog", "google", "alphabet"),
    "AVGO": ("avgo", "broadcom"), "NFLX": ("nflx", "netflix"),
}
SERIES_INTERVAL = {"5m": "5m", "1h": "1H", "1d": "1D"}
INTRADAY_5M_DAYS = 50  # Yahoo keeps ~60 days of 5-minute bars
COST_BPS = 6.0


def _news(finnhub: FinnhubNewsClient, symbol: str, start: date, end: date, cache: Path) -> list[NewsItem]:
    path = cache / f"news_{symbol}_{start}_{end}.json"
    if path.exists():
        rows = json.loads(path.read_text())
        return [NewsItem(**{**r, "published": datetime.fromisoformat(r["published"])}) for r in rows]
    items = finnhub.fetch_range(symbol, start, end)
    path.write_text(json.dumps([{**asdict(i), "published": i.published.isoformat()} for i in items]))
    return list(items)


def _bars(loader, symbol: str, series: str, start: date, end: date) -> pd.DataFrame:
    if series == "5m":
        start = max(start, date.fromordinal(end.toordinal() - INTRADAY_5M_DAYS))
    raw = loader.fetch([symbol], start.isoformat(), end.isoformat(), interval=SERIES_INTERVAL[series]).get(symbol)
    if raw is None or raw.empty:
        return pd.DataFrame()
    return localize_daily(raw) if series == "1d" else localize_bars(raw)


def _row(s: HorizonStats, n_tests: int) -> str:
    adj = min(1.0, s.p_value * n_tests)
    return (f"  {s.label:<4} n={s.n:<4} hit={s.hit_rate:5.1%} mean={s.mean_abnormal_bps:+7.1f}bps "
            f"median={s.median_abnormal_bps:+7.1f} t={s.t_stat:+5.2f} p={s.p_value:.4f} "
            f"p_adj={adj:.4f} net_raw={s.mean_raw_net_bps:+7.1f}bps")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=list(ALIASES)[:8])
    ap.add_argument("--start", type=date.fromisoformat, default=date(2026, 2, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 30))
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    load_dotenv(ROOT.parent / ".env")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    loader = get_loader("yfinance")()

    events: list[Event] = []
    with FinnhubNewsClient(os.environ["FINNHUB_API_KEY"], min_interval_s=1.05) as finnhub, TypeSafeClient() as jev:
        analyzer = JevNewsAnalyzer(jev, model=os.environ.get("TYPESAFE_MODEL", "jev-1.13.0"),
                                   cache=JudgmentCache(args.cache_dir / "jev"))
        for symbol in args.symbols:
            try:
                items = _news(finnhub, symbol, args.start, args.end, args.cache_dir)
            except NewsFetchError as exc:
                print(f"{symbol}: news fetch failed ({exc}); skipped", flush=True)
                continue
            seen, relevant = set(), []
            for item in items:
                key = headline_key(item.headline)
                if key not in seen and mentions_company(item.headline, ALIASES.get(symbol, (symbol.lower(),))):
                    seen.add(key)
                    relevant.append(item)
            analyzed = analyzer.analyze_many(relevant)
            found = strong_events(analyzed)
            events.extend(found)
            print(f"{symbol}: {len(items)} headlines -> {len(relevant)} mention the company -> {len(found)} strong events", flush=True)

    print(f"\nTOTAL strong events: {len(events)} "
          f"(up={sum(e.side > 0 for e in events)}, down={sum(e.side < 0 for e in events)})", flush=True)
    symbols = sorted({e.symbol for e in events})
    train, test = split_by_time(events, 0.6)
    cutoff = test[0].published if test else None
    print(f"train: {len(train)} events | test: {len(test)} events (from {cutoff})\n", flush=True)

    all_stats: dict[str, dict[str, HorizonStats]] = {"all": {}, "train": {}, "test": {}}
    for series in ("5m", "1h", "1d"):
        horizons = {k: h for k, h in DEFAULT_HORIZONS.items() if h.series == series}
        market = _bars(loader, "SPY", series, args.start, args.end)
        stock = {s: _bars(loader, s, series, args.start, args.end) for s in symbols}
        for name, subset in (("all", events), ("train", train), ("test", test)):
            all_stats[name].update(run_study(subset, stock, market, horizons, cost_bps=COST_BPS, seed=0))

    n_tests = len(DEFAULT_HORIZONS)
    for name in ("all", "train", "test"):
        print(f"== {name.upper()} (market-adjusted; Bonferroni over {n_tests} horizons)")
        for label in DEFAULT_HORIZONS:
            print(_row(all_stats[name][label], n_tests))
        print(flush=True)

    usable = {k: v for k, v in all_stats["train"].items() if v.n >= 30}
    if usable:
        best = max(usable.values(), key=lambda s: s.t_stat)
        print(f"Best horizon chosen on TRAIN only (highest t-stat, n>=30): {best.label}")
        print("  train:", _row(best, n_tests).strip())
        print("  TEST :", _row(all_stats["test"][best.label], 1).strip(), "  (out of sample, single pre-chosen test)")
    if args.out:
        args.out.write_text(json.dumps({k: {l: asdict(s) for l, s in v.items()} for k, v in all_stats.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
