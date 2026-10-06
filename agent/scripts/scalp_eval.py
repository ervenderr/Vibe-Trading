"""Paper-trade evaluation: price-only vs price+Jev-news scalping on real intraday bars.

Simulation only; no orders are placed.

    python scripts/scalp_eval.py --symbols NVDA TSLA AMD --start 2026-09-08 --end 2026-09-30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from typesafe_sdk import TypeSafeClient  # noqa: E402

from src.market_data import get_loader  # noqa: E402
from src.signals.jev_analyzer import JevNewsAnalyzer, JudgmentCache  # noqa: E402
from src.signals.news_source import FinnhubNewsClient, NewsFetchError  # noqa: E402
from src.signals.replay import ReplayConfig, Stats, simulate, summarize  # noqa: E402
from src.signals.scalp_eval import localize_bars, make_event_verdict_fn, make_verdict_fn, strong_event_signals  # noqa: E402
from src.signals.verdict import TimedNewsSignal  # noqa: E402

DEFAULT_CACHE = Path.home() / ".vibe-trading" / "cache" / "jev_news"


def _fmt(label: str, s: Stats) -> str:
    pf = "inf" if s.profit_factor == float("inf") else f"{s.profit_factor:.2f}"
    return (
        f"  {label:<12} trades={s.trades:<4} win={s.win_rate:5.1%} "
        f"expectancy={s.expectancy:+.4%} total={s.total_net_return:+.2%} "
        f"PF={pf:<5} maxDD={s.max_drawdown:.2%} hold={s.avg_bars_held:.1f} bars"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", nargs="+", default=["NVDA", "TSLA", "AMD"])
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--slippage-bps", type=float, default=1.0)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--out", type=Path, default=None, help="write JSON report here")
    args = ap.parse_args()

    load_dotenv(ROOT.parent / ".env")
    cfg = ReplayConfig(cost_bps=args.cost_bps, slippage_bps=args.slippage_bps)
    loader = get_loader("yfinance")()
    long_cfg = replace(cfg, max_hold_bars=36)
    report: dict[str, dict] = {}
    pooled: dict[str, list] = defaultdict(list)
    split_by_variant: dict[str, dict[str, list]] = defaultdict(lambda: {"train": [], "test": []})

    with FinnhubNewsClient(os.environ["FINNHUB_API_KEY"], min_interval_s=1.05) as finnhub, TypeSafeClient() as jev:
        analyzer = JevNewsAnalyzer(jev, model=os.environ.get("TYPESAFE_MODEL", "jev-1.13.0"), cache=JudgmentCache(args.cache_dir))
        for symbol in args.symbols:
            raw = loader.fetch([symbol], args.start.isoformat(), args.end.isoformat(), interval="5m").get(symbol)
            if raw is None or raw.empty:
                print(f"{symbol}: no intraday data, skipped")
                continue
            bars = localize_bars(raw)
            sessions = sorted({ts.date() for ts in bars.index})
            cutoff = sessions[int(len(sessions) * 0.6)]
            try:
                items = finnhub.fetch_range(symbol, args.start, args.end)
            except NewsFetchError as exc:
                print(f"{symbol}: news fetch failed ({exc}); skipped")
                continue
            analyzed = analyzer.analyze_many(items)
            timed = [TimedNewsSignal(a.signal, a.item.published) for a in analyzed]
            events = strong_event_signals(analyzed)
            runs = {
                "price-only": simulate(bars, make_verdict_fn(timed, use_news=False), cfg),
                "price+news": simulate(bars, make_verdict_fn(timed, use_news=True), cfg),
                "event@12": simulate(bars, make_event_verdict_fn(events), cfg),
                "event@36": simulate(bars, make_event_verdict_fn(events), long_cfg),
            }
            print(f"{symbol}: {len(bars)} bars, {len(items)} headlines, {len(events)} strong events")
            for name, trades in runs.items():
                pooled[name].extend(trades)
                for t in trades:
                    split_by_variant[name]["train" if t.entry_time.date() < cutoff else "test"].append(t)
                print(_fmt(name, summarize(trades)))
            report[symbol] = {"headlines": len(items), "strong_events": len(events),
                              **{n: asdict(summarize(t)) for n, t in runs.items()}}

    print("\nALL SYMBOLS (pooled)")
    for name, trades in pooled.items():
        print(_fmt(name, summarize(trades)))
    print("\nTRAIN (first 60% of sessions) vs TEST (last 40%), pooled")
    for name, parts in split_by_variant.items():
        print(_fmt(name + " train", summarize(parts["train"])))
        print(_fmt(name + " test", summarize(parts["test"])))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
