"""Pre-registered confirmatory test: 4h market-adjusted drift after strong Jev-judged news.

Rules are fixed in PREREGISTRATION-4h-news-drift.md. Research only; places no orders.

    python scripts/confirm_4h.py --cache-dir ~/.vibe-trading/research/confirm_4h --out result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402
from typesafe_sdk import TypeSafeClient  # noqa: E402

import event_study as es  # noqa: E402  (the first study's helpers: news cache, bars, aliases)
from src.market_data import get_loader  # noqa: E402
from src.signals.event_study import DEFAULT_HORIZONS, Event, HorizonStats, run_study  # noqa: E402
from src.signals.jev_analyzer import JevNewsAnalyzer, JudgmentCache  # noqa: E402
from src.signals.news_source import FinnhubNewsClient, NewsFetchError  # noqa: E402
from src.signals.scalp_eval import strong_events  # noqa: E402
from src.signals.verdict import headline_key  # noqa: E402

ALL10 = ["NVDA", "TSLA", "AMD", "AAPL", "META", "MSFT", "AMZN", "GOOGL", "AVGO", "NFLX"]
WINDOWS = (  # (start, end, symbols): none of this was in the first study
    (date(2025, 10, 15), date(2026, 1, 31), ALL10),
    (date(2026, 10, 1), date(2026, 10, 13), ALL10),
    (date(2026, 2, 1), date(2026, 9, 30), ["AVGO", "NFLX"]),
)
HORIZON = {"4h": DEFAULT_HORIZONS["4h"]}
MIN_EVENTS, MAX_P, COST_BPS = 100, 0.05, 6.0
MIN_SPLIT_N = 30


def verdict(stats: HorizonStats) -> tuple[bool, list[str]]:
    """The pre-registered success rule: every criterion must hold."""
    checks = [
        (stats.n >= MIN_EVENTS, f"n={stats.n} (need >= {MIN_EVENTS})"),
        (stats.p_value <= MAX_P, f"p={stats.p_value:.4f} (need <= {MAX_P})"),
        (stats.mean_abnormal_bps > COST_BPS, f"mean {stats.mean_abnormal_bps:+.1f} bps (need > {COST_BPS})"),
        (stats.hit_rate > 0.5, f"hit rate {stats.hit_rate:.1%} (need > 50%)"),
    ]
    return all(ok for ok, _ in checks), [("[x] " if ok else "[ ] ") + text for ok, text in checks]


def splits(events: list[Event]) -> dict[str, list[Event]]:
    """Exploratory groupings, each with enough events to say anything."""
    groups: dict[str, list[Event]] = {}
    for e in events:
        for key in (f"side={'up' if e.side > 0 else 'down'}", f"type={e.event_type}", f"symbol={e.symbol}"):
            groups.setdefault(key, []).append(e)
    return {k: v for k, v in sorted(groups.items()) if len(v) >= MIN_SPLIT_N}


def collect_events(cache: Path) -> list[Event]:
    events: list[Event] = []
    with FinnhubNewsClient(os.environ["FINNHUB_API_KEY"], min_interval_s=1.05) as finnhub, TypeSafeClient() as jev:
        analyzer = JevNewsAnalyzer(jev, model=os.environ.get("TYPESAFE_MODEL", "jev-1.13.0"), cache=JudgmentCache(cache / "jev"))
        for start, end, symbols in WINDOWS:
            for symbol in symbols:
                try:
                    items = es._news(finnhub, symbol, start, end, cache)
                except NewsFetchError as exc:
                    print(f"{symbol} {start}: news fetch failed ({exc}); window skipped", flush=True)
                    continue
                seen, relevant = set(), []
                for item in items:
                    key = headline_key(item.headline)
                    if key not in seen and es.mentions_company(item.headline, es.ALIASES.get(symbol, (symbol.lower(),))):
                        seen.add(key)
                        relevant.append(item)
                found = strong_events(analyzer.analyze_many(relevant))
                events.extend(found)
                print(f"{symbol} {start}..{end}: {len(items)} headlines -> {len(relevant)} relevant -> {len(found)} strong", flush=True)
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    load_dotenv(ROOT.parent / ".env")
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    events = collect_events(args.cache_dir)
    print(f"\nTOTAL strong events: {len(events)}", flush=True)
    loader = get_loader("yfinance")()
    start, end = min(w[0] for w in WINDOWS), max(w[1] for w in WINDOWS)
    market = es._bars(loader, "SPY", "1h", start, end)
    stock = {s: es._bars(loader, s, "1h", start, end) for s in sorted({e.symbol for e in events})}

    main_stats = run_study(events, stock, market, HORIZON, cost_bps=COST_BPS, seed=0)["4h"]
    ok, lines = verdict(main_stats)
    print("\n== CONFIRMATORY (pre-registered) ==\n" + es._row(main_stats, 1))
    print("\n".join(lines))
    print("RESULT:", "SUCCESS" if ok else "NO EDGE FOUND", flush=True)

    groups = splits(events)
    exploratory = {k: run_study(v, stock, market, HORIZON, cost_bps=COST_BPS, seed=0)["4h"] for k, v in groups.items()}
    print(f"\n== EXPLORATORY ({len(exploratory)} splits, Bonferroni-adjusted; cannot establish success) ==")
    for key, s in exploratory.items():
        print(f"{key:<26}" + es._row(s, len(exploratory)).strip())
    if args.out:
        args.out.write_text(json.dumps({"success": ok, "confirmatory": asdict(main_stats),
                                        "exploratory": {k: asdict(v) for k, v in exploratory.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
