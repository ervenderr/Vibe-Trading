"""`vibe-trading signal` command: one-shot, --watch loop, --history track record."""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from argparse import Namespace
from typing import Callable

import pandas as pd

from src.signals.ledger import DEFAULT_LEDGER, SignalLedger, grade_entry, summarize_grades
from src.signals.news_source import is_valid_symbol
from src.signals.report_format import render_text, report_to_dict

logger = logging.getLogger(__name__)
MIN_WATCH_SECONDS = 30.0  # keeps Finnhub's free tier (~60 calls/min) comfortable


def run_signal_command(
    args: Namespace,
    *,
    runner: Callable | None = None,
    ledger: SignalLedger | None = None,
    out: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    bars_fetcher: Callable[[str], pd.DataFrame] | None = None,
) -> int:
    store = ledger or SignalLedger(DEFAULT_LEDGER)
    if args.history:
        return _history(store, out, bars_fetcher)
    if args.watch is not None and not (math.isfinite(args.watch) and args.watch > 0):
        out("--watch must be a positive, finite number of seconds")
        return 2
    if runner is None:
        from src.signals.live_sources import run_signal as runner  # lazy: heavy imports
    symbols = [s.strip().upper() for s in args.symbols]
    if not args.watch:
        return _once(symbols, runner, store, out, args.json, last={})
    interval = max(float(args.watch), MIN_WATCH_SECONDS)
    last: dict[str, str] = {}
    try:
        while True:
            _once(symbols, runner, store, out, args.json, last=last)
            sleep(interval)
    except KeyboardInterrupt:
        out("stopped")
    return 0


def _once(symbols, runner, store, out, as_json: bool, *, last: dict[str, str]) -> int:
    failures = 0
    for symbol in symbols:
        if not is_valid_symbol(symbol):
            out(f"{symbol}: invalid symbol")
            failures += 1
            continue
        try:
            report = runner(symbol)
        except Exception as exc:
            logger.exception("signal failed for %s", symbol)
            out(f"{symbol}: failed ({type(exc).__name__})")
            failures += 1
            continue
        try:
            store.record(report)
        except OSError:
            logger.warning("could not write signal ledger", exc_info=True)
        action = report.verdict.action.value
        if last and last.get(symbol) == action:
            continue  # watch mode: only print on change
        last[symbol] = action
        out(json.dumps(report_to_dict(report), ensure_ascii=False) if as_json else render_text(report))
    return 1 if failures else 0


def _history(store: SignalLedger, out, bars_fetcher) -> int:
    entries = store.load()
    if not entries:
        out("no signals recorded yet; run `vibe-trading signal SYMBOL` first")
        return 0
    now = datetime.now(timezone.utc)
    if bars_fetcher is None:
        from src.signals.live_sources import fetch_recent_bars

        oldest = {e.symbol: min(x.generated_at for x in entries if x.symbol == e.symbol) for e in entries}

        def bars_fetcher(symbol: str) -> pd.DataFrame:  # reach back to the oldest signal (5m data keeps ~60d)
            return fetch_recent_bars(symbol, days=min(55, max(2, (now - oldest[symbol]).days + 2)))

    cache: dict[str, pd.DataFrame] = {}
    grades = []
    for e in entries:
        if e.symbol not in cache:
            try:
                cache[e.symbol] = bars_fetcher(e.symbol)
            except Exception:
                logger.warning("no bars for %s", e.symbol, exc_info=True)
                cache[e.symbol] = pd.DataFrame()
        g = grade_entry(e, cache[e.symbol], now=now)
        grades.append(g)
        out(f"{e.generated_at:%m-%d %H:%M} {e.symbol:<8} {e.action:<4} conf {e.confidence:.0%}  -> {g.outcome} ({g.gross_return:+.2%})")
    s = summarize_grades(grades)
    out(f"resolved {s.resolved}, pending {s.pending}, win rate {s.win_rate:.0%}, avg return {s.avg_return:+.3%} (before costs)")
    return 0
