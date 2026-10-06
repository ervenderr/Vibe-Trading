"""Live signal service: news + price + risk flags -> a single verdict report.

Data sources are injected so the orchestration is testable and the same code
serves the CLI, the agent tool and the watch loop. Advisory only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Callable, Sequence

import pandas as pd

from src.signals.jev_analyzer import AnalyzedNews, JevNewsAnalyzer
from src.signals.news_source import NewsItem, is_valid_symbol
from src.signals.technicals import state_at, technical_frame
from src.signals.verdict import (
    Action,
    TimedNewsSignal,
    Verdict,
    VerdictConfig,
    aggregate_news,
    decide,
    headline_key,
)

logger = logging.getLogger(__name__)

BAR_MINUTES = 5
MAX_ANALYZED_ITEMS = 40
NewsSource = Callable[[str], Sequence[NewsItem]]
BarsSource = Callable[[str], pd.DataFrame]


@dataclass(frozen=True)
class SignalReport:
    symbol: str
    generated_at: datetime
    verdict: Verdict
    headlines: tuple[AnalyzedNews, ...]
    bar_time: datetime | None
    stale: bool
    news_count: int
    news_error: str | None = None
    fallback_count: int = 0


def _recent(items: Sequence[NewsItem], now: datetime, window_minutes: float) -> list[NewsItem]:
    cutoff = now - timedelta(minutes=window_minutes)
    recent = [i for i in items if cutoff <= i.published <= now]
    return sorted(recent, key=lambda i: i.published, reverse=True)[:MAX_ANALYZED_ITEMS]


def _fetch_news(symbol: str, source: NewsSource) -> tuple[Sequence[NewsItem], str | None]:
    try:
        return source(symbol), None
    except Exception as exc:  # degrade to price-only; the failure is reported, not hidden
        logger.warning("news source failed for %s: %s", symbol, type(exc).__name__)
        return (), f"news unavailable ({type(exc).__name__}); price-only verdict"


def _top_headlines(analyzed: Sequence[AnalyzedNews], limit: int) -> tuple[AnalyzedNews, ...]:
    ranked = sorted(analyzed, key=lambda a: (abs(a.signal.score), a.item.published), reverse=True)
    return tuple(ranked[:limit])


def generate_signal(
    symbol: str,
    *,
    news_source: NewsSource,
    analyzer: JevNewsAnalyzer,
    bars_source: BarsSource,
    now: datetime,
    config: VerdictConfig | None = None,
    risk_flags: Sequence[str] = (),
    max_data_age_minutes: float = 15.0,
    max_headlines: int = 5,
) -> SignalReport:
    """Build a :class:`SignalReport` for ``symbol`` as of ``now`` (tz-aware UTC)."""
    sym = symbol.strip().upper()
    if not is_valid_symbol(sym):
        raise ValueError(f"invalid symbol: {symbol!r}")
    cfg = config or VerdictConfig()

    items, news_error = _fetch_news(sym, news_source)
    analyzed = analyzer.analyze_many(_recent(items, now, cfg.news_window_minutes)) if items else ()
    # Keyword-fallback scores are never trusted to steer a verdict; report them instead.
    trusted = [a for a in analyzed if a.origin != "fallback"]
    fallback_count = len(analyzed) - len(trusted)
    timed = [TimedNewsSignal(a.signal, a.item.published, headline_key(a.item.headline)) for a in trusted]
    news = aggregate_news(timed, now, cfg)

    bars = bars_source(sym)
    if len(bars) and bars.index[-1] + timedelta(minutes=BAR_MINUTES) > now:
        bars = bars.iloc[:-1]  # the newest bar is still forming; the replay only uses closed bars
    frame = technical_frame(bars) if len(bars) else None
    state = state_at(frame, len(frame) - 1) if frame is not None else None
    bar_time = bars.index[-1].to_pydatetime() if len(bars) else None
    age_limit = timedelta(minutes=max_data_age_minutes)
    stale = bar_time is not None and now - (bar_time + timedelta(minutes=BAR_MINUTES)) > age_limit

    verdict = decide(news, None if stale else state, cfg, now=now, risk_flags=risk_flags)
    if stale and verdict.action is not Action.AVOID:
        verdict = replace(verdict, reasons=("price data stale (market closed or feed delayed)",))
    return SignalReport(
        symbol=sym,
        generated_at=now,
        verdict=verdict,
        headlines=_top_headlines(analyzed, max_headlines),
        bar_time=bar_time,
        stale=bool(stale),
        news_count=len(analyzed),
        news_error=news_error,
        fallback_count=fallback_count,
    )
