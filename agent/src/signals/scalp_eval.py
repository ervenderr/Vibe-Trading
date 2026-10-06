"""Glue between market data, analysed news and the replay simulator."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence

import pandas as pd

from src.signals.event_study import Event
from src.signals.jev_analyzer import AnalyzedNews
from src.signals.technicals import TechnicalState
from src.signals.verdict import (
    AggregatedNews,
    TimedNewsSignal,
    Verdict,
    VerdictConfig,
    aggregate_news,
    decide,
    headline_key,
)

EXCHANGE_TZ = "America/New_York"
_CRYPTO_RE = re.compile(r"^[A-Z0-9]{2,10}-(USD|USDT|USDC)$")
_US_RE = re.compile(r"^[A-Z]{1,5}(\.[A-Z])?$")
_NO_NEWS = AggregatedNews(score=0.0, block=False, count=0, reason="")


def exchange_tz_for(symbol: str) -> str | None:
    """Timezone of the naive timestamps the loaders return, or None if unsupported.

    Crypto bars are UTC wall-clock; US stocks are New York wall-clock. Other markets are
    refused rather than guessed: a wrong zone shifts every bar against the news.
    """
    s = symbol.strip().upper()
    if _CRYPTO_RE.match(s):
        return "UTC"
    if _US_RE.match(s):
        return EXCHANGE_TZ
    return None


def localize_bars(bars: pd.DataFrame, tz: str = EXCHANGE_TZ) -> pd.DataFrame:
    """Return ``bars`` with a UTC index (rows in a DST gap/fold are dropped).

    Loaders return naive exchange wall-clock timestamps, while news is stamped in
    UTC. Aligning them is essential: a one-zone mistake leaks news into earlier bars.
    """
    index = pd.DatetimeIndex(bars.index)
    if index.tz is None:
        index = index.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    out = bars.set_axis(index.tz_convert("UTC"), axis=0)
    return out[out.index.notna()]


def make_verdict_fn(
    news: Sequence[TimedNewsSignal],
    config: VerdictConfig | None = None,
    *,
    use_news: bool = True,
) -> Callable[[TechnicalState, datetime], Verdict]:
    """Verdict function for the simulator; ``use_news=False`` is the price-only baseline."""
    cfg = config or VerdictConfig()
    frozen = tuple(news)

    def verdict(state: TechnicalState, now: datetime) -> Verdict:
        aggregated = aggregate_news(frozen, now, cfg) if use_news else _NO_NEWS
        return decide(aggregated, state, cfg, now=now)

    return verdict


@dataclass(frozen=True)
class EventCriteria:
    """What counts as a tradeable event, and how long it stays actionable."""

    min_relevance: float = 0.7
    min_confirmed: float = 0.6
    min_sentiment_confidence: float = 0.7
    min_abs_sentiment: float = 0.5
    trigger_minutes: float = 30.0
    half_life_minutes: float = 15.0


def strong_event_signals(
    analyzed: Sequence[AnalyzedNews], criteria: EventCriteria | None = None
) -> tuple[TimedNewsSignal, ...]:
    """Keep only confident, relevant, confirmed, clearly directional Jev judgments."""
    c = criteria or EventCriteria()
    kept = []
    for a in analyzed:
        j = a.judgment
        if j is None:  # lexicon fallback is not trusted as an event trigger
            continue
        if (
            j.relevance >= c.min_relevance
            and j.confirmed >= c.min_confirmed
            and j.sentiment_confidence >= c.min_sentiment_confidence
            and abs(j.sentiment) >= c.min_abs_sentiment
        ):
            kept.append(TimedNewsSignal(a.signal, a.item.published, headline_key(a.item.headline)))
    return tuple(kept)


def make_event_verdict_fn(
    events: Sequence[TimedNewsSignal], criteria: EventCriteria | None = None
) -> Callable[[TechnicalState, datetime], Verdict]:
    """News is the trigger; trend only confirms direction. No event means no trade."""
    c = criteria or EventCriteria()
    cfg = VerdictConfig(
        news_window_minutes=c.trigger_minutes,
        news_half_life_minutes=c.half_life_minutes,
        tech_only_threshold=float("inf"),  # price alone may never open a trade
    )
    return make_verdict_fn(events, cfg, use_news=True)


def strong_events(
    analyzed: Sequence[AnalyzedNews], criteria: EventCriteria | None = None
) -> tuple[Event, ...]:
    """Directional events for the event study: strong Jev judgments with side and type."""
    c = criteria or EventCriteria()
    events = []
    for a in analyzed:
        j = a.judgment
        if j is None:
            continue
        if (
            j.relevance >= c.min_relevance
            and j.confirmed >= c.min_confirmed
            and j.sentiment_confidence >= c.min_sentiment_confidence
            and abs(j.sentiment) >= c.min_abs_sentiment
        ):
            events.append(
                Event(a.item.symbol, a.item.published, 1 if j.sentiment > 0 else -1,
                      j.event_type, abs(j.sentiment) * j.relevance)
            )
    return tuple(events)
