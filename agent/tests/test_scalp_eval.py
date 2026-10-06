"""Tests for the glue between data, news and the replay simulator."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.signals.news_judgment import NewsSignal
from src.signals.scalp_eval import localize_bars, make_verdict_fn
from src.signals.technicals import TechnicalState
from src.signals.verdict import Action, TimedNewsSignal

NOW = datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc)
STATE = TechnicalState(close=100.0, ema_fast=101.0, ema_slow=100.0, rsi=55.0, atr=1.0, vol_ratio=1.0)


class TestLocalizeBars:
    def test_naive_eastern_becomes_utc(self) -> None:
        idx = pd.DatetimeIndex(["2026-09-30 09:30:00"])
        df = pd.DataFrame({"close": [1.0]}, index=idx)
        out = localize_bars(df)
        assert str(out.index.tz) == "UTC"
        assert out.index[0] == pd.Timestamp("2026-09-30 13:30:00", tz="UTC")  # EDT = UTC-4

    def test_winter_offset_is_handled(self) -> None:
        df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2026-12-15 09:30:00"]))
        assert localize_bars(df).index[0] == pd.Timestamp("2026-12-15 14:30:00", tz="UTC")  # EST = UTC-5

    def test_already_aware_index_is_converted_not_relabelled(self) -> None:
        idx = pd.DatetimeIndex(["2026-09-30 13:30:00"], tz="UTC")
        out = localize_bars(pd.DataFrame({"close": [1.0]}, index=idx))
        assert out.index[0] == pd.Timestamp("2026-09-30 13:30:00", tz="UTC")

    def test_input_not_mutated(self) -> None:
        df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2026-09-30 09:30:00"]))
        localize_bars(df)
        assert df.index.tz is None


class TestMakeVerdictFn:
    def _news(self, score: float, minutes_ago: float) -> TimedNewsSignal:
        return TimedNewsSignal(NewsSignal(score, False, "ok"), NOW - timedelta(minutes=minutes_ago))

    def test_with_news_uses_it(self) -> None:
        fn = make_verdict_fn([self._news(0.8, 1)], use_news=True)
        assert fn(STATE, NOW).action is Action.BUY

    def test_without_news_flag_ignores_news(self) -> None:
        fn = make_verdict_fn([self._news(0.8, 1)], use_news=False)
        # tech score tanh(1)=0.76 >= 0.6, so tech-only still buys, but news is not reflected
        v = fn(STATE, NOW)
        assert v.news_score == 0.0

    def test_news_published_after_now_is_never_used(self) -> None:
        fn = make_verdict_fn([self._news(-0.9, -30)], use_news=True)
        assert fn(STATE, NOW).news_score == 0.0


from src.signals.jev_analyzer import AnalyzedNews
from src.signals.news_judgment import NewsJudgment, news_signal
from src.signals.news_source import NewsItem
from src.signals.scalp_eval import EventCriteria, make_event_verdict_fn, strong_event_signals


def _analyzed(minutes_ago: float = 1.0, **judgment: float | str) -> AnalyzedNews:
    base = dict(relevance=0.95, sentiment=0.9, sentiment_confidence=0.9, event_type="earnings",
                event_confidence=0.9, binary_event=0.9, confirmed=0.95)
    j = NewsJudgment(**{**base, **judgment})
    item = NewsItem("NVDA", "h", "s", NOW - timedelta(minutes=minutes_ago), "src", "u", "1")
    return AnalyzedNews(item, j, news_signal(j), "jev")


class TestStrongEvents:
    def test_keeps_confident_relevant_confirmed_events(self) -> None:
        assert len(strong_event_signals([_analyzed()])) == 1

    @pytest.mark.parametrize(
        "field,value",
        [("relevance", 0.4), ("confirmed", 0.2), ("sentiment_confidence", 0.4), ("sentiment", 0.1)],
    )
    def test_drops_weak_items(self, field: str, value: float) -> None:
        assert strong_event_signals([_analyzed(**{field: value})]) == ()

    def test_drops_fallback_items_without_judgment(self) -> None:
        item = NewsItem("NVDA", "h", "s", NOW, "src", "u", "1")
        fb = AnalyzedNews(item, None, NewsSignal(0.9, False, "fallback"), "fallback")
        assert strong_event_signals([fb]) == ()

    def test_criteria_are_configurable(self) -> None:
        loose = EventCriteria(min_relevance=0.3)
        assert len(strong_event_signals([_analyzed(relevance=0.4)], loose)) == 1


class TestEventVerdictFn:
    def test_no_event_means_wait_even_with_strong_trend(self) -> None:
        fn = make_event_verdict_fn(())
        assert fn(STATE, NOW).action is Action.WAIT

    def test_recent_event_plus_confirming_trend_trades(self) -> None:
        fn = make_event_verdict_fn(strong_event_signals([_analyzed(minutes_ago=2)]))
        assert fn(STATE, NOW).action is Action.BUY

    def test_stale_event_does_not_trigger(self) -> None:
        fn = make_event_verdict_fn(strong_event_signals([_analyzed(minutes_ago=600)]))
        assert fn(STATE, NOW).action is Action.WAIT

    def test_event_against_the_trend_does_not_trade(self) -> None:
        fn = make_event_verdict_fn(strong_event_signals([_analyzed(minutes_ago=2, sentiment=-0.9)]))
        assert fn(STATE, NOW).action is Action.WAIT


from src.signals.scalp_eval import strong_events


class TestStrongEventsForStudy:
    def test_direction_type_and_strength_come_from_the_judgment(self) -> None:
        (e,) = strong_events([_analyzed(sentiment=-0.8, event_type="guidance")])
        assert e.side == -1 and e.event_type == "guidance" and e.symbol == "NVDA"
        assert 0 < e.strength <= 1

    def test_positive_sentiment_is_a_buy_side(self) -> None:
        assert strong_events([_analyzed(sentiment=0.9)])[0].side == 1

    def test_weak_items_are_excluded_like_before(self) -> None:
        assert strong_events([_analyzed(relevance=0.3)]) == ()

    def test_uses_the_publication_time(self) -> None:
        (e,) = strong_events([_analyzed(minutes_ago=7)])
        assert e.published == NOW - timedelta(minutes=7)
