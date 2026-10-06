"""Tests for turning a Jev response into a NewsJudgment and a NewsSignal."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.signals.news_judgment import (
    EVENT_TYPES,
    SENTIMENT_LEVELS,
    NewsJudgment,
    SignalThresholds,
    build_questions,
    judgment_from_response,
    news_signal,
)


def _response(
    *,
    about: float = 0.99,
    sentiment: float = 4.0,
    sentiment_conf: float = 0.95,
    event: str = "earnings",
    event_conf: float = 0.9,
    binary: float = 0.9,
    confirmed: float = 0.95,
) -> SimpleNamespace:
    return SimpleNamespace(
        nouls={
            "about": SimpleNamespace(noul=about),
            "binary_event": SimpleNamespace(noul=binary),
            "confirmed": SimpleNamespace(noul=confirmed),
        },
        scores={"sentiment": SimpleNamespace(score=sentiment, confidence=sentiment_conf)},
        choices={"event_type": SimpleNamespace(choice=event, confidence=event_conf)},
    )


def _judgment(**overrides: float | str | bool) -> NewsJudgment:
    base = dict(
        relevance=0.99,
        sentiment=1.0,
        sentiment_confidence=0.95,
        event_type="earnings",
        event_confidence=0.9,
        binary_event=0.9,
        confirmed=0.95,
    )
    return NewsJudgment(**{**base, **overrides})


class TestBuildQuestions:
    def test_has_every_question_the_parser_reads(self) -> None:
        questions = build_questions()
        assert set(questions) == {"about", "sentiment", "event_type", "binary_event", "confirmed"}

    def test_sentiment_levels_are_ordered_and_odd(self) -> None:
        assert len(SENTIMENT_LEVELS) == 5
        assert "negative" in SENTIMENT_LEVELS[0].lower()
        assert "positive" in SENTIMENT_LEVELS[-1].lower()

    def test_event_types_have_a_no_match_option(self) -> None:
        assert "other" in EVENT_TYPES


class TestJudgmentFromResponse:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [(0.0, -1.0), (2.0, 0.0), (4.0, 1.0), (3.0, 0.5), (1.0, -0.5)],
    )
    def test_sentiment_is_normalised_to_minus_one_plus_one(
        self, score: float, expected: float
    ) -> None:
        j = judgment_from_response(_response(sentiment=score))
        assert j.sentiment == pytest.approx(expected)

    def test_out_of_range_score_is_clamped(self) -> None:
        assert judgment_from_response(_response(sentiment=9.0)).sentiment == 1.0
        assert judgment_from_response(_response(sentiment=-3.0)).sentiment == -1.0

    def test_fields_are_copied(self) -> None:
        j = judgment_from_response(_response())
        assert j.relevance == 0.99
        assert j.event_type == "earnings"
        assert j.binary_event == 0.9
        assert j.confirmed == 0.95

    def test_unknown_event_type_becomes_other(self) -> None:
        assert judgment_from_response(_response(event="aliens")).event_type == "other"

    def test_judgment_is_immutable(self) -> None:
        with pytest.raises(AttributeError):
            _judgment().sentiment = -1.0  # type: ignore[misc]


class TestNewsSignal:
    def test_confirmed_relevant_positive_news_scores_high(self) -> None:
        s = news_signal(_judgment())
        assert s.score > 0.9
        assert s.block is False

    def test_negative_news_scores_negative(self) -> None:
        assert news_signal(_judgment(sentiment=-1.0)).score < -0.9

    def test_irrelevant_news_is_ignored(self) -> None:
        s = news_signal(_judgment(relevance=0.05))
        assert abs(s.score) < 0.1
        assert s.block is False

    def test_low_confidence_direction_scores_zero(self) -> None:
        s = news_signal(_judgment(sentiment_confidence=0.2))
        assert s.score == 0.0

    def test_rumor_is_discounted(self) -> None:
        confirmed = news_signal(_judgment(confirmed=0.95)).score
        rumor = news_signal(_judgment(confirmed=0.05)).score
        assert 0 < rumor < confirmed

    def test_unconfirmed_binary_event_blocks_entry(self) -> None:
        s = news_signal(_judgment(binary_event=0.9, confirmed=0.1))
        assert s.block is True
        assert "binary" in s.reason.lower()

    def test_binary_event_with_unclear_direction_blocks_entry(self) -> None:
        s = news_signal(_judgment(binary_event=0.9, sentiment_confidence=0.3))
        assert s.block is True

    def test_confirmed_clear_binary_event_does_not_block(self) -> None:
        assert news_signal(_judgment(binary_event=0.9)).block is False

    def test_irrelevant_binary_event_does_not_block(self) -> None:
        s = news_signal(_judgment(relevance=0.1, binary_event=0.95, confirmed=0.1))
        assert s.block is False

    def test_score_is_bounded(self) -> None:
        for sentiment in (-1.0, 1.0):
            assert -1.0 <= news_signal(_judgment(sentiment=sentiment)).score <= 1.0

    def test_thresholds_are_configurable(self) -> None:
        strict = SignalThresholds(min_confidence=0.99)
        assert news_signal(_judgment(sentiment_confidence=0.95), strict).score == 0.0
