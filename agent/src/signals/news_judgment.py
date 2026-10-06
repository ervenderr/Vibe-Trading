"""Pure logic for judging a news item with Jev and turning it into a signal.

Jev supplies *judgments* about text (relevance, direction, event type, whether
it is confirmed). Everything numeric and every policy decision lives here, in
code: Jev is not reliable at arithmetic, so it never computes the signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from typesafe_sdk import Choice, Noul, Score

#: Ordered Score levels, worst to best for a holder of the stock.
SENTIMENT_LEVELS: tuple[str, ...] = (
    "Strongly negative for holders: a large loss of value is likely, such as a fraud finding, "
    "a guidance cut, a failed trial or a major lawsuit loss",
    "Negative for holders: a modest loss of value is likely, such as a downgrade or a miss",
    "Neutral: no clear effect on the value of the stock, or a mix of good and bad",
    "Positive for holders: a modest gain in value is likely, such as an upgrade or a small beat",
    "Strongly positive for holders: a large gain in value is likely, such as a big beat with "
    "raised guidance, an approval or an acquisition at a premium",
)

#: Closed set of event types. ``other`` is the explicit no-match outcome.
EVENT_TYPES: tuple[str, ...] = (
    "earnings",
    "guidance",
    "merger_acquisition",
    "regulatory_legal",
    "analyst_rating",
    "product_launch",
    "macro_economic",
    "security_incident",
    "other",
)

_EVENT_TYPE_HINTS: Mapping[str, str] = {
    "earnings": "Reports quarterly or annual results",
    "guidance": "Company changes its own forecast",
    "merger_acquisition": "A merger, acquisition, stake sale or buyout",
    "regulatory_legal": "A regulator, court, lawsuit, approval or investigation",
    "analyst_rating": "An analyst upgrade, downgrade or price target change",
    "product_launch": "A product, partnership or contract announcement",
    "macro_economic": "Interest rates, inflation, jobs data or other economy-wide news",
    "security_incident": "A hack, breach, exploit or outage",
    "other": "None of the above",
}


@dataclass(frozen=True)
class NewsJudgment:
    """What Jev concluded about one headline for one symbol."""

    relevance: float  # P(headline is directly about the symbol)
    sentiment: float  # -1 (very bad for holders) .. +1 (very good)
    sentiment_confidence: float
    event_type: str
    event_confidence: float
    binary_event: float  # P(one-time event that can move price sharply)
    confirmed: float  # P(already happened or officially announced)


@dataclass(frozen=True)
class SignalThresholds:
    """Policy knobs. Tune on your own data; these are starting points."""

    min_confidence: float = 0.5  # below this the direction is treated as unknown
    block_binary_event: float = 0.6  # P(binary event) that makes it risky
    block_relevance: float = 0.5  # only block on news that is about the symbol
    block_direction_confidence: float = 0.6  # binary event with direction less sure than this
    block_confirmed: float = 0.5  # binary event less confirmed than this
    rumor_discount: float = 0.5  # weight applied to a fully unconfirmed item


@dataclass(frozen=True)
class NewsSignal:
    """The actionable result: a bounded score and an entry-block flag."""

    score: float  # -1 .. +1
    block: bool
    reason: str


def build_questions() -> dict[str, Noul | Choice | Score]:
    """The questions asked about every ``{symbol, headline, summary}`` state."""
    return {
        "about": Noul(
            instructions=(
                "Is `headline` directly about the company or asset identified by `symbol`, "
                "rather than only mentioning it in passing or being about a different company?"
            )
        ),
        "sentiment": Score(
            instructions=(
                "How good or bad is the news in `headline` and `summary` for someone who "
                "holds `symbol`?"
            ),
            criteria=list(SENTIMENT_LEVELS),
        ),
        "event_type": Choice(
            instructions="What kind of event does `headline` report?",
            criteria=dict(_EVENT_TYPE_HINTS),
        ),
        "binary_event": Noul(
            instructions=(
                "Does `headline` report or anticipate a one-time event that can move the price "
                "of `symbol` sharply in either direction, such as earnings, a regulatory "
                "decision, a court ruling, a merger, or a guidance change?"
            )
        ),
        "confirmed": Noul(
            instructions=(
                "Does `headline` report something that has already happened or has been "
                "officially announced, rather than a rumor, an opinion, a prediction or a "
                "plan?"
            )
        ),
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalise_sentiment(score: float) -> float:
    top = len(SENTIMENT_LEVELS) - 1
    return _clamp((score - top / 2) / (top / 2), -1.0, 1.0)


def judgment_from_response(response: Any) -> NewsJudgment:
    """Convert a Jev ``system_one`` response into a :class:`NewsJudgment`."""
    sentiment = response.scores["sentiment"]
    event = response.choices["event_type"]
    event_type = event.choice if event.choice in EVENT_TYPES else "other"
    return NewsJudgment(
        relevance=response.nouls["about"].noul,
        sentiment=_normalise_sentiment(sentiment.score),
        sentiment_confidence=sentiment.confidence,
        event_type=event_type,
        event_confidence=event.confidence,
        binary_event=response.nouls["binary_event"].noul,
        confirmed=response.nouls["confirmed"].noul,
    )


def news_signal(
    judgment: NewsJudgment, thresholds: SignalThresholds | None = None
) -> NewsSignal:
    """Combine a judgment into a score and an entry-block flag, in code."""
    t = thresholds or SignalThresholds()

    block, block_reason = _should_block(judgment, t)
    if judgment.sentiment_confidence < t.min_confidence:
        return NewsSignal(0.0, block, block_reason or "direction unclear (low confidence)")

    evidence = 1.0 - t.rumor_discount * (1.0 - judgment.confirmed)
    score = _clamp(judgment.sentiment * judgment.relevance * evidence, -1.0, 1.0)
    return NewsSignal(score, block, block_reason or "ok")


def _should_block(j: NewsJudgment, t: SignalThresholds) -> tuple[bool, str]:
    if j.relevance < t.block_relevance or j.binary_event < t.block_binary_event:
        return False, ""
    if j.confirmed < t.block_confirmed:
        return True, "unconfirmed binary event pending"
    if j.sentiment_confidence < t.block_direction_confidence:
        return True, "binary event with unclear direction"
    return False, ""
