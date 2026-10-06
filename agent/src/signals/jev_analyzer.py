"""Judge news items with Jev, with a content-hash cache and a safe fallback.

Jev failures (auth, rate limit, timeout, outage) degrade to the deterministic
lexicon scorer and are logged; they never block a trade and are never cached.
Programming errors are *not* swallowed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Sequence

from typesafe_sdk import TypeSafeError

from backtest.loaders.rsshub_events import default_lexicon_scorer
from src.signals.news_judgment import (
    EVENT_TYPES,
    NewsJudgment,
    NewsSignal,
    SignalThresholds,
    build_questions,
    judgment_from_response,
    news_signal,
)
from src.signals.news_source import NewsItem

logger = logging.getLogger(__name__)

Origin = Literal["jev", "cache", "fallback"]
DEFAULT_MAX_WORKERS = 8
FALLBACK_SCALE = 0.25  # keyword scores are weak evidence: keep them far below an entry
#: Changes whenever the questions change, so stale judgments are never reused.
PROMPT_VERSION = hashlib.sha256(repr(build_questions()).encode()).hexdigest()[:12]
_PARSE_ERRORS = (KeyError, AttributeError, TypeError, ValueError, IndexError)


@dataclass(frozen=True)
class AnalyzedNews:
    """A news item with its judgment (None on fallback) and resulting signal."""

    item: NewsItem
    judgment: NewsJudgment | None
    signal: NewsSignal
    origin: Origin


def cache_key(item: NewsItem, model: str) -> str:
    """Stable, filesystem-safe key over everything that affects the answer."""
    raw = json.dumps([PROMPT_VERSION, model, item.symbol, item.headline, item.summary], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _is_number(value: object, low: float, high: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and low <= value <= high
    )


def _is_valid(j: NewsJudgment) -> bool:
    return (
        all(_is_number(v, 0.0, 1.0) for v in (j.relevance, j.sentiment_confidence, j.event_confidence, j.binary_event, j.confirmed))
        and _is_number(j.sentiment, -1.0, 1.0)
        and isinstance(j.event_type, str)
        and j.event_type in EVENT_TYPES
    )


class JudgmentCache:
    """One JSON file per judgment, written atomically."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)

    def get(self, key: str) -> NewsJudgment | None:
        path = self._dir / f"{key}.json"
        try:
            judgment = NewsJudgment(**json.loads(path.read_text()))
        except FileNotFoundError:
            return None
        except (ValueError, TypeError, OSError) as exc:
            logger.warning("ignoring unreadable cache entry %s: %s", path.name, exc)
            return None
        if not _is_valid(judgment):
            logger.warning("ignoring invalid cache entry %s", path.name)
            return None
        return judgment

    def put(self, key: str, judgment: NewsJudgment) -> None:
        tmp: str | None = None
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
            with os.fdopen(fd, "w") as handle:
                json.dump(asdict(judgment), handle)
            os.replace(tmp, self._dir / f"{key}.json")
            tmp = None
        except OSError as exc:
            logger.warning("could not write judgment cache: %s", exc)
        finally:
            if tmp is not None:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


class JevNewsAnalyzer:
    """Turns :class:`NewsItem` into :class:`AnalyzedNews` using a Jev client."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        cache: JudgmentCache,
        thresholds: SignalThresholds | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._cache = cache
        self._thresholds = thresholds or SignalThresholds()

    def analyze(self, item: NewsItem) -> AnalyzedNews:
        key = cache_key(item, self._model)
        cached = self._cache.get(key)
        if cached is not None:
            return self._result(item, cached, "cache")
        try:
            response = self._client.system_one(
                {"symbol": item.symbol, "headline": item.headline, "summary": item.summary},
                build_questions(),
                model=self._model,
            )
            judgment = judgment_from_response(response)
        except TypeSafeError as exc:
            logger.warning("Jev unavailable (%s); using lexicon fallback", type(exc).__name__)
            return self._fallback(item)
        except _PARSE_ERRORS as exc:
            logger.warning("malformed Jev response (%s); using lexicon fallback", type(exc).__name__)
            return self._fallback(item)
        if not _is_valid(judgment):
            logger.warning("out-of-range Jev response; using lexicon fallback")
            return self._fallback(item)
        self._cache.put(key, judgment)
        return self._result(item, judgment, "jev")

    def analyze_many(
        self, items: Sequence[NewsItem], *, max_workers: int = DEFAULT_MAX_WORKERS
    ) -> tuple[AnalyzedNews, ...]:
        """Analyze items concurrently; results keep the input order."""
        if not items:
            return ()
        unique: dict[str, NewsItem] = {}
        for item in items:
            unique.setdefault(cache_key(item, self._model), item)
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(unique)))) as pool:
            done = dict(zip(unique, pool.map(self.analyze, unique.values())))
        return tuple(
            done[cache_key(item, self._model)] if item is unique[cache_key(item, self._model)]
            else replace(done[cache_key(item, self._model)], item=item)
            for item in items
        )

    def _result(self, item: NewsItem, judgment: NewsJudgment, origin: Origin) -> AnalyzedNews:
        return AnalyzedNews(item, judgment, news_signal(judgment, self._thresholds), origin)

    @staticmethod
    def _fallback(item: NewsItem) -> AnalyzedNews:
        score = default_lexicon_scorer(item.headline, item.summary) * FALLBACK_SCALE
        signal = NewsSignal(score, False, "fallback: Jev unavailable, keyword score only (capped)")
        return AnalyzedNews(item, None, signal, "fallback")
