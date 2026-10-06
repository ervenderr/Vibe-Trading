"""Tests for the Jev news analyzer: caching, fallback, error handling."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from typesafe_sdk import TypeSafeError

from src.signals.jev_analyzer import JevNewsAnalyzer, JudgmentCache, cache_key
from src.signals.news_judgment import NewsJudgment
from src.signals.news_source import NewsItem

_ITEM = NewsItem(
    symbol="NVDA",
    headline="NVIDIA beats earnings estimates and raises guidance",
    summary="Strong quarter.",
    published=datetime(2026, 10, 1, tzinfo=timezone.utc),
    source="Yahoo",
    url="https://example.com",
    item_id="1",
)


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        nouls={
            "about": SimpleNamespace(noul=0.99),
            "binary_event": SimpleNamespace(noul=0.9),
            "confirmed": SimpleNamespace(noul=0.95),
        },
        scores={"sentiment": SimpleNamespace(score=4.0, confidence=0.95)},
        choices={"event_type": SimpleNamespace(choice="earnings", confidence=0.9)},
    )


class FakeClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._error = error

    def system_one(self, state, questions, *, model=None, **_):
        self.calls.append({"state": state, "questions": questions, "model": model})
        if self._error:
            raise self._error
        return _response()


def _analyzer(client: FakeClient, tmp_path) -> JevNewsAnalyzer:
    return JevNewsAnalyzer(client, model="jev-1.13.0", cache=JudgmentCache(tmp_path))


class TestAnalyze:
    def test_returns_signal_from_jev(self, tmp_path) -> None:
        result = _analyzer(FakeClient(), tmp_path).analyze(_ITEM)
        assert result.origin == "jev"
        assert result.signal.score > 0.9
        assert result.judgment is not None

    def test_sends_symbol_headline_and_summary_as_state(self, tmp_path) -> None:
        client = FakeClient()
        _analyzer(client, tmp_path).analyze(_ITEM)
        call = client.calls[0]
        assert call["state"] == {
            "symbol": "NVDA",
            "headline": _ITEM.headline,
            "summary": _ITEM.summary,
        }
        assert call["model"] == "jev-1.13.0"

    def test_second_call_is_served_from_cache(self, tmp_path) -> None:
        client = FakeClient()
        analyzer = _analyzer(client, tmp_path)
        analyzer.analyze(_ITEM)
        again = analyzer.analyze(_ITEM)
        assert len(client.calls) == 1
        assert again.origin == "cache"
        assert again.signal.score > 0.9

    def test_cache_persists_across_instances(self, tmp_path) -> None:
        _analyzer(FakeClient(), tmp_path).analyze(_ITEM)
        client = FakeClient()
        result = _analyzer(client, tmp_path).analyze(_ITEM)
        assert client.calls == [] and result.origin == "cache"

    def test_api_error_falls_back_to_lexicon_never_blocks(self, tmp_path) -> None:
        result = _analyzer(FakeClient(TypeSafeError("boom")), tmp_path).analyze(_ITEM)
        assert result.origin == "fallback"
        assert result.judgment is None
        assert result.signal.block is False
        assert "fallback" in result.signal.reason

    def test_fallback_result_is_not_cached(self, tmp_path) -> None:
        analyzer = _analyzer(FakeClient(TypeSafeError("boom")), tmp_path)
        analyzer.analyze(_ITEM)
        client = FakeClient()
        recovered = JevNewsAnalyzer(client, model="jev-1.13.0", cache=JudgmentCache(tmp_path))
        assert recovered.analyze(_ITEM).origin == "jev"

    def test_unexpected_error_is_not_swallowed(self, tmp_path) -> None:
        with pytest.raises(ZeroDivisionError):
            _analyzer(FakeClient(ZeroDivisionError()), tmp_path).analyze(_ITEM)

    def test_corrupt_cache_entry_is_ignored(self, tmp_path) -> None:
        cache = JudgmentCache(tmp_path)
        (tmp_path / f"{cache_key(_ITEM, 'jev-1.13.0')}.json").write_text("{not json")
        client = FakeClient()
        result = JevNewsAnalyzer(client, model="jev-1.13.0", cache=cache).analyze(_ITEM)
        assert result.origin == "jev" and len(client.calls) == 1


class TestCacheKey:
    def test_depends_on_model_and_text(self) -> None:
        other = NewsItem(**{**_ITEM.__dict__, "headline": "different"})
        assert cache_key(_ITEM, "m1") != cache_key(_ITEM, "m2")
        assert cache_key(_ITEM, "m1") != cache_key(other, "m1")

    def test_is_stable_and_filesystem_safe(self) -> None:
        key = cache_key(_ITEM, "m1")
        assert key == cache_key(_ITEM, "m1")
        assert key.isalnum()


class TestAnalyzeMany:
    def test_preserves_order_and_isolates_failures(self, tmp_path) -> None:
        items = [
            NewsItem(**{**_ITEM.__dict__, "headline": f"headline {i}", "item_id": str(i)})
            for i in range(5)
        ]
        results = _analyzer(FakeClient(), tmp_path).analyze_many(items, max_workers=3)
        assert [r.item.item_id for r in results] == ["0", "1", "2", "3", "4"]


class TestReviewFindings:
    def test_malformed_response_degrades_that_item_not_the_batch(self, tmp_path) -> None:
        class Flaky(FakeClient):
            def system_one(self, state, questions, *, model=None, **_):
                if "bad" in state["headline"]:
                    return SimpleNamespace(nouls={}, scores={}, choices={})
                return super().system_one(state, questions, model=model)

        items = [NewsItem(**{**_ITEM.__dict__, "headline": h, "item_id": h}) for h in ("good one", "bad one", "good two")]
        results = _analyzer(Flaky(), tmp_path).analyze_many(items)
        assert [r.origin for r in results] == ["jev", "fallback", "jev"]

    def test_fallback_score_is_capped_well_below_an_entry(self, tmp_path) -> None:
        item = NewsItem(**{**_ITEM.__dict__, "headline": "surges upgraded record beat strong growth"})
        r = _analyzer(FakeClient(TypeSafeError("down")), tmp_path).analyze(item)
        assert r.origin == "fallback" and abs(r.signal.score) <= 0.25

    def test_cache_key_changes_with_prompt_version(self, monkeypatch) -> None:
        import src.signals.jev_analyzer as mod

        before = cache_key(_ITEM, "m")
        monkeypatch.setattr(mod, "PROMPT_VERSION", "different")
        assert cache_key(_ITEM, "m") != before

    @pytest.mark.parametrize("bad", [
        {"relevance": "high"}, {"relevance": float("nan")}, {"sentiment": 5.0}, {"event_type": 7},
    ])
    def test_cache_entry_with_bad_types_or_ranges_is_ignored(self, tmp_path, bad) -> None:
        cache = JudgmentCache(tmp_path)
        good = dict(relevance=0.9, sentiment=0.5, sentiment_confidence=0.9, event_type="earnings",
                    event_confidence=0.9, binary_event=0.5, confirmed=0.9)
        (tmp_path / "k.json").write_text(json.dumps({**good, **bad}, allow_nan=True))
        assert cache.get("k") is None

    def test_duplicate_items_in_a_batch_are_judged_once(self, tmp_path) -> None:
        client = FakeClient()
        results = _analyzer(client, tmp_path).analyze_many([_ITEM, _ITEM, _ITEM])
        assert len(client.calls) == 1 and len(results) == 3

    def test_temp_file_is_not_leaked_when_write_fails(self, tmp_path, monkeypatch) -> None:
        import src.signals.jev_analyzer as mod

        monkeypatch.setattr(mod.json, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
        JudgmentCache(tmp_path).put("k", _judgment())
        assert list(tmp_path.glob("*.tmp")) == []


def _judgment() -> NewsJudgment:
    return NewsJudgment(0.9, 0.5, 0.9, "earnings", 0.9, 0.5, 0.9)
