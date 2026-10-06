"""Tests for Finnhub news parsing and fetching."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from src.signals.news_source import (
    FinnhubNewsClient,
    NewsFetchError,
    NewsItem,
    parse_finnhub_items,
)

_RAW = {
    "category": "company",
    "datetime": 1790848200,
    "headline": "NVIDIA beats estimates",
    "id": 142549037,
    "related": "NVDA",
    "source": "Yahoo",
    "summary": "Strong quarter.",
    "url": "https://example.com/a",
}


class TestParse:
    def test_parses_valid_item(self) -> None:
        (item,) = parse_finnhub_items("NVDA", [_RAW])
        assert item.symbol == "NVDA"
        assert item.headline == "NVIDIA beats estimates"
        assert item.published == datetime.fromtimestamp(1790848200, tz=timezone.utc)
        assert item.item_id == "142549037"

    def test_skips_items_without_headline_or_time(self) -> None:
        bad = [{**_RAW, "headline": ""}, {k: v for k, v in _RAW.items() if k != "datetime"}]
        assert parse_finnhub_items("NVDA", bad) == ()

    def test_skips_non_dict_entries(self) -> None:
        assert parse_finnhub_items("NVDA", ["x", None, 3]) == ()

    def test_rejects_non_list_payload(self) -> None:
        with pytest.raises(NewsFetchError):
            parse_finnhub_items("NVDA", {"error": "bad key"})

    def test_sorted_newest_first_and_deduplicated(self) -> None:
        older = {**_RAW, "id": 1, "datetime": 1000}
        newer = {**_RAW, "id": 2, "datetime": 2000}
        items = parse_finnhub_items("NVDA", [older, newer, newer])
        assert [i.item_id for i in items] == ["2", "1"]

    def test_item_is_immutable(self) -> None:
        (item,) = parse_finnhub_items("NVDA", [_RAW])
        with pytest.raises(AttributeError):
            item.headline = "x"  # type: ignore[misc]

    def test_text_is_length_capped(self) -> None:
        (item,) = parse_finnhub_items("NVDA", [{**_RAW, "summary": "x" * 10_000}])
        assert len(item.summary) <= 1500


def _client(handler) -> FinnhubNewsClient:
    return FinnhubNewsClient("k", transport=httpx.MockTransport(handler))


class TestClient:
    def test_requires_api_key(self) -> None:
        with pytest.raises(NewsFetchError):
            FinnhubNewsClient("")

    def test_fetch_sends_symbol_and_token(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(dict(request.url.params))
            return httpx.Response(200, json=[_RAW])

        items = _client(handler).fetch("nvda", days=2)
        assert seen["symbol"] == "NVDA"
        assert "token" not in seen  # the key travels in a header, never in the URL
        assert len(items) == 1

    @pytest.mark.parametrize("bad", ["", "NV DA", "A" * 40, "../x"])
    def test_rejects_invalid_symbol(self, bad: str) -> None:
        with pytest.raises(ValueError):
            _client(lambda r: httpx.Response(200, json=[])).fetch(bad)

    def test_http_error_raises_without_leaking_token(self) -> None:
        with pytest.raises(NewsFetchError) as exc:
            _client(lambda r: httpx.Response(401, json={"error": "no"})).fetch("NVDA")
        assert "k" != str(exc.value)
        assert "token=k" not in str(exc.value)

    def test_rate_limit_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.signals.news_source as mod

        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        with pytest.raises(NewsFetchError, match="rate"):
            _client(lambda r: httpx.Response(429)).fetch("NVDA")


class TestMarketNews:
    def test_market_category_uses_news_endpoint(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["category"] = request.url.params["category"]
            return httpx.Response(200, json=[_RAW])

        _client(handler).fetch_market("crypto")
        assert seen == {"path": "/api/v1/news", "category": "crypto"}

    def test_rejects_unknown_category(self) -> None:
        with pytest.raises(ValueError):
            _client(lambda r: httpx.Response(200, json=[])).fetch_market("memes")


def test_fallback_id_is_stable_across_calls() -> None:
    raw = {k: v for k, v in _RAW.items() if k != "id"}
    a = parse_finnhub_items("NVDA", [raw])[0].item_id
    b = parse_finnhub_items("NVDA", [raw])[0].item_id
    assert a == b and a.startswith("NVDA:")


class TestFetchRange:
    def test_chunks_by_day_and_merges_deduplicated(self) -> None:
        from datetime import date

        calls: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append((request.url.params["from"], request.url.params["to"]))
            return httpx.Response(200, json=[{**_RAW, "id": 7}, {**_RAW, "id": len(calls) + 100}])

        items = _client(handler).fetch_range("NVDA", date(2026, 9, 1), date(2026, 9, 3))
        assert calls == [("2026-09-01", "2026-09-01"), ("2026-09-02", "2026-09-02"), ("2026-09-03", "2026-09-03")]
        assert len({i.item_id for i in items}) == len(items) == 4

    def test_rejects_reversed_range(self) -> None:
        from datetime import date

        with pytest.raises(ValueError):
            _client(lambda r: httpx.Response(200, json=[])).fetch_range("NVDA", date(2026, 9, 3), date(2026, 9, 1))


class TestThrottle:
    def test_spaces_consecutive_requests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.signals.news_source as mod

        now = [100.0]
        sleeps: list[float] = []
        monkeypatch.setattr(mod.time, "monotonic", lambda: now[0])
        monkeypatch.setattr(mod.time, "sleep", lambda s: (sleeps.append(s), now.__setitem__(0, now[0] + s)))
        client = FinnhubNewsClient("k", min_interval_s=1.0, transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[])))
        client.fetch("NVDA")
        client.fetch("NVDA")
        assert len(sleeps) == 1 and sleeps[0] == pytest.approx(1.0)


class TestRetries:
    def _client(self, handler, monkeypatch):
        import src.signals.news_source as mod

        self.sleeps: list[float] = []
        monkeypatch.setattr(mod.time, "sleep", lambda s: self.sleeps.append(s))
        return FinnhubNewsClient("k", transport=httpx.MockTransport(handler))

    def test_timeout_then_success_recovers(self, monkeypatch) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("slow", request=request)
            return httpx.Response(200, json=[_RAW])

        assert len(self._client(handler, monkeypatch).fetch("NVDA")) == 1
        assert calls["n"] == 2 and len(self.sleeps) == 1

    def test_server_error_is_retried(self, monkeypatch) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json=[])

        self._client(handler, monkeypatch).fetch("NVDA")
        assert calls["n"] == 3

    def test_persistent_failure_raises_after_bounded_attempts(self, monkeypatch) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            raise httpx.ConnectTimeout("x", request=request)

        with pytest.raises(NewsFetchError):
            self._client(handler, monkeypatch).fetch("NVDA")
        assert calls["n"] == 3

    @pytest.mark.parametrize("status", [401, 403, 404])
    def test_client_errors_are_not_retried(self, status: int, monkeypatch) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(status)

        with pytest.raises(NewsFetchError):
            self._client(handler, monkeypatch).fetch("NVDA")
        assert calls["n"] == 1

    def test_rate_limit_is_retried_with_backoff_then_reported(self, monkeypatch) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(429)

        with pytest.raises(NewsFetchError, match="rate"):
            self._client(handler, monkeypatch).fetch("NVDA")
        assert calls["n"] == 3


class TestProtocolErrorsAreRetried:
    def test_dropped_connection_mid_response_is_retried(self, monkeypatch) -> None:
        import src.signals.news_source as mod

        monkeypatch.setattr(mod.time, "sleep", lambda s: None)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.RemoteProtocolError("server disconnected", request=request)
            return httpx.Response(200, json=[_RAW])

        client = FinnhubNewsClient("k", transport=httpx.MockTransport(handler))
        assert len(client.fetch("NVDA")) == 1 and calls["n"] == 2
