"""Finnhub news source: fetch, validate and normalise into :class:`NewsItem`."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import httpx

from src.signals.text_safety import sanitize_text

logger = logging.getLogger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
MARKET_CATEGORIES = frozenset({"general", "forex", "crypto", "merger"})
MAX_SUMMARY_CHARS = 1500
MAX_HEADLINE_CHARS = 500
MAX_SOURCE_CHARS = 40
MAX_URL_CHARS = 300
DEFAULT_TIMEOUT_S = 10.0
MAX_ATTEMPTS = 3
RETRY_BACKOFF_S = 1.0
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")


def is_valid_symbol(symbol: str) -> bool:
    """True for plain tickers such as ``NVDA``, ``BRK.B`` or ``BTC-USD``."""
    return bool(_SYMBOL_RE.match(symbol.strip().upper()))


class NewsFetchError(RuntimeError):
    """News could not be fetched or the response was not usable."""


@dataclass(frozen=True)
class NewsItem:
    """One normalised news item. ``symbol`` is the asset it was fetched for."""

    symbol: str
    headline: str
    summary: str
    published: datetime  # timezone-aware UTC
    source: str
    url: str
    item_id: str


def _fallback_id(symbol: str, stamp: int, headline: str) -> str:
    digest = hashlib.sha256(f"{symbol}|{stamp}|{headline}".encode()).hexdigest()[:16]
    return f"{symbol}:{stamp}:{digest}"


def _item_from_raw(symbol: str, raw: Any) -> NewsItem | None:
    if not isinstance(raw, dict):
        return None
    headline = sanitize_text(raw.get("headline"), MAX_HEADLINE_CHARS)
    stamp = raw.get("datetime")
    if not headline or isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or stamp <= 0:
        return None
    try:
        published = datetime.fromtimestamp(stamp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None  # one malformed timestamp must not discard the whole batch
    return NewsItem(
        symbol=symbol,
        headline=headline,
        summary=sanitize_text(raw.get("summary"), MAX_SUMMARY_CHARS),
        published=published,
        source=sanitize_text(raw.get("source"), MAX_SOURCE_CHARS),
        url=sanitize_text(raw.get("url"), MAX_URL_CHARS),
        item_id=sanitize_text(raw.get("id"), 64) or _fallback_id(symbol, int(stamp), headline),
    )


def parse_finnhub_items(symbol: str, payload: Any) -> tuple[NewsItem, ...]:
    """Validate a Finnhub payload; return deduplicated items, newest first."""
    if not isinstance(payload, list):
        raise NewsFetchError("Finnhub returned an unexpected (non-list) response")
    unique: dict[str, NewsItem] = {}
    for raw in payload:
        item = _item_from_raw(symbol, raw)
        if item is not None:
            unique.setdefault(item.item_id, item)
    return tuple(sorted(unique.values(), key=lambda i: i.published, reverse=True))


class FinnhubNewsClient:
    """Thin, synchronous Finnhub news client (free tier: ~60 calls/minute)."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        min_interval_s: float = 0.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise NewsFetchError("FINNHUB_API_KEY is not set")
        self._api_key = api_key
        self._min_interval_s = min_interval_s
        self._last_request = float("-inf")
        self._http = httpx.Client(
            base_url=FINNHUB_BASE_URL, timeout=timeout, transport=transport
        )

    def fetch(self, symbol: str, *, days: int = 2) -> tuple[NewsItem, ...]:
        """Company news for a stock ticker over the last ``days`` days."""
        sym = symbol.strip().upper()
        if not _SYMBOL_RE.match(sym):
            raise ValueError(f"invalid symbol: {symbol!r}")
        today = datetime.now(timezone.utc).date()
        params = {
            "symbol": sym,
            "from": (today - timedelta(days=max(days, 0))).isoformat(),
            "to": today.isoformat(),
        }
        return parse_finnhub_items(sym, self._get("/company-news", params))

    def fetch_range(self, symbol: str, start: date, end: date) -> tuple[NewsItem, ...]:
        """Company news for ``start..end`` inclusive, one request per day.

        Daily chunks keep each response well under Finnhub's per-call item cap.
        """
        sym = symbol.strip().upper()
        if not _SYMBOL_RE.match(sym):
            raise ValueError(f"invalid symbol: {symbol!r}")
        if end < start:
            raise ValueError("end must not be before start")
        merged: dict[str, NewsItem] = {}
        day = start
        while day <= end:
            params = {"symbol": sym, "from": day.isoformat(), "to": day.isoformat()}
            for item in parse_finnhub_items(sym, self._get("/company-news", params)):
                merged.setdefault(item.item_id, item)
            day += timedelta(days=1)
        return tuple(sorted(merged.values(), key=lambda i: i.published, reverse=True))

    def fetch_market(self, category: str) -> tuple[NewsItem, ...]:
        """Market-wide news: ``general``, ``forex``, ``crypto`` or ``merger``."""
        if category not in MARKET_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(MARKET_CATEGORIES)}")
        return parse_finnhub_items(category.upper(), self._get("/news", {"category": category}))

    def _throttle(self) -> None:
        wait = self._last_request + self._min_interval_s - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, path: str, params: dict[str, str]) -> Any:
        """GET with bounded retries on transient failures (timeouts, 429, 5xx)."""
        last_error = "Finnhub request failed"
        for attempt in range(MAX_ATTEMPTS):
            if attempt:
                time.sleep(RETRY_BACKOFF_S * 2 ** (attempt - 1))
            self._throttle()
            try:
                response = self._http.get(path, params=params, headers={"X-Finnhub-Token": self._api_key})
            except httpx.TransportError as exc:  # timeouts, network and protocol drops
                # Message omits the request URL, which contains the API token.
                last_error = f"Finnhub request failed: {type(exc).__name__}"
                continue
            except httpx.HTTPError as exc:
                raise NewsFetchError(f"Finnhub request failed: {type(exc).__name__}") from None
            if response.status_code == 429:
                last_error = "Finnhub rate limit reached; retry shortly"
                continue
            if response.status_code >= 500:
                last_error = f"Finnhub returned HTTP {response.status_code}"
                continue
            if response.status_code != 200:
                raise NewsFetchError(f"Finnhub returned HTTP {response.status_code}")
            try:
                return response.json()
            except ValueError:
                raise NewsFetchError("Finnhub returned invalid JSON") from None
        raise NewsFetchError(last_error)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "FinnhubNewsClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def newest(items: Iterable[NewsItem], since: datetime) -> tuple[NewsItem, ...]:
    """Items published strictly after ``since``, newest first."""
    return tuple(sorted((i for i in items if i.published > since), key=lambda i: i.published, reverse=True))
