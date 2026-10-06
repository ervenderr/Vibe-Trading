"""Tests for the live signal service (orchestration with injected sources)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src.signals.jev_analyzer import AnalyzedNews, JevNewsAnalyzer, JudgmentCache
from src.signals.news_source import NewsItem
from src.signals.service import generate_signal
from src.signals.verdict import Action

NOW = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)


def _bars(n: int = 80, end: datetime = NOW, slope: float = 0.3) -> pd.DataFrame:
    idx = pd.DatetimeIndex([end - timedelta(minutes=5 * (n - i)) for i in range(n)])
    # a realistic noisy trend (RSI well below 100) rather than a perfect ramp
    close = pd.Series([100 + slope * i + (0.5 if i % 2 else -0.5) for i in range(n)], index=idx)
    return pd.DataFrame(
        {"open": close.shift(1).fillna(close.iloc[0]), "high": close + 0.3, "low": close - 0.3,
         "close": close, "volume": 1000.0}
    )


def _item(headline: str, minutes_ago: float = 3.0) -> NewsItem:
    return NewsItem("NVDA", headline, "s", NOW - timedelta(minutes=minutes_ago), "src", "u", headline)


class FakeJev:
    def __init__(self, sentiment: float = 4.0) -> None:
        self.sentiment = sentiment

    def system_one(self, state, questions, *, model=None, **_):
        return SimpleNamespace(
            nouls={"about": SimpleNamespace(noul=0.97), "binary_event": SimpleNamespace(noul=0.8),
                   "confirmed": SimpleNamespace(noul=0.95)},
            scores={"sentiment": SimpleNamespace(score=self.sentiment, confidence=0.95)},
            choices={"event_type": SimpleNamespace(choice="earnings", confidence=0.9)},
        )


def _run(tmp_path, *, items=None, bars=None, jev=None, **kw):
    analyzer = JevNewsAnalyzer(jev or FakeJev(), model="m", cache=JudgmentCache(tmp_path))
    return generate_signal(
        "NVDA",
        news_source=lambda s: items if items is not None else [_item("NVIDIA beats estimates")],
        analyzer=analyzer,
        bars_source=lambda s: bars if bars is not None else _bars(),
        now=NOW,
        **kw,
    )


class TestGenerateSignal:
    def test_bullish_news_and_uptrend_is_buy(self, tmp_path) -> None:
        report = _run(tmp_path)
        assert report.verdict.action is Action.BUY
        assert report.symbol == "NVDA"
        assert report.stale is False

    def test_bearish_news_and_downtrend_is_sell(self, tmp_path) -> None:
        report = _run(tmp_path, bars=_bars(slope=-0.3), jev=FakeJev(sentiment=0.0))
        assert report.verdict.action is Action.SELL

    def test_single_headline_does_not_chase_an_exhausted_move(self, tmp_path) -> None:
        ramp = _bars()
        ramp["close"] = [100 + 0.3 * i for i in range(len(ramp))]  # straight line: RSI pinned at 100
        ramp["open"], ramp["high"], ramp["low"] = ramp["close"].shift(1).fillna(100), ramp["close"] + 0.3, ramp["close"] - 0.3
        assert _run(tmp_path, bars=ramp).verdict.action is Action.WAIT

    def test_conflicting_news_and_trend_waits(self, tmp_path) -> None:
        report = _run(tmp_path, jev=FakeJev(sentiment=0.0))
        assert report.verdict.action is Action.WAIT

    def test_stale_price_data_never_trades(self, tmp_path) -> None:
        old = _bars(end=NOW - timedelta(hours=20))
        report = _run(tmp_path, bars=old)
        assert report.stale is True
        assert report.verdict.action is Action.WAIT
        assert "stale" in " ".join(report.verdict.reasons).lower()

    def test_not_enough_bars_waits(self, tmp_path) -> None:
        report = _run(tmp_path, bars=_bars(n=10))
        assert report.verdict.action is Action.WAIT

    def test_empty_bars_waits(self, tmp_path) -> None:
        report = _run(tmp_path, bars=_bars(n=80).iloc[0:0])
        assert report.verdict.action is Action.WAIT

    def test_risk_flags_force_avoid(self, tmp_path) -> None:
        report = _run(tmp_path, risk_flags=("honeypot",))
        assert report.verdict.action is Action.AVOID

    def test_news_source_failure_degrades_to_price_only_not_crash(self, tmp_path) -> None:
        def boom(symbol):
            raise RuntimeError("finnhub down")

        analyzer = JevNewsAnalyzer(FakeJev(), model="m", cache=JudgmentCache(tmp_path))
        report = generate_signal("NVDA", news_source=boom, analyzer=analyzer,
                                 bars_source=lambda s: _bars(), now=NOW)
        assert report.news_error is not None
        assert report.verdict.action in (Action.BUY, Action.WAIT)

    def test_top_headlines_are_ranked_and_capped(self, tmp_path) -> None:
        items = [_item(f"headline {i}", minutes_ago=i + 1) for i in range(12)]
        report = _run(tmp_path, items=items, max_headlines=3)
        assert len(report.headlines) == 3
        assert all(isinstance(h, AnalyzedNews) for h in report.headlines)

    def test_old_news_outside_lookback_is_not_analyzed(self, tmp_path) -> None:
        report = _run(tmp_path, items=[_item("ancient", minutes_ago=60 * 24)])
        assert report.headlines == ()

    def test_invalid_symbol_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            analyzer = JevNewsAnalyzer(FakeJev(), model="m", cache=JudgmentCache(tmp_path))
            generate_signal("../etc", news_source=lambda s: [], analyzer=analyzer,
                            bars_source=lambda s: _bars(), now=NOW)
