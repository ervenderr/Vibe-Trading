"""Tests for the BUY / SELL / WAIT / AVOID verdict engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.signals.news_judgment import NewsSignal
from src.signals.technicals import TechnicalState
from src.signals.verdict import (
    Action,
    TimedNewsSignal,
    VerdictConfig,
    aggregate_news,
    decide,
)

NOW = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)


def _tech(**kw: float) -> TechnicalState:
    base = dict(close=100.0, ema_fast=101.0, ema_slow=100.0, rsi=55.0, atr=1.0, vol_ratio=1.0)
    return TechnicalState(**{**base, **kw})


def _news(score: float, minutes_ago: float = 1.0, block: bool = False) -> TimedNewsSignal:
    return TimedNewsSignal(
        NewsSignal(score, block, "blocked" if block else "ok"), NOW - timedelta(minutes=minutes_ago)
    )


class TestAggregateNews:
    def test_empty_is_neutral(self) -> None:
        agg = aggregate_news([], NOW)
        assert agg.score == 0.0 and agg.block is False and agg.count == 0

    def test_recent_news_outweighs_old(self) -> None:
        agg = aggregate_news([_news(-1.0, 300), _news(1.0, 1)], NOW)
        assert agg.score > 0.5

    def test_news_outside_window_is_ignored(self) -> None:
        cfg = VerdictConfig(news_window_minutes=60)
        assert aggregate_news([_news(1.0, 600)], NOW, cfg).count == 0

    def test_future_news_is_ignored_no_lookahead(self) -> None:
        assert aggregate_news([_news(1.0, -5)], NOW).count == 0

    def test_any_recent_block_blocks(self) -> None:
        assert aggregate_news([_news(0.5), _news(0.0, 10, block=True)], NOW).block is True

    def test_score_is_bounded(self) -> None:
        agg = aggregate_news([_news(1.0), _news(1.0, 2)], NOW)
        assert -1.0 <= agg.score <= 1.0


class TestDecide:
    def test_aligned_bullish_is_buy_with_stop_and_target(self) -> None:
        v = decide(aggregate_news([_news(0.8)], NOW), _tech(), now=NOW)
        assert v.action is Action.BUY
        assert v.stop < v.entry < v.target
        assert 0.0 < v.confidence <= 1.0

    def test_aligned_bearish_is_sell_with_inverted_levels(self) -> None:
        v = decide(aggregate_news([_news(-0.8)], NOW), _tech(ema_fast=99.0), now=NOW)
        assert v.action is Action.SELL
        assert v.target < v.entry < v.stop

    def test_news_and_price_disagreeing_is_wait(self) -> None:
        v = decide(aggregate_news([_news(-0.9)], NOW), _tech(), now=NOW)
        assert v.action is Action.WAIT
        assert "disagree" in " ".join(v.reasons).lower()

    def test_weak_signal_is_wait(self) -> None:
        v = decide(aggregate_news([], NOW), _tech(ema_fast=100.05), now=NOW)
        assert v.action is Action.WAIT

    def test_news_block_forces_wait_even_if_aligned(self) -> None:
        v = decide(aggregate_news([_news(0.9, block=True)], NOW), _tech(), now=NOW)
        assert v.action is Action.WAIT
        assert any("blocked" in r for r in v.reasons)

    def test_risk_flags_force_avoid_over_everything(self) -> None:
        v = decide(
            aggregate_news([_news(0.9)], NOW), _tech(), now=NOW, risk_flags=("honeypot",)
        )
        assert v.action is Action.AVOID
        assert "honeypot" in " ".join(v.reasons)

    def test_no_technical_data_is_wait(self) -> None:
        v = decide(aggregate_news([_news(0.9)], NOW), None, now=NOW)
        assert v.action is Action.WAIT

    def test_tech_only_needs_a_higher_bar_than_with_news(self) -> None:
        cfg = VerdictConfig()
        moderate = _tech(ema_fast=100.4)
        with_news = decide(aggregate_news([_news(0.6)], NOW), moderate, cfg, now=NOW)
        without = decide(aggregate_news([], NOW), moderate, cfg, now=NOW)
        assert with_news.action is Action.BUY
        assert without.action is Action.WAIT

    def test_validity_window_is_set(self) -> None:
        v = decide(aggregate_news([_news(0.8)], NOW), _tech(), VerdictConfig(valid_minutes=10), now=NOW)
        assert v.valid_until == NOW + timedelta(minutes=10)

    def test_reasons_are_always_present(self) -> None:
        assert decide(aggregate_news([], NOW), _tech(), now=NOW).reasons

    def test_zero_atr_cannot_produce_a_trade(self) -> None:
        v = decide(aggregate_news([_news(0.9)], NOW), _tech(atr=0.0), now=NOW)
        assert v.action is Action.WAIT

    @pytest.mark.parametrize("news_score", [-1.0, -0.3, 0.0, 0.3, 1.0])
    def test_confidence_always_in_unit_interval(self, news_score: float) -> None:
        v = decide(aggregate_news([_news(news_score)], NOW), _tech(), now=NOW)
        assert 0.0 <= v.confidence <= 1.0


class TestOverboughtConflict:
    def test_bearish_news_against_overbought_uptrend_waits_not_sells(self) -> None:
        # RSI 100 damps tech_score to 0, but the trend is still clearly up.
        v = decide(aggregate_news([_news(-0.9)], NOW), _tech(rsi=100.0), now=NOW)
        assert v.action is Action.WAIT
        assert "disagree" in " ".join(v.reasons).lower()

    def test_bullish_news_against_oversold_downtrend_waits_not_buys(self) -> None:
        v = decide(aggregate_news([_news(0.9)], NOW), _tech(ema_fast=99.0, rsi=0.0), now=NOW)
        assert v.action is Action.WAIT


class TestStopWidth:
    """A stop inside one 5-minute noise band got GLD stopped out in a minute: widen it, with a price floor."""

    def _buy(self, **kw: float):
        v = decide(aggregate_news([_news(0.8)], NOW), _tech(ema_fast=102.0, **kw), now=NOW)
        assert v.action == Action.BUY
        return v

    def test_stop_is_two_atr_and_target_three_atr(self) -> None:
        v = self._buy(close=100.0, atr=1.0)
        assert round(v.entry - v.stop, 4) == 2.0
        assert round(v.target - v.entry, 4) == 3.0

    def test_calm_symbol_gets_a_price_floor_on_the_stop(self) -> None:
        v = self._buy(close=383.0, atr=0.27)  # 2 ATR = 0.54 (0.14%), below the 0.3% floor
        assert round(v.entry - v.stop, 3) == round(383.0 * 0.003, 3)
        assert round((v.target - v.entry) / (v.entry - v.stop), 3) == 1.5  # reward:risk preserved
