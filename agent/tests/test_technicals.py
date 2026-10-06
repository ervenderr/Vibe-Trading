"""Tests for causal technical indicators and the bounded tech score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.technicals import (
    MIN_BARS,
    TechnicalState,
    technical_frame,
    tech_score,
    state_at,
)


def _bars(close: list[float], spread: float = 1.0, volume: float = 1000.0) -> pd.DataFrame:
    c = pd.Series(close, dtype=float)
    return pd.DataFrame(
        {
            "open": c.shift(1).fillna(c.iloc[0]),
            "high": c + spread / 2,
            "low": c - spread / 2,
            "close": c,
            "volume": volume,
        }
    )


def _trend(n: int, step: float) -> pd.DataFrame:
    return _bars([100 + step * i for i in range(n)])


class TestFrame:
    def test_constant_price_has_flat_ema_and_neutral_rsi(self) -> None:
        f = technical_frame(_bars([50.0] * 80))
        assert f["ema_fast"].iloc[-1] == pytest.approx(50.0)
        assert f["ema_slow"].iloc[-1] == pytest.approx(50.0)
        assert f["rsi"].iloc[-1] == pytest.approx(50.0)

    def test_rising_series_has_high_rsi_and_fast_above_slow(self) -> None:
        f = technical_frame(_trend(80, 1.0))
        assert f["rsi"].iloc[-1] > 95
        assert f["ema_fast"].iloc[-1] > f["ema_slow"].iloc[-1]

    def test_falling_series_has_low_rsi(self) -> None:
        assert technical_frame(_trend(80, -0.5))["rsi"].iloc[-1] < 5

    def test_atr_equals_range_for_constant_range_bars(self) -> None:
        f = technical_frame(_bars([50.0] * 80, spread=2.0))
        assert f["atr"].iloc[-1] == pytest.approx(2.0)

    def test_volume_ratio_is_relative_to_recent_average(self) -> None:
        bars = _bars([50.0] * 80)
        bars.loc[79, "volume"] = 3000.0
        assert technical_frame(bars)["vol_ratio"].iloc[-1] > 2.0

    def test_input_is_not_mutated(self) -> None:
        bars = _trend(80, 1.0)
        before = bars.copy(deep=True)
        technical_frame(bars)
        pd.testing.assert_frame_equal(bars, before)

    def test_is_causal_future_bars_never_change_past_values(self) -> None:
        rng = np.random.default_rng(7)
        close = list(100 + np.cumsum(rng.normal(0, 1, 200)))
        full = technical_frame(_bars(close))
        part = technical_frame(_bars(close[:120]))
        pd.testing.assert_frame_equal(
            full.iloc[:120].reset_index(drop=True), part.reset_index(drop=True)
        )

    def test_missing_columns_raise(self) -> None:
        with pytest.raises(ValueError, match="close"):
            technical_frame(pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "volume": [1]}))


class TestStateAt:
    def test_none_before_enough_history(self) -> None:
        f = technical_frame(_trend(80, 1.0))
        assert state_at(f, MIN_BARS - 2) is None

    def test_returns_state_when_warm(self) -> None:
        f = technical_frame(_trend(80, 1.0))
        s = state_at(f, 79)
        assert isinstance(s, TechnicalState)
        assert s.close == pytest.approx(179.0)
        assert s.atr > 0

    def test_state_is_immutable(self) -> None:
        s = state_at(technical_frame(_trend(80, 1.0)), 79)
        with pytest.raises(AttributeError):
            s.rsi = 1.0  # type: ignore[misc]


class TestTechScore:
    def _state(self, **kw: float) -> TechnicalState:
        base = dict(close=100.0, ema_fast=101.0, ema_slow=100.0, rsi=55.0, atr=1.0, vol_ratio=1.0)
        return TechnicalState(**{**base, **kw})

    def test_uptrend_positive_downtrend_negative(self) -> None:
        assert tech_score(self._state()) > 0
        assert tech_score(self._state(ema_fast=99.0)) < 0

    def test_is_bounded(self) -> None:
        assert tech_score(self._state(ema_fast=500.0)) == 1.0
        assert tech_score(self._state(ema_fast=-500.0)) == -1.0

    def test_zero_atr_gives_zero(self) -> None:
        assert tech_score(self._state(atr=0.0)) == 0.0

    def test_overbought_dampens_long_score(self) -> None:
        normal = tech_score(self._state(rsi=55))
        hot = tech_score(self._state(rsi=85))
        assert 0 < hot < normal

    def test_oversold_dampens_short_score(self) -> None:
        normal = tech_score(self._state(ema_fast=99.0, rsi=45))
        washed = tech_score(self._state(ema_fast=99.0, rsi=15))
        assert normal < washed < 0
