"""Causal technical indicators and a bounded trend score for intraday bars.

Every column at row ``i`` depends only on bars ``0..i``; a test asserts this,
because lookahead bias is what makes backtests lie.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
FAST_SPAN = 9
SLOW_SPAN = 21
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_WINDOW = 20
#: Rows needed before every indicator is meaningful.
MIN_BARS = SLOW_SPAN + ATR_PERIOD
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0


@dataclass(frozen=True)
class TechnicalState:
    """Indicator values at a single bar."""

    close: float
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float
    vol_ratio: float


def _wilder(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _rsi(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0.0).fillna(0.0), RSI_PERIOD)
    loss = _wilder((-delta).clip(lower=0.0).fillna(0.0), RSI_PERIOD)
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0)
    return rsi.mask((loss == 0.0) & (gain == 0.0), 50.0)


def _atr(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return _wilder(true_range, ATR_PERIOD)


def technical_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Indicator frame aligned to ``bars`` (a new frame; input is untouched)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {', '.join(missing)}")
    close = bars["close"].astype(float)
    avg_volume = bars["volume"].astype(float).shift(1).rolling(VOLUME_WINDOW, min_periods=5).mean()
    return pd.DataFrame(
        {
            "close": close,
            "ema_fast": close.ewm(span=FAST_SPAN, adjust=False).mean(),
            "ema_slow": close.ewm(span=SLOW_SPAN, adjust=False).mean(),
            "rsi": _rsi(close),
            "atr": _atr(bars[list(REQUIRED_COLUMNS)].astype(float)),
            "vol_ratio": (bars["volume"] / avg_volume).replace([np.inf, -np.inf], np.nan).fillna(1.0),
        },
        index=bars.index,
    )


def state_at(frame: pd.DataFrame, i: int) -> TechnicalState | None:
    """Indicator state at row ``i``; ``None`` until enough history exists."""
    if i < MIN_BARS - 1 or i >= len(frame):
        return None
    row = frame.iloc[i]
    if row.isna().any():
        return None
    return TechnicalState(
        close=float(row["close"]),
        ema_fast=float(row["ema_fast"]),
        ema_slow=float(row["ema_slow"]),
        rsi=float(row["rsi"]),
        atr=float(row["atr"]),
        vol_ratio=float(row["vol_ratio"]),
    )


def trend_direction(state: TechnicalState) -> float:
    """Raw trend in ``[-1, 1]``: EMA spread in ATR units, not damped by RSI."""
    if state.atr <= 0.0:
        return 0.0
    return math.tanh((state.ema_fast - state.ema_slow) / state.atr)


def tech_score(state: TechnicalState) -> float:
    """Entry-sizing score in ``[-1, 1]``: the trend, damped at RSI extremes."""
    score = trend_direction(state)
    if score > 0 and state.rsi > RSI_OVERBOUGHT:
        score *= max(0.0, 1.0 - (state.rsi - RSI_OVERBOUGHT) / (100.0 - RSI_OVERBOUGHT))
    elif score < 0 and state.rsi < RSI_OVERSOLD:
        score *= max(0.0, 1.0 - (RSI_OVERSOLD - state.rsi) / RSI_OVERSOLD)
    return score
