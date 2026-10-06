"""Event study: does Jev-judged news predict returns, and at which horizon?

Honesty rules baked in:
  * the entry is the first bar that *starts at or after* the publication time;
  * returns are market-adjusted (stock minus SPY over the identical window);
  * overlapping events per symbol are thinned so windows are not double counted;
  * significance uses a sign-flip permutation test (no distributional assumptions);
  * the caller splits events chronologically so the best horizon is chosen out of sample.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

EXCHANGE_TZ = "America/New_York"
SESSION_OPEN = pd.Timedelta(hours=9, minutes=30)
PERMUTATIONS = 20_000
#: Longest wait for the next tradable bar (covers long weekends); anything later is a data gap.
MAX_ENTRY_DELAY = timedelta(days=4)
_CHUNK = 2_000


@dataclass(frozen=True)
class Horizon:
    label: str
    series: str  # "5m" | "1h" | "1d": which bar series is used
    n_bars: int  # holding period in bars
    same_session: bool  # intraday windows must not cross the session boundary
    gap: timedelta  # minimum spacing between kept events of one symbol


DEFAULT_HORIZONS: dict[str, Horizon] = {
    "30m": Horizon("30m", "5m", 6, True, timedelta(minutes=30)),
    "1h": Horizon("1h", "1h", 1, True, timedelta(hours=1)),
    "2h": Horizon("2h", "1h", 2, True, timedelta(hours=2)),
    "4h": Horizon("4h", "1h", 4, True, timedelta(hours=4)),
    "1d": Horizon("1d", "1d", 1, False, timedelta(days=1)),
    "3d": Horizon("3d", "1d", 3, False, timedelta(days=3)),
    "5d": Horizon("5d", "1d", 5, False, timedelta(days=5)),
}


@dataclass(frozen=True)
class Event:
    symbol: str
    published: datetime  # timezone-aware
    side: int  # +1 expects a rise, -1 a fall (from Jev's sentiment)
    event_type: str
    strength: float


@dataclass(frozen=True)
class ReturnResult:
    entry_time: pd.Timestamp
    stock_return: float
    market_return: float
    abnormal: float


@dataclass(frozen=True)
class HorizonStats:
    label: str
    n: int
    hit_rate: float  # share of events where the market-adjusted move went the predicted way
    mean_abnormal_bps: float
    median_abnormal_bps: float
    t_stat: float
    p_value: float  # sign-flip permutation, two-sided
    mean_raw_net_bps: float  # unhedged signed return after round-trip cost


def mentions_company(text: str, aliases: Iterable[str]) -> bool:
    """True if any alias appears as a whole word (case-insensitive)."""
    return any(re.search(rf"\b{re.escape(a)}\b", text, re.IGNORECASE) for a in aliases)


def localize_daily(bars: pd.DataFrame) -> pd.DataFrame:
    """Stamp daily bars at the New York session open, in UTC.

    Then 'first bar starting at or after publication' means: pre-market news trades the
    same day's open, news during or after the session trades the next open.
    """
    index = pd.DatetimeIndex(bars.index).normalize() + SESSION_OPEN
    return bars.set_axis(index.tz_localize(EXCHANGE_TZ).tz_convert("UTC"), axis=0)


def _ny_date(ts: pd.Timestamp):
    return ts.tz_convert(EXCHANGE_TZ).date()


def abnormal_return(
    stock: pd.DataFrame, market: pd.DataFrame, published: datetime, horizon: Horizon
) -> ReturnResult | None:
    """Market-adjusted return from the first bar at/after ``published``; None if unusable."""
    if stock.empty or market.empty:
        return None
    entry_i = int(stock.index.searchsorted(pd.Timestamp(published), side="left"))
    exit_i = entry_i + horizon.n_bars - 1
    if exit_i >= len(stock):
        return None
    entry_time, exit_time = stock.index[entry_i], stock.index[exit_i]
    if entry_time - pd.Timestamp(published) > MAX_ENTRY_DELAY:
        return None  # e.g. published before the series starts: never enter at its first bar
    if horizon.same_session and _ny_date(entry_time) != _ny_date(exit_time):
        return None
    m_entry = market.index.get_indexer([entry_time])[0]
    if m_entry < 0 or m_entry + horizon.n_bars - 1 >= len(market):
        return None
    m_exit = m_entry + horizon.n_bars - 1
    if market.index[m_exit] != exit_time:
        return None
    s_open, s_close = float(stock["open"].iloc[entry_i]), float(stock["close"].iloc[exit_i])
    m_open, m_close = float(market["open"].iloc[m_entry]), float(market["close"].iloc[m_exit])
    if not (s_open > 0 and m_open > 0):
        return None
    stock_ret, market_ret = s_close / s_open - 1.0, m_close / m_open - 1.0
    return ReturnResult(entry_time, stock_ret, market_ret, stock_ret - market_ret)


def thin_events(events: Iterable[Event], gap: timedelta) -> list[Event]:
    """Drop events that start within ``gap`` of the previous kept event of the same symbol."""
    kept: list[Event] = []
    last: dict[str, datetime] = {}
    for event in sorted(events, key=lambda e: e.published):
        prev = last.get(event.symbol)
        if prev is None or event.published - prev >= gap:
            kept.append(event)
            last[event.symbol] = event.published
    return kept


def split_by_time(events: Sequence[Event], train_fraction: float) -> tuple[list[Event], list[Event]]:
    ordered = sorted(events, key=lambda e: e.published)
    cut = int(len(ordered) * train_fraction)
    return ordered[:cut], ordered[cut:]


def sign_flip_pvalue(values: np.ndarray, *, n_iter: int = PERMUTATIONS, seed: int = 0) -> float:
    """Two-sided p-value for 'mean is zero' by randomly flipping each value's sign.

    If the events carried no information, the predicted direction would be independent
    of the outcome and flipping signs would not change the distribution of the mean.
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(x.mean())
    extreme, done = 0, 0
    while done < n_iter:
        batch = min(_CHUNK, n_iter - done)
        signs = rng.choice((-1.0, 1.0), size=(batch, x.size))
        extreme += int((np.abs((signs * x).mean(axis=1)) >= observed - 1e-15).sum())
        done += batch
    return (extreme + 1) / (n_iter + 1)


def summarize(
    label: str, abnormal: np.ndarray, raw: np.ndarray, *, cost_bps: float, seed: int
) -> HorizonStats:
    ab, rw = np.asarray(abnormal, dtype=float), np.asarray(raw, dtype=float)
    if ab.size == 0:
        return HorizonStats(label, 0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    std = ab.std(ddof=1) if ab.size > 1 else 0.0
    return HorizonStats(
        label=label,
        n=int(ab.size),
        hit_rate=float((ab > 0).mean()),
        mean_abnormal_bps=float(ab.mean() * 1e4),
        median_abnormal_bps=float(np.median(ab) * 1e4),
        t_stat=float(ab.mean() / (std / np.sqrt(ab.size))) if std > 0 else 0.0,
        p_value=sign_flip_pvalue(ab, seed=seed),
        mean_raw_net_bps=float(rw.mean() * 1e4 - cost_bps),
    )


def run_study(
    events: Sequence[Event],
    stock_bars: Mapping[str, pd.DataFrame],
    market_bars: pd.DataFrame,
    horizons: Mapping[str, Horizon],
    *,
    cost_bps: float = 6.0,
    seed: int = 0,
) -> dict[str, HorizonStats]:
    """Per-horizon statistics over the (thinned) events. Unusable events are skipped."""
    results: dict[str, HorizonStats] = {}
    for label, horizon in horizons.items():
        signed_abnormal, signed_raw = [], []
        for event in thin_events(events, horizon.gap):
            bars = stock_bars.get(event.symbol)
            if bars is None:
                continue
            r = abnormal_return(bars, market_bars, event.published, horizon)
            if r is None:
                continue
            signed_abnormal.append(event.side * r.abnormal)
            signed_raw.append(event.side * r.stock_return)
        results[label] = summarize(
            label, np.array(signed_abnormal), np.array(signed_raw), cost_bps=cost_bps, seed=seed
        )
    return results
