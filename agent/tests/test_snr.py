"""Malaysian SnR: levels from closes, freshness by wick vs body, and the pre-registered entry/exit rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signals.snr import find_levels, levels_asof, snr_signal


def frame(rows) -> pd.DataFrame:
    """rows: (open, high, low, close)."""
    idx = pd.bdate_range("2024-01-01", periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"], dtype=float)


def line(closes, half=0.5):
    """A chart whose bars open half-way between the previous and current close, with a +-half wick range."""
    rows, prev = [], closes[0]
    for c in closes:
        o = (prev + c) / 2
        rows.append((o, max(o, c) + half, min(o, c) - half, c))
        prev = c
    return frame(rows)


def test_peaks_and_valleys_of_the_close_line_become_a_and_v_levels():
    levels = find_levels(line([10, 12, 11, 13, 9, 10]))
    kinds = {(lv.kind, lv.price) for lv in levels if lv.kind in ("A", "V")}
    assert kinds == {("A", 12.0), ("A", 13.0), ("V", 11.0), ("V", 9.0)}
    a = next(lv for lv in levels if lv.kind == "A" and lv.price == 12.0)
    assert a.formed == 1 and a.usable_from == 3            # confirmed by the next close, usable one bar later


def test_gap_level_needs_two_same_colour_candles_with_a_gap():
    up = frame([(10, 11.2, 9.8, 11), (12, 13.2, 11.8, 13), (13, 13.5, 12.5, 13.2)])     # up, gap up, up
    gaps = [lv for lv in find_levels(up) if lv.kind.startswith("GAP")]
    assert [(g.kind, g.price) for g in gaps] == [("GAP_UP", 11.0)]
    mixed = frame([(10, 11.2, 9.8, 11), (12, 12.2, 10.8, 11), (11, 11.5, 10.5, 11.2)])  # second candle is down
    assert not [lv for lv in find_levels(mixed) if lv.kind.startswith("GAP")]
    touching = frame([(10, 11.2, 9.8, 11), (11, 12.2, 10.8, 12)])                        # no gap: open == previous close
    assert not [lv for lv in find_levels(touching) if lv.kind.startswith("GAP")]
    down = frame([(11, 11.2, 9.8, 10), (9, 9.5, 8.0, 8.5)])
    assert [(g.kind, g.price) for g in find_levels(down) if g.kind.startswith("GAP")] == [("GAP_DOWN", 10.0)]


def test_wick_touch_makes_a_level_unfresh_and_a_body_cross_makes_it_fresh_again():
    base_rows = [(105, 105.5, 103.5, 104), (104, 104.5, 101.5, 102), (102, 102.5, 99.8, 100),   # valley close 100
                 (101, 102.5, 100.6, 102), (102, 104.5, 101.8, 104), (104, 105.5, 103.8, 105), (105, 106.5, 104.8, 106)]
    v = lambda df: next(l for l in levels_asof(df, max_age=250) if l["kind"] == "V" and l["price"] == 100.0)
    assert v(frame(base_rows))["fresh"] is True
    wick = base_rows + [(105, 105.5, 99.8, 105.2)]                     # wick dips through 100, body stays above
    assert v(frame(wick))["fresh"] is False
    crossed = wick + [(101, 101.5, 98.0, 99.0)]                        # body (101 -> 99) crosses 100
    assert v(frame(crossed))["fresh"] is True


def _retest_chart(wick_touch_early=False, tail=None):
    """Rising chart with a valley close at 103 (bar 12), then a bullish retest of it at bar 26."""
    closes = [100 + 0.3 * i for i in range(12)] + [103.0] + [104.5 + 0.5 * k for k in range(13)]   # bars 0..25
    rows = [tuple(r) for r in line(closes, half=0.5).itertuples(index=False)]
    if wick_touch_early:
        c18 = rows[18][3]
        rows[18] = (c18 + 0.5, c18 + 1.0, 102.9, c18)                 # bearish bar: a wick reaches the level, body stays above
    rows.append((104.0, 105.0, 103.4, 104.5))                         # bar 26: touches, holds, closes up
    rows += tail or []
    return frame(rows)


KW = dict(trend_window=None, touch_atr=1.0, stop_atr=0.5, target_atr=3.0, max_hold=40, max_age=250)


def test_fresh_support_retest_enters_at_the_bar_close():
    sig = snr_signal(_retest_chart(), **KW)
    assert sig.iloc[26] == 1 and sig.iloc[:26].sum() == 0


def test_an_unfresh_level_is_not_traded():
    assert snr_signal(_retest_chart(wick_touch_early=True), **KW).sum() == 0


def test_trend_filter_blocks_entries_below_the_average():
    falling = frame([(200 - i, 200.5 - i, 199 - i, 199.5 - i) for i in range(30)])
    assert snr_signal(falling, **{**KW, "trend_window": 3}).sum() == 0


def test_stop_target_and_timeout_exits():
    stop = snr_signal(_retest_chart(tail=[(104.4, 104.6, 100.5, 101.0)]), **KW)
    assert stop.iloc[26] == 1 and stop.iloc[27] == 0
    target = snr_signal(_retest_chart(tail=[(104.6, 121.0, 104.5, 120.0)]), **KW)
    assert target.iloc[26] == 1 and target.iloc[27] == 0
    drift = [(104.5, 105.0, 104.0, 104.5)] * 8
    timeout = snr_signal(_retest_chart(tail=drift), **{**KW, "max_hold": 5})
    assert timeout.iloc[26] == 1 and timeout.iloc[31] == 0 and timeout.iloc[28] == 1


def test_future_bars_cannot_change_earlier_signals():
    rng = np.random.default_rng(4)
    closes = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 400))
    base = line(list(closes), half=0.6)
    shocked = base.copy()
    shocked.iloc[-30:] = shocked.iloc[-30:] * 1.6
    a = snr_signal(base, **{**KW, "trend_window": 20})
    b = snr_signal(shocked, **{**KW, "trend_window": 20})
    pd.testing.assert_series_equal(a.iloc[:-40], b.iloc[:-40])


def test_inputs_are_not_mutated_and_output_aligns():
    df = _retest_chart()
    before = df.copy()
    sig = snr_signal(df, **KW)
    pd.testing.assert_frame_equal(df, before)
    assert sig.index.equals(df.index) and set(sig.unique()) <= {0, 1}
