"""Tests for the event-study core: timing, thinning, statistics, honesty checks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.signals.event_study import (
    DEFAULT_HORIZONS,
    Event,
    Horizon,
    abnormal_return,
    localize_daily,
    mentions_company,
    run_study,
    sign_flip_pvalue,
    split_by_time,
    summarize,
    thin_events,
)

UTC = timezone.utc


def _hourly(day: str, n: int = 7, price: float = 100.0, drift: float = 0.0) -> pd.DataFrame:
    """Hourly bars starting 13:30 UTC (09:30 NY) like yfinance 1h bars."""
    start = pd.Timestamp(f"{day} 13:30", tz="UTC")
    idx = pd.DatetimeIndex([start + timedelta(hours=i) for i in range(n)])
    close = pd.Series([price * (1 + drift * (i + 1)) for i in range(n)], index=idx)
    return pd.DataFrame({"open": close.shift(1).fillna(price), "high": close, "low": close, "close": close, "volume": 1.0})


H1 = Horizon("1h", "1h", 1, True, timedelta(hours=1))
H4 = Horizon("4h", "1h", 4, True, timedelta(hours=4))


class TestMentionsCompany:
    def test_matches_ticker_or_alias_case_insensitively(self) -> None:
        assert mentions_company("Nvidia beats estimates", ("nvda", "nvidia"))
        assert mentions_company("NVDA stock jumps", ("nvda", "nvidia"))

    def test_ticker_must_be_a_whole_word(self) -> None:
        assert not mentions_company("Candidate Anvda report", ("nvda",))

    def test_unrelated_listicle_does_not_match(self) -> None:
        assert not mentions_company("The 3 Best Stocks to Buy in October", ("nvda", "nvidia"))


class TestAbnormalReturn:
    def test_entry_is_first_bar_at_or_after_publication(self) -> None:
        bars = _hourly("2026-09-30")
        pub = pd.Timestamp("2026-09-30 14:10", tz="UTC").to_pydatetime()  # between 13:30 and 14:30 bars
        r = abnormal_return(bars, bars, pub, H1)
        assert r is not None and r.entry_time == pd.Timestamp("2026-09-30 14:30", tz="UTC")

    def test_bar_that_started_before_publication_is_never_used(self) -> None:
        bars = _hourly("2026-09-30")
        bars.iloc[1, bars.columns.get_loc("open")] = 1.0  # the 14:30 bar's open is wildly different
        pub = pd.Timestamp("2026-09-30 14:45", tz="UTC").to_pydatetime()  # after 14:30 began
        r = abnormal_return(bars, bars, pub, H1)
        assert r.entry_time == pd.Timestamp("2026-09-30 15:30", tz="UTC")

    def test_exit_is_close_of_the_nth_bar(self) -> None:
        bars = _hourly("2026-09-30", drift=0.01)
        pub = pd.Timestamp("2026-09-30 13:00", tz="UTC").to_pydatetime()
        r = abnormal_return(bars, _hourly("2026-09-30", drift=0.0), pub, H4)
        # stock close after 4 bars = 100*1.04, entry open = 100 -> +4%; market flat
        assert r.stock_return == pytest.approx(0.04, abs=1e-9) and r.abnormal == pytest.approx(0.04, abs=1e-9)

    def test_market_return_is_subtracted_over_the_same_window(self) -> None:
        stock, mkt = _hourly("2026-09-30", drift=0.01), _hourly("2026-09-30", drift=0.004)
        pub = pd.Timestamp("2026-09-30 13:00", tz="UTC").to_pydatetime()
        r = abnormal_return(stock, mkt, pub, H4)
        assert r.abnormal == pytest.approx(0.04 - 0.016, abs=1e-9)

    def test_intraday_window_crossing_the_session_is_dropped(self) -> None:
        bars = pd.concat([_hourly("2026-09-30"), _hourly("2026-10-01")])
        pub = pd.Timestamp("2026-09-30 18:00", tz="UTC").to_pydatetime()  # entry 18:30; 4 bars -> next day
        assert abnormal_return(bars, bars, pub, H4) is None

    def test_after_close_news_enters_at_next_session_open(self) -> None:
        bars = pd.concat([_hourly("2026-09-30"), _hourly("2026-10-01")])
        pub = pd.Timestamp("2026-09-30 22:00", tz="UTC").to_pydatetime()
        assert abnormal_return(bars, bars, pub, H1).entry_time == pd.Timestamp("2026-10-01 13:30", tz="UTC")

    def test_not_enough_future_bars_returns_none(self) -> None:
        bars = _hourly("2026-09-30", n=3)
        pub = pd.Timestamp("2026-09-30 13:00", tz="UTC").to_pydatetime()
        assert abnormal_return(bars, bars, pub, H4) is None

    def test_market_missing_the_entry_bar_returns_none(self) -> None:
        stock = _hourly("2026-09-30")
        assert abnormal_return(stock, stock.iloc[3:], pd.Timestamp("2026-09-30 13:00", tz="UTC").to_pydatetime(), H1) is None


class TestLocalizeDaily:
    def test_daily_bars_are_stamped_at_the_session_open_in_utc(self) -> None:
        df = pd.DataFrame({"open": [1.0], "close": [1.0]}, index=pd.DatetimeIndex(["2026-09-30"]))
        assert localize_daily(df).index[0] == pd.Timestamp("2026-09-30 13:30", tz="UTC")  # 09:30 EDT

    def test_premarket_news_trades_same_day_but_intraday_news_trades_next_day(self) -> None:
        df = localize_daily(pd.DataFrame({"open": [1.0, 2.0, 3.0], "close": [1.5, 2.5, 3.5]},
                                         index=pd.DatetimeIndex(["2026-09-29", "2026-09-30", "2026-10-01"])))
        h = Horizon("1d", "1d", 1, False, timedelta(days=1))
        pre = abnormal_return(df, df, pd.Timestamp("2026-09-30 12:00", tz="UTC").to_pydatetime(), h)
        mid = abnormal_return(df, df, pd.Timestamp("2026-09-30 15:00", tz="UTC").to_pydatetime(), h)
        assert pre.entry_time.day == 30 and mid.entry_time.day == 1


class TestThinning:
    def _ev(self, minutes: int, symbol: str = "NVDA") -> Event:
        return Event(symbol, datetime(2026, 9, 30, 14, 0, tzinfo=UTC) + timedelta(minutes=minutes), 1, "earnings", 0.9)

    def test_overlapping_events_for_one_symbol_are_dropped(self) -> None:
        kept = thin_events([self._ev(0), self._ev(30), self._ev(90)], timedelta(hours=1))
        assert [e.published for e in kept] == [self._ev(0).published, self._ev(90).published]

    def test_symbols_are_thinned_independently(self) -> None:
        kept = thin_events([self._ev(0, "NVDA"), self._ev(10, "TSLA")], timedelta(hours=1))
        assert len(kept) == 2

    def test_order_independent(self) -> None:
        a = thin_events([self._ev(90), self._ev(0), self._ev(30)], timedelta(hours=1))
        assert [e.published for e in a] == sorted(e.published for e in a)


class TestSignFlip:
    def test_clear_positive_effect_has_tiny_p(self) -> None:
        rng = np.random.default_rng(1)
        assert sign_flip_pvalue(rng.normal(0.01, 0.01, 300), seed=0) < 0.001

    def test_pure_noise_is_not_significant(self) -> None:
        rng = np.random.default_rng(2)
        assert sign_flip_pvalue(rng.normal(0.0, 0.01, 300), seed=0) > 0.05

    def test_deterministic_for_a_seed_and_handles_empty(self) -> None:
        x = np.array([0.01, -0.02, 0.03, 0.01])
        assert sign_flip_pvalue(x, seed=5) == sign_flip_pvalue(x, seed=5)
        assert sign_flip_pvalue(np.array([]), seed=0) == 1.0


class TestSummarize:
    def test_core_numbers(self) -> None:
        s = summarize("1h", np.array([0.01, 0.02, -0.01, 0.03]), np.array([0.011, 0.021, -0.009, 0.031]), cost_bps=6.0, seed=0)
        assert s.n == 4 and s.hit_rate == pytest.approx(0.75)
        assert s.mean_abnormal_bps == pytest.approx(125.0)  # (1+2-1+3)/4 = 1.25%
        assert s.mean_raw_net_bps == pytest.approx(135.0 - 6.0)  # raw mean 1.35% minus 6 bps cost

    def test_empty_is_safe(self) -> None:
        s = summarize("1h", np.array([]), np.array([]), cost_bps=6.0, seed=0)
        assert s.n == 0 and s.p_value == 1.0 and s.hit_rate == 0.0


class TestSplit:
    def test_chronological_not_random(self) -> None:
        evs = [Event("A", datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i), 1, "x", 1.0) for i in range(10)]
        train, test = split_by_time(evs, 0.6)
        assert len(train) == 6 and max(e.published for e in train) < min(e.published for e in test)


def _market(days: int, drift_by_hour: dict[int, float] | None = None, seed: int = 0, noise: float = 0.0015):
    """Many sessions of hourly bars with optional planted post-event drift handled by the test."""
    rng = np.random.default_rng(seed)
    frames = []
    for d in range(days):
        day = (datetime(2026, 3, 2, tzinfo=UTC) + timedelta(days=d))
        if day.weekday() >= 5:
            continue
        frames.append(_hourly(day.strftime("%Y-%m-%d"), price=100.0))
    base = pd.concat(frames)
    ret = pd.Series(rng.normal(0, noise, len(base)), index=base.index)
    close = 100.0 * (1 + ret).cumprod()
    out = pd.DataFrame({"open": close.shift(1).fillna(100.0), "high": close, "low": close, "close": close, "volume": 1.0})
    return out


class TestRunStudy:
    def _events(self, bars: pd.DataFrame, n: int = 120, side: int = 1) -> list[Event]:
        times = bars.index[bars.index.hour == 14][:: max(1, len(bars) // (n * 7))][:n]  # 14:30 UTC bars
        return [Event("SYM", (t - timedelta(minutes=20)).to_pydatetime(), side, "earnings", 0.9) for t in times]

    def test_planted_one_hour_drift_is_detected_at_1h(self) -> None:
        mkt = _market(160, seed=1)
        stock = mkt.copy()
        events = self._events(stock)
        # plant: after each event entry, the next bar's close is +60 bps (a real drift) in the stock only
        for e in events:
            entry = pd.Timestamp(e.published + timedelta(minutes=20))
            i = stock.index.get_loc(entry)
            stock.iloc[i, stock.columns.get_loc("close")] *= 1.006
        results = run_study(events, {"SYM": stock}, mkt, {"1h": H1}, cost_bps=6.0, seed=0)
        s = results["1h"]
        assert s.n >= 50 and s.hit_rate > 0.8 and s.p_value < 0.001

    def test_pure_noise_has_no_significant_horizon(self) -> None:
        mkt = _market(160, seed=3)
        stock = _market(160, seed=4)  # independent noise: no information in the events
        events = self._events(stock)
        results = run_study(events, {"SYM": stock}, mkt, {"1h": H1, "4h": H4}, cost_bps=6.0, seed=0)
        assert all(s.p_value > 0.01 for s in results.values())

    def test_negative_side_events_profit_from_a_fall(self) -> None:
        mkt = _market(160, seed=5)
        stock = mkt.copy()
        events = self._events(stock, side=-1)
        for e in events:
            i = stock.index.get_loc(pd.Timestamp(e.published + timedelta(minutes=20)))
            stock.iloc[i, stock.columns.get_loc("close")] *= 0.994
        s = run_study(events, {"SYM": stock}, mkt, {"1h": H1}, cost_bps=6.0, seed=0)["1h"]
        assert s.hit_rate > 0.8 and s.mean_abnormal_bps > 30

    def test_missing_symbol_bars_skip_events_without_crashing(self) -> None:
        mkt = _market(30)
        s = run_study([Event("NOPE", datetime(2026, 3, 5, 14, tzinfo=UTC), 1, "x", 1.0)], {}, mkt, {"1h": H1}, cost_bps=6.0, seed=0)["1h"]
        assert s.n == 0


class TestDefaults:
    def test_horizons_are_fixed_up_front_and_cover_the_requested_range(self) -> None:
        assert list(DEFAULT_HORIZONS) == ["30m", "1h", "2h", "4h", "1d", "3d", "5d"]
        assert DEFAULT_HORIZONS["30m"].series == "5m" and DEFAULT_HORIZONS["1d"].series == "1d"
        assert all(h.same_session for k, h in DEFAULT_HORIZONS.items() if h.series != "1d")


class TestEntryMustFollowPublicationClosely:
    def test_event_published_long_before_the_bars_begin_is_dropped_not_entered_at_the_first_bar(self) -> None:
        bars = _hourly("2026-09-30")
        months_earlier = datetime(2026, 6, 1, 14, 0, tzinfo=UTC)
        assert abnormal_return(bars, bars, months_earlier, H1) is None

    def test_long_weekend_gap_is_still_allowed(self) -> None:
        # Friday 21:00 UTC news, bars resume Tuesday after a Monday holiday (< 4 days)
        fri = _hourly("2026-09-25")
        tue = _hourly("2026-09-29")
        bars = pd.concat([fri, tue])
        pub = datetime(2026, 9, 25, 21, 0, tzinfo=UTC)
        r = abnormal_return(bars, bars, pub, H1)
        assert r is not None and r.entry_time == pd.Timestamp("2026-09-29 13:30", tz="UTC")

    def test_gap_in_the_data_longer_than_the_limit_is_dropped(self) -> None:
        bars = pd.concat([_hourly("2026-09-01"), _hourly("2026-09-30")])
        assert abnormal_return(bars, bars, datetime(2026, 9, 1, 21, 0, tzinfo=UTC), H1) is None
