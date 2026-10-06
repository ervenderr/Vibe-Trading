"""Regression tests for the second batch of review findings."""

from __future__ import annotations

import json
import math
import os
import stat
import sys
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from typesafe_sdk import TypeSafeError

from src.signals.jev_analyzer import JevNewsAnalyzer, JudgmentCache
from src.signals.ledger import Grade, LedgerEntry, SignalLedger, grade_entry, summarize_grades
from src.signals.news_source import NewsItem
from src.signals.replay import ReplayConfig, simulate
from src.signals.report_format import report_to_dict
from src.signals.scalp_eval import exchange_tz_for, localize_bars
from src.signals.service import generate_signal
from src.signals.technicals import MIN_BARS, TechnicalState
from src.signals.verdict import Action, Verdict

NOW = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- timezone (H3)
class TestExchangeTimezone:
    @pytest.mark.parametrize("symbol,tz", [
        ("BTC-USD", "UTC"), ("ETH-USDT", "UTC"), ("DOGE-USD", "UTC"),
        ("NVDA", "America/New_York"), ("BRK.B", "America/New_York"), ("AAPL", "America/New_York"),
    ])
    def test_supported_symbols(self, symbol: str, tz: str) -> None:
        assert exchange_tz_for(symbol) == tz

    @pytest.mark.parametrize("symbol", ["0700.HK", "RELIANCE.NS", "EURUSD=X", "600519.SH", "^GSPC"])
    def test_unsupported_markets_are_refused_not_guessed(self, symbol: str) -> None:
        assert exchange_tz_for(symbol) is None

    def test_crypto_bars_are_kept_in_utc(self) -> None:
        df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2026-09-29 00:00:00"]))
        out = localize_bars(df, "UTC")
        assert out.index[0] == pd.Timestamp("2026-09-29 00:00:00", tz="UTC")

    def test_dst_ambiguous_and_nonexistent_times_are_dropped_not_fatal(self) -> None:
        idx = pd.DatetimeIndex(["2026-11-01 01:30:00", "2026-03-08 02:30:00", "2026-09-30 10:00:00"])
        out = localize_bars(pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx))
        assert list(out["close"]) == [3.0] and out.index.notna().all()


# --------------------------------------------------- Jev degradation + partial bar
def _bars(n: int = 80, end: datetime = NOW, slope: float = 0.3) -> pd.DataFrame:
    idx = pd.DatetimeIndex([end - timedelta(minutes=5 * (n - i)) for i in range(n)])
    close = pd.Series([100 + slope * i for i in range(n)], index=idx)
    return pd.DataFrame({"open": close.shift(1).fillna(close.iloc[0]), "high": close + 0.3, "low": close - 0.3,
                         "close": close, "volume": 1000.0})


def _item() -> NewsItem:
    return NewsItem("NVDA", "NVIDIA surges record beat upgraded", "s", NOW - timedelta(minutes=3), "src", "u", "1")


class Down:
    def system_one(self, *a, **k):
        raise TypeSafeError("down")


class TestJevDegradation:
    def test_fallback_news_is_excluded_and_reported(self, tmp_path) -> None:
        analyzer = JevNewsAnalyzer(Down(), model="m", cache=JudgmentCache(tmp_path))
        report = generate_signal("NVDA", news_source=lambda s: [_item()], analyzer=analyzer,
                                 bars_source=lambda s: _bars(), now=NOW)
        assert report.fallback_count == 1
        assert report.verdict.news_score == 0.0  # keyword score must not steer the verdict
        d = report_to_dict(report)
        assert any("jev" in w.lower() for w in d["warnings"])


class TestPartialBar:
    def test_still_forming_last_bar_is_not_used(self, tmp_path) -> None:
        bars = _bars()  # last completed bar ends exactly at NOW
        forming = bars.iloc[[-1]].copy()
        forming.index = pd.DatetimeIndex([NOW - timedelta(minutes=2)])  # started 2 min ago, 3 left
        forming["close"] = 1.0  # absurd partial print that would flip the trend
        both = pd.concat([bars, forming])
        analyzer = JevNewsAnalyzer(SimpleNamespace(), model="m", cache=JudgmentCache(tmp_path))
        report = generate_signal("NVDA", news_source=lambda s: [], analyzer=analyzer,
                                 bars_source=lambda s: both, now=NOW)
        assert report.bar_time == bars.index[-1].to_pydatetime()


# ------------------------------------------------------------ replay gap fills (M3)
START = datetime(2026, 9, 30, 14, 0, tzinfo=timezone.utc)
FREE = ReplayConfig(cost_bps=0.0, slippage_bps=0.0, cooldown_bars=0, max_hold_bars=10)
SIGNAL_BAR = MIN_BARS + 5


def _flat(n: int = 80, price: float = 100.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex([START + timedelta(minutes=5 * i) for i in range(n)])
    return pd.DataFrame({"open": price, "high": price + 0.5, "low": price - 0.5, "close": price, "volume": 1000.0}, index=idx)


def _once(action: Action):
    calls = {"n": 0}

    def fn(state: TechnicalState, now: datetime) -> Verdict:
        i = calls["n"] + MIN_BARS - 1
        calls["n"] += 1
        if i != SIGNAL_BAR:
            return Verdict(Action.WAIT, 0.0, ("x",), now, 0.0, 0.0)
        side = 1.0 if action is Action.BUY else -1.0
        return Verdict(action, 0.8, ("s",), now, 0.5, 0.5, entry=state.close,
                       stop=state.close - side, target=state.close + 1.5 * side)

    return fn


class TestGapFills:
    def test_gap_down_through_long_stop_fills_at_the_open_not_the_stop(self) -> None:
        bars = _flat()
        j = bars.columns.get_loc
        bars.iloc[SIGNAL_BAR + 3, [j("open"), j("high"), j("low"), j("close")]] = [95.0, 95.5, 94.5, 95.0]
        (t,) = simulate(bars, _once(Action.BUY), FREE)
        assert t.exit_reason == "stop" and t.exit == pytest.approx(95.0)

    def test_gap_up_through_long_target_fills_at_the_open(self) -> None:
        bars = _flat()
        j = bars.columns.get_loc
        bars.iloc[SIGNAL_BAR + 3, [j("open"), j("high"), j("low"), j("close")]] = [105.0, 105.5, 104.5, 105.0]
        (t,) = simulate(bars, _once(Action.BUY), FREE)
        assert t.exit_reason == "target" and t.exit == pytest.approx(105.0)

    def test_gap_up_through_short_stop_fills_at_the_open(self) -> None:
        bars = _flat()
        j = bars.columns.get_loc
        bars.iloc[SIGNAL_BAR + 3, [j("open"), j("high"), j("low"), j("close")]] = [105.0, 105.5, 104.5, 105.0]
        (t,) = simulate(bars, _once(Action.SELL), FREE)
        assert t.exit_reason == "stop" and t.exit == pytest.approx(105.0)


# ------------------------------------------------------------------ ledger (M4, L9)
T0 = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)


def _entry(**kw) -> LedgerEntry:
    base = dict(symbol="NVDA", action="BUY", confidence=0.7, generated_at=T0,
                valid_until=T0 + timedelta(minutes=15), entry=100.0, stop=99.0, target=101.5)
    return LedgerEntry(**{**base, **kw})


def _after(opens, highs, lows, closes=None) -> pd.DataFrame:
    idx = pd.DatetimeIndex([T0 + timedelta(minutes=5 * (i + 1)) for i in range(len(highs))])
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes or opens, "volume": 1.0}, index=idx)


class TestLedgerGrading:
    def test_entry_is_the_next_bar_open_with_distances_preserved(self) -> None:
        # signal entry 100 / target +1.5, but the next bar opens at 100.8: the real target is 102.3
        miss = grade_entry(_entry(), _after([100.8, 100.9], [101.0, 101.8], [100.5, 100.6]))
        assert miss.outcome == "expired" and miss.gross_return < 0.01  # 101.8 < 102.3, not a win
        hit = grade_entry(_entry(), _after([100.8, 100.9], [101.0, 102.5], [100.5, 100.6]))
        assert hit.outcome == "target" and hit.gross_return == pytest.approx(1.5 / 100.8)

    def test_gap_down_through_stop_is_charged_at_the_gap_open(self) -> None:
        # enters at 100.0 (stop 99.0); next bar gaps open at 96.0
        g = grade_entry(_entry(), _after([100.0, 96.0], [100.2, 96.5], [99.8, 95.5]))
        assert g.outcome == "stop" and g.gross_return == pytest.approx((96.0 - 100.0) / 100.0)

    def test_expired_when_now_is_past_validity_even_if_session_closed_early(self) -> None:
        bars = _after([100.0], [100.4], [99.6], [100.2])  # only one bar, window not covered
        g = grade_entry(_entry(), bars, now=T0 + timedelta(hours=3))
        assert g.outcome == "expired"

    def test_still_pending_while_window_open_and_now_is_inside_it(self) -> None:
        bars = _after([100.0], [100.4], [99.6])
        assert grade_entry(_entry(), bars, now=T0 + timedelta(minutes=8)).outcome == "pending"

    def test_no_data_long_after_validity_is_nodata_not_pending_forever(self) -> None:
        g = grade_entry(_entry(), _after([], [], []), now=T0 + timedelta(days=3))
        assert g.outcome == "nodata"

    def test_nodata_is_excluded_from_win_rate(self) -> None:
        s = summarize_grades([Grade("target", 0.01), Grade("nodata", 0.0)])
        assert s.resolved == 1 and s.pending == 0 and s.win_rate == 1.0

    def test_zero_entry_price_cannot_divide_by_zero(self) -> None:
        g = grade_entry(_entry(entry=0.0), _after([0.0], [1.0], [0.0]))
        assert g.outcome in {"pending", "nodata"}


class TestLedgerFile:
    def _report(self):
        from src.signals.service import SignalReport

        v = Verdict(Action.BUY, 0.7, ("r",), T0 + timedelta(minutes=15), 0.5, 0.5, entry=100.0, stop=99.0, target=101.5)
        return SignalReport("NVDA", T0, v, (), T0, False, 0)

    def test_naive_datetime_or_bad_values_in_file_are_skipped_not_fatal(self, tmp_path) -> None:
        path = tmp_path / "l.jsonl"
        row = dict(symbol="NVDA", action="BUY", confidence=0.7, generated_at="2026-10-01T15:00:00",
                   valid_until="2026-10-01T15:15:00", entry=100.0, stop=99.0, target=101.5)
        bad_entry = {**row, "generated_at": "2026-10-01T15:00:00+00:00", "valid_until": "2026-10-01T15:15:00+00:00", "entry": 0}
        path.write_text(json.dumps(row) + "\n" + json.dumps(bad_entry) + "\n")
        ledger = SignalLedger(path)
        assert ledger.load() == ()
        assert ledger.record(self._report()) is True  # duplicate check must not crash on bad rows

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
    def test_files_and_dirs_are_private(self, tmp_path) -> None:
        path = tmp_path / "sub" / "l.jsonl"
        SignalLedger(path).record(self._report())
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700

    def test_concurrent_records_do_not_duplicate(self, tmp_path) -> None:
        import threading

        ledger = SignalLedger(tmp_path / "l.jsonl")
        threads = [threading.Thread(target=lambda: ledger.record(self._report())) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(ledger.load()) == 1


# ------------------------------------------------------------- CLI / tool (L10)
class TestCliAndToolLimits:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -5.0])
    def test_non_finite_or_negative_watch_interval_is_rejected(self, bad, tmp_path) -> None:
        from src.signals.cli import run_signal_command

        out: list[str] = []
        code = run_signal_command(Namespace(symbols=["NVDA"], watch=bad, history=False, json=False),
                                  runner=lambda s, **k: None, ledger=SignalLedger(tmp_path / "l"), out=out.append,
                                  sleep=lambda s: (_ for _ in ()).throw(AssertionError("must not sleep")))
        assert code == 2 and any("watch" in line.lower() for line in out)

    def test_tool_serves_repeat_calls_from_a_short_cache(self, monkeypatch, tmp_path) -> None:
        from src.tools import signal_tool
        from src.signals.service import SignalReport

        calls = {"n": 0}
        v = Verdict(Action.WAIT, 0.0, ("r",), T0, 0.0, 0.0)

        def fake(symbol, **kw):
            calls["n"] += 1
            return SignalReport(symbol, T0, v, (), None, False, 0)

        monkeypatch.setattr(signal_tool, "run_signal", fake)
        monkeypatch.setattr(signal_tool, "DEFAULT_LEDGER", tmp_path / "l")
        signal_tool.clear_signal_cache()
        tool = signal_tool.TradeSignalTool()
        tool.execute(symbol="NVDA")
        tool.execute(symbol="nvda")
        assert calls["n"] == 1
        signal_tool.clear_signal_cache()

    def test_tool_output_marks_external_text_as_untrusted(self, monkeypatch, tmp_path) -> None:
        from src.tools import signal_tool
        from src.signals.service import SignalReport

        v = Verdict(Action.WAIT, 0.0, ("r",), T0, 0.0, 0.0)
        monkeypatch.setattr(signal_tool, "run_signal", lambda s, **k: SignalReport(s, T0, v, (), None, False, 0))
        monkeypatch.setattr(signal_tool, "DEFAULT_LEDGER", tmp_path / "l")
        signal_tool.clear_signal_cache()
        out = json.loads(signal_tool.TradeSignalTool().execute(symbol="NVDA"))
        assert "untrusted" in out["note"].lower()
        signal_tool.clear_signal_cache()
