"""Tests for the `vibe-trading signal` command."""

from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.signals.cli import MIN_WATCH_SECONDS, run_signal_command
from src.signals.ledger import SignalLedger
from src.signals.service import SignalReport
from src.signals.verdict import Action, Verdict

T0 = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)


def _report(symbol: str = "NVDA", action: Action = Action.BUY) -> SignalReport:
    v = Verdict(action, 0.7, ("r",), T0 + timedelta(minutes=15), 0.5, 0.5, entry=100.0, stop=99.0, target=101.5)
    return SignalReport(symbol, T0, v, (), T0, False, 0)


def _args(**kw) -> Namespace:
    return Namespace(**{**dict(symbols=["NVDA"], watch=None, history=False, json=False), **kw})


class TestOneShot:
    def test_prints_each_symbol(self, tmp_path) -> None:
        out: list[str] = []
        code = run_signal_command(_args(symbols=["nvda", "tsla"]), runner=lambda s, **k: _report(s),
                                  ledger=SignalLedger(tmp_path / "l"), out=out.append)
        text = "\n".join(out)
        assert code == 0 and "NVDA" in text and "TSLA" in text

    def test_json_mode_emits_parseable_json_per_symbol(self, tmp_path) -> None:
        out: list[str] = []
        run_signal_command(_args(json=True), runner=lambda s, **k: _report(s),
                           ledger=SignalLedger(tmp_path / "l"), out=out.append)
        assert json.loads(out[0])["action"] == "BUY"

    def test_records_actionable_signals(self, tmp_path) -> None:
        ledger = SignalLedger(tmp_path / "l")
        run_signal_command(_args(), runner=lambda s, **k: _report(s), ledger=ledger, out=lambda _: None)
        assert len(ledger.load()) == 1

    def test_one_failing_symbol_does_not_stop_the_rest(self, tmp_path) -> None:
        def runner(s, **k):
            if s == "BAD":
                raise RuntimeError("x")
            return _report(s)

        out: list[str] = []
        code = run_signal_command(_args(symbols=["BAD", "NVDA"]), runner=runner,
                                  ledger=SignalLedger(tmp_path / "l"), out=out.append)
        assert code == 1 and any("NVDA" in line for line in out)
        assert any("BAD" in line and "failed" in line for line in out)

    def test_invalid_symbol_reported_without_calling_runner(self, tmp_path) -> None:
        called: list[str] = []
        out: list[str] = []
        code = run_signal_command(_args(symbols=["../x"]), runner=lambda s, **k: called.append(s),
                                  ledger=SignalLedger(tmp_path / "l"), out=out.append)
        assert code == 1 and called == []


class TestWatch:
    def test_repeats_until_interrupted_and_sleeps_between(self, tmp_path) -> None:
        sleeps: list[float] = []
        calls = {"n": 0}

        def runner(s, **k):
            calls["n"] += 1
            return _report(s)

        def sleeper(seconds: float) -> None:
            sleeps.append(seconds)
            if len(sleeps) == 2:
                raise KeyboardInterrupt

        code = run_signal_command(_args(watch=60), runner=runner, ledger=SignalLedger(tmp_path / "l"),
                                  out=lambda _: None, sleep=sleeper)
        assert code == 0 and calls["n"] == 2 and sleeps == [60, 60]

    def test_interval_is_floored_to_respect_rate_limits(self, tmp_path) -> None:
        sleeps: list[float] = []

        def sleeper(seconds: float) -> None:
            sleeps.append(seconds)
            raise KeyboardInterrupt

        run_signal_command(_args(watch=1), runner=lambda s, **k: _report(s),
                           ledger=SignalLedger(tmp_path / "l"), out=lambda _: None, sleep=sleeper)
        assert sleeps == [MIN_WATCH_SECONDS]

    def test_unchanged_verdict_is_not_reprinted(self, tmp_path) -> None:
        out: list[str] = []
        sleeps = {"n": 0}

        def sleeper(seconds: float) -> None:
            sleeps["n"] += 1
            if sleeps["n"] == 3:
                raise KeyboardInterrupt

        run_signal_command(_args(watch=60), runner=lambda s, **k: _report(s),
                           ledger=SignalLedger(tmp_path / "l"), out=out.append, sleep=sleeper)
        assert sum("NVDA: BUY" in line for line in out) == 1


class TestHistory:
    def test_grades_ledger_entries_with_provided_bars(self, tmp_path) -> None:
        ledger = SignalLedger(tmp_path / "l")
        ledger.record(_report())
        idx = pd.DatetimeIndex([T0 + timedelta(minutes=5 * (i + 1)) for i in range(2)])
        bars = pd.DataFrame({"open": 100.0, "high": [100.5, 102.0], "low": [99.6, 100.0], "close": 100.0, "volume": 1.0}, index=idx)
        out: list[str] = []
        code = run_signal_command(_args(history=True), runner=None, ledger=ledger, out=out.append,
                                  bars_fetcher=lambda s: bars)
        text = "\n".join(out)
        assert code == 0 and "target" in text and "win rate" in text.lower()

    def test_empty_ledger_says_so(self, tmp_path) -> None:
        out: list[str] = []
        run_signal_command(_args(history=True), runner=None, ledger=SignalLedger(tmp_path / "l"),
                           out=out.append, bars_fetcher=lambda s: pd.DataFrame())
        assert "no signals" in " ".join(out).lower()
