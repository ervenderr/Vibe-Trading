"""Autopilot loop: market hours, end-of-day flatten, kill switch, fail-closed behaviour."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.execution.autopilot import Autopilot
from src.execution.broker import BrokerError
from src.execution.models import ExecutionPlan, ExecutionResult, MarketClock
from src.signals.service import SignalReport
from src.signals.verdict import Action, Verdict

T = datetime(2026, 10, 2, 17, 0, tzinfo=timezone.utc)


def _report(symbol: str) -> SignalReport:
    v = Verdict(Action.WAIT, 0.0, ("x",), T, 0.0, 0.0)
    return SignalReport(symbol, T, v, (), T, False, 0)


class FakeBroker:
    def __init__(self, *, is_open: bool = True, minutes_to_close: float = 120.0, clock_error: bool = False,
                 flatten_result: dict | None = None) -> None:
        self.is_open, self.minutes_to_close, self.clock_error = is_open, minutes_to_close, clock_error
        self.flatten_calls = 0
        self.flatten_result = flatten_result or {"status": "ok", "cancelled": 1, "closed": 1}

    def market_clock(self) -> MarketClock:
        if self.clock_error:
            raise BrokerError("down")
        return MarketClock(self.is_open, T, T + timedelta(hours=16), T + timedelta(minutes=self.minutes_to_close))

    def flatten(self, symbols) -> dict:
        self.flatten_calls += 1
        return self.flatten_result

    def exposure(self, symbols) -> list:
        return []


class FakeExecutor:
    def __init__(self, status: str = "skipped", fail_for: str | None = None) -> None:
        self.calls: list[tuple[str, bool]] = []
        self.status, self.fail_for = status, fail_for

    def execute(self, report, *, now=None, dry_run=True):
        if report.symbol == self.fail_for:
            raise RuntimeError("boom")
        self.calls.append((report.symbol, dry_run))
        plan = ExecutionPlan(report.symbol, "buy", 1, 100.0, 99.0, 101.5, "id", 0.8) if self.status == "submitted" else None
        return ExecutionResult(self.status, ("because",), plan, {"order_id": "o"} if self.status == "submitted" else None)


def _pilot(broker=None, executor=None, *, symbols=("NVDA", "TSLA"), dry_run=True, halted=False, runner=None, out=None):
    broker = broker or FakeBroker()
    executor = executor or FakeExecutor()
    seen: list[str] = []
    pilot = Autopilot(
        list(symbols), runner=runner or (lambda s, **k: (seen.append(s), _report(s))[1]), executor=executor,
        broker=broker, dry_run=dry_run, interval_s=60.0, flatten_minutes=10.0,
        halt_check=lambda: halted, now_fn=lambda: T, sleep=lambda s: None, out=(out.append if out is not None else (lambda _: None)),
        managed_symbols=lambda: {"NVDA", "TSLA"},
    )
    pilot.seen = seen  # type: ignore[attr-defined]
    return pilot, broker, executor


class TestMarketHours:
    def test_closed_market_does_nothing_and_waits_longer(self) -> None:
        pilot, _, ex = _pilot(FakeBroker(is_open=False))
        o = pilot.cycle()
        assert o.status == "closed" and o.sleep_s >= 300 and ex.calls == [] and pilot.seen == []

    def test_open_market_evaluates_every_symbol(self) -> None:
        pilot, _, ex = _pilot()
        o = pilot.cycle()
        assert o.status == "idle" and pilot.seen == ["NVDA", "TSLA"] and [c[0] for c in ex.calls] == ["NVDA", "TSLA"]

    def test_dry_run_flag_is_passed_through(self) -> None:
        pilot, _, ex = _pilot(dry_run=True)
        pilot.cycle()
        assert ex.calls and all(call[1] is True for call in ex.calls)
        pilot, _, ex = _pilot(dry_run=False)
        pilot.cycle()
        assert all(call[1] is False for call in ex.calls)

    def test_submission_is_reported_as_traded(self) -> None:
        out: list[str] = []
        pilot, _, _ = _pilot(executor=FakeExecutor("submitted"), dry_run=False, out=out)
        assert pilot.cycle().status == "traded" and any("submitted" in l.lower() for l in out)


class TestEndOfDayFlatten:
    def test_flattens_once_before_the_close_and_opens_nothing_new(self) -> None:
        pilot, broker, ex = _pilot(FakeBroker(minutes_to_close=5), dry_run=False)
        assert pilot.cycle().status == "flattened"
        assert pilot.cycle().status == "closing"  # second look the same day: no second flatten
        assert broker.flatten_calls == 1 and ex.calls == [] and pilot.seen == []

    def test_dry_run_never_flattens(self) -> None:
        pilot, broker, _ = _pilot(FakeBroker(minutes_to_close=5), dry_run=True)
        pilot.cycle()
        assert broker.flatten_calls == 0

    def test_failed_flatten_is_retried_next_cycle_and_reported(self) -> None:
        out: list[str] = []
        pilot, broker, _ = _pilot(FakeBroker(minutes_to_close=5, flatten_result={"status": "error", "error": "api down"}), dry_run=False, out=out)
        assert pilot.cycle().status == "flatten_failed"
        pilot.cycle()
        assert broker.flatten_calls == 2 and any("api down" in l for l in out)

    def test_outside_the_flatten_window_trading_continues(self) -> None:
        pilot, broker, ex = _pilot(FakeBroker(minutes_to_close=45), dry_run=False)
        pilot.cycle()
        assert broker.flatten_calls == 0 and len(ex.calls) == 2


class TestSafety:
    def test_kill_switch_stops_everything(self) -> None:
        pilot, broker, ex = _pilot(halted=True)
        o = pilot.cycle()
        assert o.status == "halted" and ex.calls == [] and pilot.seen == [] and broker.flatten_calls == 0

    def test_unreadable_clock_fails_closed(self) -> None:
        pilot, _, ex = _pilot(FakeBroker(clock_error=True))
        o = pilot.cycle()
        assert o.status == "no_clock" and ex.calls == [] and pilot.seen == []

    def test_one_failing_symbol_does_not_stop_the_others(self) -> None:
        pilot, _, ex = _pilot(executor=FakeExecutor(fail_for="NVDA"))
        pilot.cycle()
        assert [c[0] for c in ex.calls] == ["TSLA"]

    def test_runner_failure_is_isolated_too(self) -> None:
        def runner(s, **k):
            if s == "NVDA":
                raise RuntimeError("news down")
            return _report(s)

        pilot, _, ex = _pilot(runner=runner)
        pilot.cycle()
        assert [c[0] for c in ex.calls] == ["TSLA"]

    def test_invalid_symbols_are_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError):
            _pilot(symbols=("NVDA", "../x"))


class TestRun:
    def test_runs_the_requested_number_of_cycles_and_sleeps_between(self) -> None:
        sleeps: list[float] = []
        pilot, _, ex = _pilot()
        pilot._sleep = sleeps.append  # type: ignore[attr-defined]
        assert pilot.run(max_cycles=3) == 0
        assert len(sleeps) == 2 and len(ex.calls) == 6

    def test_stops_when_the_kill_switch_is_set(self) -> None:
        pilot, _, ex = _pilot(halted=True)
        assert pilot.run(max_cycles=5) == 0 and ex.calls == []

    def test_keyboard_interrupt_exits_cleanly(self) -> None:
        pilot, _, _ = _pilot()
        pilot._sleep = lambda s: (_ for _ in ()).throw(KeyboardInterrupt)  # type: ignore[attr-defined]
        assert pilot.run(max_cycles=5) == 0

    def test_repeated_identical_results_are_not_reprinted(self) -> None:
        out: list[str] = []
        pilot, _, _ = _pilot(out=out)
        pilot.run(max_cycles=3)
        assert sum("NVDA" in l for l in out) == 1
