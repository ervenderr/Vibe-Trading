"""Tests for `vibe-trading execute`."""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timedelta, timezone

from src.execution.cli import run_execute_command
from src.execution.models import ExecutionPlan, ExecutionResult
from src.signals.service import SignalReport
from src.signals.verdict import Action, Verdict

NOW = datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc)


def _report(symbol: str) -> SignalReport:
    v = Verdict(Action.BUY, 0.8, ("r",), NOW + timedelta(minutes=15), 0.5, 0.5, entry=100.0, stop=99.0, target=101.5)
    return SignalReport(symbol, NOW, v, (), NOW, False, 0)


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def execute(self, report, *, now=None, dry_run=True):
        self.calls.append((report.symbol, dry_run))
        plan = ExecutionPlan(report.symbol, "buy", 100, 100.0, 99.0, 101.5, "vt-x", 0.8)
        status = "dry_run" if dry_run else "submitted"
        return ExecutionResult(status, ("buy 100 @ 100",), plan, None if dry_run else {"order_id": "o-1"})


def _args(**kw) -> Namespace:
    return Namespace(**{**dict(symbols=["NVDA"], execute=False), **kw})


class TestExecuteCommand:
    def test_default_is_dry_run_and_says_so(self) -> None:
        ex, out = FakeExecutor(), []
        code = run_execute_command(_args(), runner=lambda s, **k: _report(s), executor=ex, out=out.append)
        assert code == 0 and ex.calls == [("NVDA", True)]
        text = "\n".join(out)
        assert "DRY RUN" in text and "no order was sent" in text.lower()

    def test_execute_flag_submits_to_paper_and_labels_it(self) -> None:
        ex, out = FakeExecutor(), []
        run_execute_command(_args(execute=True), runner=lambda s, **k: _report(s), executor=ex, out=out.append)
        assert ex.calls == [("NVDA", False)]
        assert "PAPER" in "\n".join(out) and "submitted" in "\n".join(out).lower()

    def test_multiple_symbols_each_get_a_result(self) -> None:
        ex, out = FakeExecutor(), []
        run_execute_command(_args(symbols=["nvda", "tsla"]), runner=lambda s, **k: _report(s), executor=ex, out=out.append)
        assert [c[0] for c in ex.calls] == ["NVDA", "TSLA"]

    def test_signal_failure_is_reported_and_does_not_stop_others(self) -> None:
        def runner(s, **k):
            if s == "BAD":
                raise RuntimeError("down")
            return _report(s)

        ex, out = FakeExecutor(), []
        code = run_execute_command(_args(symbols=["BAD", "NVDA"]), runner=runner, executor=ex, out=out.append)
        assert code == 1 and [c[0] for c in ex.calls] == ["NVDA"] and any("BAD" in l and "failed" in l for l in out)

    def test_invalid_symbol_never_reaches_the_runner_or_executor(self) -> None:
        ex, out, called = FakeExecutor(), [], []
        code = run_execute_command(_args(symbols=["../x"]), runner=lambda s, **k: called.append(s), executor=ex, out=out.append)
        assert code == 1 and called == [] and ex.calls == []

    def test_refusal_and_skip_reasons_are_shown(self) -> None:
        class Refuser:
            def execute(self, report, *, now=None, dry_run=True):
                return ExecutionResult("refused", ("kill switch (HALT) is set",))

        out = []
        code = run_execute_command(_args(), runner=lambda s, **k: _report(s), executor=Refuser(), out=out.append)
        assert code == 1 and any("kill switch" in l for l in out)
