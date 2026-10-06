"""`vibe-trading autopilot` and its --report track record."""

from __future__ import annotations

from argparse import Namespace
from datetime import date, datetime, timedelta, timezone

from src.execution.autopilot_cli import run_autopilot_command
from src.execution.models import AccountState
from src.execution.report import Fill

T = datetime(2026, 10, 2, 14, 0, tzinfo=timezone.utc)


def _args(**kw) -> Namespace:
    return Namespace(**{**dict(symbols=["NVDA"], execute=False, interval=120.0, once=False, report=False, since=None), **kw})


class FakeBroker:
    is_paper = True

    def __init__(self, fills=()) -> None:
        self._fills = tuple(fills)
        self.since = None

    def fills(self, since):
        self.since = since
        return self._fills

    def account(self):
        return AccountState(99_990.0, 400_000.0, False)


class FakePilot:
    instances: list = []

    def __init__(self, symbols, **kw) -> None:
        self.symbols, self.kw, self.ran = symbols, kw, None
        FakePilot.instances.append(self)

    def run(self, max_cycles=None) -> int:
        self.ran = max_cycles
        return 0


def _factory(broker=None):
    FakePilot.instances = []
    return dict(broker=broker or FakeBroker(), executor=object(), runner=lambda s, **k: None, pilot_cls=FakePilot)


class TestRun:
    def test_default_is_dry_run_and_announces_it(self) -> None:
        out: list[str] = []
        code = run_autopilot_command(_args(), out=out.append, **_factory())
        pilot = FakePilot.instances[0]
        assert code == 0 and pilot.kw["dry_run"] is True and "DRY RUN" in "\n".join(out)

    def test_execute_flag_means_paper_orders_and_is_announced(self) -> None:
        out: list[str] = []
        run_autopilot_command(_args(execute=True), out=out.append, **_factory())
        assert FakePilot.instances[0].kw["dry_run"] is False and "PAPER" in "\n".join(out)

    def test_once_runs_a_single_cycle(self) -> None:
        run_autopilot_command(_args(once=True), out=lambda _: None, **_factory())
        assert FakePilot.instances[0].ran == 1

    def test_continuous_mode_has_no_cycle_limit(self) -> None:
        run_autopilot_command(_args(), out=lambda _: None, **_factory())
        assert FakePilot.instances[0].ran is None

    def test_needs_symbols(self) -> None:
        out: list[str] = []
        assert run_autopilot_command(_args(symbols=[]), out=out.append, **_factory()) == 2 and FakePilot.instances == []

    def test_invalid_symbol_is_rejected(self) -> None:
        out: list[str] = []
        assert run_autopilot_command(_args(symbols=["../x"]), out=out.append, **_factory()) == 2

    def test_interval_below_the_floor_or_non_finite_is_rejected(self) -> None:
        for bad in (5.0, float("nan"), float("inf")):
            out: list[str] = []
            assert run_autopilot_command(_args(interval=bad), out=out.append, **_factory()) == 2

    def test_live_account_is_refused(self) -> None:
        b = FakeBroker()
        b.is_paper = False
        out: list[str] = []
        assert run_autopilot_command(_args(), out=out.append, **_factory(b)) == 1 and FakePilot.instances == []


class TestReport:
    def _fills(self):
        return [Fill(T, "AAPL", "buy", 10, 100.0), Fill(T + timedelta(minutes=5), "AAPL", "sell", 10, 101.0),
                Fill(T + timedelta(minutes=10), "TSLA", "buy", 5, 200.0), Fill(T + timedelta(minutes=15), "TSLA", "sell", 5, 198.0)]

    def test_prints_the_record_and_every_promotion_check(self) -> None:
        out: list[str] = []
        code = run_autopilot_command(_args(report=True, symbols=[]), out=out.append, **_factory(FakeBroker(self._fills())))
        text = "\n".join(out).lower()
        assert code == 0 and "closed trades" in text and "profit factor" in text and "win rate" in text
        assert "not eligible" in text and "manual" in text

    def test_since_date_is_forwarded_and_defaults_to_30_days(self) -> None:
        b = FakeBroker()
        run_autopilot_command(_args(report=True, symbols=[], since=date(2026, 9, 1)), out=lambda _: None, **_factory(b))
        assert b.since.date() == date(2026, 9, 1)
        b2 = FakeBroker()
        run_autopilot_command(_args(report=True, symbols=[]), out=lambda _: None, **_factory(b2))
        assert (datetime.now(timezone.utc) - b2.since).days in (29, 30)

    def test_empty_history_is_explained_not_an_error(self) -> None:
        out: list[str] = []
        assert run_autopilot_command(_args(report=True, symbols=[]), out=out.append, **_factory()) == 0
        assert any("no closed trades" in l.lower() for l in out)
