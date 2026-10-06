"""Review round 3: store integrity and locking, executor gates, autopilot resilience."""

from __future__ import annotations

import os
import stat
import sys
import threading
import time
from argparse import Namespace
from datetime import datetime, timedelta, timezone

import pytest

from src.execution.autopilot import Autopilot
from src.execution.broker import BrokerError
from src.execution.executor import PaperExecutor
from src.execution.models import AccountState, ExecutionLimits, ExecutionPlan, ExecutionResult, MarketClock
from src.execution.store import ExecutionStore, StoreError
from src.signals.service import SignalReport
from src.signals.verdict import Action, Verdict

NOW = datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc)


def _plan(symbol="NVDA", cid="vt-NVDA-b-1") -> ExecutionPlan:
    return ExecutionPlan(symbol, "buy", 1, 100.0, 99.0, 101.5, cid, 0.8)


class TestStoreIntegrity:
    def test_corrupt_state_fails_closed_instead_of_resetting_the_limits(self, tmp_path) -> None:
        (tmp_path / "state.json").write_text("{not json")
        with pytest.raises(StoreError):
            ExecutionStore(tmp_path).day_state(NOW, 100_000.0)

    def test_missing_state_is_a_normal_fresh_start(self, tmp_path) -> None:
        assert ExecutionStore(tmp_path).day_state(NOW, 100_000.0).trades == 0

    def test_baseline_from_the_broker_overrides_the_first_look_equity(self, tmp_path) -> None:
        d = ExecutionStore(tmp_path).day_state(NOW, 97_000.0, baseline=100_000.0)
        assert d.start_equity == 100_000.0

    def test_unusable_baseline_falls_back_to_current_equity(self, tmp_path) -> None:
        assert ExecutionStore(tmp_path).day_state(NOW, 97_000.0, baseline=float("nan")).start_equity == 97_000.0
        assert ExecutionStore(tmp_path / "b").day_state(NOW, 97_000.0, baseline=-5.0).start_equity == 97_000.0

    def test_managed_symbols_and_last_submission_come_from_the_log(self, tmp_path) -> None:
        s = ExecutionStore(tmp_path)
        s.day_state(NOW, 100_000.0)
        s.record(_plan("NVDA", "a"), {"order_id": "1"}, NOW)
        s.record(_plan("TSLA", "b"), {"order_id": "2"}, NOW + timedelta(minutes=5))
        assert s.managed_symbols() == {"NVDA", "TSLA"}
        assert s.last_submission("NVDA") == NOW and s.last_submission("AAPL") is None

    def test_transaction_is_reentrant_and_serialises_threads(self, tmp_path) -> None:
        s = ExecutionStore(tmp_path)
        counter = {"n": 0}

        def work() -> None:
            with s.transaction():
                with s.transaction():  # re-entering must not deadlock
                    v = counter["n"]
                    time.sleep(0.01)
                    counter["n"] = v + 1

        threads = [threading.Thread(target=work) for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert counter["n"] == 6

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
    def test_lock_file_is_private(self, tmp_path) -> None:
        s = ExecutionStore(tmp_path / "x")
        with s.transaction():
            pass
        assert stat.S_IMODE(os.stat(tmp_path / "x" / "store.lock").st_mode) == 0o600


def _report(symbol="NVDA", at=NOW, action=Action.BUY) -> SignalReport:
    v = Verdict(action, 0.8, ("r",), at + timedelta(minutes=15), 0.5, 0.5, entry=100.0, stop=99.0, target=101.5)
    return SignalReport(symbol, at, v, (), at, False, 0)


class GateBroker:
    is_paper = True

    def __init__(self, *, open_=True, minutes_to_close=120.0, clock_error=False, price=100.0, price_error=False,
                 last_equity=None, equity=100_000.0, submit_delay=0.0) -> None:
        self.open_, self.mtc, self.clock_error = open_, minutes_to_close, clock_error
        self.price, self.price_error, self.last_equity, self.equity = price, price_error, last_equity, equity
        self.submitted: list[ExecutionPlan] = []
        self.submit_delay = submit_delay

    def market_clock(self) -> MarketClock:
        if self.clock_error:
            raise BrokerError("down")
        return MarketClock(self.open_, NOW, NOW + timedelta(hours=16), NOW + timedelta(minutes=self.mtc))

    def last_price(self, symbol, side) -> float:
        if self.price_error:
            raise BrokerError("no quote")
        return self.price

    def account(self):
        return AccountState(self.equity, 400_000.0, False, self.last_equity)

    def positions(self):
        return []

    def open_orders(self):
        return []

    def submit_bracket(self, plan):
        time.sleep(self.submit_delay)
        self.submitted.append(plan)
        return {"status": "ok", "order_id": f"o-{len(self.submitted)}", "order_status": "accepted"}


def _ex(tmp_path, broker=None, *, halt=lambda: False, limits=None):
    broker = broker or GateBroker()
    return PaperExecutor(broker, ExecutionStore(tmp_path), limits or ExecutionLimits(), halt_check=halt), broker


class TestExecutorGates:
    def test_closed_market_blocks_real_orders_but_not_dry_runs(self, tmp_path) -> None:
        ex, broker = _ex(tmp_path, GateBroker(open_=False))
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "skipped" and "market" in " ".join(r.reasons).lower() and broker.submitted == []
        assert ex.execute(_report(), now=NOW, dry_run=True).status == "dry_run"

    def test_no_new_entries_near_the_close(self, tmp_path) -> None:
        ex, broker = _ex(tmp_path, GateBroker(minutes_to_close=10))
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "skipped" and "close" in " ".join(r.reasons).lower() and broker.submitted == []

    def test_unreadable_clock_blocks_real_orders(self, tmp_path) -> None:
        ex, broker = _ex(tmp_path, GateBroker(clock_error=True))
        assert ex.execute(_report(), now=NOW, dry_run=False).status == "skipped" and broker.submitted == []

    def test_kill_switch_is_rechecked_right_before_submission(self, tmp_path) -> None:
        calls = {"n": 0}

        def halt() -> bool:
            calls["n"] += 1
            return calls["n"] >= 2  # clear at the start, tripped mid-flight

        ex, broker = _ex(tmp_path, halt=halt)
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "refused" and broker.submitted == []

    def test_plan_is_reanchored_to_the_live_price(self, tmp_path) -> None:
        ex, _ = _ex(tmp_path, GateBroker(price=100.5))
        r = ex.execute(_report(), now=NOW, dry_run=True)
        assert r.plan.entry == 100.5 and r.plan.stop == pytest.approx(99.5)

    def test_market_far_from_the_signal_is_refused(self, tmp_path) -> None:
        ex, broker = _ex(tmp_path, GateBroker(price=110.0))
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "skipped" and "moved" in " ".join(r.reasons) and broker.submitted == []

    def test_missing_live_price_fails_closed(self, tmp_path) -> None:
        ex, broker = _ex(tmp_path, GateBroker(price_error=True))
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "skipped" and "price" in " ".join(r.reasons).lower() and broker.submitted == []

    def test_non_actionable_verdicts_do_not_need_a_quote(self, tmp_path) -> None:
        ex, _ = _ex(tmp_path, GateBroker(price_error=True))
        v = Verdict(Action.WAIT, 0.0, ("x",), NOW, 0.0, 0.0)
        assert ex.execute(SignalReport("NVDA", NOW, v, (), NOW, False, 0), now=NOW).status == "skipped"

    def test_cooldown_blocks_a_quick_reentry_then_allows_it(self, tmp_path) -> None:
        ex, broker = _ex(tmp_path)
        ex.execute(_report(at=NOW), now=NOW, dry_run=False)
        t1 = NOW + timedelta(minutes=30)
        r = ex.execute(_report(at=t1), now=t1, dry_run=False)
        assert r.status == "skipped" and "cooldown" in " ".join(r.reasons).lower()
        t2 = NOW + timedelta(minutes=70)
        assert ex.execute(_report(at=t2), now=t2, dry_run=False).status == "submitted" and len(broker.submitted) == 2

    def test_a_logging_failure_after_a_successful_order_does_not_hide_the_order(self, tmp_path, monkeypatch) -> None:
        ex, broker = _ex(tmp_path)
        monkeypatch.setattr(ExecutionStore, "record", lambda self, *a, **k: (_ for _ in ()).throw(OSError("disk full")))
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "submitted" and len(broker.submitted) == 1
        assert any("could not be logged" in x for x in r.reasons)

    def test_corrupt_state_stops_trading(self, tmp_path) -> None:
        (tmp_path / "state.json").write_text("garbage")
        ex, broker = _ex(tmp_path)
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "skipped" and "state" in " ".join(r.reasons).lower() and broker.submitted == []

    def test_daily_loss_uses_the_brokers_previous_close_as_baseline(self, tmp_path) -> None:
        # first look of the day happens AFTER a 3% loss; baseline must still be yesterday's close
        ex, broker = _ex(tmp_path, GateBroker(equity=97_000.0, last_equity=100_000.0))
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "skipped" and "loss" in " ".join(r.reasons) and broker.submitted == []

    def test_concurrent_executors_cannot_both_submit_the_same_signal(self, tmp_path) -> None:
        broker = GateBroker(submit_delay=0.05)
        results: list[ExecutionResult] = []

        def run() -> None:
            ex = PaperExecutor(broker, ExecutionStore(tmp_path), ExecutionLimits(), halt_check=lambda: False)
            results.append(ex.execute(_report(), now=NOW, dry_run=False))

        threads = [threading.Thread(target=run) for _ in range(4)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(broker.submitted) == 1 and sorted(r.status for r in results).count("submitted") == 1


T = datetime(2026, 10, 2, 17, 0, tzinfo=timezone.utc)


class PilotBroker:
    def __init__(self, *, is_open=True, minutes_to_close=120.0, flatten_status="ok", exposure=(), clock_boom=False) -> None:
        self.is_open, self.mtc, self.flatten_status = is_open, minutes_to_close, flatten_status
        self._exposure, self.clock_boom = list(exposure), clock_boom
        self.flatten_calls: list[set] = []

    def market_clock(self):
        if self.clock_boom:
            raise RuntimeError("unexpected")
        return MarketClock(self.is_open, T, T + timedelta(hours=16), T + timedelta(minutes=self.mtc))

    def flatten(self, symbols):
        self.flatten_calls.append(set(symbols))
        return {"status": self.flatten_status, "cancelled": 0, "closed": 0, "error": "x", "remaining": ["NVDA (position)"]}

    def exposure(self, symbols):
        return list(self._exposure)


class QuietExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, report, *, now=None, dry_run=True):
        self.calls += 1
        return ExecutionResult("skipped", ("quiet",))


def _pilot(broker, *, dry_run=False, interval=60.0, halted=False, out=None, managed=("NVDA", "TSLA")):
    sleeps: list[float] = []
    p = Autopilot(["NVDA"], runner=lambda s, **k: _report(s), executor=QuietExecutor(), broker=broker,
                  dry_run=dry_run, interval_s=interval, flatten_minutes=10.0, halt_check=lambda: halted,
                  now_fn=lambda: T, sleep=sleeps.append, out=(out.append if out is not None else (lambda _: None)),
                  managed_symbols=lambda: set(managed))
    p.sleeps = sleeps  # type: ignore[attr-defined]
    return p


class TestAutopilotNeverMissesTheFlattenWindow:
    def test_sleep_is_clamped_so_a_wakeup_lands_inside_the_window(self) -> None:
        o = _pilot(PilotBroker(minutes_to_close=25), interval=1200.0).cycle()
        assert o.sleep_s <= (25 - 10) * 60 + 1  # next wake-up at or before the flatten window opens

    def test_just_before_the_window_the_sleep_is_short(self) -> None:
        assert _pilot(PilotBroker(minutes_to_close=11), interval=900.0).cycle().sleep_s <= 61

    def test_flatten_targets_only_the_symbols_this_system_manages(self) -> None:
        b = PilotBroker(minutes_to_close=5)
        p = _pilot(b, managed=("AAPL", "NVDA"))
        p.cycle()
        assert b.flatten_calls == [{"AAPL", "NVDA"}]

    def test_leftover_exposure_after_the_close_triggers_one_flatten_per_day(self) -> None:
        b = PilotBroker(is_open=False, exposure=["NVDA (position)"])
        p = _pilot(b)
        p.cycle()
        p.cycle()
        assert len(b.flatten_calls) == 1

    def test_closed_market_dry_run_or_no_exposure_never_flattens(self) -> None:
        b1, b2 = PilotBroker(is_open=False, exposure=["NVDA (position)"]), PilotBroker(is_open=False)
        _pilot(b1, dry_run=True).cycle()
        _pilot(b2).cycle()
        assert b1.flatten_calls == [] and b2.flatten_calls == []


class TestAutopilotResilience:
    def test_unexpected_errors_do_not_kill_the_loop(self) -> None:
        out: list[str] = []
        p = _pilot(PilotBroker(clock_boom=True), out=out)
        assert p.run(max_cycles=2) == 1 or p.run(max_cycles=2) == 0
        assert any("error" in l.lower() for l in out)

    def test_repeated_failures_attempt_a_flatten_and_exit_nonzero(self) -> None:
        b = PilotBroker(clock_boom=True)
        p = _pilot(b)
        assert p.run(max_cycles=10) == 1 and b.flatten_calls  # gave up after consecutive failures, flattened

    def test_kill_switch_warns_about_open_exposure_without_trading(self) -> None:
        out: list[str] = []
        b = PilotBroker(exposure=["NVDA (position)"])
        o = _pilot(b, halted=True, out=out).cycle()
        assert o.status == "halted" and b.flatten_calls == []
        assert any("NVDA (position)" in l for l in out) and any("still open" in l.lower() for l in out)


class TestAutopilotCliInterval:
    def test_interval_above_the_cap_is_rejected(self) -> None:
        from src.execution.autopilot_cli import MAX_INTERVAL_S, run_autopilot_command

        out: list[str] = []
        args = Namespace(symbols=["NVDA"], execute=False, interval=MAX_INTERVAL_S + 1, once=False, report=False, since=None)

        class B:
            is_paper = True

        assert run_autopilot_command(args, broker=B(), executor=object(), runner=lambda s, **k: None, pilot_cls=object, out=out.append) == 2
        assert MAX_INTERVAL_S <= 300
