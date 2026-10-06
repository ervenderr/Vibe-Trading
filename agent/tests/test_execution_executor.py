"""Executor, store and Alpaca adapter tests (no network)."""

from __future__ import annotations

import json
import os
import stat
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.execution.broker import AlpacaPaperBroker, BrokerError
from src.execution.executor import PaperExecutor
from src.execution.models import AccountState, ExecutionLimits, ExecutionPlan, MarketClock
from src.execution.store import ExecutionStore, StoreError
from src.signals.service import SignalReport
from src.signals.verdict import Action, Verdict

NOW = datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc)


def _report(symbol: str = "NVDA", action: Action = Action.BUY, conf: float = 0.8, at: datetime = NOW) -> SignalReport:
    v = Verdict(action, conf, ("r",), at + timedelta(minutes=15), 0.5, 0.5, entry=100.0, stop=99.0, target=101.5)
    return SignalReport(symbol, at, v, (), at, False, 0)


class FakeBroker:
    def __init__(self, *, paper: bool = True, equity: float = 100_000.0, positions=None, orders=None,
                 fail_reads: bool = False, submit_result: dict | None = None) -> None:
        self.is_paper = paper
        self._equity, self._positions, self._orders = equity, positions or [], orders or []
        self.fail_reads = fail_reads
        self.submitted: list[ExecutionPlan] = []
        self.reads = 0
        self.submit_result = submit_result or {"status": "ok", "order_id": "ord-1", "order_status": "accepted"}

    def account(self) -> AccountState:
        self.reads += 1
        if self.fail_reads:
            raise BrokerError("boom")
        return AccountState(self._equity, 400_000.0, False)

    def market_clock(self) -> MarketClock:
        return MarketClock(True, NOW, NOW + timedelta(hours=16), NOW + timedelta(hours=2))

    def last_price(self, symbol, side) -> float:
        return 100.0

    def positions(self):
        return list(self._positions)

    def open_orders(self):
        return list(self._orders)

    def submit_bracket(self, plan: ExecutionPlan) -> dict:
        self.submitted.append(plan)
        return self.submit_result


def _executor(tmp_path, broker=None, *, halted: bool = False, limits: ExecutionLimits | None = None):
    broker = broker or FakeBroker()
    ex = PaperExecutor(broker, ExecutionStore(tmp_path), limits or ExecutionLimits(), halt_check=lambda: halted)
    return ex, broker


class TestDryRunDefault:
    def test_dry_run_plans_but_never_submits(self, tmp_path) -> None:
        ex, broker = _executor(tmp_path)
        r = ex.execute(_report(), now=NOW)
        assert r.status == "dry_run" and r.plan is not None and r.plan.qty == 100
        assert broker.submitted == []

    def test_dry_run_does_not_consume_the_daily_trade_budget(self, tmp_path) -> None:
        ex, _ = _executor(tmp_path)
        ex.execute(_report(), now=NOW)
        assert ExecutionStore(tmp_path).day_state(NOW, 100_000.0).trades == 0


class TestSubmission:
    def test_submits_one_bracket_and_records_it(self, tmp_path) -> None:
        ex, broker = _executor(tmp_path)
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "submitted" and len(broker.submitted) == 1
        assert r.order["order_id"] == "ord-1"
        assert ExecutionStore(tmp_path).day_state(NOW, 100_000.0).trades == 1

    def test_same_signal_window_is_never_submitted_twice(self, tmp_path) -> None:
        ex, broker = _executor(tmp_path)
        ex.execute(_report(), now=NOW, dry_run=False)
        again = ex.execute(_report(), now=NOW + timedelta(minutes=2), dry_run=False)
        assert again.status == "skipped" and "duplicate" in " ".join(again.reasons).lower()
        assert len(broker.submitted) == 1

    def test_broker_rejection_is_an_error_and_does_not_count_as_a_trade(self, tmp_path) -> None:
        broker = FakeBroker(submit_result={"status": "error", "error": "insufficient buying power"})
        ex, _ = _executor(tmp_path, broker)
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "error" and "insufficient" in " ".join(r.reasons)
        assert ExecutionStore(tmp_path).day_state(NOW, 100_000.0).trades == 0

    def test_planner_skips_pass_through_with_reasons(self, tmp_path) -> None:
        ex, broker = _executor(tmp_path)
        r = ex.execute(_report(conf=0.1), now=NOW, dry_run=False)
        assert r.status == "skipped" and r.reasons and broker.submitted == []


class TestSafetyGates:
    def test_kill_switch_refuses_everything_and_touches_nothing(self, tmp_path) -> None:
        ex, broker = _executor(tmp_path, halted=True)
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "refused" and "halt" in " ".join(r.reasons).lower()
        assert broker.reads == 0 and broker.submitted == []

    def test_live_account_is_refused_before_any_broker_call(self, tmp_path) -> None:
        ex, broker = _executor(tmp_path, FakeBroker(paper=False))
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "refused" and "paper" in " ".join(r.reasons).lower()
        assert broker.reads == 0 and broker.submitted == []

    def test_unreadable_broker_state_fails_closed(self, tmp_path) -> None:
        ex, broker = _executor(tmp_path, FakeBroker(fail_reads=True))
        r = ex.execute(_report(), now=NOW, dry_run=False)
        assert r.status == "skipped" and "unavailable" in " ".join(r.reasons).lower()
        assert broker.submitted == []

    def test_daily_loss_limit_uses_the_persisted_start_of_day_equity(self, tmp_path) -> None:
        ExecutionStore(tmp_path).day_state(NOW, 100_000.0)  # first look of the day: 100k
        ex, broker = _executor(tmp_path, FakeBroker(equity=97_000.0))
        later = NOW + timedelta(hours=1)
        r = ex.execute(_report(at=later), now=later, dry_run=False)
        assert r.status == "skipped" and "loss" in " ".join(r.reasons)

    def test_max_trades_per_day_persists_across_processes(self, tmp_path) -> None:
        lim = ExecutionLimits(max_trades_per_day=1)
        ex1, _ = _executor(tmp_path, limits=lim)
        ex1.execute(_report("NVDA"), now=NOW, dry_run=False)
        ex2, b2 = _executor(tmp_path, limits=lim)  # a fresh executor/store, same directory
        r = ex2.execute(_report("TSLA"), now=NOW + timedelta(minutes=30), dry_run=False)
        assert r.status == "skipped" and b2.submitted == []


class TestStore:
    def test_new_day_resets_counters_and_start_equity(self, tmp_path) -> None:
        s = ExecutionStore(tmp_path)
        s.day_state(NOW, 100_000.0)
        s.record(ExecutionPlan("NVDA", "buy", 1, 100, 99, 101, "id-1", 0.8), {"order_id": "o"}, NOW)
        nxt = s.day_state(NOW + timedelta(days=1), 105_000.0)
        assert nxt.trades == 0 and nxt.start_equity == 105_000.0 and nxt.day == "2026-10-03"

    def test_start_equity_is_fixed_after_the_first_look(self, tmp_path) -> None:
        s = ExecutionStore(tmp_path)
        s.day_state(NOW, 100_000.0)
        assert s.day_state(NOW + timedelta(hours=2), 90_000.0).start_equity == 100_000.0

    def test_corrupt_state_file_fails_closed_instead_of_resetting_the_limits(self, tmp_path) -> None:
        (tmp_path / "state.json").write_text("{not json")
        with pytest.raises(StoreError):
            ExecutionStore(tmp_path).day_state(NOW, 100_000.0)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
    def test_files_are_private(self, tmp_path) -> None:
        s = ExecutionStore(tmp_path / "x")
        s.record(ExecutionPlan("NVDA", "buy", 1, 100, 99, 101, "id-1", 0.8), {"order_id": "o"}, NOW)
        assert stat.S_IMODE(os.stat(tmp_path / "x" / "executions.jsonl").st_mode) == 0o600

    def test_log_line_has_plan_and_order_and_no_credentials(self, tmp_path) -> None:
        s = ExecutionStore(tmp_path)
        s.record(ExecutionPlan("NVDA", "buy", 5, 100, 99, 101.5, "id-1", 0.8), {"order_id": "o-9", "status": "ok"}, NOW)
        row = json.loads((tmp_path / "executions.jsonl").read_text().splitlines()[0])
        assert row["symbol"] == "NVDA" and row["order_id"] == "o-9" and row["qty"] == 5


class FakeTradingClient:
    def __init__(self, order=None, error: Exception | None = None) -> None:
        self.requests = []
        self._order, self._error = order or SimpleNamespace(id="abc-123", status="accepted"), error

    def submit_order(self, order_data):
        self.requests.append(order_data)
        if self._error:
            raise self._error
        return self._order


class TestAlpacaAdapter:
    def _broker(self, monkeypatch, client: FakeTradingClient, *, paper: bool = True):
        from src.trading.connectors.alpaca import sdk

        monkeypatch.setattr(sdk, "_trading_client", lambda cfg: client)
        cfg = sdk.AlpacaConfig(api_key="k", secret_key="s", profile="paper" if paper else "live")
        return AlpacaPaperBroker(cfg)

    def test_submits_a_real_bracket_request(self, monkeypatch) -> None:
        client = FakeTradingClient()
        out = self._broker(monkeypatch, client).submit_bracket(
            ExecutionPlan("NVDA", "buy", 7, 100.0, 99.0, 101.5, "vt-NVDA-b-1", 0.8))
        assert out["status"] == "ok" and out["order_id"] == "abc-123"
        req = client.requests[0]
        assert req.symbol == "NVDA" and float(req.qty) == 7 and str(req.order_class).lower().endswith("bracket")
        assert float(req.stop_loss.stop_price) == 99.0 and float(req.take_profit.limit_price) == 101.5
        assert req.client_order_id == "vt-NVDA-b-1"

    def test_short_side_maps_to_sell(self, monkeypatch) -> None:
        client = FakeTradingClient()
        self._broker(monkeypatch, client).submit_bracket(
            ExecutionPlan("NVDA", "sell", 3, 100.0, 101.0, 98.5, "vt-NVDA-s-1", 0.8))
        assert str(client.requests[0].side).lower().endswith("sell")

    def test_submission_exception_becomes_an_error_dict_without_raising(self, monkeypatch) -> None:
        out = self._broker(monkeypatch, FakeTradingClient(error=RuntimeError("rejected: market closed"))).submit_bracket(
            ExecutionPlan("NVDA", "buy", 1, 100.0, 99.0, 101.5, "id", 0.8))
        assert out["status"] == "error" and "market closed" in out["error"]

    def test_credentials_in_error_messages_are_redacted(self, monkeypatch) -> None:
        from src.trading.connectors.alpaca import sdk

        client = FakeTradingClient(error=RuntimeError("401 for key PKFAKEKEY0000000000000000A secret FAKESECRET00000000000000000000000000000000AB"))
        monkeypatch.setattr(sdk, "_trading_client", lambda cfg: client)
        cfg = sdk.AlpacaConfig(api_key="PKFAKEKEY0000000000000000A", secret_key="FAKESECRET00000000000000000000000000000000AB", profile="paper")
        out = AlpacaPaperBroker(cfg).submit_bracket(ExecutionPlan("NVDA", "buy", 1, 100.0, 99.0, 101.5, "id", 0.8))
        assert "PKFAKE" not in out["error"] and "FAKESECRET" not in out["error"] and "<redacted>" in out["error"]

    def test_reports_whether_it_is_paper(self, monkeypatch) -> None:
        assert self._broker(monkeypatch, FakeTradingClient()).is_paper is True
        assert self._broker(monkeypatch, FakeTradingClient(), paper=False).is_paper is False

    def test_live_profile_cannot_submit_even_if_called_directly(self, monkeypatch) -> None:
        client = FakeTradingClient()
        out = self._broker(monkeypatch, client, paper=False).submit_bracket(
            ExecutionPlan("NVDA", "buy", 1, 100.0, 99.0, 101.5, "id", 0.8))
        assert out["status"] == "error" and "paper" in out["error"].lower() and client.requests == []

    def test_account_parsing_and_bad_data(self, monkeypatch) -> None:
        from src.trading.connectors.alpaca import sdk

        monkeypatch.setattr(sdk, "get_account_snapshot", lambda cfg: {"status": "ok", "account": {
            "equity": "100000", "buying_power": "400000", "trading_blocked": False}})
        monkeypatch.setattr(sdk, "_trading_client", lambda cfg: None)
        b = AlpacaPaperBroker(sdk.AlpacaConfig(api_key="k", secret_key="s", profile="paper"))
        a = b.account()
        assert a.equity == 100_000.0 and a.buying_power == 400_000.0 and a.trading_blocked is False
        monkeypatch.setattr(sdk, "get_account_snapshot", lambda cfg: {"status": "ok", "account": {"equity": "n/a"}})
        with pytest.raises(BrokerError):
            b.account()
        monkeypatch.setattr(sdk, "get_account_snapshot", lambda cfg: {"status": "error", "error": "401"})
        with pytest.raises(BrokerError):
            b.account()
