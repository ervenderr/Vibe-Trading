"""Tests for the pure execution planner: eligibility, sizing and safety limits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.execution.models import AccountState, DayState, ExecutionLimits
from src.execution.planner import client_order_id, plan_execution, position_size
from src.signals.service import SignalReport
from src.signals.verdict import Action, Verdict

NOW = datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc)
ACCOUNT = AccountState(equity=100_000.0, buying_power=400_000.0, trading_blocked=False)
DAY = DayState(day="2026-10-02", start_equity=100_000.0, trades=0)
LIMITS = ExecutionLimits()


def _report(action: Action = Action.BUY, *, conf: float = 0.8, entry: float = 100.0, stop: float = 99.0,
            target: float = 101.5, valid_min: float = 15.0, stale: bool = False, symbol: str = "NVDA") -> SignalReport:
    v = Verdict(action, conf, ("r",), NOW + timedelta(minutes=valid_min), 0.5, 0.5, entry=entry, stop=stop, target=target)
    return SignalReport(symbol, NOW, v, (), NOW, stale, 0)


def _plan(report=None, *, account=ACCOUNT, positions=(), orders=(), day=DAY, limits=LIMITS, now=NOW):
    return plan_execution(report or _report(), account, positions, orders, day, limits, now)


class TestSizing:
    def test_risk_budget_sets_quantity(self) -> None:
        # equity 100k * 0.5% = 500 risk; stop distance 5 -> 100 shares (notional 10k cap not binding at price 100? 100*100=10k)
        assert position_size(100_000, 100.0, 95.0, 0.005, 50_000.0, 400_000.0) == 100

    def test_notional_cap_binds(self) -> None:
        assert position_size(100_000, 100.0, 99.0, 0.005, 10_000.0, 400_000.0) == 100  # risk says 500, cap says 100

    def test_buying_power_cap_binds(self) -> None:
        assert position_size(100_000, 100.0, 99.0, 0.005, 1e9, 4_000.0) == 20  # 50% of 4k / 100

    def test_zero_when_stop_distance_is_zero_or_inverted(self) -> None:
        assert position_size(100_000, 100.0, 100.0, 0.005, 1e9, 1e9) == 0

    def test_never_returns_fractional_or_negative(self) -> None:
        q = position_size(1_000, 333.33, 330.0, 0.005, 1e9, 1e9)
        assert isinstance(q, int) and q >= 0


class TestEligibility:
    def test_clean_buy_produces_a_bracket_plan(self) -> None:
        d = _plan()
        assert d.plan is not None and d.status == "planned"
        p = d.plan
        assert (p.symbol, p.side) == ("NVDA", "buy")
        assert p.qty == 100 and p.stop < p.entry < p.target

    @pytest.mark.parametrize("action", [Action.WAIT, Action.AVOID])
    def test_non_actionable_verdicts_are_skipped(self, action: Action) -> None:
        v = Verdict(action, 0.0, ("x",), NOW, 0.0, 0.0)
        d = _plan(SignalReport("NVDA", NOW, v, (), NOW, False, 0))
        assert d.plan is None and d.status == "skipped"

    def test_stale_data_is_skipped(self) -> None:
        assert _plan(_report(stale=True)).plan is None

    def test_expired_signal_is_skipped(self) -> None:
        assert _plan(_report(valid_min=-1)).plan is None

    def test_low_confidence_is_skipped(self) -> None:
        d = _plan(_report(conf=0.5))
        assert d.plan is None and "confidence" in " ".join(d.reasons)

    def test_shorts_are_disabled_by_default(self) -> None:
        d = _plan(_report(Action.SELL, stop=101.0, target=98.5))
        assert d.plan is None and "short" in " ".join(d.reasons)

    def test_shorts_allowed_when_enabled(self) -> None:
        d = _plan(_report(Action.SELL, stop=101.0, target=98.5), limits=ExecutionLimits(allow_shorts=True))
        assert d.plan is not None and d.plan.side == "sell" and d.plan.target < d.plan.entry < d.plan.stop

    def test_blocked_account_is_skipped(self) -> None:
        d = _plan(account=AccountState(100_000.0, 400_000.0, True))
        assert d.plan is None and "blocked" in " ".join(d.reasons)

    def test_existing_position_or_open_order_in_symbol_is_skipped(self) -> None:
        assert _plan(positions=[{"symbol": "NVDA"}]).plan is None
        assert _plan(orders=[{"symbol": "NVDA"}]).plan is None
        assert _plan(positions=[{"symbol": "TSLA"}]).plan is not None  # other symbols are fine

    def test_max_open_positions_is_enforced(self) -> None:
        pos = [{"symbol": s} for s in ("A", "B", "C")]
        assert _plan(positions=pos).plan is None

    def test_max_trades_per_day_is_enforced(self) -> None:
        assert _plan(day=DayState("2026-10-02", 100_000.0, trades=10)).plan is None

    def test_daily_loss_limit_halts_new_trades(self) -> None:
        d = _plan(account=AccountState(97_500.0, 400_000.0, False), day=DayState("2026-10-02", 100_000.0, 0))
        assert d.plan is None and "loss" in " ".join(d.reasons)

    def test_small_loss_below_the_limit_still_trades(self) -> None:
        assert _plan(account=AccountState(99_000.0, 400_000.0, False)).plan is not None

    def test_penny_stocks_are_skipped(self) -> None:
        assert _plan(_report(entry=2.0, stop=1.9, target=2.3)).plan is None

    def test_stop_wider_than_the_risk_budget_allows_zero_shares_is_skipped(self) -> None:
        d = _plan(_report(entry=100.0, stop=10.0, target=300.0), limits=ExecutionLimits(risk_per_trade=0.00001))
        assert d.plan is None and "size" in " ".join(d.reasons)

    @pytest.mark.parametrize("entry,stop,target", [(100.0, 101.0, 102.0), (100.0, 99.0, 99.5), (100.0, 100.0, 101.0)])
    def test_inconsistent_levels_are_rejected(self, entry, stop, target) -> None:
        assert _plan(_report(entry=entry, stop=stop, target=target)).plan is None

    def test_missing_levels_are_rejected(self) -> None:
        v = Verdict(Action.BUY, 0.8, ("x",), NOW + timedelta(minutes=5), 0.5, 0.5)  # no entry/stop/target
        assert _plan(SignalReport("NVDA", NOW, v, (), NOW, False, 0)).plan is None

    def test_prices_are_rounded_to_cents_and_stay_distinct(self) -> None:
        d = _plan(_report(entry=100.123456, stop=99.123456, target=101.623456))
        p = d.plan
        assert p is not None and round(p.stop, 2) == p.stop and round(p.target, 2) == p.target and p.stop < p.target

    def test_every_decision_explains_itself(self) -> None:
        assert _plan().reasons and _plan(_report(conf=0.1)).reasons


class TestClientOrderId:
    def test_stable_within_a_window_and_distinct_across_symbols_and_windows(self) -> None:
        a = client_order_id("NVDA", "buy", NOW)
        assert a == client_order_id("NVDA", "buy", NOW + timedelta(minutes=5))
        assert a != client_order_id("NVDA", "buy", NOW + timedelta(minutes=20))
        assert a != client_order_id("TSLA", "buy", NOW)
        assert a != client_order_id("NVDA", "sell", NOW)

    def test_is_short_and_safe_for_the_broker(self) -> None:
        cid = client_order_id("BRK.B", "buy", NOW)
        assert len(cid) <= 48 and cid.replace("-", "").replace("_", "").replace(".", "").isalnum()
