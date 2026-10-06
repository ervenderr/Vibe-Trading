"""Review round 1: numeric fail-closed, symbol validation, pending orders, live price, ids."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.execution.broker import AlpacaPaperBroker, BrokerError
from src.execution.models import AccountState, DayState, ExecutionLimits
from src.execution.planner import client_order_id, plan_execution
from src.signals.service import SignalReport
from src.signals.verdict import Action, Verdict
from src.trading.connectors.alpaca import sdk

NOW = datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc)
ACCT = AccountState(100_000.0, 400_000.0, False)
DAY = DayState("2026-10-02", 100_000.0, 0)
LIM = ExecutionLimits()


def _rep(*, conf=0.8, entry=100.0, stop=99.0, target=101.5, symbol="NVDA", action=Action.BUY) -> SignalReport:
    v = Verdict(action, conf, ("r",), NOW + timedelta(minutes=15), 0.5, 0.5, entry=entry, stop=stop, target=target)
    return SignalReport(symbol, NOW, v, (), NOW, False, 0)


def _plan(rep=None, acct=ACCT, positions=(), orders=(), live_price=None, limits=LIM):
    return plan_execution(rep or _rep(), acct, positions, orders, DAY, limits, NOW, live_price=live_price)


class TestNonFiniteInputsAreRejected:
    @pytest.mark.parametrize("kw", [dict(conf=math.nan), dict(conf=math.inf), dict(target=math.inf),
                                    dict(entry=math.nan), dict(stop=math.nan), dict(target=math.nan)])
    def test_bad_signal_numbers_never_plan(self, kw) -> None:
        assert _plan(_rep(**kw)).plan is None

    @pytest.mark.parametrize("acct", [AccountState(1e5, math.nan, False), AccountState(math.nan, 4e5, False),
                                      AccountState(math.inf, 4e5, False), AccountState(1e5, math.inf, False),
                                      AccountState(-5.0, 4e5, False), AccountState(0.0, 4e5, False)])
    def test_bad_account_numbers_never_plan(self, acct) -> None:
        assert _plan(acct=acct).plan is None

    def test_string_numbers_are_not_coerced(self) -> None:
        assert _plan(_rep(entry="100", stop=99.0, target=101.5)).plan is None  # type: ignore[arg-type]


class TestSymbolValidation:
    @pytest.mark.parametrize("sym", ["nvda\n;x", "../etc", "", "A B", "X" * 40])
    def test_odd_symbols_never_plan(self, sym) -> None:
        d = _plan(_rep(symbol=sym))
        assert d.plan is None and "symbol" in " ".join(d.reasons).lower()

    def test_normal_and_class_share_symbols_are_fine(self) -> None:
        assert _plan(_rep(symbol="BRK.B")).plan is not None


class TestPendingOrdersCountAgainstTheCap:
    def test_open_entry_orders_for_other_symbols_count_as_positions(self) -> None:
        orders = [{"symbol": "A"}, {"symbol": "B"}, {"symbol": "C"}]
        d = _plan(orders=orders)
        assert d.plan is None and "max open" in " ".join(d.reasons).lower()

    def test_a_symbol_with_both_a_position_and_an_order_counts_once(self) -> None:
        d = _plan(positions=[{"symbol": "A"}, {"symbol": "B"}], orders=[{"symbol": "A"}])
        assert d.plan is not None  # 2 distinct symbols < max 3


class TestLivePriceReanchoring:
    def test_levels_shift_with_the_live_price_and_size_uses_it(self) -> None:
        d = _plan(live_price=100.5)
        p = d.plan
        assert p.entry == 100.5 and p.stop == pytest.approx(99.5) and p.target == pytest.approx(102.0)

    def test_large_gap_between_signal_and_market_is_refused(self) -> None:
        d = _plan(live_price=110.0)
        assert d.plan is None and "moved" in " ".join(d.reasons).lower()

    @pytest.mark.parametrize("px", [math.nan, math.inf, 0.0, -1.0])
    def test_unusable_live_price_is_refused(self, px) -> None:
        assert _plan(live_price=px).plan is None

    def test_notional_cap_is_enforced_on_the_live_price(self) -> None:
        # signal says 5.01 but the market is 5.05 (within 1.5%): size must use the live price
        d = _plan(_rep(entry=5.01, stop=5.0, target=5.2), live_price=5.05)
        assert d.plan is not None and d.plan.qty * d.plan.entry <= LIM.max_order_notional + 1e-6

    def test_without_a_live_price_behaviour_is_unchanged(self) -> None:
        assert _plan().plan.entry == 100.0


class TestClientOrderIds:
    def test_long_distinct_symbols_get_distinct_ids(self) -> None:
        a = client_order_id("ABCDEFGHIJKLMN1", "buy", NOW)
        b = client_order_id("ABCDEFGHIJKLMN2", "buy", NOW)
        assert a != b and len(a) <= 48

    def test_planner_derives_the_window_from_now_not_the_report(self) -> None:
        rep = _rep()
        p1 = plan_execution(rep, ACCT, [], [], DAY, LIM, NOW).plan
        later = NOW + timedelta(minutes=20)
        rep2 = SignalReport(rep.symbol, NOW, Verdict(Action.BUY, 0.8, ("r",), later + timedelta(minutes=5), 0.5, 0.5, entry=100.0, stop=99.0, target=101.5), (), NOW, False, 0)
        p2 = plan_execution(rep2, ACCT, [], [], DAY, LIM, later).plan
        assert p1.client_order_id != p2.client_order_id


class FakeClient:
    def __init__(self, account=None, quote=None) -> None:
        self._account = account

    def get_account(self):
        return self._account


def _broker(monkeypatch, snapshot, *, last_equity="100000", paper=True) -> AlpacaPaperBroker:
    monkeypatch.setattr(sdk, "get_account_snapshot", lambda cfg: snapshot)
    monkeypatch.setattr(sdk, "_trading_client", lambda cfg: FakeClient(SimpleNamespace(last_equity=last_equity)))
    return AlpacaPaperBroker(sdk.AlpacaConfig(api_key="k" * 10, secret_key="s" * 10, profile="paper" if paper else "live"))


class TestBrokerAccountNumerics:
    @pytest.mark.parametrize("equity,bp", [("nan", "4e5"), ("inf", "1"), ("1e5", "nan"), ("1e5", "inf"), ("-1", "5"), ("0", "5"), ("1e5", "-1")])
    def test_non_finite_or_non_positive_values_raise(self, monkeypatch, equity, bp) -> None:
        snap = {"status": "ok", "account": {"equity": equity, "buying_power": bp, "trading_blocked": False}}
        with pytest.raises(BrokerError):
            _broker(monkeypatch, snap).account()

    def test_last_equity_is_read_for_the_daily_baseline(self, monkeypatch) -> None:
        snap = {"status": "ok", "account": {"equity": "99000", "buying_power": "4e5", "trading_blocked": False}}
        assert _broker(monkeypatch, snap, last_equity="100000").account().last_equity == 100_000.0

    def test_missing_or_bad_last_equity_is_none_not_an_error(self, monkeypatch) -> None:
        snap = {"status": "ok", "account": {"equity": "99000", "buying_power": "4e5", "trading_blocked": False}}
        assert _broker(monkeypatch, snap, last_equity=None).account().last_equity is None
        assert _broker(monkeypatch, snap, last_equity="abc").account().last_equity is None


class TestLastPrice:
    def _b(self, monkeypatch, quote) -> AlpacaPaperBroker:
        monkeypatch.setattr(sdk, "get_quote", lambda sym, config=None, **k: quote)
        return AlpacaPaperBroker(sdk.AlpacaConfig(api_key="k" * 10, secret_key="s" * 10, profile="paper"))

    def test_buy_uses_the_ask_and_sell_uses_the_bid(self, monkeypatch) -> None:
        b = self._b(monkeypatch, {"status": "ok", "quote": {"bid": 99.9, "ask": 100.1}})
        assert b.last_price("NVDA", "buy") == 100.1 and b.last_price("NVDA", "sell") == 99.9

    @pytest.mark.parametrize("quote", [{"status": "error"}, {"status": "ok", "quote": {"bid": 0, "ask": 0}},
                                       {"status": "ok", "quote": {"bid": "x", "ask": "y"}}, {"status": "ok", "quote": {"bid": float("nan"), "ask": float("nan")}}])
    def test_unusable_quotes_raise(self, monkeypatch, quote) -> None:
        with pytest.raises(BrokerError):
            self._b(monkeypatch, quote).last_price("NVDA", "buy")
