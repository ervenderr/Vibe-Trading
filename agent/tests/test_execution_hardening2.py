"""Review round 2: verified flatten of own symbols, paginated own-order fills, report integrity."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.execution.broker import AlpacaPaperBroker, BrokerError
from src.execution.report import Fill, pair_fills
from src.trading.connectors.alpaca import sdk

T = datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc)


def _pos(symbol: str, qty: str = "10", side: str = "PositionSide.LONG") -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, qty=qty, side=side)


def _ord(oid: str, symbol: str, client_id: str = "vt-x") -> SimpleNamespace:
    return SimpleNamespace(id=oid, symbol=symbol, client_order_id=client_id)


class FlatClient:
    def __init__(self, positions, orders, *, close_fails=(), stuck=False) -> None:
        self.positions, self.orders = list(positions), list(orders)
        self.cancelled: list[str] = []
        self.submitted: list = []
        self.close_fails, self.stuck = set(close_fails), stuck

    def get_orders(self, filter=None):
        return list(self.orders)

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)
        self.orders = [o for o in self.orders if o.id != oid]

    def get_all_positions(self):
        return list(self.positions)

    def submit_order(self, order_data):
        if order_data.symbol in self.close_fails:
            raise RuntimeError(f"cannot close {order_data.symbol}")
        self.submitted.append(order_data)
        if not self.stuck:
            self.positions = [p for p in self.positions if p.symbol != order_data.symbol]
        return SimpleNamespace(id="close-1")


def _broker(monkeypatch, client, *, paper=True, sleeps=None) -> AlpacaPaperBroker:
    monkeypatch.setattr(sdk, "_trading_client", lambda cfg: client)
    cfg = sdk.AlpacaConfig(api_key="k" * 10, secret_key="s" * 10, profile="paper" if paper else "live")
    return AlpacaPaperBroker(cfg, sleep=(sleeps.append if sleeps is not None else (lambda s: None)))


class TestFlattenOwnSymbolsOnly:
    def test_other_symbols_orders_and_positions_are_untouched(self, monkeypatch) -> None:
        client = FlatClient([_pos("AAPL"), _pos("MY_HOLD", "5")], [_ord("o1", "AAPL"), _ord("o2", "MY_HOLD", "manual-1")])
        out = _broker(monkeypatch, client).flatten({"AAPL"})
        assert out["status"] == "ok" and client.cancelled == ["o1"]
        assert [s.symbol for s in client.submitted] == ["AAPL"]
        assert [p.symbol for p in client.positions] == ["MY_HOLD"]

    def test_closing_order_is_a_tagged_opposite_side_market_order(self, monkeypatch) -> None:
        client = FlatClient([_pos("AAPL", "10"), _pos("TSLA", "3", "PositionSide.SHORT")], [])
        _broker(monkeypatch, client).flatten({"AAPL", "TSLA"})
        by = {s.symbol: s for s in client.submitted}
        assert str(by["AAPL"].side).lower().endswith("sell") and float(by["AAPL"].qty) == 10
        assert str(by["TSLA"].side).lower().endswith("buy") and float(by["TSLA"].qty) == 3
        assert all(s.client_order_id.startswith("vt-flat-") for s in client.submitted)

    def test_nothing_to_flatten_makes_no_broker_calls(self, monkeypatch) -> None:
        client = FlatClient([_pos("AAPL")], [])
        out = _broker(monkeypatch, client).flatten(set())
        assert out["status"] == "ok" and client.submitted == [] and client.cancelled == []

    def test_live_profile_is_refused_before_any_call(self, monkeypatch) -> None:
        client = FlatClient([_pos("AAPL")], [])
        out = _broker(monkeypatch, client, paper=False).flatten({"AAPL"})
        assert out["status"] == "error" and "paper" in out["error"].lower() and client.submitted == []


class TestFlattenIsVerified:
    def test_success_is_only_reported_when_positions_are_actually_gone(self, monkeypatch) -> None:
        client = FlatClient([_pos("AAPL")], [], stuck=True)  # the close never takes effect
        sleeps: list[float] = []
        out = _broker(monkeypatch, client, sleeps=sleeps).flatten({"AAPL"})
        assert out["status"] == "error" and "AAPL" in " ".join(out["remaining"]) and sleeps  # it polled and waited

    def test_one_failed_close_is_reported_and_the_others_still_run(self, monkeypatch) -> None:
        client = FlatClient([_pos("AAPL"), _pos("TSLA")], [], close_fails={"AAPL"})
        out = _broker(monkeypatch, client).flatten({"AAPL", "TSLA"})
        assert out["status"] == "error" and "AAPL" in " ".join(out["remaining"])
        assert [s.symbol for s in client.submitted] == ["TSLA"]

    def test_read_failure_is_an_error_not_a_crash(self, monkeypatch) -> None:
        class Boom(FlatClient):
            def get_orders(self, filter=None):
                raise RuntimeError("api down")

        out = _broker(monkeypatch, Boom([], [])).flatten({"AAPL"})
        assert out["status"] == "error"

    def test_exposure_lists_open_positions_and_orders_for_our_symbols(self, monkeypatch) -> None:
        client = FlatClient([_pos("AAPL"), _pos("OTHER")], [_ord("o1", "TSLA")])
        assert sorted(_broker(monkeypatch, client).exposure({"AAPL", "TSLA"})) == ["AAPL (position)", "TSLA (order)"]


def _fill_order(oid: str, symbol: str, side: str, qty: str, price: str, minutes: int, cid: str = "vt-a", legs=()) -> SimpleNamespace:
    t = T + timedelta(minutes=minutes)
    return SimpleNamespace(id=oid, symbol=symbol, side=side, filled_qty=qty, filled_avg_price=price,
                           filled_at=t, submitted_at=t, client_order_id=cid, legs=list(legs))


class PagedClient:
    def __init__(self, pages) -> None:
        self.pages, self.filters = pages, []

    def get_orders(self, filter=None):
        self.filters.append(filter)
        return self.pages.pop(0) if self.pages else []


class TestFillsAreOurOwnAndComplete:
    def test_manual_and_foreign_orders_are_excluded(self, monkeypatch) -> None:
        orders = [_fill_order("1", "AAPL", "OrderSide.BUY", "1", "100", 0, cid="vt-AAPL-b-1"),
                  _fill_order("2", "AAPL", "OrderSide.BUY", "5", "100", 1, cid="manual-xyz"),
                  _fill_order("3", "AAPL", "OrderSide.SELL", "1", "101", 2, cid="vt-flat-AAPL-9")]
        fills = _broker(monkeypatch, PagedClient([orders])).fills(since=T - timedelta(days=1))
        assert [(f.side, f.qty) for f in fills] == [("buy", 1.0), ("sell", 1.0)]

    def test_filled_bracket_legs_are_included_via_their_parent(self, monkeypatch) -> None:
        leg = _fill_order("L", "AAPL", "OrderSide.SELL", "1", "102", 5, cid="auto-leg")
        parent = _fill_order("P", "AAPL", "OrderSide.BUY", "1", "100", 0, cid="vt-AAPL-b-1", legs=[leg])
        fills = _broker(monkeypatch, PagedClient([[parent]])).fills(since=T - timedelta(days=1))
        assert sorted((f.side, f.price) for f in fills) == [("buy", 100.0), ("sell", 102.0)]

    def test_unfilled_legs_and_orders_are_skipped(self, monkeypatch) -> None:
        leg = _fill_order("L", "AAPL", "OrderSide.SELL", "0", "0", 5, cid="auto")
        parent = _fill_order("P", "AAPL", "OrderSide.BUY", "1", "100", 0, cid="vt-1", legs=[leg])
        assert len(_broker(monkeypatch, PagedClient([[parent]])).fills(since=T)) == 1

    def test_pages_are_followed_until_exhausted_and_deduplicated(self, monkeypatch) -> None:
        p1 = [_fill_order("a", "AAPL", "OrderSide.BUY", "1", "100", 10), _fill_order("b", "AAPL", "OrderSide.SELL", "1", "101", 9)]
        p2 = [_fill_order("b", "AAPL", "OrderSide.SELL", "1", "101", 9), _fill_order("c", "TSLA", "OrderSide.BUY", "1", "50", 1)]
        client = PagedClient([p1, p2, []])
        b = _broker(monkeypatch, client)
        b._page_size = 2
        fills = b.fills(since=T - timedelta(days=1))
        assert len(fills) == 3 and len(client.filters) >= 2
        assert client.filters[1].until is not None and client.filters[0].nested is True

    def test_runaway_history_raises_instead_of_silently_truncating(self, monkeypatch) -> None:
        full_page = [_fill_order(str(i), "A", "OrderSide.BUY", "1", "1", i) for i in range(2)]
        client = PagedClient([list(full_page) for _ in range(50)])
        b = _broker(monkeypatch, client)
        b._page_size, b._max_pages = 2, 3
        with pytest.raises(BrokerError, match="history"):
            b.fills(since=T - timedelta(days=1))


def _f(m: int, side: str, qty: float, price: float, symbol: str = "A") -> Fill:
    return Fill(T + timedelta(minutes=m), symbol, side, qty, price)


class TestPairFillsIntegrity:
    def test_a_scaled_exit_is_one_trade_not_one_per_lot(self) -> None:
        trades = pair_fills([_f(0, "buy", 10, 100.0), _f(1, "buy", 10, 110.0), _f(2, "sell", 15, 120.0)])
        assert len(trades) == 1 and trades[0].qty == 15
        assert trades[0].entry == pytest.approx((10 * 100 + 5 * 110) / 15)
        assert trades[0].pnl == pytest.approx(10 * 20 + 5 * 10)

    def test_orphan_exit_does_not_become_a_phantom_short_when_shorts_are_off(self) -> None:
        fills = [_f(0, "sell", 10, 105.0), _f(1, "buy", 10, 100.0), _f(2, "sell", 10, 101.0)]
        trades = pair_fills(fills, allow_short=False)
        assert [(t.side, t.pnl) for t in trades] == [("long", pytest.approx(10.0))]

    def test_default_still_supports_real_shorts(self) -> None:
        (t,) = pair_fills([_f(0, "sell", 5, 200.0), _f(1, "buy", 5, 196.0)])
        assert t.side == "short" and t.pnl == pytest.approx(20.0)
