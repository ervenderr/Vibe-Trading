"""Broker additions: market clock, flatten (paper-only), fills."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.execution.broker import AlpacaPaperBroker, BrokerError
from src.trading.connectors.alpaca import sdk

T = datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc)


class FakeClient:
    def __init__(self, *, clock=None, orders=None, fail: Exception | None = None) -> None:
        self.calls: list[str] = []
        self._clock, self._orders, self._fail = clock, orders or [], fail
        self.last_filter = None

    def get_clock(self):
        self.calls.append("get_clock")
        if self._fail:
            raise self._fail
        return self._clock

    def get_orders(self, filter=None):
        self.calls.append("get_orders")
        self.last_filter = filter
        if self._fail:
            raise self._fail
        return self._orders


def _broker(monkeypatch, client: FakeClient, *, paper: bool = True) -> AlpacaPaperBroker:
    monkeypatch.setattr(sdk, "_trading_client", lambda cfg: client)
    return AlpacaPaperBroker(sdk.AlpacaConfig(api_key="k" * 10, secret_key="s" * 10, profile="paper" if paper else "live"))


class TestMarketClock:
    def test_parses_the_broker_clock(self, monkeypatch) -> None:
        clock = SimpleNamespace(is_open=True, timestamp=T, next_open=T + timedelta(hours=18), next_close=T + timedelta(hours=1))
        c = _broker(monkeypatch, FakeClient(clock=clock)).market_clock()
        assert c.is_open is True and c.next_close == T + timedelta(hours=1)

    def test_failure_raises_broker_error_so_callers_fail_closed(self, monkeypatch) -> None:
        with pytest.raises(BrokerError):
            _broker(monkeypatch, FakeClient(fail=RuntimeError("down"))).market_clock()

    def test_malformed_clock_is_an_error(self, monkeypatch) -> None:
        with pytest.raises(BrokerError):
            _broker(monkeypatch, FakeClient(clock=SimpleNamespace(is_open="maybe"))).market_clock()


def _order(minutes: int, symbol: str, side: str, qty: str, price: str | None) -> SimpleNamespace:
    return SimpleNamespace(id=f"{symbol}-{minutes}", symbol=symbol, side=side, filled_qty=qty, filled_avg_price=price,
                           filled_at=T + timedelta(minutes=minutes), submitted_at=T,
                           client_order_id="vt-test", legs=[])


class TestFills:
    def test_returns_only_real_fills_as_fill_objects(self, monkeypatch) -> None:
        orders = [_order(0, "AAPL", "OrderSide.BUY", "1", "332.36"), _order(5, "AAPL", "OrderSide.SELL", "1", "332.15"),
                  _order(6, "TSLA", "OrderSide.BUY", "0", None), _order(7, "TSLA", "OrderSide.BUY", None, None)]
        fills = _broker(monkeypatch, FakeClient(orders=orders)).fills(since=T - timedelta(days=1))
        assert [(f.symbol, f.side, f.qty, f.price) for f in fills] == [("AAPL", "buy", 1.0, 332.36), ("AAPL", "sell", 1.0, 332.15)]

    def test_requests_closed_orders_since_the_given_time(self, monkeypatch) -> None:
        client = FakeClient()
        _broker(monkeypatch, client).fills(since=T)
        assert "closed" in str(client.last_filter.status).lower() and client.last_filter.after == T

    def test_garbage_rows_are_skipped_and_failure_raises(self, monkeypatch) -> None:
        bad = [_order(0, "AAPL", "OrderSide.BUY", "abc", "1"), _order(1, "AAPL", "OrderSide.BUY", "1", "x")]
        assert _broker(monkeypatch, FakeClient(orders=bad)).fills(since=T) == ()
        with pytest.raises(BrokerError):
            _broker(monkeypatch, FakeClient(fail=RuntimeError("x"))).fills(since=T)
