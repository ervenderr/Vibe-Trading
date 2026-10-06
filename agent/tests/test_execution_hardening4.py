"""Track-record integrity against stale/untagged lots (found by re-running on the real account)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.execution.report import Fill, pair_fills

T = datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc)


def _f(m: int, side: str, qty: float, price: float, tag: str = "", symbol: str = "A", days: int = 0) -> Fill:
    return Fill(T + timedelta(days=days, minutes=m), symbol, side, qty, price, tag)


class TestFlatClosesTheNewestLotsAndDiscardsStaleOnes:
    def test_stale_untagged_entry_does_not_steal_the_exit(self) -> None:
        # the real-account case: an old entry whose exit was never captured, then a fresh round trip
        fills = [_f(0, "buy", 1, 332.36), _f(60, "buy", 1, 332.86), _f(61, "sell", 1, 332.86, tag="flat")]
        trades = pair_fills(fills, allow_short=False)
        assert len(trades) == 1 and trades[0].entry == pytest.approx(332.86) and trades[0].pnl == pytest.approx(0.0)

    def test_after_a_flat_no_older_lot_can_be_matched_later(self) -> None:
        fills = [_f(0, "buy", 1, 100.0), _f(10, "buy", 1, 110.0), _f(11, "sell", 1, 111.0, tag="flat"),
                 _f(30, "buy", 1, 120.0), _f(31, "sell", 1, 121.0)]
        trades = pair_fills(fills, allow_short=False)
        assert [round(t.pnl, 2) for t in trades] == [1.0, 1.0]  # the 100.0 lot was discarded, not paired with 121

    def test_flat_larger_than_the_known_lots_just_closes_what_is_known(self) -> None:
        trades = pair_fills([_f(0, "buy", 2, 100.0), _f(5, "sell", 5, 101.0, tag="flat")], allow_short=False)
        assert len(trades) == 1 and trades[0].qty == 2

    def test_untagged_fills_still_use_plain_fifo(self) -> None:
        fills = [_f(0, "buy", 1, 100.0), _f(1, "buy", 1, 110.0), _f(2, "sell", 1, 120.0)]
        (t,) = pair_fills(fills)
        assert t.entry == 100.0

    def test_flat_scaled_exit_is_still_one_trade(self) -> None:
        fills = [_f(0, "buy", 5, 100.0), _f(1, "buy", 5, 102.0), _f(2, "sell", 10, 104.0, tag="flat")]
        (t,) = pair_fills(fills, allow_short=False)
        assert t.qty == 10 and t.pnl == pytest.approx(5 * 4 + 5 * 2)


class TestIntradayExpiry:
    def test_lots_from_an_earlier_session_expire_when_intraday_only(self) -> None:
        fills = [_f(0, "buy", 1, 100.0), _f(5, "buy", 1, 200.0, days=1), _f(10, "sell", 1, 201.0, days=1)]
        trades = pair_fills(fills, allow_short=False, intraday_only=True)
        assert len(trades) == 1 and trades[0].entry == 200.0

    def test_default_keeps_old_behaviour(self) -> None:
        fills = [_f(0, "buy", 1, 100.0), _f(5, "sell", 1, 101.0, days=1)]
        assert len(pair_fills(fills)) == 1

    def test_same_session_lots_are_unaffected(self) -> None:
        fills = [_f(0, "buy", 1, 100.0), _f(5, "sell", 1, 101.0)]
        assert len(pair_fills(fills, allow_short=False, intraday_only=True)) == 1


class TestBrokerTagsFlatFills:
    def test_flatten_orders_are_tagged_flat_and_entries_are_not(self, monkeypatch) -> None:
        from src.execution.broker import AlpacaPaperBroker
        from src.trading.connectors.alpaca import sdk

        def order(oid, cid, side, price, minutes):
            t = T + timedelta(minutes=minutes)
            return SimpleNamespace(id=oid, symbol="AAPL", side=side, filled_qty="1", filled_avg_price=price,
                                   filled_at=t, submitted_at=t, client_order_id=cid, legs=[])

        class C:
            def get_orders(self, filter=None):
                return [order("1", "vt-AAPL-b-1", "OrderSide.BUY", "100", 0), order("2", "vt-flat-AAPL-9", "OrderSide.SELL", "101", 5)]

        monkeypatch.setattr(sdk, "_trading_client", lambda cfg: C())
        b = AlpacaPaperBroker(sdk.AlpacaConfig(api_key="k" * 10, secret_key="s" * 10, profile="paper"))
        fills = b.fills(since=T - timedelta(days=1))
        assert [(f.side, f.tag) for f in fills] == [("buy", ""), ("sell", "flat")]
