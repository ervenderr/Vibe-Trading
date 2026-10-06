"""Track record: FIFO fill pairing, statistics and the promotion gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.execution.report import (
    Fill,
    PromotionCriteria,
    TrackRecord,
    evaluate_promotion,
    pair_fills,
    summarize_trades,
)

T0 = datetime(2026, 10, 2, 14, 0, tzinfo=timezone.utc)


def _fill(minutes: int, symbol: str, side: str, qty: float, price: float) -> Fill:
    return Fill(T0 + timedelta(minutes=minutes), symbol, side, qty, price)


class TestPairFills:
    def test_simple_long_round_trip(self) -> None:
        (t,) = pair_fills([_fill(0, "AAPL", "buy", 10, 100.0), _fill(5, "AAPL", "sell", 10, 101.0)])
        assert t.side == "long" and t.qty == 10 and t.pnl == pytest.approx(10.0)
        assert t.return_pct == pytest.approx(0.01)

    def test_short_round_trip(self) -> None:
        (t,) = pair_fills([_fill(0, "TSLA", "sell", 5, 200.0), _fill(5, "TSLA", "buy", 5, 196.0)])
        assert t.side == "short" and t.pnl == pytest.approx(20.0) and t.return_pct == pytest.approx(0.02)

    def test_scaled_exit_is_matched_fifo_and_reported_as_one_trade(self) -> None:
        trades = pair_fills([
            _fill(0, "A", "buy", 10, 100.0), _fill(1, "A", "buy", 10, 110.0),
            _fill(2, "A", "sell", 15, 120.0),
        ])
        assert len(trades) == 1 and trades[0].qty == 15
        assert trades[0].entry == pytest.approx((10 * 100 + 5 * 110) / 15)  # FIFO: all 10 @100, then 5 @110
        assert trades[0].pnl == pytest.approx(10 * 20 + 5 * 10)

    def test_open_position_is_not_a_closed_trade(self) -> None:
        assert pair_fills([_fill(0, "AAPL", "buy", 10, 100.0)]) == ()

    def test_symbols_are_independent_and_order_does_not_matter(self) -> None:
        fills = [_fill(5, "A", "sell", 1, 11.0), _fill(0, "A", "buy", 1, 10.0),
                 _fill(1, "B", "buy", 1, 20.0), _fill(6, "B", "sell", 1, 19.0)]
        trades = pair_fills(fills)
        assert {t.symbol: round(t.pnl, 2) for t in trades} == {"A": 1.0, "B": -1.0}

    def test_flipping_through_zero_opens_the_opposite_side(self) -> None:
        trades = pair_fills([_fill(0, "A", "buy", 5, 100.0), _fill(1, "A", "sell", 8, 102.0), _fill(2, "A", "buy", 3, 101.0)])
        assert [(t.side, t.qty) for t in trades] == [("long", 5), ("short", 3)]

    def test_ignores_zero_or_invalid_fills(self) -> None:
        assert pair_fills([_fill(0, "A", "buy", 0, 100.0), _fill(1, "A", "sell", 1, 0.0)]) == ()


def _trades(pnls: list[float], price: float = 100.0, qty: float = 10.0):
    fills = []
    for i, p in enumerate(pnls):
        fills += [_fill(i * 10, "A", "buy", qty, price), _fill(i * 10 + 5, "A", "sell", qty, price + p / qty)]
    return pair_fills(fills)


class TestSummarize:
    def test_core_numbers(self) -> None:
        s = summarize_trades(_trades([30.0, -10.0, 20.0, -10.0]), start_equity=100_000.0)
        assert s.trades == 4 and s.win_rate == pytest.approx(0.5)
        assert s.total_pnl == pytest.approx(30.0)
        assert s.profit_factor == pytest.approx(50.0 / 20.0)

    def test_max_drawdown_is_peak_to_trough_on_equity(self) -> None:
        s = summarize_trades(_trades([100.0, -50.0, -60.0, 20.0]), start_equity=1_000.0)
        assert s.max_drawdown == pytest.approx(110.0 / 1_100.0)

    def test_no_losses_gives_infinite_profit_factor_and_no_trades_is_safe(self) -> None:
        assert summarize_trades(_trades([5.0, 5.0]), 1_000.0).profit_factor == float("inf")
        e = summarize_trades((), 1_000.0)
        assert e.trades == 0 and e.win_rate == 0.0 and e.p_value == 1.0

    def test_trading_days_counts_distinct_exit_days(self) -> None:
        a = pair_fills([_fill(0, "A", "buy", 1, 10), _fill(5, "A", "sell", 1, 11)])
        b = pair_fills([Fill(T0 + timedelta(days=1), "A", "buy", 1, 10), Fill(T0 + timedelta(days=1, minutes=5), "A", "sell", 1, 11)])
        assert summarize_trades(a + b, 1_000.0).trading_days == 2

    def test_p_value_is_small_for_a_clear_edge_and_large_for_noise(self) -> None:
        winning = summarize_trades(_trades([10.0] * 60 + [-4.0] * 20), 100_000.0)
        noise = summarize_trades(_trades([5.0, -5.0] * 40), 100_000.0)
        assert winning.p_value < 0.01 and noise.p_value > 0.3


def _record(**kw) -> TrackRecord:
    base = dict(trades=150, wins=90, win_rate=0.6, total_pnl=900.0, expectancy_pct=0.002, profit_factor=1.6,
                max_drawdown=0.03, p_value=0.01, trading_days=30)
    return TrackRecord(**{**base, **kw})


class TestPromotion:
    def test_all_criteria_met_is_eligible_but_still_manual(self) -> None:
        p = evaluate_promotion(_record())
        assert p.eligible and "manual" in p.note.lower()

    @pytest.mark.parametrize("field,value,needle", [
        ("trades", 40, "trades"), ("trading_days", 8, "days"), ("profit_factor", 1.1, "profit factor"),
        ("max_drawdown", 0.08, "drawdown"), ("p_value", 0.2, "significance"), ("total_pnl", -5.0, "profit"),
    ])
    def test_any_single_failure_blocks_promotion(self, field: str, value: float, needle: str) -> None:
        p = evaluate_promotion(_record(**{field: value}))
        assert not p.eligible and any(needle in c.name.lower() and not c.ok for c in p.checks)

    def test_criteria_are_fixed_up_front_and_visible(self) -> None:
        c = PromotionCriteria()
        assert (c.min_trades, c.min_days, c.min_profit_factor, c.max_drawdown, c.max_p_value) == (100, 20, 1.3, 0.05, 0.05)

    def test_empty_record_is_not_eligible(self) -> None:
        assert not evaluate_promotion(summarize_trades((), 100_000.0)).eligible
