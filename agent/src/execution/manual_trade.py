"""One managed paper stock trade: entry, a broker-side safety stop, our own stop/target/timeout, retries.

Fixes the weak spots of the throwaway script: a dead process no longer leaves the position unprotected
(the broker holds a stop slightly below ours), a failed sell is retried, one bad price poll does not end
the trade, stale bars are never trusted, and the exit sells the quantity the broker really holds.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol

from src.execution.orb_bot import AlpacaVenue

MAX_BAR_AGE_S = 180
BAD_POLLS = 5          # consecutive missing/stale polls before we give up and sell
SELL_TRIES = 3
SAFETY_GAP = 0.003     # broker stop sits this far below our own stop


class TradeVenue(Protocol):
    def last_bar(self, symbol: str) -> tuple[datetime, float]: ...
    def market(self, symbol: str, side: str, qty: int) -> float: ...
    def position_qty(self, symbol: str) -> int: ...
    def place_stop(self, symbol: str, qty: int, price: float) -> str | None: ...
    def cancel(self, order_id: str) -> None: ...


def decide(close: float, stop: float, target: float, held_s: float, max_hold_s: float) -> str | None:
    if close <= stop:
        return "stop"
    if close >= target:
        return "target"
    return "timeout" if held_s >= max_hold_s else None


def run_trade(venue: TradeVenue, symbol: str, *, notional: float, stop_pct: float, target_pct: float,
              max_hold_s: float, log: Callable[[str], None], halted: Callable[[], bool],
              now_fn: Callable[[], datetime], sleep: Callable[[float], None] = time.sleep,
              poll_s: float = 15.0) -> float | None:
    """Returns net P&L, or None when the broker-side stop closed the trade (read the account for the fill)."""
    qty = int(notional // venue.last_bar(symbol)[1])
    if qty < 1:
        raise ValueError("notional too small for one share")
    entry = venue.market(symbol, "buy", qty)
    stop, target = entry * (1 - stop_pct), entry * (1 + target_pct)
    safety = None
    try:
        safety = venue.place_stop(symbol, qty, round(stop * (1 - SAFETY_GAP), 2))
    except Exception as exc:
        log(f"TRADE {symbol}: WARNING no broker-side safety stop ({type(exc).__name__})")
    log(f"TRADE {symbol}: bought {qty} @ {entry:.2f}, stop {stop:.2f}, target {target:.2f}, safety stop at broker: {bool(safety)}")
    t0, bad, why, net = now_fn(), 0, None, None
    try:
        while why is None:
            sleep(poll_s)
            try:
                ts, close = venue.last_bar(symbol)
                fresh = (now_fn() - ts).total_seconds() <= MAX_BAR_AGE_S
            except Exception:
                fresh = False
            if fresh:
                bad = 0
                why = decide(close, stop, target, (now_fn() - t0).total_seconds(), max_hold_s)
            else:
                bad += 1
                why = "no-data" if bad >= BAD_POLLS else None
            if why is None and halted():
                why = "halt"
            if why is None and venue.position_qty(symbol) == 0:
                log(f"TRADE {symbol}: position gone, the broker-side stop must have filled; check the account")
                return None
    finally:
        why = why or "error"
        if safety:
            try:
                venue.cancel(safety)
            except Exception:
                pass  # already filled or cancelled
        held = venue.position_qty(symbol)
        if held:
            fill = _sell(venue, symbol, held, log)
            net = (fill - entry) * held
            log(f"TRADE {symbol}: sold {held} @ {fill:.2f} on {why}: net {net:+.2f} USD")
    return net


def _sell(venue: TradeVenue, symbol: str, qty: int, log: Callable[[str], None]) -> float:
    for attempt in range(1, SELL_TRIES + 1):
        try:
            return venue.market(symbol, "sell", qty)
        except Exception as exc:
            log(f"TRADE {symbol}: sell attempt {attempt}/{SELL_TRIES} failed: {type(exc).__name__}")
    log(f"TRADE {symbol}: SELL FAILED, {qty} shares are STILL OPEN at the broker")
    raise RuntimeError("sell failed after retries")


class AlpacaTradeVenue(AlpacaVenue):
    def last_bar(self, symbol: str) -> tuple[datetime, float]:
        from alpaca.data.requests import StockBarsRequest  # type: ignore
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore
        from src.trading.connectors.alpaca import sdk

        now = datetime.now(timezone.utc)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                               start=now - timedelta(minutes=10), end=now, feed=sdk._data_feed(self._cfg))
        rows = sdk._data_client(self._cfg).get_stock_bars(req).data.get(symbol, [])
        if not rows:
            raise RuntimeError("no recent bars")
        return rows[-1].timestamp + timedelta(minutes=1), float(rows[-1].close)  # a bar is final at its end

    def position_qty(self, symbol: str) -> int:
        from src.trading.connectors.alpaca import sdk

        held = [p for p in sdk.get_positions(self._cfg).get("positions", []) if p.get("symbol") == symbol]
        return int(float(held[0]["quantity"])) if held else 0

    def place_stop(self, symbol: str, qty: int, price: float) -> str | None:
        from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore
        from alpaca.trading.requests import StopOrderRequest  # type: ignore
        from src.trading.connectors.alpaca import sdk

        order = sdk._trading_client(self._cfg).submit_order(StopOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY, stop_price=price))
        return str(order.id)

    def cancel(self, order_id: str) -> None:
        from src.trading.connectors.alpaca import sdk

        sdk._trading_client(self._cfg).cancel_order_by_id(order_id)
