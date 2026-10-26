"""Paper-only opening range breakout bot for SPY and QQQ. Rules: scripts/PREREGISTRATION-orb-paper.md.

One trade per symbol per day, flat before the close. Never touches a live account; respects the kill switch.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

SYMBOLS = ("SPY", "QQQ")
OR_MINUTES = 30
BAR_MINUTES = 5
EXIT_BEFORE_CLOSE = timedelta(minutes=5)
MIN_SESSION = timedelta(hours=6)  # a shorter session is an early close: no trade
NOTIONAL = 5000.0
MIN_RANGE, MAX_RANGE = 0.0015, 0.015
MAX_LOSS = 1000.0
POLL_S = 20.0
FILL_TIMEOUT_S = 20.0
TRADES = Path.home() / ".vibe-trading" / "execution" / "orb_trades.jsonl"


class Venue(Protocol):
    def session(self) -> tuple[datetime, datetime]: ...
    def bars(self, symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]: ...
    def price(self, symbol: str) -> float: ...
    def market(self, symbol: str, side: str, qty: int) -> float: ...


@dataclass
class Position:
    side: str  # "long" or "short"
    qty: int
    entry: float
    stop: float


def opening_range(bars: list[dict[str, Any]]) -> tuple[float, float]:
    return max(b["high"] for b in bars), min(b["low"] for b in bars)


def range_ok(high: float, low: float) -> bool:
    return MIN_RANGE <= (high - low) / ((high + low) / 2) <= MAX_RANGE


def breakout(high: float, low: float, close: float) -> str | None:
    return "long" if close > high else "short" if close < low else None


def stopped(pos: Position, close: float) -> bool:
    return close <= pos.stop if pos.side == "long" else close >= pos.stop


def pnl(side: str, entry: float, exit_: float, qty: int) -> float:
    return (exit_ - entry) * qty * (1 if side == "long" else -1)


def _completed(bars: list[dict[str, Any]], now: datetime) -> list[dict[str, Any]]:
    return [b for b in bars if b["time"] + timedelta(minutes=BAR_MINUTES) <= now]


def run_day(venue: Venue, *, log: Callable[[str], None], halted: Callable[[], bool], now_fn: Callable[[], datetime],
            sleep: Callable[[float], None] = time.sleep, record: Callable[[dict], None] = lambda r: None,
            symbols: tuple[str, ...] = SYMBOLS, loss_so_far: float = 0.0) -> float:
    """Run one session. Returns the day's own net P&L."""
    open_t, close_t = venue.session()
    if close_t - open_t < MIN_SESSION:
        log("ORB: early-close session, no trades today")
        return 0.0
    or_end = open_t + timedelta(minutes=OR_MINUTES)
    while now_fn() < or_end:
        if halted():
            log("ORB: kill switch set, no trades")
            return 0.0
        sleep(POLL_S)
    ranges = {}
    for s in symbols:
        bars = venue.bars(s, open_t, or_end)
        if not bars:
            log(f"ORB {s}: no opening-range bars, skipping")
            continue
        hi, lo = opening_range(bars)
        if range_ok(hi, lo):
            ranges[s] = (hi, lo)
        log(f"ORB {s}: opening range {lo:.2f}-{hi:.2f}" + ("" if s in ranges else " (outside 0.15%-1.5%, skipping)"))
    open_pos: dict[str, Position] = {}
    done: set[str] = set()
    total = 0.0

    def close_out(s: str, why: str) -> None:
        nonlocal total
        pos = open_pos.pop(s)
        fill = venue.market(s, "sell" if pos.side == "long" else "buy", pos.qty)
        net = pnl(pos.side, pos.entry, fill, pos.qty)
        total += net
        record({"t": now_fn().isoformat(), "symbol": s, "side": pos.side, "qty": pos.qty, "entry": pos.entry,
                "exit": fill, "net": net, "why": why})
        log(f"ORB {s}: closed {pos.side} on {why}: {pos.entry:.2f} -> {fill:.2f}, net {net:+.2f} USD")

    try:
        while now_fn() < close_t - EXIT_BEFORE_CLOSE:
            if halted() or loss_so_far + total <= -MAX_LOSS:
                log("ORB: kill switch or loss limit, flattening")
                break
            for s, (hi, lo) in ranges.items():
                if s in done:
                    continue
                now = now_fn()
                bars = _completed(venue.bars(s, or_end, now), now)
                if not bars:
                    continue
                close = bars[-1]["close"]
                if s in open_pos:
                    if stopped(open_pos[s], close):
                        close_out(s, "stop")
                        done.add(s)
                    continue
                side = breakout(hi, lo, close)
                if side:
                    qty = int(NOTIONAL // close)
                    if qty < 1:
                        continue
                    fill = venue.market(s, "buy" if side == "long" else "sell", qty)
                    open_pos[s] = Position(side, qty, fill, lo if side == "long" else hi)
                    log(f"ORB {s}: {side} {qty} @ {fill:.2f}, stop {open_pos[s].stop:.2f}")
            sleep(POLL_S)
    finally:
        for s in list(open_pos):
            close_out(s, "end of day")
    log(f"ORB: day finished, own net {total:+.2f} USD")
    return total


class AlpacaVenue:
    """Alpaca paper adapter. Refuses a non-paper profile."""

    def __init__(self, cfg: Any) -> None:
        if not cfg.is_paper:
            raise RuntimeError("ORB bot only runs against a paper account")
        self._cfg = cfg

    def session(self) -> tuple[datetime, datetime]:
        from alpaca.trading.requests import GetCalendarRequest  # type: ignore
        from src.trading.connectors.alpaca import sdk

        client = sdk._trading_client(self._cfg)
        today = client.get_clock().timestamp.date()
        days = client.get_calendar(GetCalendarRequest(start=today, end=today))
        if not days or days[0].date != today:
            raise RuntimeError("no trading session today")
        return _to_utc(days[0].open), _to_utc(days[0].close)

    def bars(self, symbol: str, start: datetime, end: datetime) -> list[dict[str, Any]]:
        from alpaca.data.requests import StockBarsRequest  # type: ignore
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore
        from src.trading.connectors.alpaca import sdk

        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame(BAR_MINUTES, TimeFrameUnit.Minute),
                               start=start, end=end, feed=sdk._data_feed(self._cfg))
        rows = sdk._data_client(self._cfg).get_stock_bars(req).data.get(symbol, [])
        return [{"time": r.timestamp, "open": float(r.open), "high": float(r.high), "low": float(r.low),
                 "close": float(r.close)} for r in rows]

    def price(self, symbol: str) -> float:
        from src.trading.connectors.alpaca import sdk

        return float(sdk.get_quote(symbol, config=self._cfg)["quote"]["ask"])

    def market(self, symbol: str, side: str, qty: int) -> float:
        from src.trading.connectors.alpaca import sdk

        res = sdk.place_order(self._cfg, symbol=symbol, side=side, quantity=qty)
        if res.get("status") != "ok":
            raise RuntimeError(f"order failed: {res.get('error')}")
        client = sdk._trading_client(self._cfg)
        deadline = time.monotonic() + FILL_TIMEOUT_S
        while time.monotonic() < deadline:
            order = client.get_order_by_id(res["order_id"])
            if order.filled_avg_price is not None and float(order.filled_qty or 0) >= qty:
                return float(order.filled_avg_price)
            time.sleep(1.0)
        client.cancel_order_by_id(res["order_id"])
        raise RuntimeError("order did not fill in time and was cancelled; check the account for a partial fill")


def _to_utc(naive_exchange_time: datetime) -> datetime:
    """Alpaca calendar times are naive US/Eastern wall clock."""
    from zoneinfo import ZoneInfo

    return naive_exchange_time.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)


def append_trade(row: dict) -> None:
    TRADES.parent.mkdir(parents=True, exist_ok=True)
    with TRADES.open("a") as f:
        f.write(json.dumps(row) + "\n")


def own_net() -> float:
    if not TRADES.exists():
        return 0.0
    return sum(json.loads(l)["net"] for l in TRADES.read_text().splitlines() if l.strip())
