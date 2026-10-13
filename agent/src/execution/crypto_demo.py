"""Paper-only crypto demo of the real signal strategy.

Alpaca has no stop/target (bracket) orders for crypto, so this script watches the price itself and
sells at the stop, the target or a time limit. Long only (spot crypto cannot be shorted). It never
touches a live account and respects the kill switch.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from src.execution.models import ExecutionLimits
from src.signals.verdict import Action

logger = logging.getLogger(__name__)
MAX_SPREAD = 0.003  # no entry when buy and sell prices are more than 0.3% apart
MAX_SIGNAL_AGE_S = 1800
FILL_TIMEOUT_S = 20.0


@dataclass(frozen=True)
class DemoConfig:
    symbol: str = "BTC/USD"        # the Alpaca trading symbol
    signal_symbol: str = "BTC-USD"  # the symbol the signal engine understands
    notional: float = 1000.0
    wait_minutes: float = 30.0      # how long to wait for a qualifying BUY
    max_hold_s: float = 900.0
    signal_every_s: float = 60.0
    poll_s: float = 2.0
    status_every_s: float = 15.0


@dataclass(frozen=True)
class DemoResult:
    exit_reason: str | None = None
    entry: float | None = None
    exit: float | None = None
    qty: float | None = None
    stop: float | None = None
    target: float | None = None
    gross_pnl: float | None = None
    net_pnl: float | None = None


def exit_reason(entry: float, stop: float, target: float, *, bid: float, held_s: float, max_hold_s: float) -> str | None:
    if bid <= stop:
        return "stop"
    if bid >= target:
        return "target"
    if held_s >= max_hold_s:
        return "timeout"
    return None


def _spread(quote: dict) -> float:
    bid, ask = float(quote["bid"]), float(quote["ask"])
    return (ask - bid) / ask if ask > 0 and ask >= bid > 0 else math.inf


def _entry_ok(report: Any, quote: dict, now: datetime, min_confidence: float) -> str | None:
    """None when a BUY may be taken, otherwise the reason it may not."""
    v = report.verdict
    if v.action is not Action.BUY:
        return f"{v.action.value}: " + "; ".join(v.reasons)
    if report.stale:
        return "price data stale"
    if v.confidence < min_confidence:
        return f"confidence {v.confidence:.2f} below {min_confidence:.2f}"
    if _spread(quote) > MAX_SPREAD:
        return "spread too wide"
    if (now - report.generated_at).total_seconds() > MAX_SIGNAL_AGE_S:
        return "signal too old"
    return None


def run_demo(
    cfg: DemoConfig, *, trader: Any, signal_fn: Callable[[str], Any], quote_fn: Callable[[str], dict],
    log: Callable[[str], None], halted: Callable[[], bool],
    now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc), sleep: Callable[[float], None] = time.sleep,
) -> DemoResult:
    if not getattr(trader, "is_paper", False):
        raise RuntimeError("crypto demo refuses to run: the account is not marked as paper")
    min_confidence = ExecutionLimits().min_confidence
    deadline = now_fn().timestamp() + cfg.wait_minutes * 60
    last_note = None
    while now_fn().timestamp() < deadline:
        if halted():
            log("CRYPTO DEMO: kill switch is on; stopping")
            return DemoResult()
        try:
            report, quote = signal_fn(cfg.signal_symbol), quote_fn(cfg.symbol)
            why_not = _entry_ok(report, quote, now_fn(), min_confidence)
        except Exception as exc:
            why_not = f"could not evaluate ({type(exc).__name__})"
            report = None
        if why_not is None:
            return _hold(cfg, report, quote, trader, quote_fn, log, halted, now_fn, sleep)
        if why_not != last_note:
            log(f"CRYPTO DEMO {cfg.symbol}: no entry - {why_not}")
            last_note = why_not
        sleep(cfg.signal_every_s)
    log(f"CRYPTO DEMO {cfg.symbol}: no qualifying BUY in {cfg.wait_minutes:.0f} minutes; no trade made")
    return DemoResult()


def _hold(cfg, report, quote, trader, quote_fn, log, halted, now_fn, sleep) -> DemoResult:
    v = report.verdict
    stop_dist, target_dist = v.entry - v.stop, v.target - v.entry
    qty = math.floor(cfg.notional / float(quote["ask"]) * 1e6) / 1e6
    equity_before = trader.equity_now()
    entry, held_qty = trader.buy(cfg.symbol, qty)  # held < qty: the fee is taken in coins
    stop, target = entry - stop_dist, entry + target_dist
    log(f"CRYPTO DEMO entered: bought {qty} {cfg.symbol} at {entry:,.2f} (confidence {v.confidence:.2f}); "
        f"stop {stop:,.2f}, target {target:,.2f}, max hold {cfg.max_hold_s:.0f}s")
    started, last_status, reason, bid = now_fn(), now_fn(), None, entry
    try:
        while reason is None:
            sleep(cfg.poll_s)
            bid = float(quote_fn(cfg.symbol)["bid"])
            held_s = (now_fn() - started).total_seconds()
            reason = "halt" if halted() else exit_reason(entry, stop, target, bid=bid, held_s=held_s, max_hold_s=cfg.max_hold_s)
            if reason is None and (now_fn() - last_status).total_seconds() >= cfg.status_every_s:
                last_status = now_fn()
                log(f"CRYPTO DEMO holding: bid {bid:,.2f} ({(bid / entry - 1) * 100:+.3f}%), {held_s:.0f}s in")
    except Exception as exc:
        reason = f"error ({type(exc).__name__})"
        logger.exception("crypto demo hold loop failed")
    finally:
        try:
            exit_price = trader.sell(cfg.symbol, held_qty)  # always try to get flat
        except Exception as exc:
            log(f"CRYPTO DEMO SELL FAILED ({type(exc).__name__}): the {cfg.symbol} position may still be open")
            raise
    gross = (exit_price - entry) * held_qty
    net = trader.equity_now() - equity_before
    log(f"CRYPTO DEMO closed on {reason}: sold at {exit_price:,.2f}; gross {gross:+.2f}, net after fees {net:+.2f} USD")
    return DemoResult(reason, entry, exit_price, held_qty, stop, target, gross, net)


class AlpacaCryptoTrader:
    """Market buys and sells on the Alpaca PAPER account. Reads fills back to get the real prices."""

    def __init__(self, broker: Any) -> None:
        from alpaca.trading.client import TradingClient

        self._broker = broker
        cfg = broker._cfg
        self._client = TradingClient(cfg.api_key, cfg.secret_key, paper=True)

    @property
    def is_paper(self) -> bool:
        return bool(self._broker.is_paper)

    def equity_now(self) -> float:
        return float(self._client.get_account().equity)

    def _market(self, symbol: str, qty: float, side: str) -> tuple[float, float]:
        """Returns (average fill price, quantity filled after fees)."""
        import uuid

        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order = self._client.submit_order(MarketOrderRequest(
            symbol=symbol, qty=qty, side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC, client_order_id=f"demo-{side}-{uuid.uuid4().hex[:12]}"))
        deadline = time.monotonic() + FILL_TIMEOUT_S
        while time.monotonic() < deadline:
            order = self._client.get_order_by_id(order.id)
            if str(order.status).lower().endswith("filled") and order.filled_avg_price:
                return float(order.filled_avg_price), float(order.filled_qty)
            time.sleep(0.5)
        raise RuntimeError(f"{side} order not filled within {FILL_TIMEOUT_S:.0f}s")

    def buy(self, symbol: str, qty: float) -> tuple[float, float]:
        return self._market(symbol, qty, "buy")

    def sell(self, symbol: str, qty: float) -> float:
        return self._market(symbol, qty, "sell")[0]
