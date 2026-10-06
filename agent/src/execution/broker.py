"""Alpaca paper broker adapter: reads account state and submits bracket orders.

Paper-only: ``submit_bracket`` refuses a live profile even if called directly, and
every failure is returned or raised as a sanitised message, never with credentials.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from typing import Any, Iterable

from src.execution.models import AccountState, ExecutionPlan, MarketClock
from src.execution.report import Fill
from src.signals.text_safety import sanitize_text
from src.trading.connectors.alpaca import sdk

logger = logging.getLogger(__name__)
MAX_ERROR_CHARS = 200
MIN_SECRET_LEN = 8
MAX_FILL_ORDERS = 500
MAX_FILL_PAGES = 10
FLATTEN_POLLS = 5
OWN_PREFIX = "vt-"
FLAT_PREFIX = "vt-flat-"


class BrokerError(RuntimeError):
    """Broker state could not be read or was malformed. Callers must fail closed."""


class AlpacaPaperBroker:
    def __init__(
        self,
        config: "sdk.AlpacaConfig | None" = None,
        *,
        sleep=time.sleep,
        page_size: int = MAX_FILL_ORDERS,
        max_pages: int = MAX_FILL_PAGES,
    ) -> None:
        self._cfg = config or sdk.load_config()
        self._sleep = sleep
        self._page_size = page_size
        self._max_pages = max_pages

    @property
    def is_paper(self) -> bool:
        return bool(self._cfg.is_paper)

    def _clean(self, message: object) -> str:
        text = sanitize_text(message, MAX_ERROR_CHARS)
        for secret in (self._cfg.api_key, self._cfg.secret_key):
            if secret and len(secret) >= MIN_SECRET_LEN:  # never mangle text with a trivially short "secret"
                text = text.replace(secret, "<redacted>")
        return text

    def _read(self, fn_name: str) -> dict[str, Any]:
        try:
            result = getattr(sdk, fn_name)(self._cfg)
        except Exception as exc:
            raise BrokerError(f"{fn_name} failed: {self._clean(type(exc).__name__)}") from None
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise BrokerError(f"{fn_name} returned an error")
        return result

    def account(self) -> AccountState:
        acct = self._read("get_account_snapshot").get("account") or {}
        try:
            equity, buying_power = float(acct["equity"]), float(acct["buying_power"])
        except (KeyError, TypeError, ValueError):
            raise BrokerError("account data missing or malformed") from None
        if not (math.isfinite(equity) and equity > 0 and math.isfinite(buying_power) and buying_power >= 0):
            raise BrokerError("account numbers are not usable")
        # Fail closed: only an explicit False means trading is allowed.
        blocked = acct.get("trading_blocked") is not False
        return AccountState(equity, buying_power, blocked, self._last_equity())

    def _last_equity(self) -> float | None:
        """Previous-close equity: the stable baseline for the daily loss limit (None if unavailable)."""
        try:
            value = float(sdk._trading_client(self._cfg).get_account().last_equity)
        except Exception:
            return None
        return value if math.isfinite(value) and value > 0 else None

    def last_price(self, symbol: str, side: str) -> float:
        """Live price a market order would hit: the ask for buys, the bid for sells."""
        try:
            quote = sdk.get_quote(symbol, config=self._cfg)
        except Exception as exc:
            raise BrokerError(f"quote unavailable: {type(exc).__name__}") from None
        if not isinstance(quote, dict) or quote.get("status") != "ok":
            raise BrokerError("quote unavailable")
        try:
            value = float((quote.get("quote") or {})["ask" if side == "buy" else "bid"])
        except (KeyError, TypeError, ValueError):
            raise BrokerError("quote malformed") from None
        if not (math.isfinite(value) and value > 0):
            raise BrokerError("quote not usable")
        return value

    def positions(self) -> list[dict]:
        return list(self._read("get_positions").get("positions") or [])

    def open_orders(self) -> list[dict]:
        return list(self._read("get_open_orders").get("open_orders") or [])

    def submit_bracket(self, plan: ExecutionPlan) -> dict[str, Any]:
        """Submit entry + stop-loss + take-profit as one atomic bracket order."""
        if not self.is_paper:
            return {"status": "error", "error": "paper-only: refusing to submit to a live profile"}
        try:
            from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

            request = MarketOrderRequest(
                symbol=plan.symbol,
                qty=plan.qty,
                side=OrderSide.BUY if plan.side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=plan.target),
                stop_loss=StopLossRequest(stop_price=plan.stop),
                client_order_id=plan.client_order_id,
            )
            order = sdk._trading_client(self._cfg).submit_order(order_data=request)
        except Exception as exc:  # reported to the caller, never raised
            return {"status": "error", "error": self._clean(exc)}
        return {
            "status": "ok",
            "order_id": str(getattr(order, "id", "")),
            "order_status": str(getattr(order, "status", "")),
        }

    def market_clock(self) -> MarketClock:
        """Broker clock (knows holidays and early closes). Failure means 'unknown': fail closed."""
        try:
            clock = sdk._trading_client(self._cfg).get_clock()
        except Exception as exc:
            raise BrokerError(f"clock unavailable: {type(exc).__name__}") from None
        if not isinstance(getattr(clock, "is_open", None), bool):
            raise BrokerError("clock data malformed")
        try:
            return MarketClock(clock.is_open, clock.timestamp, clock.next_open, clock.next_close)
        except AttributeError:
            raise BrokerError("clock data malformed") from None

    def _remaining(self, client: Any, wanted: set[str], *, ignore_closers: bool) -> list[str]:
        left = [f"{str(p.symbol).upper()} (position)" for p in client.get_all_positions() if str(p.symbol).upper() in wanted]
        for o in client.get_orders():
            if str(o.symbol).upper() not in wanted:
                continue
            if ignore_closers and str(getattr(o, "client_order_id", "")).startswith(FLAT_PREFIX):
                continue  # our own pending closing order
            left.append(f"{str(o.symbol).upper()} (order)")
        return sorted(left)

    def exposure(self, symbols: Iterable[str]) -> list[str]:
        """Open positions/orders in ``symbols`` (read-only)."""
        wanted = {str(s).upper() for s in symbols}
        if not wanted:
            return []
        try:
            return self._remaining(sdk._trading_client(self._cfg), wanted, ignore_closers=False)
        except Exception as exc:
            raise BrokerError(f"exposure unavailable: {type(exc).__name__}") from None

    def flatten(self, symbols: Iterable[str]) -> dict[str, Any]:
        """Cancel orders and close positions in ``symbols`` only, then VERIFY they are gone.

        Paper-only. Anything else in the account (manual positions, other bots) is left
        alone. Success means the positions are actually closed, not that closes were sent.
        """
        if not self.is_paper:
            return {"status": "error", "error": "paper-only: refusing to flatten a live profile"}
        wanted = {str(s).upper() for s in symbols}
        if not wanted:
            return {"status": "ok", "cancelled": 0, "closed": 0}
        cancelled = closed = 0
        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest

            client = sdk._trading_client(self._cfg)
            for order in client.get_orders():
                if str(order.symbol).upper() in wanted:
                    try:
                        client.cancel_order_by_id(order.id)
                        cancelled += 1
                    except Exception as exc:
                        logger.warning("cancel failed for %s: %s", order.symbol, type(exc).__name__)
            for pos in client.get_all_positions():
                symbol = str(pos.symbol).upper()
                if symbol not in wanted:
                    continue
                short = str(pos.side).lower().endswith("short")
                try:
                    client.submit_order(order_data=MarketOrderRequest(
                        symbol=symbol, qty=abs(float(pos.qty)),
                        side=OrderSide.BUY if short else OrderSide.SELL, time_in_force=TimeInForce.DAY,
                        client_order_id=f"{FLAT_PREFIX}{symbol}-{int(time.time())}",
                    ))
                    closed += 1
                except Exception as exc:
                    logger.warning("close failed for %s: %s", symbol, type(exc).__name__)
            remaining = self._remaining(client, wanted, ignore_closers=True)
            for _ in range(FLATTEN_POLLS):
                if not remaining:
                    break
                self._sleep(1.0)
                remaining = self._remaining(client, wanted, ignore_closers=True)
        except Exception as exc:
            return {"status": "error", "error": self._clean(exc), "remaining": sorted(wanted)}
        if remaining:
            return {"status": "error", "error": f"still open after flatten: {', '.join(remaining)}", "remaining": remaining}
        return {"status": "ok", "cancelled": cancelled, "closed": closed}

    def _fill_rows(self, order: Any, tag: str = "") -> list[Fill]:
        try:
            qty, price = float(order.filled_qty), float(order.filled_avg_price)
            when = getattr(order, "filled_at", None) or order.submitted_at
            if not (qty > 0 and price > 0 and math.isfinite(qty) and math.isfinite(price)) or when is None:
                return []
            side = "buy" if str(order.side).lower().endswith("buy") else "sell"
            return [Fill(when, str(order.symbol).upper(), side, qty, price, tag)]
        except (TypeError, ValueError, AttributeError):
            return []

    def fills(self, since: datetime) -> tuple[Fill, ...]:
        """Fills of THIS system's orders (``vt-`` ids plus their bracket legs) since ``since``.

        Manual and foreign orders are excluded so they cannot pollute the track record.
        Pages through history; raises if it cannot be read completely rather than
        silently truncating (a truncated history creates phantom trades).
        """
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            client = sdk._trading_client(self._cfg)
            seen: set[str] = set()
            parents: list[Any] = []
            cursor = None
            for _ in range(self._max_pages):
                page = list(client.get_orders(filter=GetOrdersRequest(
                    status=QueryOrderStatus.CLOSED, after=since, until=cursor,
                    limit=self._page_size, nested=True)) or [])
                parents.extend(o for o in page if str(o.id) not in seen and not seen.add(str(o.id)))
                if len(page) < self._page_size:
                    break
                stamps = [o.submitted_at for o in page if getattr(o, "submitted_at", None)]
                if not stamps:
                    raise BrokerError("cannot page through order history")
                cursor = min(stamps)
            else:
                raise BrokerError("order history too long to read completely")
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"fills unavailable: {type(exc).__name__}") from None
        rows: list[Fill] = []
        for order in parents:
            if not str(getattr(order, "client_order_id", "")).startswith(OWN_PREFIX):
                continue
            cid = str(getattr(order, "client_order_id", ""))
            rows.extend(self._fill_rows(order, "flat" if cid.startswith(FLAT_PREFIX) else ""))
            for leg in getattr(order, "legs", None) or []:
                rows.extend(self._fill_rows(leg))
        return tuple(rows)
