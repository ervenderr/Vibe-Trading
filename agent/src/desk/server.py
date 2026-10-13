"""Paper Desk: a local, read-only dashboard for the paper-trading autopilot.

GET-only and bound to localhost. It reads the paper account and the signal engine; it has no
code path that submits, cancels or modifies an order.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from src.desk.lab import run_lab
from src.desk.payloads import jsonable, signal_payload, state_payload, track_payload
from src.execution.broker import BrokerError
from src.signals.news_source import NewsFetchError, is_valid_symbol

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
STATE_TTL_S, TRACK_TTL_S, SIGNAL_TTL_S, TICK_TTL_S, BARS_TTL_S, LAB_TTL_S = 4.0, 30.0, 120.0, 0.8, 5.0, 300.0
TIMEFRAMES = {"1Min": (1, "Minute"), "5Min": (5, "Minute"), "15Min": (15, "Minute")}
MAX_BARS = 300
MAX_TRACK_DAYS = 90
CSP = ("default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")


def _default_broker():
    from src.execution.broker import AlpacaPaperBroker

    return AlpacaPaperBroker()


def _default_signal(symbol: str):
    from src.signals.live_sources import run_signal

    return run_signal(symbol)


_CRYPTO_RE = re.compile(r"^[A-Z]{2,5}/USD$")
_data_clients: dict[str, Any] = {}


def _default_tick(symbol: str) -> dict:
    """Latest bid, ask and last trade from Alpaca's market-data feed (read-only)."""
    from src.trading.connectors.alpaca import sdk

    cfg = sdk.load_config()
    if _CRYPTO_RE.match(symbol):
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient as Client
        from alpaca.data.requests import CryptoLatestQuoteRequest as QuoteReq, CryptoLatestTradeRequest as TradeReq
    else:
        from alpaca.data.historical.stock import StockHistoricalDataClient as Client
        from alpaca.data.requests import StockLatestQuoteRequest as QuoteReq, StockLatestTradeRequest as TradeReq
    client = _data_clients.setdefault(Client.__name__, Client(cfg.api_key, cfg.secret_key))
    quote = client.get_crypto_latest_quote(QuoteReq(symbol_or_symbols=symbol))[symbol] if _CRYPTO_RE.match(symbol) \
        else client.get_stock_latest_quote(QuoteReq(symbol_or_symbols=symbol))[symbol]
    trade = client.get_crypto_latest_trade(TradeReq(symbol_or_symbols=symbol))[symbol] if _CRYPTO_RE.match(symbol) \
        else client.get_stock_latest_trade(TradeReq(symbol_or_symbols=symbol))[symbol]
    return {"symbol": symbol, "bid": quote.bid_price, "ask": quote.ask_price, "last": trade.price, "t": trade.timestamp}


def _default_bars(symbol: str, tf: str) -> list[dict]:
    """The most recent OHLC candles (oldest first) from Alpaca's free market-data feed."""
    from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import DataFeed
    from alpaca.common.enums import Sort
    from src.trading.connectors.alpaca import sdk

    cfg = sdk.load_config()
    amount, unit = TIMEFRAMES[tf]
    frame = TimeFrame(amount, getattr(TimeFrameUnit, unit))
    crypto = bool(_CRYPTO_RE.match(symbol))
    start = datetime.now(timezone.utc) - timedelta(days=2 if crypto else 5)
    if crypto:
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient as Client
        request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=frame, start=start, limit=MAX_BARS, sort=Sort.DESC)
    else:
        from alpaca.data.historical.stock import StockHistoricalDataClient as Client
        request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=frame, start=start, limit=MAX_BARS,
                                   sort=Sort.DESC, feed=DataFeed.IEX)
    client = _data_clients.setdefault(Client.__name__, Client(cfg.api_key, cfg.secret_key))
    raw = (client.get_crypto_bars(request) if crypto else client.get_stock_bars(request)).data.get(symbol, [])
    return [{"time": int(b.timestamp.timestamp()), "open": b.open, "high": b.high, "low": b.low,
             "close": b.close, "volume": b.volume} for b in reversed(raw)]


def _default_fills(symbol: str) -> list[dict]:
    """This account's recent filled orders for one symbol, for chart markers (read-only)."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    from src.trading.connectors.alpaca import sdk

    cfg = sdk.load_config()
    client = _data_clients.setdefault("trading", TradingClient(cfg.api_key, cfg.secret_key, paper=True))
    orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], limit=100))
    return [{"time": int(o.filled_at.timestamp()), "side": str(o.side).split(".")[-1].lower(),
             "price": float(o.filled_avg_price), "qty": float(o.filled_qty)}
            for o in orders if o.filled_at and o.filled_avg_price and float(o.filled_qty or 0) > 0]


def _default_history(symbol: str, interval: str, years: float):
    """Closing prices for the backtester from the free Yahoo feed (read-only)."""
    import pandas as pd
    from src.market_data import get_loader

    ticker = symbol.replace("/", "-")
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=int(years * 366))
    frame = get_loader("yfinance")().fetch([ticker], start.isoformat(), end.isoformat(), interval=interval).get(ticker)
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    return frame["close"]


def _default_halt() -> bool:
    from src.execution.executor import _default_halt as halted

    return halted()


def _default_journal(limit: int) -> list[dict]:
    from src.execution.journal import Journal

    return Journal().recent(limit)


class Desk:
    def __init__(self, *, broker_factory: Callable[[], Any] = _default_broker,
                 signal_fn: Callable[[str], Any] = _default_signal,
                 tick_fn: Callable[[str], dict] = _default_tick,
                 bars_fn: Callable[[str, str], list[dict]] = _default_bars,
                 fills_fn: Callable[[str], list[dict]] = _default_fills,
                 history_fn: Callable[[str, str, float], Any] = _default_history,
                 halt_fn: Callable[[], bool] = _default_halt,
                 journal_fn: Callable[[int], list[dict]] = _default_journal,
                 now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self._broker_factory, self._signal, self._halt, self._journal, self._now = (
            broker_factory, signal_fn, halt_fn, journal_fn, now_fn)
        self._tick, self._bars, self._fills, self._history = tick_fn, bars_fn, fills_fn, history_fn
        self._broker: Any = None
        self._cache: dict[str, tuple[float, Any]] = {}

    def _cached(self, key: str, ttl: float, build: Callable[[], Any]) -> Any:
        hit = self._cache.get(key)
        if hit and time.monotonic() - hit[0] < ttl:
            return hit[1]
        value = build()
        self._cache[key] = (time.monotonic(), value)
        return value

    def _get_broker(self) -> Any:
        if self._broker is None:
            self._broker = self._broker_factory()
        return self._broker

    def handle(self, method: str, path: str, query: dict, *, host: str) -> tuple[int, str, bytes]:
        if method != "GET":
            return self._json({"error": "read-only"}, 405)
        if host.rsplit(":", 1)[0] not in ALLOWED_HOSTS and host not in ALLOWED_HOSTS:
            return self._json({"error": "forbidden host"}, 403)
        if path == "/api/state":
            return self._api_state()
        if path == "/api/track":
            return self._api_track(query)
        if path == "/api/signal":
            return self._api_signal(query)
        if path == "/api/tick":
            return self._api_tick(query)
        if path == "/api/backtest":
            return self._api_backtest(query)
        if path == "/api/fills":
            return self._api_fills(query)
        if path == "/api/bars":
            return self._api_bars(query)
        if path.startswith("/api/"):
            return self._json({"error": "not found"}, 404)
        return self._static(path)

    def _api_state(self) -> tuple[int, str, bytes]:
        def build() -> dict:
            try:
                broker = self._get_broker()
            except Exception as exc:  # missing credentials etc.
                logger.warning("broker unavailable: %s", exc)
                return state_payload(_Unavailable(str(exc)), halted=self._halt(), journal=self._journal(400))
            return state_payload(broker, halted=self._halt(), journal=self._journal(400))

        return self._json(self._cached("state", STATE_TTL_S, build))

    def _api_track(self, query: dict) -> tuple[int, str, bytes]:
        try:
            days = max(1, min(MAX_TRACK_DAYS, int(query.get("days", ["30"])[0])))
        except ValueError:
            return self._json({"error": "days must be an integer"}, 400)

        def build() -> dict:
            try:
                broker = self._get_broker()
            except Exception as exc:
                logger.warning("broker unavailable: %s", exc)
                return track_payload(_Unavailable(str(exc)), since=self._now())
            return track_payload(broker, since=self._now() - timedelta(days=days))

        return self._json(self._cached(f"track:{days}", TRACK_TTL_S, build))

    def _api_signal(self, query: dict) -> tuple[int, str, bytes]:
        symbol = (query.get("symbol", [""])[0] or "").strip().upper()
        if not is_valid_symbol(symbol):
            return self._json({"error": "invalid symbol"}, 400)
        try:
            return self._json(self._cached(f"sig:{symbol}", SIGNAL_TTL_S, lambda: signal_payload(self._signal(symbol))))
        except NewsFetchError as exc:  # messages are written in-house and carry no secrets
            logger.warning("signal failed for %s: %s", symbol, exc)
            return self._json({"error": f"News is unavailable: {exc}"}, 502)
        except Exception:
            logger.exception("signal failed for %s", symbol)
            return self._json({"error": "signal unavailable right now"}, 502)

    def _api_tick(self, query: dict) -> tuple[int, str, bytes]:
        symbol = (query.get("symbol", [""])[0] or "").strip().upper()
        if not (_CRYPTO_RE.match(symbol) or is_valid_symbol(symbol)):
            return self._json({"error": "invalid symbol"}, 400)
        try:
            tick = self._cached(f"tick:{symbol}", TICK_TTL_S, lambda: jsonable(self._tick(symbol)))
            return self._json(tick)
        except Exception:
            logger.exception("tick failed for %s", symbol)
            return self._json({"error": "price feed unavailable right now"}, 502)

    def _api_bars(self, query: dict) -> tuple[int, str, bytes]:
        symbol = (query.get("symbol", [""])[0] or "").strip().upper()
        tf = query.get("tf", ["1Min"])[0]
        if not (_CRYPTO_RE.match(symbol) or is_valid_symbol(symbol)) or tf not in TIMEFRAMES:
            return self._json({"error": "invalid symbol or timeframe"}, 400)
        try:
            bars = self._cached(f"bars:{symbol}:{tf}", BARS_TTL_S, lambda: jsonable(self._bars(symbol, tf)))
            return self._json({"symbol": symbol, "tf": tf, "bars": bars})
        except Exception:
            logger.exception("bars failed for %s", symbol)
            return self._json({"error": "price history unavailable right now"}, 502)

    def _api_backtest(self, query: dict) -> tuple[int, str, bytes]:
        key = "lab:" + "&".join(f"{k}={v[0]}" for k, v in sorted(query.items()) if v)
        status, payload = self._cached(key, LAB_TTL_S, lambda: run_lab(query, history_fn=self._history))
        if status != 200:
            self._cache.pop(key, None)       # never cache failures
        return self._json(payload, status)

    def _api_fills(self, query: dict) -> tuple[int, str, bytes]:
        symbol = (query.get("symbol", [""])[0] or "").strip().upper()
        if not (_CRYPTO_RE.match(symbol) or is_valid_symbol(symbol)):
            return self._json({"error": "invalid symbol"}, 400)
        try:
            fills = self._cached(f"fills:{symbol}", STATE_TTL_S, lambda: jsonable(self._fills(symbol)))
            return self._json({"symbol": symbol, "fills": fills})
        except Exception:
            logger.exception("fills failed for %s", symbol)
            return self._json({"error": "trade history unavailable right now"}, 502)

    def _static(self, path: str) -> tuple[int, str, bytes]:
        rel = "index.html" if path in ("", "/") else unquote(path).lstrip("/")
        target = (STATIC / rel).resolve()
        if STATIC.resolve() not in target.parents or not target.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        return 200, ctype, target.read_bytes()

    @staticmethod
    def _json(payload: Any, status: int = 200) -> tuple[int, str, bytes]:
        return status, "application/json", json.dumps(payload, allow_nan=False).encode()


class _Unavailable:
    """Stands in for a broker that could not be created; every read reports the reason."""

    is_paper = False

    def __init__(self, reason: str) -> None:
        self._reason = reason

    def _fail(self, *_a: Any, **_k: Any):
        raise BrokerError(self._reason[:200])

    account = market_clock = positions = open_orders = fills = _fail


def make_handler(desk: Desk) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _respond(self, method: str) -> None:
            url = urlparse(self.path)
            status, ctype, body = desk.handle(method, url.path, parse_qs(url.query), host=self.headers.get("Host", ""))
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            self._respond("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._respond("POST")

        do_PUT = do_DELETE = do_PATCH = do_POST

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug(fmt, *args)

    return Handler


def serve(port: int = 8765) -> None:
    try:
        from src.providers.llm import _ensure_dotenv

        _ensure_dotenv()  # same key loading as the CLI
    except Exception:  # keys may also come from the real environment
        logger.warning("could not load .env; relying on the process environment")
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(Desk()))
    print(f"Paper Desk (read-only) on http://127.0.0.1:{port}  ·  Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
