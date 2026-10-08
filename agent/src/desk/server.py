"""Paper Desk: a local, read-only dashboard for the paper-trading autopilot.

GET-only and bound to localhost. It reads the paper account and the signal engine; it has no
code path that submits, cancels or modifies an order.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import os
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from src.desk.payloads import signal_payload, state_payload, track_payload
from src.execution.broker import BrokerError
from src.signals.news_source import NewsFetchError, is_valid_symbol

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
STATE_TTL_S, TRACK_TTL_S, SIGNAL_TTL_S = 4.0, 30.0, 120.0
MAX_TRACK_DAYS = 90
CSP = ("default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")


def _default_broker():
    from src.execution.broker import AlpacaPaperBroker

    return AlpacaPaperBroker()


def _default_signal(symbol: str):
    from src.signals.live_sources import run_signal

    return run_signal(symbol)


def _default_halt() -> bool:
    from src.execution.executor import _default_halt as halted

    return halted()


def _default_journal(limit: int) -> list[dict]:
    from src.execution.journal import Journal

    return Journal().recent(limit)


class Desk:
    def __init__(self, *, broker_factory: Callable[[], Any] = _default_broker,
                 signal_fn: Callable[[str], Any] = _default_signal,
                 halt_fn: Callable[[], bool] = _default_halt,
                 journal_fn: Callable[[int], list[dict]] = _default_journal,
                 now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self._broker_factory, self._signal, self._halt, self._journal, self._now = (
            broker_factory, signal_fn, halt_fn, journal_fn, now_fn)
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
