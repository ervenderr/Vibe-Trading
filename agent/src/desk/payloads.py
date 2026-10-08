"""Read-only, JSON-safe views for the desk dashboard. Nothing here can place an order."""

from __future__ import annotations

import dataclasses
import logging
import math
from datetime import date, datetime
from enum import Enum
from typing import Any

from src.execution.broker import BrokerError
from src.execution.models import ExecutionLimits
from src.execution.report import evaluate_promotion, pair_fills, summarize_trades

logger = logging.getLogger(__name__)
LOGBOOK_ROWS = 80


def jsonable(value: Any) -> Any:
    """Convert to strictly valid JSON data: NaN/inf become null, times become ISO text."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _unreachable(exc: Exception) -> str:
    if isinstance(exc, BrokerError):
        return f"paper account unreachable: {exc}"
    logger.exception("desk read failed")
    return f"paper account read failed ({type(exc).__name__})"


def state_payload(broker: Any, *, halted: bool, journal: list[dict]) -> dict:
    beats = [row for row in journal if row.get("k") == "beat"]
    logbook = [row for row in journal if row.get("k") != "beat"][-LOGBOOK_ROWS:]
    base = {"paper": bool(getattr(broker, "is_paper", False)), "halted": halted,
            "journal": logbook, "last_beat": beats[-1] if beats else None, "account": None, "clock": None, "positions": [], "orders": [], "error": None}
    try:
        account = broker.account()
        clock = broker.market_clock()
        positions = broker.positions()
        orders = broker.open_orders()
    except Exception as exc:
        return jsonable({**base, "error": _unreachable(exc)})
    day_change = None if account.last_equity is None else account.equity - account.last_equity
    return jsonable({
        **base,
        "account": {"equity": account.equity, "buying_power": account.buying_power,
                    "blocked": account.trading_blocked, "last_equity": account.last_equity,
                    "day_change": day_change},
        "clock": clock,
        "positions": positions,
        "orders": orders,
    })


def track_payload(broker: Any, *, since: datetime) -> dict:
    """Closed trades, running P&L curve and the fixed promotion checks."""
    try:
        trades = pair_fills(broker.fills(since), allow_short=ExecutionLimits().allow_shorts, intraday_only=True)
        equity = broker.account().equity
    except Exception as exc:
        return jsonable({"error": _unreachable(exc), "trades": [], "curve": [],
                         "summary": None, "promotion": None})
    record = summarize_trades(trades, start_equity=equity - sum(t.pnl for t in trades))
    running, curve = 0.0, []
    for trade in trades:
        running += trade.pnl
        curve.append({"t": trade.exit_time, "cum_pnl": running})
    return jsonable({"error": None, "trades": trades, "curve": curve, "summary": record,
                     "promotion": evaluate_promotion(record)})


def _safe_url(url: str) -> str | None:
    return url if isinstance(url, str) and url.startswith(("https://", "http://")) else None


def signal_payload(report: Any) -> dict:
    """One symbol's verdict and the news behind it. Headline text is untrusted: the UI must not parse it as HTML."""
    verdict = report.verdict
    levels = {"entry": verdict.entry, "stop": verdict.stop, "target": verdict.target}
    headlines = []
    for analyzed in report.headlines:
        judgment = analyzed.judgment
        headlines.append({
            "headline": analyzed.item.headline, "source": analyzed.item.source,
            "published": analyzed.item.published, "url": _safe_url(analyzed.item.url),
            "origin": analyzed.origin, "score": analyzed.signal.score, "block": analyzed.signal.block,
            "reason": analyzed.signal.reason,
            "event_type": judgment.event_type if judgment else None,
            "sentiment": judgment.sentiment if judgment else None,
            "relevance": judgment.relevance if judgment else None,
            "confirmed": judgment.confirmed if judgment else None,
        })
    return jsonable({
        "symbol": report.symbol, "generated_at": report.generated_at, "action": verdict.action,
        "confidence": verdict.confidence, "reasons": verdict.reasons, "valid_until": verdict.valid_until,
        "news_score": verdict.news_score, "tech_score": verdict.tech_score, "levels": levels,
        "stale": report.stale, "news_count": report.news_count, "news_error": report.news_error,
        "headlines": headlines,
    })
