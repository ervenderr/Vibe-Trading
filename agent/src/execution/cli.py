"""`vibe-trading execute`: signal -> guarded plan -> (optionally) a paper bracket order."""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any, Callable

from src.execution.models import ExecutionResult
from src.signals.news_source import is_valid_symbol

logger = logging.getLogger(__name__)


def _render(symbol: str, result: ExecutionResult, dry_run: bool) -> list[str]:
    mode = "DRY RUN" if dry_run else "PAPER"
    lines = [f"{symbol}: {result.status.upper()} [{mode}]"]
    if result.plan is not None:
        p = result.plan
        lines.append(f"  {p.side} {p.qty} @ ~{p.entry:.2f}  stop {p.stop:.2f}  target {p.target:.2f}  (confidence {p.confidence:.0%})")
    lines.extend(f"  - {r}" for r in result.reasons)
    if result.status == "dry_run":
        lines.append("  No order was sent. Re-run with --execute to place it in the PAPER account.")
    if result.status == "submitted" and result.order:
        lines.append(f"  PAPER order id: {result.order.get('order_id')}")
    return lines


def run_execute_command(
    args: Namespace,
    *,
    runner: Callable[..., Any] | None = None,
    executor: Any = None,
    out: Callable[[str], None] = print,
) -> int:
    if runner is None:
        from src.signals.live_sources import run_signal as runner  # lazy: heavy imports
    if executor is None:
        from src.execution.broker import AlpacaPaperBroker
        from src.execution.executor import PaperExecutor
        from src.execution.store import ExecutionStore

        executor = PaperExecutor(AlpacaPaperBroker(), ExecutionStore())
    dry_run = not args.execute
    failures = 0
    for raw in args.symbols:
        symbol = raw.strip().upper()
        if not is_valid_symbol(symbol):
            out(f"{symbol}: invalid symbol")
            failures += 1
            continue
        try:
            report = runner(symbol)
            result = executor.execute(report, dry_run=dry_run)
        except Exception as exc:
            logger.exception("execute failed for %s", symbol)
            out(f"{symbol}: failed ({type(exc).__name__})")
            failures += 1
            continue
        for line in _render(symbol, result, dry_run):
            out(line)
        if result.status in ("refused", "error"):
            failures += 1
    return 1 if failures else 0
