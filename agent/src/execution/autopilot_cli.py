"""`vibe-trading autopilot [SYMBOLS] [--execute] [--once]` and `--report`."""

from __future__ import annotations

import logging
import math
from argparse import Namespace
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any, Callable

from src.execution.broker import BrokerError
from src.execution.models import ExecutionLimits
from src.execution.report import evaluate_promotion, pair_fills, summarize_trades
from src.signals.news_source import is_valid_symbol

logger = logging.getLogger(__name__)
MAX_INTERVAL_S = 300.0  # must stay well inside the end-of-day flatten window
MIN_INTERVAL_S = 30.0  # keeps Finnhub's free tier (~60 calls/min) comfortable across symbols
DEFAULT_REPORT_DAYS = 30


def run_autopilot_command(
    args: Namespace,
    *,
    broker: Any = None,
    executor: Any = None,
    runner: Callable[..., Any] | None = None,
    pilot_cls: Any = None,
    out: Callable[[str], None] = print,
) -> int:
    if broker is None:
        from src.execution.broker import AlpacaPaperBroker

        broker = AlpacaPaperBroker()
    if args.report:
        return _report(args, broker, out)
    if not args.symbols:
        out("usage: vibe-trading autopilot SYMBOL [SYMBOL ...] [--execute] [--once] [--interval N] | --report")
        return 2
    bad = [s for s in args.symbols if not is_valid_symbol(s)]
    if bad:
        out(f"invalid symbol(s): {bad}")
        return 2
    if not (math.isfinite(args.interval) and MIN_INTERVAL_S <= args.interval <= MAX_INTERVAL_S):
        out(f"--interval must be between {MIN_INTERVAL_S:.0f} and {MAX_INTERVAL_S:.0f} seconds")
        return 2
    if not broker.is_paper:
        out("refused: autopilot only runs against a PAPER account")
        return 1

    from src.execution.store import ExecutionStore

    store = ExecutionStore()
    if runner is None:
        from src.signals.live_sources import run_signal as runner
    if executor is None:
        from src.execution.executor import PaperExecutor
        executor = PaperExecutor(broker, store)
    if pilot_cls is None:
        from src.execution.autopilot import Autopilot as pilot_cls

    dry_run = not args.execute
    out("autopilot: DRY RUN (no orders will be sent; add --execute for PAPER orders)" if dry_run
        else "autopilot: PAPER orders enabled (simulated money, real order flow)")
    from src.execution.journal import Journal

    journal = Journal()
    pilot = pilot_cls(args.symbols, runner=runner, executor=executor, broker=broker,
                      dry_run=dry_run, interval_s=args.interval, out=journal.tee(out), heartbeat=journal.beat,
                      managed_symbols=store.managed_symbols)
    return pilot.run(max_cycles=1 if args.once else None)


def _report(args: Namespace, broker: Any, out: Callable[[str], None]) -> int:
    since = (datetime.combine(args.since, dtime.min, tzinfo=timezone.utc) if args.since
             else datetime.now(timezone.utc) - timedelta(days=DEFAULT_REPORT_DAYS))
    try:
        trades = pair_fills(broker.fills(since), allow_short=ExecutionLimits().allow_shorts, intraday_only=True)
        equity = broker.account().equity
    except BrokerError as exc:
        out(f"could not read the paper account: {exc}")
        return 1
    if not trades:
        out(f"no closed trades since {since:%Y-%m-%d}; nothing to evaluate yet")
        return 0
    record = summarize_trades(trades, start_equity=equity - sum(t.pnl for t in trades))
    pf = "inf" if record.profit_factor == float("inf") else f"{record.profit_factor:.2f}"
    out(f"paper track record since {since:%Y-%m-%d} (this system's own orders only)")
    out(f"  closed trades {record.trades} over {record.trading_days} day(s) | win rate {record.win_rate:.1%}")
    out(f"  net P&L ${record.total_pnl:,.2f} | profit factor {pf} | expectancy {record.expectancy_pct:+.3%}/trade")
    out(f"  max drawdown {record.max_drawdown:.2%} | significance p={record.p_value:.3f}")
    promo = evaluate_promotion(record)
    out("promotion checks (fixed in advance):")
    for check in promo.checks:
        out(f"  [{'x' if check.ok else ' '}] {check.name}: {check.detail}")
    out(("ELIGIBLE" if promo.eligible else "NOT ELIGIBLE") + " for a live-trading discussion. " + promo.note)
    return 0
