"""One managed PAPER stock trade (entry, broker-side safety stop, own stop/target/timeout).

    python scripts/manual_trade.py --execute --symbol TSLA [--notional 3000] [--stop 1.0] [--target 1.5] [--hours 3]

Run the news check and the Scout / Risk Officer / Devil's Advocate reviews BEFORE using this.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.execution.executor import _default_halt  # noqa: E402
from src.execution.journal import Journal  # noqa: E402
from src.execution.manual_trade import AlpacaTradeVenue, run_trade  # noqa: E402
from src.trading.connectors.alpaca import sdk  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="place PAPER orders (required)")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--notional", type=float, default=3000.0)
    ap.add_argument("--stop", type=float, default=1.0, help="stop distance, percent")
    ap.add_argument("--target", type=float, default=1.5, help="target distance, percent")
    ap.add_argument("--hours", type=float, default=3.0)
    args = ap.parse_args()
    if not args.execute:
        print("add --execute to place PAPER orders on the demo account")
        return 2
    if not (10 <= args.notional <= 5000 and 0 < args.stop <= 5 and 0 < args.target <= 10 and 0 < args.hours <= 6):
        print("--notional 10-5000, --stop 0-5, --target 0-10, --hours 0-6")
        return 2
    journal = Journal()

    def log(message: str) -> None:
        print(message, flush=True)
        journal.append(message)

    run_trade(AlpacaTradeVenue(sdk.load_config()), args.symbol.upper(), notional=args.notional,
              stop_pct=args.stop / 100, target_pct=args.target / 100, max_hold_s=args.hours * 3600,
              log=log, halted=_default_halt, now_fn=lambda: datetime.now(timezone.utc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
