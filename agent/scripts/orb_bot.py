"""Run one session of the paper opening range breakout bot (rules: PREREGISTRATION-orb-paper.md).

    python scripts/orb_bot.py --execute

Start it before the US open (or any time before the 30-minute range ends). It exits after the close.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.execution.executor import _default_halt  # noqa: E402
from src.execution.journal import Journal  # noqa: E402
from src.execution.orb_bot import MAX_LOSS, AlpacaVenue, append_trade, own_net, run_day  # noqa: E402
from src.trading.connectors.alpaca import sdk  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="place PAPER orders (required)")
    if not ap.parse_args().execute:
        print("add --execute to place PAPER orders on the demo account")
        return 2
    journal = Journal()

    def log(message: str) -> None:
        print(message, flush=True)
        journal.append(message)

    prior = own_net()
    if prior <= -MAX_LOSS:
        log(f"ORB: own net {prior:+.2f} already at the -{MAX_LOSS:,.0f} limit, not trading")
        return 1
    run_day(AlpacaVenue(sdk.load_config()), log=log, halted=_default_halt,
            now_fn=lambda: datetime.now(timezone.utc), record=append_trade, loss_so_far=prior)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
