"""Run the real signal strategy on a crypto pair against the PAPER account.

    python scripts/crypto_demo.py --execute [--notional 1000] [--wait-minutes 30] [--max-hold 900]

Progress is printed and written to the autopilot journal, so the Paper Desk logbook shows it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.desk.server import _default_tick  # noqa: E402
from src.execution.broker import AlpacaPaperBroker  # noqa: E402
from src.execution.crypto_demo import AlpacaCryptoTrader, DemoConfig, run_demo  # noqa: E402
from src.execution.executor import _default_halt  # noqa: E402
from src.execution.journal import Journal  # noqa: E402
from src.signals.live_sources import run_signal  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="place PAPER orders (required)")
    ap.add_argument("--notional", type=float, default=1000.0)
    ap.add_argument("--wait-minutes", type=float, default=30.0)
    ap.add_argument("--max-hold", type=float, default=900.0)
    ap.add_argument("--pair", default="BTC", help="coin, e.g. SOL (avoid BTC/ETH while the trend bot holds them)")
    ap.add_argument("--cycles", type=int, default=1, help="how many entry-and-exit rounds to run")
    ap.add_argument("--max-loss", type=float, default=400.0, help="stop for good once this bot's own net loss reaches this (USD)")
    args = ap.parse_args()
    if not args.execute:
        print("add --execute to place PAPER orders on the demo account")
        return 2
    if not (10 <= args.notional <= 5000):
        print("--notional must be between 10 and 5000 USD")
        return 2
    journal = Journal()

    def log(message: str) -> None:
        print(message, flush=True)
        journal.append(message)

    broker = AlpacaPaperBroker()
    trader = AlpacaCryptoTrader(broker)
    coin = args.pair.upper()
    cfg = DemoConfig(symbol=f"{coin}/USD", signal_symbol=f"{coin}-USD", notional=args.notional,
                     wait_minutes=args.wait_minutes, max_hold_s=args.max_hold)
    result, total = None, 0.0
    for i in range(args.cycles):
        if _default_halt() or total <= -args.max_loss:
            log(f"CRYPTO DEMO: stopping before cycle {i + 1} (kill switch, or own net {total:+.2f} hit the -{args.max_loss:,.0f} limit)")
            break
        result = run_demo(cfg, trader=trader, signal_fn=run_signal, quote_fn=_default_tick, log=log, halted=_default_halt)
        total += result.net_pnl or 0.0
    return 0 if result is None or result.exit_reason or result.entry is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
