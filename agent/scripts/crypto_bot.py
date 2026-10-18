"""Daily crypto trend check on the PAPER account (rules: PREREGISTRATION-crypto-trend-bot.md).

    python scripts/crypto_bot.py --execute        # run once, after the 00:00 UTC daily close

Safe to run repeatedly: it acts at most once per UTC day. Output also goes to the autopilot journal.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.desk.server import _default_tick  # noqa: E402
from src.execution.broker import AlpacaPaperBroker  # noqa: E402
from src.execution.crypto_bot import BotConfig, run_check  # noqa: E402
from src.execution.crypto_demo import AlpacaCryptoTrader  # noqa: E402
from src.execution.executor import _default_halt  # noqa: E402
from src.execution.journal import Journal  # noqa: E402


def daily_closes(symbol: str) -> list[float]:
    """Closed UTC daily bars only: the still-forming bar for today is dropped."""
    import yfinance as yf

    df = yf.Ticker(symbol.replace("/", "-")).history(period="2y", interval="1d", auto_adjust=False)
    closes = [float(c) for c in df["Close"].dropna()]
    if not closes or df.index[-1].date() >= datetime.now(timezone.utc).date():
        closes = closes[:-1]
    if len(closes) < 200:
        raise RuntimeError(f"only {len(closes)} daily bars for {symbol}")
    return closes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="place PAPER orders (required)")
    if not ap.parse_args().execute:
        print("add --execute to place PAPER orders")
        return 2
    journal = Journal()

    def log(message: str) -> None:
        print(message, flush=True)
        journal.append(message)

    run_check(BotConfig(), trader=AlpacaCryptoTrader(AlpacaPaperBroker()), closes_fn=daily_closes,
              quote_fn=_default_tick, log=log, halted=_default_halt, today=datetime.now(timezone.utc).date())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
