"""Daily crypto trend bot on the Alpaca PAPER account (rules: scripts/PREREGISTRATION-crypto-trend-bot.md).

Two fixed rules (50-day breakout, 50/200 SMA cross) on BTC/USD and ETH/USD, long/flat. Each (asset, rule)
keeps its own quantity in a small state file, so rules sharing a coin never sell each other's holdings.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Sequence

STATE_PATH = Path.home() / ".vibe-trading" / "execution" / "crypto_bot.json"
BREAKOUT_IN, BREAKOUT_OUT, FAST, SLOW = 50, 20, 50, 200
MAX_SPREAD_NOTIONAL = 5000.0  # hard ceiling on the fixed size per position


@dataclass(frozen=True)
class BotConfig:
    symbols: tuple[str, ...] = ("BTC/USD", "ETH/USD")
    rules: tuple[str, ...] = ("breakout", "cross")
    notional: float = 5000.0


def desired_long(rule: str, closes: Sequence[float], is_long: bool) -> bool:
    """Target position from CLOSED daily closes (oldest first). Too little history means flat."""
    if rule == "cross":
        if len(closes) < SLOW:
            return False
        return sum(closes[-FAST:]) / FAST > sum(closes[-SLOW:]) / SLOW
    if rule == "breakout":
        if len(closes) < BREAKOUT_IN + 1:
            return False
        last = closes[-1]
        if last > max(closes[-BREAKOUT_IN - 1:-1]):
            return True
        if last < min(closes[-BREAKOUT_OUT - 1:-1]):
            return False
        return is_long
    raise ValueError(f"unknown rule: {rule}")


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {"checked": None, "qty": {}}


def _save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def run_check(
    cfg: BotConfig, *, trader: Any, closes_fn: Callable[[str], list[float]], quote_fn: Callable[[str], dict],
    log: Callable[[str], None], halted: Callable[[], bool], today: date, state_path: Path = STATE_PATH,
) -> None:
    """One daily pass. Safe to call repeatedly: it acts at most once per UTC day."""
    if not getattr(trader, "is_paper", False):
        raise RuntimeError("crypto bot refuses to run: the account is not marked as paper")
    if cfg.notional > MAX_SPREAD_NOTIONAL:
        raise ValueError("notional above the hard ceiling")
    if halted():
        log("CRYPTO BOT: kill switch is on; no orders")
        return
    state = _load(state_path)
    if state["checked"] == today.isoformat():
        return
    qty: dict[str, float] = dict(state["qty"])
    for symbol in cfg.symbols:
        closes = closes_fn(symbol)
        for rule in cfg.rules:
            key = f"{symbol}|{rule}"
            held = qty.get(key, 0.0)
            want = desired_long(rule, closes, held > 0)
            if want and held == 0:
                buy_qty = math.floor(cfg.notional / float(quote_fn(symbol)["ask"]) * 1e6) / 1e6
                price, got = trader.buy(symbol, buy_qty)
                qty[key] = got
                log(f"CRYPTO BOT {key}: bought {buy_qty} at {price:,.2f}")
            elif not want and held > 0:
                price = trader.sell(symbol, held)
                qty.pop(key)
                log(f"CRYPTO BOT {key}: sold {held} at {price:,.2f}")
            else:
                log(f"CRYPTO BOT {key}: {'holding' if held else 'flat'}")
            _save(state_path, {"checked": state["checked"], "qty": qty})  # survive a crash mid-pass
    _save(state_path, {"checked": today.isoformat(), "qty": qty})
