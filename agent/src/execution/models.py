"""Plain data types for the execution bridge."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

PlanStatus = Literal["planned", "skipped"]
ExecStatus = Literal["dry_run", "submitted", "skipped", "refused", "error"]


@dataclass(frozen=True)
class ExecutionLimits:
    """Hard limits. Conservative defaults; every one is enforced before an order."""

    min_confidence: float = 0.6
    risk_per_trade: float = 0.005  # fraction of equity risked between entry and stop
    max_order_notional: float = 10_000.0
    max_open_positions: int = 3
    max_trades_per_day: int = 10
    daily_loss_limit: float = 0.02  # stop opening trades after losing this fraction of the day's start equity
    allow_shorts: bool = False
    min_price: float = 5.0
    buying_power_fraction: float = 0.5  # never commit more than this share of buying power to one order
    max_price_deviation: float = 0.015  # refuse if the live price is further than this from the signal entry
    cooldown_minutes: float = 60.0  # no re-entry in a symbol this soon after our last submission
    min_minutes_to_close: float = 15.0  # no new entries this close to the bell


@dataclass(frozen=True)
class AccountState:
    equity: float
    buying_power: float
    trading_blocked: bool
    last_equity: float | None = None  # previous close; the stable baseline for the daily loss limit


@dataclass(frozen=True)
class DayState:
    day: str  # YYYY-MM-DD (UTC)
    start_equity: float
    trades: int


@dataclass(frozen=True)
class ExecutionPlan:
    symbol: str
    side: str  # "buy" | "sell"
    qty: int
    entry: float
    stop: float
    target: float
    client_order_id: str
    confidence: float


@dataclass(frozen=True)
class PlanDecision:
    status: PlanStatus
    plan: ExecutionPlan | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionResult:
    status: ExecStatus
    reasons: tuple[str, ...]
    plan: ExecutionPlan | None = None
    order: dict | None = None


@dataclass(frozen=True)
class MarketClock:
    is_open: bool
    now: datetime
    next_open: datetime
    next_close: datetime
