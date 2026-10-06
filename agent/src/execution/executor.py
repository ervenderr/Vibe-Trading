"""Paper executor: kill switch, paper-only guard, market gate, planning, dedupe, submission.

Defaults to dry run. Nothing is read from or sent to the broker until the kill switch and
paper-only checks pass. Real orders additionally require an open market that is not about
to close, a fresh live price, and an exclusive transaction around check -> submit -> record
so concurrent runners cannot double-submit. The kill switch is re-checked immediately
before the order leaves.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol

from src.execution.broker import BrokerError
from src.execution.models import AccountState, ExecutionLimits, ExecutionPlan, ExecutionResult
from src.execution.planner import plan_execution
from src.execution.store import ExecutionStore, StoreError
from src.live.halt import halt_flag_set
from src.signals.news_source import is_valid_symbol
from src.signals.text_safety import sanitize_text
from src.signals.verdict import Action

logger = logging.getLogger(__name__)
BROKER_KEY = "alpaca"


class Broker(Protocol):
    is_paper: bool

    def account(self) -> AccountState: ...
    def positions(self) -> list[dict]: ...
    def open_orders(self) -> list[dict]: ...
    def market_clock(self): ...
    def last_price(self, symbol: str, side: str) -> float: ...
    def submit_bracket(self, plan: ExecutionPlan) -> dict: ...


def _default_halt() -> bool:
    return halt_flag_set(BROKER_KEY)


def _skip(*reasons: str, plan: ExecutionPlan | None = None) -> ExecutionResult:
    return ExecutionResult("skipped", tuple(reasons), plan)


class PaperExecutor:
    def __init__(
        self,
        broker: Broker,
        store: ExecutionStore,
        limits: ExecutionLimits | None = None,
        halt_check: Callable[[], bool] = _default_halt,
    ) -> None:
        self._broker = broker
        self._store = store
        self._limits = limits or ExecutionLimits()
        self._halted = halt_check

    @property
    def store(self) -> ExecutionStore:
        return self._store

    def execute(self, report, *, now: datetime | None = None, dry_run: bool = True) -> ExecutionResult:
        now = now or datetime.now(timezone.utc)
        if self._halted():
            return ExecutionResult("refused", ("kill switch (HALT) is set; all execution stopped",))
        if not self._broker.is_paper:
            return ExecutionResult("refused", ("live account refused: this executor is paper-only",))
        try:
            if not dry_run:
                gate = self._market_gate()
                if gate:
                    return _skip(gate)
            account = self._broker.account()
            positions = self._broker.positions()
            orders = self._broker.open_orders()
            live_price, price_problem = self._live_price(report)
        except BrokerError as exc:
            logger.warning("broker state unavailable: %s", exc)
            return _skip("broker state unavailable; not trading (fail-closed)")
        if price_problem:
            return _skip(price_problem)

        with self._store.transaction():
            try:
                day = self._store.day_state(now, account.equity, baseline=account.last_equity)
                decision = plan_execution(report, account, positions, orders, day, self._limits, now, live_price=live_price)
                if decision.plan is None:
                    return _skip(*decision.reasons)
                plan = decision.plan
                if self._store.already_submitted(plan.client_order_id):
                    return _skip("duplicate: this signal window was already submitted", plan=plan)
                last = self._store.last_submission(plan.symbol)
            except StoreError as exc:
                logger.warning("execution state problem: %s", exc)
                return _skip("execution state unreadable; not trading (fail-closed)")
            cooldown = timedelta(minutes=self._limits.cooldown_minutes)
            if last is not None and now - last < cooldown:
                return _skip(f"cooldown: {plan.symbol} was traded {(now - last).total_seconds() / 60:.0f} min ago", plan=plan)
            if dry_run:
                return ExecutionResult("dry_run", decision.reasons, plan)
            if self._halted():  # re-check at the last moment before the order leaves
                return ExecutionResult("refused", ("kill switch (HALT) tripped before submission",), plan)

            order = self._broker.submit_bracket(plan)
            if order.get("status") != "ok":
                return ExecutionResult("error", (f"broker rejected the order: {sanitize_text(order.get('error'), 160)}",), plan, order)
            reasons = decision.reasons
            try:
                self._store.record(plan, order, now)
            except Exception:
                logger.exception("order %s was placed but could not be logged", order.get("order_id"))
                reasons = reasons + ("WARNING: order placed but could not be logged; check the account",)
            return ExecutionResult("submitted", reasons, plan, order)

    def _market_gate(self) -> str | None:
        """Why a real order must not go out right now, or None. Unknown means no."""
        clock = self._broker.market_clock()  # BrokerError propagates -> fail closed
        if not clock.is_open:
            return "market is closed"
        minutes_left = (clock.next_close - clock.now).total_seconds() / 60.0
        if minutes_left <= self._limits.min_minutes_to_close:
            return f"too close to the close ({minutes_left:.0f} min left); no new entries"
        return None

    def _live_price(self, report) -> tuple[float | None, str | None]:
        v = report.verdict
        if v.action not in (Action.BUY, Action.SELL) or not is_valid_symbol(report.symbol):
            return None, None
        side = "buy" if v.action is Action.BUY else "sell"
        try:
            return self._broker.last_price(report.symbol.upper(), side), None
        except BrokerError:
            return None, "live price unavailable; not trading (fail-closed)"
