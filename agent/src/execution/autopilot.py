"""Autopilot: signal -> guarded execution on a schedule, during market hours only.

Safety properties, each covered by a test:
  * the kill switch is checked every cycle and stops the loop;
  * an unreadable market clock means 'do nothing' (fail closed);
  * bracket legs are DAY orders and would expire at the close, leaving a position
    unprotected overnight, so everything is flattened shortly before the bell;
  * a failure on one symbol never stops the others.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

from src.execution.broker import BrokerError
from src.execution.executor import _default_halt
from src.signals.news_source import is_valid_symbol

logger = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")
CLOSED_SLEEP_S = 300.0
MIN_WAKE_S = 5.0
MAX_CONSECUTIVE_ERRORS = 3


@dataclass(frozen=True)
class CycleOutcome:
    status: str  # halted | no_clock | closed | closing | flattened | flatten_failed | traded | idle
    sleep_s: float


class Autopilot:
    def __init__(
        self,
        symbols: Sequence[str],
        *,
        runner: Callable[..., Any],
        executor: Any,
        broker: Any,
        dry_run: bool = True,
        interval_s: float = 120.0,
        flatten_minutes: float = 10.0,
        halt_check: Callable[[], bool] = _default_halt,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        sleep: Callable[[float], None] = time.sleep,
        out: Callable[[str], None] = print,
        heartbeat: Callable[[str, float], None] = lambda message, sleep_s: None,
        managed_symbols: Callable[[], set[str]] = lambda: set(),
    ) -> None:
        cleaned = [s.strip().upper() for s in symbols]
        bad = [s for s in cleaned if not is_valid_symbol(s)]
        if bad:
            raise ValueError(f"invalid symbol(s): {bad}")
        self._symbols = cleaned
        self._runner, self._executor, self._broker = runner, executor, broker
        self._dry_run, self._interval = dry_run, interval_s
        self._flatten_minutes = flatten_minutes
        self._halted, self._now, self._sleep, self._out = halt_check, now_fn, sleep, out
        self._managed = managed_symbols
        self._heartbeat = heartbeat
        self._checked = self._submitted = 0
        self._closed_flatten_day = None
        self._flattened_day = None
        self._last: dict[str, tuple] = {}

    def cycle(self) -> CycleOutcome:
        """One pass. Never raises: an unexpected error becomes an 'error' outcome."""
        self._checked = self._submitted = 0
        try:
            outcome = self._cycle()
        except Exception as exc:
            logger.exception("autopilot cycle failed")
            self._out(f"cycle error ({type(exc).__name__}); will retry")
            outcome = CycleOutcome("error", self._interval)
        self._beat(outcome)
        return outcome

    def _beat(self, outcome: CycleOutcome) -> None:
        """Leave a trace every cycle, even a quiet one. Never allowed to disturb trading."""
        n = self._checked
        trades = f"{self._submitted} trade{'s' if self._submitted != 1 else ''}" if self._submitted else "no trades"
        messages = {
            "closed": "market closed; waiting for the open",
            "halted": "kill switch is on; autopilot stopped",
            "no_clock": "market clock unavailable; did nothing",
            "closing": "end-of-day window; no new trades",
            "flattened": "end of day: closed everything it opened",
            "flatten_failed": "end of day: could not confirm everything is closed",
            "error": "cycle error; will retry",
        }
        message = messages.get(outcome.status, f"checked {n} symbol{'s' if n != 1 else ''}, {trades}")
        try:
            self._heartbeat(message, outcome.sleep_s)
        except Exception:
            logger.warning("heartbeat failed", exc_info=True)

    def _cycle(self) -> CycleOutcome:
        if self._halted():
            self._warn_open_exposure()
            self._out("kill switch is set: autopilot stopped")
            return CycleOutcome("halted", 0.0)
        try:
            clock = self._broker.market_clock()
        except BrokerError as exc:
            self._out(f"market clock unavailable ({exc}); doing nothing this cycle")
            return CycleOutcome("no_clock", self._interval)
        if not clock.is_open:
            self._after_hours_cleanup(clock.now)
            return CycleOutcome("closed", max(self._interval, CLOSED_SLEEP_S))

        minutes_left = (clock.next_close - clock.now).total_seconds() / 60.0
        if minutes_left <= self._flatten_minutes:
            return self._closing(clock.now)
        return self._trade(self._wake_before_window(minutes_left))

    def _wake_before_window(self, minutes_left: float) -> float:
        """Sleep no longer than the time until the flatten window opens, so one cycle always lands in it."""
        return max(MIN_WAKE_S, min(self._interval, (minutes_left - self._flatten_minutes) * 60.0))

    def _warn_open_exposure(self) -> None:
        """The kill switch stops trading but cannot safely trade out of positions: say what is still open."""
        try:
            open_items = self._broker.exposure(self._managed())
        except Exception:
            return
        if open_items:
            self._out("WARNING: positions/orders still open (bracket legs expire at the close): " + ", ".join(open_items))

    def _after_hours_cleanup(self, now: datetime) -> None:
        """If a position survived past the close (missed or failed flatten), try once per day to queue its exit."""
        if self._dry_run:
            return
        day = now.astimezone(NY).date()
        if self._closed_flatten_day == day:
            return
        try:
            leftover = self._broker.exposure(self._managed())
        except Exception:
            return
        if not leftover:
            return
        self._closed_flatten_day = day
        result = self._broker.flatten(self._managed())
        self._out(f"after hours: found {', '.join(leftover)}; flatten {'queued' if result.get('status') == 'ok' else 'incomplete (' + str(result.get('error')) + ')'}")

    def _closing(self, now: datetime) -> CycleOutcome:
        day = now.astimezone(NY).date()
        if self._dry_run:
            if self._flattened_day != day:
                self._out("dry run: would flatten all positions before the close")
                self._flattened_day = day
            return CycleOutcome("closing", self._interval)
        if self._flattened_day == day:
            return CycleOutcome("closing", self._interval)
        result = self._broker.flatten(self._managed())
        if result.get("status") != "ok":
            self._out(f"FLATTEN FAILED, will retry: {result.get('error')}")
            return CycleOutcome("flatten_failed", self._interval)
        self._flattened_day = day
        self._out(f"end of day: cancelled {result.get('cancelled')} orders, closed {result.get('closed')} positions")
        return CycleOutcome("flattened", self._interval)

    def _trade(self, sleep_s: float) -> CycleOutcome:
        traded = False
        for symbol in self._symbols:
            try:
                result = self._executor.execute(self._runner(symbol), dry_run=self._dry_run)
            except Exception as exc:
                logger.exception("autopilot failed for %s", symbol)
                self._report(symbol, ("failed", (type(exc).__name__,)))
                continue
            self._checked += 1
            self._submitted += result.status == "submitted"
            traded = traded or result.status == "submitted"
            self._report(symbol, (result.status, result.reasons), result)
        return CycleOutcome("traded" if traded else "idle", sleep_s)

    def _report(self, symbol: str, key: tuple, result: Any = None) -> None:
        if self._last.get(symbol) == key and not (result is not None and result.status == "submitted"):
            return  # unchanged since last cycle: stay quiet
        self._last[symbol] = key
        status, reasons = key
        line = f"{symbol}: {status}" + (f" - {'; '.join(reasons)}" if reasons else "")
        if result is not None and result.plan is not None:
            p = result.plan
            line += f" [{p.side} {p.qty} @ {p.entry:.2f}, stop {p.stop:.2f}, target {p.target:.2f}]"
        self._out(line)

    def run(self, max_cycles: int | None = None) -> int:
        count = errors = 0
        try:
            while max_cycles is None or count < max_cycles:
                outcome = self.cycle()
                count += 1
                if outcome.status == "halted":
                    return 0
                errors = errors + 1 if outcome.status == "error" else 0
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    return self._give_up()
                if max_cycles is not None and count >= max_cycles:
                    break
                self._sleep(outcome.sleep_s)
        except KeyboardInterrupt:
            self._out("stopped")
        return 0

    def _give_up(self) -> int:
        self._out("too many consecutive errors: stopping")
        if not self._dry_run:
            try:
                result = self._broker.flatten(self._managed())
                self._out(f"emergency flatten: {result.get('status')}")
            except Exception as exc:
                self._out(f"emergency flatten failed ({type(exc).__name__}); check the account")
        return 1
