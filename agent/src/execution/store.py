"""Persistent execution state: per-day counters and an append-only order log.

Private files (0600 in a 0700 directory). Everything that reads then writes runs inside
``transaction()``: a thread lock plus a cross-process ``flock``, so two runners (the
autopilot and a manual ``execute``) cannot both pass a check and both submit.

Corrupt state fails CLOSED (``StoreError``): silently resetting would also reset the
daily loss and trade-count limits.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

try:  # POSIX only; threads are still serialised by the module lock on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from src.execution.models import DayState, ExecutionPlan

logger = logging.getLogger(__name__)
DEFAULT_DIR = Path.home() / ".vibe-trading" / "execution"
_THREAD_LOCK = threading.Lock()


class StoreError(RuntimeError):
    """Execution state is unreadable. Callers must not trade."""


def _day(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y-%m-%d")


class ExecutionStore:
    def __init__(self, directory: Path = DEFAULT_DIR) -> None:
        self._dir = Path(directory)
        self._state_path = self._dir / "state.json"
        self._log_path = self._dir / "executions.jsonl"
        self._lock_path = self._dir / "store.lock"
        self._local = threading.local()

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self._dir, 0o700)
        except OSError:
            pass

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Exclusive section across threads and processes; safe to nest."""
        depth = getattr(self._local, "depth", 0)
        if depth:
            self._local.depth = depth + 1
            try:
                yield
            finally:
                self._local.depth -= 1
            return
        self._ensure_dir()
        with _THREAD_LOCK:
            fd = None
            if fcntl is not None:
                fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX)
            self._local.depth = 1
            try:
                yield
            finally:
                self._local.depth = 0
                if fd is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)

    def _read_state(self) -> dict:
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (ValueError, OSError) as exc:
            raise StoreError(f"execution state unreadable: {type(exc).__name__}") from None
        if not isinstance(raw, dict):
            raise StoreError("execution state malformed")
        return raw

    def _write_state(self, state: dict) -> None:
        self._ensure_dir()
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._state_path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def day_state(self, now: datetime, equity: float, baseline: float | None = None) -> DayState:
        """Today's counters. ``start_equity`` is fixed at the first look of the day, using
        the broker's previous-close equity as ``baseline`` when it is usable."""
        today = _day(now)
        with self.transaction():
            state = self._read_state()
            valid = state.get("day") == today and isinstance(state.get("start_equity"), (int, float))
            if not valid or state["start_equity"] <= 0:
                good = baseline is not None and math.isfinite(baseline) and baseline > 0
                state = {"day": today, "start_equity": float(baseline if good else equity),
                         "trades": int(state.get("trades", 0)) if state.get("day") == today else 0}
                self._write_state(state)
            return DayState(today, float(state["start_equity"]), int(state.get("trades", 0)))

    def _rows(self) -> list[dict]:
        try:
            lines = self._log_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        rows = []
        for line in lines:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except ValueError:
                continue
        return rows

    def already_submitted(self, client_order_id: str) -> bool:
        return any(r.get("client_order_id") == client_order_id for r in self._rows())

    def managed_symbols(self) -> set[str]:
        """Every symbol this system has ever submitted an order for."""
        return {str(r["symbol"]).upper() for r in self._rows() if r.get("symbol")}

    def last_submission(self, symbol: str) -> datetime | None:
        times = []
        for r in self._rows():
            if str(r.get("symbol", "")).upper() == symbol.upper():
                try:
                    times.append(datetime.fromisoformat(r["time"]))
                except (KeyError, ValueError, TypeError):
                    continue
        return max(times) if times else None

    def record(self, plan: ExecutionPlan, order: dict, now: datetime) -> None:
        """Log a submitted order and count it against today's budget."""
        row = {
            "time": now.astimezone(timezone.utc).isoformat(),
            "symbol": plan.symbol, "side": plan.side, "qty": plan.qty,
            "entry": plan.entry, "stop": plan.stop, "target": plan.target,
            "confidence": plan.confidence, "client_order_id": plan.client_order_id,
            "order_id": order.get("order_id"), "order_status": order.get("order_status"),
        }
        with self.transaction():
            self._ensure_dir()
            fd = os.open(self._log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            state = self._read_state()
            today = _day(now)
            if state.get("day") != today:
                state = {"day": today, "start_equity": 0.0, "trades": 0}
            state["trades"] = int(state.get("trades", 0)) + 1
            self._write_state(state)
