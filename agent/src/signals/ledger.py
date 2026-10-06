"""Signal ledger: record every BUY/SELL signal, then grade it against what price did.

This is the honest scoreboard. Signals are appended to a JSONL file and graded
later with the same conservative rules as the replay (stop wins a same-bar tie).
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from src.signals.service import BAR_MINUTES, SignalReport
from src.signals.verdict import Action

try:  # POSIX only; threads are still serialised by the module lock on Windows
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)
_WRITE_LOCK = threading.Lock()
DEFAULT_LEDGER = Path.home() / ".vibe-trading" / "signals" / "ledger.jsonl"


@dataclass(frozen=True)
class LedgerEntry:
    symbol: str
    action: str  # BUY | SELL
    confidence: float
    generated_at: datetime
    valid_until: datetime
    entry: float
    stop: float
    target: float


@dataclass(frozen=True)
class Grade:
    outcome: str  # target | stop | expired | pending | nodata
    gross_return: float  # fractional, before costs


@dataclass(frozen=True)
class GradeSummary:
    resolved: int
    pending: int
    win_rate: float
    avg_return: float


def _valid_entry(e: LedgerEntry) -> bool:
    prices = (e.entry, e.stop, e.target)
    return (
        e.generated_at.tzinfo is not None
        and e.valid_until.tzinfo is not None
        and e.action in (Action.BUY.value, Action.SELL.value)
        and all(isinstance(p, (int, float)) and math.isfinite(p) and p > 0 for p in prices)
    )


class SignalLedger:
    """Append-only JSONL store of actionable signals (private files, process-safe)."""

    def __init__(self, path: Path = DEFAULT_LEDGER) -> None:
        self._path = Path(path)

    def record(self, report: SignalReport) -> bool:
        """Append an actionable signal; returns False if skipped (WAIT/AVOID/duplicate)."""
        v = report.verdict
        if v.action not in (Action.BUY, Action.SELL) or v.entry is None:
            return False
        entry = LedgerEntry(
            report.symbol, v.action.value, v.confidence, report.generated_at,
            v.valid_until, v.entry, v.stop, v.target,
        )
        if not _valid_entry(entry):
            return False
        row = {k: (x.isoformat() if isinstance(x, datetime) else x) for k, x in asdict(entry).items()}
        with self._locked():
            if self._is_duplicate(entry):
                return False
            fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        return True

    def load(self) -> tuple[LedgerEntry, ...]:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return ()
        entries = []
        for line in lines:
            try:
                raw = json.loads(line)
                raw["generated_at"] = datetime.fromisoformat(raw["generated_at"])
                raw["valid_until"] = datetime.fromisoformat(raw["valid_until"])
                entry = LedgerEntry(**raw)
            except (ValueError, KeyError, TypeError):
                logger.warning("skipping unreadable ledger line")
                continue
            if _valid_entry(entry):
                entries.append(entry)
            else:
                logger.warning("skipping invalid ledger entry")
        return tuple(entries)

    def _is_duplicate(self, new: LedgerEntry) -> bool:
        return any(
            e.symbol == new.symbol and e.action == new.action and new.generated_at < e.valid_until
            for e in self.load()
        )

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Serialise check-then-append across threads and processes."""
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self._path.parent, 0o700)
        except OSError:
            pass
        with _WRITE_LOCK:
            lock_fd = None
            if fcntl is not None:
                lock_fd = os.open(self._path.with_suffix(".lock"), os.O_RDWR | os.O_CREAT, 0o600)
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if lock_fd is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    os.close(lock_fd)


def grade_entry(entry: LedgerEntry, bars: pd.DataFrame, now: datetime | None = None) -> Grade:
    """Grade one signal against later OHLC bars.

    The fill is the first bar's *open* after the signal, with the stop and target kept
    at the signal's distances (as in the replay). A stop wins a same-bar tie and a gap
    through a level fills at the gap open. ``now`` lets old signals resolve to
    ``expired``/``nodata`` instead of staying ``pending`` forever.
    """
    over = now is not None and now >= entry.valid_until
    if bars.empty or not entry.entry > 0:
        return Grade("nodata" if over else "pending", 0.0)
    window = bars[(bars.index >= entry.generated_at) & (bars.index < entry.valid_until)]
    if window.empty:
        return Grade("nodata" if over else "pending", 0.0)
    side = 1.0 if entry.action == Action.BUY.value else -1.0
    fill = float(window["open"].iloc[0])
    if not fill > 0:
        return Grade("nodata", 0.0)
    stop = fill - side * abs(entry.entry - entry.stop)
    target = fill + side * abs(entry.target - entry.entry)
    for _, bar in window.iterrows():
        open_ = float(bar["open"])
        hit_stop = bar["low"] <= stop if side > 0 else bar["high"] >= stop
        hit_target = bar["high"] >= target if side > 0 else bar["low"] <= target
        if hit_stop:
            px = min(open_, stop) if side > 0 else max(open_, stop)
            return Grade("stop", side * (px - fill) / fill)
        if hit_target:
            px = max(open_, target) if side > 0 else min(open_, target)
            return Grade("target", side * (px - fill) / fill)
    covered = window.index[-1] + timedelta(minutes=BAR_MINUTES) >= entry.valid_until
    if not (covered or over):
        return Grade("pending", 0.0)
    return Grade("expired", side * (float(window["close"].iloc[-1]) - fill) / fill)


def summarize_grades(grades: Iterable[Grade]) -> GradeSummary:
    items = list(grades)
    resolved = [g for g in items if g.outcome not in ("pending", "nodata")]
    wins = sum(1 for g in resolved if g.gross_return > 0)
    return GradeSummary(
        resolved=len(resolved),
        pending=sum(1 for g in items if g.outcome == "pending"),
        win_rate=wins / len(resolved) if resolved else 0.0,
        avg_return=sum(g.gross_return for g in resolved) / len(resolved) if resolved else 0.0,
    )
