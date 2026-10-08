"""Append-only journal of autopilot output, so a dashboard can show what happened."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)
DEFAULT_PATH = Path.home() / ".vibe-trading" / "execution" / "autopilot.jsonl"
MAX_MESSAGE_CHARS = 600
TAIL_BYTES = 256 * 1024


class Journal:
    def __init__(self, path: Path | None = None, *, now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self._path = Path(path) if path is not None else DEFAULT_PATH
        self._now = now_fn

    def beat(self, message: str, next_s: float) -> None:
        """A heartbeat row: proof the autopilot is alive, plus when it will look again."""
        self.append(message, kind="beat", next_s=next_s)

    def append(self, message: str, *, kind: str | None = None, next_s: float | None = None) -> None:
        row: dict = {"t": self._now().isoformat(), "msg": str(message)[:MAX_MESSAGE_CHARS]}
        if kind:
            row["k"] = kind
        if next_s is not None:
            row["next_s"] = next_s
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("journal write failed: %s", exc)  # never let logging stop trading logic

    def tee(self, out: Callable[[str], None]) -> Callable[[str], None]:
        def write(message: str) -> None:
            self.append(message)
            out(message)

        return write

    def recent(self, limit: int = 200) -> list[dict]:
        try:
            with open(self._path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - TAIL_BYTES))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        rows = []
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and "msg" in row and "t" in row:
                rows.append(row)
        return rows[-limit:]
