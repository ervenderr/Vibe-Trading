"""Autopilot journal: timestamped JSONL of what the autopilot said, readable by a dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.execution.journal import Journal

T = datetime(2026, 10, 6, 14, 0, tzinfo=timezone.utc)


def test_tee_writes_event_and_forwards(tmp_path):
    seen: list[str] = []
    j = Journal(tmp_path / "j.jsonl", now_fn=lambda: T)
    j.tee(seen.append)("NVDA: idle - no edge")
    assert seen == ["NVDA: idle - no edge"]
    row = json.loads((tmp_path / "j.jsonl").read_text().splitlines()[0])
    assert row == {"t": T.isoformat(), "msg": "NVDA: idle - no edge"}


def test_recent_returns_newest_last_and_limits(tmp_path):
    j = Journal(tmp_path / "j.jsonl", now_fn=lambda: T)
    for i in range(5):
        j.append(f"m{i}")
    assert [e["msg"] for e in j.recent(3)] == ["m2", "m3", "m4"]


def test_recent_skips_corrupt_lines_and_missing_file(tmp_path):
    p = tmp_path / "j.jsonl"
    assert Journal(p).recent(10) == []
    p.write_text('not json\n{"t": "x", "msg": "ok"}\n[1]\n')
    assert [e["msg"] for e in Journal(p).recent(10)] == ["ok"]


def test_write_failure_never_breaks_the_autopilot(tmp_path):
    seen: list[str] = []
    j = Journal(tmp_path / "missing" / "deep" / "j.jsonl")
    (tmp_path / "missing").write_text("a file, so mkdir fails")
    j.tee(seen.append)("still printed")
    assert seen == ["still printed"]


def test_messages_are_truncated(tmp_path):
    j = Journal(tmp_path / "j.jsonl", now_fn=lambda: T)
    j.append("x" * 5000)
    assert len(j.recent(1)[0]["msg"]) <= 600
