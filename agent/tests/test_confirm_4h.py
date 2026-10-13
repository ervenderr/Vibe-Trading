"""The pre-registered success rule and exploratory splits of scripts/confirm_4h.py."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import confirm_4h as c  # noqa: E402
from src.signals.event_study import Event, HorizonStats  # noqa: E402


def stats(n=150, p=0.01, mean=12.0, hit=0.55) -> HorizonStats:
    return HorizonStats("4h", n, hit, mean, mean, 2.5, p, mean - 6)


def test_success_needs_every_criterion():
    assert c.verdict(stats())[0] is True
    for bad in (stats(n=99), stats(p=0.06), stats(mean=5.9), stats(hit=0.5)):
        assert c.verdict(bad)[0] is False


def test_splits_drop_small_groups():
    t = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = [Event("NVDA", t, 1, "earnings", 0.5)] * 30 + [Event("TSLA", t, -1, "other", 0.5)] * 5
    groups = c.splits(events)
    assert "symbol=NVDA" in groups and "side=up" in groups and "symbol=TSLA" not in groups
