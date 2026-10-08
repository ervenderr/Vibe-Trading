"""run_signal must find the news/AI keys in the repo-root .env, like the dashboard does."""

from __future__ import annotations

import os

from src.signals.live_sources import ensure_signal_keys


def test_loads_only_signal_keys_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FINNHUB_API_KEY='abc'\nTYPESAFE_API_KEY=xyz\nOTHER_SECRET=nope\n")
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "already")
    monkeypatch.delenv("OTHER_SECRET", raising=False)
    ensure_signal_keys(env)
    assert os.environ["FINNHUB_API_KEY"] == "abc"
    assert os.environ["TYPESAFE_API_KEY"] == "already"
    assert "OTHER_SECRET" not in os.environ


def test_missing_file_is_a_no_op(tmp_path):
    ensure_signal_keys(tmp_path / "nope.env")
