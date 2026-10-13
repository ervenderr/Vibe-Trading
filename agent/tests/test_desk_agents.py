"""AI research desk: safe, read-only access to TradingAgents' saved reports."""

from __future__ import annotations

import os

import pytest

from src.desk.agents import MAX_REPORT_CHARS, list_runs, read_run


def make(tmp_path, ticker="NVDA", date="2026-10-13", **reports):
    folder = tmp_path / ticker / date / "reports"
    folder.mkdir(parents=True)
    for name, text in reports.items():
        (folder / f"{name}.md").write_text(text)
    return tmp_path


def test_lists_runs_newest_first_with_report_names(tmp_path):
    make(tmp_path, "NVDA", "2026-10-12", market_report="a")
    make(tmp_path, "AAPL", "2026-10-13", market_report="a", final_trade_decision="b")
    os.utime(tmp_path / "NVDA" / "2026-10-12", (1, 1))
    import time
    runs = list_runs(tmp_path, now=time.time() + 600)   # long after the files were written
    assert [(r["ticker"], r["date"]) for r in runs] == [("AAPL", "2026-10-13"), ("NVDA", "2026-10-12")]
    assert runs[0]["complete"] is True and runs[1]["complete"] is False
    assert "market_report" in runs[1]["reports"]


def test_read_returns_ordered_reports_and_separates_reasoning(tmp_path):
    make(tmp_path, market_report="# Market\nfine", investment_plan="### Plan\n<think>secret chain</think>\nBuy it")
    run = read_run(tmp_path, "NVDA", "2026-10-13")
    assert [r["name"] for r in run["reports"]] == ["market_report", "investment_plan"]
    plan = run["reports"][1]
    assert "<think>" not in plan["text"] and "Buy it" in plan["text"] and "secret chain" in plan["reasoning"]


def test_unterminated_think_block_is_not_shown_as_the_answer(tmp_path):
    make(tmp_path, investment_plan="intro\n<think>never closed ... long reasoning")
    plan = read_run(tmp_path, "NVDA", "2026-10-13")["reports"][0]
    assert "long reasoning" not in plan["text"] and "long reasoning" in plan["reasoning"]


@pytest.mark.parametrize("ticker,date", [("../etc", "2026-10-13"), ("NVDA", "../../x"), ("NV DA", "2026-10-13"), ("NVDA", "2026-13-45x"), ("", "")])
def test_rejects_traversal_and_bad_names(tmp_path, ticker, date):
    make(tmp_path)
    assert read_run(tmp_path, ticker, date) is None


def test_missing_run_is_none_and_empty_dir_lists_nothing(tmp_path):
    assert read_run(tmp_path, "NVDA", "2026-10-13") is None
    assert list_runs(tmp_path / "does-not-exist") == []


def test_symlinks_and_huge_files_are_handled(tmp_path):
    make(tmp_path, market_report="x" * (MAX_REPORT_CHARS + 500))
    outside = tmp_path.parent / "outside.md"
    outside.write_text("TOP SECRET")
    (tmp_path / "NVDA" / "2026-10-13" / "reports" / "leak.md").symlink_to(outside)
    run = read_run(tmp_path, "NVDA", "2026-10-13")
    names = [r["name"] for r in run["reports"]]
    assert "leak" not in names                                  # a link out of the folder is never followed
    assert len(run["reports"][0]["text"]) <= MAX_REPORT_CHARS + 40 and run["reports"][0]["truncated"] is True


def test_a_report_still_being_written_is_not_complete(tmp_path):
    import time
    make(tmp_path, market_report="a", final_trade_decision="partial")
    final = tmp_path / "NVDA" / "2026-10-13" / "reports" / "final_trade_decision.md"
    assert read_run(tmp_path, "NVDA", "2026-10-13", now=time.time())["complete"] is False        # just written
    assert list_runs(tmp_path, now=time.time())[0]["complete"] is False
    assert read_run(tmp_path, "NVDA", "2026-10-13", now=final.stat().st_mtime + 600)["complete"] is True   # settled
