"""Read-only access to TradingAgents' saved reports, for the dashboard's AI research desk.

TradingAgents writes ``<base>/<TICKER>/<YYYY-MM-DD>/reports/*.md``. Report text is untrusted model
output: it is returned as data and the page renders it as text, never as HTML.
"""

from __future__ import annotations

import re
from pathlib import Path

DEFAULT_BASE = Path.home() / ".tradingagents" / "logs"
MAX_REPORT_CHARS = 60_000
MAX_RUNS = 50
TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,11}$")
DATE_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")
NAME_RE = re.compile(r"^[a-z0-9_]{1,60}$")
FINAL_REPORT = "final_trade_decision"
ORDER = ("market_report", "sentiment_report", "news_report", "fundamentals_report", "investment_plan",
         "trader_investment_plan", "risk_report", "final_trade_decision")
THINK_RE = re.compile(r"<think>(.*?)(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)


def _reports_dir(base: Path, ticker: str, date: str) -> Path | None:
    if not (TICKER_RE.match(ticker) and DATE_RE.match(date)):
        return None
    folder = (base / ticker / date / "reports")
    try:
        resolved = folder.resolve(strict=True)
    except OSError:
        return None
    return resolved if base.resolve() in resolved.parents else None


def _names(folder: Path) -> list[str]:
    found = [p.stem for p in folder.glob("*.md") if NAME_RE.match(p.stem) and p.is_file() and not p.is_symlink()]
    return sorted(found, key=lambda n: (ORDER.index(n) if n in ORDER else len(ORDER), n))


def list_runs(base: Path = DEFAULT_BASE) -> list[dict]:
    runs = []
    try:
        tickers = [p for p in base.iterdir() if p.is_dir() and TICKER_RE.match(p.name)]
    except OSError:
        return []
    for ticker in tickers:
        for day in ticker.iterdir():
            folder = _reports_dir(base, ticker.name, day.name)
            if folder is None:
                continue
            names = _names(folder)
            runs.append({"ticker": ticker.name, "date": day.name, "reports": names,
                         "complete": FINAL_REPORT in names, "updated": day.stat().st_mtime})
    return sorted(runs, key=lambda r: r["updated"], reverse=True)[:MAX_RUNS]


def read_run(base: Path, ticker: str, date: str) -> dict | None:
    folder = _reports_dir(base, ticker, date)
    if folder is None:
        return None
    reports = []
    for name in _names(folder):
        raw = (folder / f"{name}.md").read_text(encoding="utf-8", errors="replace")
        reasoning = "\n".join(m.strip() for m in THINK_RE.findall(raw))
        text = THINK_RE.sub("", raw).strip()
        truncated = len(text) > MAX_REPORT_CHARS
        reports.append({"name": name, "text": text[:MAX_REPORT_CHARS] + ("\n[cut: report is longer]" if truncated else ""),
                        "reasoning": reasoning[:MAX_REPORT_CHARS], "truncated": truncated})
    return {"ticker": ticker, "date": date, "complete": any(r["name"] == FINAL_REPORT for r in reports), "reports": reports}
