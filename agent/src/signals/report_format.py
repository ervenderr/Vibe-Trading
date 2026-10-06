"""Render a :class:`SignalReport` as JSON-safe data or readable text."""

from __future__ import annotations

from typing import Any

from src.signals.service import SignalReport
from src.signals.verdict import Action

DISCLAIMER = (
    "Advisory only: no orders were placed. Signals can be wrong; this is not financial "
    "advice. Check the track record before relying on it."
)


def report_to_dict(report: SignalReport) -> dict[str, Any]:
    v = report.verdict
    warnings = []
    if report.stale:
        warnings.append("Price data is stale (market closed or feed delayed).")
    if report.news_error:
        warnings.append(report.news_error)
    if report.fallback_count:
        warnings.append(
            f"Jev unavailable for {report.fallback_count} headline(s); keyword fallback ignored (price-only verdict)."
        )
    return {
        "symbol": report.symbol,
        "action": v.action.value,
        "confidence": round(v.confidence, 3),
        "entry": v.entry,
        "stop": v.stop,
        "target": v.target,
        "valid_until": v.valid_until.isoformat(),
        "reasons": list(v.reasons),
        "news_score": round(v.news_score, 3),
        "trend_score": round(v.tech_score, 3),
        "generated_at": report.generated_at.isoformat(),
        "price_as_of": report.bar_time.isoformat() if report.bar_time else None,
        "headlines": [
            {
                "headline": h.item.headline,
                "source": h.item.source,
                "published": h.item.published.isoformat(),
                "score": round(h.signal.score, 3),
                "event_type": h.judgment.event_type if h.judgment else None,
                "relevance": round(h.judgment.relevance, 3) if h.judgment else None,
                "origin": h.origin,
            }
            for h in report.headlines
        ],
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }


def render_text(report: SignalReport) -> str:
    v = report.verdict
    lines = [f"{report.symbol}: {v.action.value}" + (f"  (confidence {v.confidence:.0%})" if v.action in (Action.BUY, Action.SELL) else "")]
    if v.action in (Action.BUY, Action.SELL):
        lines.append(f"  entry {v.entry:.2f}   stop {v.stop:.2f}   target {v.target:.2f}")
        lines.append(f"  valid until {v.valid_until:%H:%M} UTC")
    lines.extend(f"  - {r}" for r in v.reasons)
    for h in report.headlines:
        lines.append(f"  news {h.signal.score:+.2f}  {h.item.headline[:90]}")
    lines.extend(f"  ! {w}" for w in report_to_dict(report)["warnings"])
    lines.append(f"  {DISCLAIMER}")
    return "\n".join(lines)
