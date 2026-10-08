"""Rule-based verdict: combine news, price action and risk flags.

Advisory only. Jev supplies the news judgment; every threshold and decision is
explicit code so it can be inspected, tuned and backtested.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable, Sequence

from src.signals.news_judgment import NewsSignal
from src.signals.technicals import TechnicalState, tech_score, trend_direction


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"
    AVOID = "AVOID"


@dataclass(frozen=True)
class VerdictConfig:
    """Tunable policy. Defaults are starting points to be tuned on real results."""

    news_window_minutes: float = 240.0
    news_half_life_minutes: float = 30.0
    weight_news: float = 0.5
    weight_tech: float = 0.5
    entry_threshold: float = 0.35  # |combined| needed when fresh news exists
    tech_only_threshold: float = 0.6  # |tech| needed when there is no news
    disagree_min: float = 0.2  # both sides at least this strong to count as conflict
    max_item_score: float = 0.7  # one item can never saturate the news score
    min_trend_confirmation: float = 0.15  # news alone may not open a trade
    stop_atr: float = 2.0  # one 5-minute ATR is inside normal noise; 2 gives a trade room to work
    target_atr: float = 3.0
    min_stop_pct: float = 0.003  # floor for calm symbols (e.g. gold ETFs): stop is never closer than 0.3% of price
    valid_minutes: float = 15.0


_KEY_RE = re.compile(r"[^a-z0-9]+")


def headline_key(headline: str) -> str:
    """Normalised headline used to collapse syndicated duplicates of one story."""
    return _KEY_RE.sub(" ", headline.lower()).strip()[:80]


@dataclass(frozen=True)
class TimedNewsSignal:
    signal: NewsSignal
    published: datetime  # timezone-aware
    key: str = ""  # headline_key(); items sharing a key count once


@dataclass(frozen=True)
class AggregatedNews:
    score: float  # -1 .. +1
    block: bool
    count: int
    reason: str


@dataclass(frozen=True)
class Verdict:
    action: Action
    confidence: float  # 0 .. 1
    reasons: tuple[str, ...]
    valid_until: datetime
    news_score: float
    tech_score: float
    entry: float | None = None
    stop: float | None = None
    target: float | None = None


def aggregate_news(
    signals: Iterable[TimedNewsSignal], now: datetime, config: VerdictConfig | None = None
) -> AggregatedNews:
    """Recency-decayed sum of recent signals, clipped to ``[-1, 1]``.

    Each item is capped at ``max_item_score`` and items sharing a headline key count
    once (the strongest wins), so neither one item nor a syndicated story can dominate.
    Items published after ``now`` are ignored (no lookahead). Items that carry no
    direction do not count as news.
    """
    cfg = config or VerdictConfig()
    contributions: dict[str, float] = {}
    block, reason, n_items = False, "", 0
    for i, timed in enumerate(signals):
        age = (now - timed.published).total_seconds() / 60.0
        if age < 0 or age > cfg.news_window_minutes:
            continue
        if timed.signal.block:
            block, reason = True, timed.signal.reason
        capped = max(-cfg.max_item_score, min(cfg.max_item_score, timed.signal.score))
        value = capped * math.pow(0.5, age / cfg.news_half_life_minutes)
        slot = timed.key or f"#{i}"
        if abs(value) > abs(contributions.get(slot, 0.0)):
            contributions[slot] = value
    n_items = sum(1 for v in contributions.values() if abs(v) > 1e-9)
    total = sum(contributions.values())
    return AggregatedNews(max(-1.0, min(1.0, total)), block, n_items, reason)


def _hold(
    action: Action, reasons: Sequence[str], now: datetime, cfg: VerdictConfig, n: float, t: float
) -> Verdict:
    return Verdict(action, 0.0, tuple(reasons), now + timedelta(minutes=cfg.valid_minutes), n, t)


def decide(
    news: AggregatedNews,
    tech: TechnicalState | None,
    config: VerdictConfig | None = None,
    *,
    now: datetime,
    risk_flags: Sequence[str] = (),
) -> Verdict:
    """Return BUY / SELL / WAIT / AVOID with confidence, levels and reasons."""
    cfg = config or VerdictConfig()
    if risk_flags:
        return _hold(Action.AVOID, [f"risk flag: {f}" for f in risk_flags], now, cfg, news.score, 0.0)
    if tech is None or tech.atr <= 0.0:
        return _hold(Action.WAIT, ["no usable price data"], now, cfg, news.score, 0.0)

    t = tech_score(tech)
    n = news.score
    if news.block:
        return _hold(Action.WAIT, [f"entry blocked: {news.reason}"], now, cfg, n, t)
    trend = trend_direction(tech)  # conflict uses the raw trend, not the RSI-damped score
    if n * trend < 0 and min(abs(n), abs(trend)) >= cfg.disagree_min:
        return _hold(Action.WAIT, ["news and price disagree"], now, cfg, n, t)

    has_news = news.count > 0
    combined = cfg.weight_news * n + cfg.weight_tech * t if has_news else t
    if has_news and trend * combined < cfg.min_trend_confirmation:
        return _hold(Action.WAIT, ["price does not confirm the news"], now, cfg, n, t)
    threshold = cfg.entry_threshold if has_news else cfg.tech_only_threshold
    if abs(combined) < threshold:
        why = "signal too weak" if has_news else "no fresh news and price signal too weak"
        return _hold(Action.WAIT, [why], now, cfg, n, t)

    return _entry(combined, has_news, tech, now, cfg, n, t)


def _entry(
    combined: float,
    has_news: bool,
    tech: TechnicalState,
    now: datetime,
    cfg: VerdictConfig,
    n: float,
    t: float,
) -> Verdict:
    side = 1.0 if combined > 0 else -1.0
    entry = tech.close
    stop_distance = max(cfg.stop_atr * tech.atr, cfg.min_stop_pct * entry)
    reasons = [
        f"news {n:+.2f}" if has_news else "no fresh news (price only)",
        f"trend {t:+.2f} (RSI {tech.rsi:.0f}, vol x{tech.vol_ratio:.1f})",
    ]
    return Verdict(
        action=Action.BUY if side > 0 else Action.SELL,
        confidence=min(1.0, abs(combined)),
        reasons=tuple(reasons),
        valid_until=now + timedelta(minutes=cfg.valid_minutes),
        news_score=n,
        tech_score=t,
        entry=entry,
        stop=entry - side * stop_distance,
        target=entry + side * stop_distance * cfg.target_atr / cfg.stop_atr,
    )
