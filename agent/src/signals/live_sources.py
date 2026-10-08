"""Real data sources wired into :func:`generate_signal` (Finnhub, Jev, yfinance)."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
from typesafe_sdk import TypeSafeClient

from src.config.accessor import get_env_config
from src.market_data import get_loader
from src.signals.jev_analyzer import JevNewsAnalyzer, JudgmentCache
from src.signals.news_source import FinnhubNewsClient, NewsFetchError
from src.signals.scalp_eval import exchange_tz_for, localize_bars
from src.signals.service import SignalReport, generate_signal

logger = logging.getLogger(__name__)
DEFAULT_CACHE = Path.home() / ".vibe-trading" / "cache" / "jev_news"
BARS_LOOKBACK_DAYS = 6


def fetch_recent_bars(symbol: str, days: int = BARS_LOOKBACK_DAYS) -> pd.DataFrame:
    """Recent 5-minute bars with a UTC index (empty frame when unavailable)."""
    tz = exchange_tz_for(symbol)
    if tz is None:
        logger.warning("unsupported market for %s: only US stocks and crypto (-USD/-USDT) are supported", symbol)
        return pd.DataFrame()
    today = datetime.now(timezone.utc).date()
    loader = get_loader("yfinance")()
    data = loader.fetch([symbol], (today - timedelta(days=days)).isoformat(), today.isoformat(), interval="5m")
    raw = data.get(symbol)
    return localize_bars(raw, tz) if raw is not None and not raw.empty else pd.DataFrame()


REPO_ENV = Path(__file__).resolve().parents[3] / ".env"


SIGNAL_KEYS = ("FINNHUB_API_KEY", "TYPESAFE_API_KEY", "TYPESAFE_MODEL")


def ensure_signal_keys(env_file: Path = REPO_ENV) -> None:
    """The news/AI keys may sit in the repo-root .env, which the CLI's search order skips when agent/.env exists."""
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    found = False
    for line in lines:
        name, sep, value = line.partition("=")
        name = name.strip()
        if sep and name in SIGNAL_KEYS and not os.environ.get(name):
            os.environ[name] = value.strip().strip("'\"")
            found = True
    if found:
        from src.config.accessor import reset_env_config

        reset_env_config()


def run_signal(symbol: str, *, risk_flags: Sequence[str] = (), now: datetime | None = None) -> SignalReport:
    """Generate a live signal using the configured Finnhub and TypeSafe keys."""
    ensure_signal_keys()
    data_cfg = get_env_config().data
    if not data_cfg.finnhub_api_key or not data_cfg.typesafe_api_key:
        raise NewsFetchError("FINNHUB_API_KEY and TYPESAFE_API_KEY must both be set")
    with FinnhubNewsClient(data_cfg.finnhub_api_key) as finnhub, TypeSafeClient(
        api_key=data_cfg.typesafe_api_key
    ) as jev:
        analyzer = JevNewsAnalyzer(jev, model=data_cfg.typesafe_model, cache=JudgmentCache(DEFAULT_CACHE))
        return generate_signal(
            symbol,
            news_source=lambda s: finnhub.fetch(s, days=2),
            analyzer=analyzer,
            bars_source=fetch_recent_bars,
            now=now or datetime.now(timezone.utc),
            risk_flags=risk_flags,
        )
