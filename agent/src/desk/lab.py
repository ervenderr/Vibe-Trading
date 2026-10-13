"""Backtest endpoint logic for the desk: parse and validate a request, run it, shape the JSON."""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd

from src.desk.payloads import jsonable
from src.signals.news_source import is_valid_symbol
from src.signals.strategy_lab import Costs, backtest, breakout_signal, rsi_reversion_signal, sma_cross_signal

logger = logging.getLogger(__name__)
MAX_POINTS = 600
MAX_TRADES = 300
MIN_BARS = 100
INTERVALS = ("1D", "1H")
# name -> (signal function, {param: (default, min, max)})
STRATEGIES: dict[str, tuple[Callable[..., pd.Series], dict[str, tuple[float, float, float]]]] = {
    "sma_cross": (sma_cross_signal, {"fast": (20, 2, 400), "slow": (100, 3, 1000)}),
    "rsi_reversion": (rsi_reversion_signal, {"period": (14, 2, 100), "low": (30, 1, 49), "high": (70, 51, 99)}),
    "breakout": (breakout_signal, {"window": (20, 4, 400)}),
}
DEFAULT_FEE_BPS, DEFAULT_SLIPPAGE_BPS, MAX_COST_BPS = 5.0, 5.0, 500.0


class BadRequest(ValueError):
    pass


def _first(query: dict, key: str, default: str) -> str:
    return (query.get(key, [default])[0] or default).strip()


def _number(query: dict, key: str, default: float, lo: float, hi: float) -> float:
    raw = _first(query, key, str(default))
    try:
        value = float(raw)
    except ValueError:
        raise BadRequest(f"{key} must be a number") from None
    if not (lo <= value <= hi):
        raise BadRequest(f"{key} must be between {lo:g} and {hi:g}")
    return value


def parse(query: dict) -> dict:
    symbol = _first(query, "symbol", "").upper()
    if not (is_valid_symbol(symbol) or symbol.endswith("/USD") and is_valid_symbol(symbol.replace("/", "-"))):
        raise BadRequest("invalid symbol")
    strategy = _first(query, "strategy", "sma_cross")
    if strategy not in STRATEGIES:
        raise BadRequest("unknown strategy")
    interval = _first(query, "interval", "1D")
    if interval not in INTERVALS:
        raise BadRequest("interval must be 1D or 1H")
    params = {name: _number(query, name, d, lo, hi) for name, (d, lo, hi) in STRATEGIES[strategy][1].items()}
    if strategy == "sma_cross" and params["fast"] >= params["slow"]:
        raise BadRequest("fast must be smaller than slow")
    if strategy == "rsi_reversion" and params["low"] >= params["high"]:
        raise BadRequest("low must be smaller than high")
    years = _number(query, "years", 5, 0.5, 20 if interval == "1D" else 2)
    return {"symbol": symbol, "strategy": strategy, "interval": interval, "params": {k: int(v) for k, v in params.items()},
            "years": years, "fee_bps": _number(query, "fee_bps", DEFAULT_FEE_BPS, 0, MAX_COST_BPS),
            "slippage_bps": _number(query, "slippage_bps", DEFAULT_SLIPPAGE_BPS, 0, MAX_COST_BPS)}


def _thin(frame: pd.DataFrame) -> pd.DataFrame:
    """At most ~MAX_POINTS rows: the last value of each bucket, but the deepest drawdown so the crash stays visible."""
    step = max(1, -(-len(frame) // MAX_POINTS))
    bucket = pd.Series(range(len(frame)), index=frame.index) // step
    grouped = frame.groupby(bucket.to_numpy())
    thin = grouped.last()
    thin["drawdown"] = grouped["drawdown"].max()
    thin.index = grouped.apply(lambda g: g.index[-1])
    return thin


def run_lab(query: dict, *, history_fn: Callable[[str, str, float], pd.Series]) -> tuple[int, dict[str, Any]]:
    try:
        req = parse(query)
    except BadRequest as exc:
        return 400, {"error": str(exc)}
    try:
        close = history_fn(req["symbol"], req["interval"], req["years"]).dropna()
    except Exception:
        logger.exception("history failed for %s", req["symbol"])
        return 502, {"error": "price history unavailable right now"}
    if len(close) < MIN_BARS:
        return 422, {"error": f"not enough history for {req['symbol']} ({len(close)} bars; need {MIN_BARS})"}
    signal_fn = STRATEGIES[req["strategy"]][0]
    result = backtest(close, signal_fn(close, **req["params"]), Costs(req["fee_bps"], req["slippage_bps"]))
    bench = close.pct_change().fillna(0.0)
    curves = pd.DataFrame({
        "strategy": result.equity, "buy_hold": (1 + bench).cumprod(), "drawdown": result.drawdown,
    })
    thin = _thin(curves)
    times = [int(ts.timestamp()) for ts in thin.index]
    return 200, jsonable({
        **{k: req[k] for k in ("symbol", "strategy", "interval", "params", "years")},
        "costs": {"fee_bps": req["fee_bps"], "slippage_bps": req["slippage_bps"]},
        "bars": len(close), "metrics": result.metrics, "benchmark": result.benchmark, "halves": result.halves,
        "equity": [{"time": t, "strategy": float(r.strategy), "buy_hold": float(r.buy_hold)} for t, r in zip(times, thin.itertuples())],
        "drawdown": [{"time": t, "value": float(v)} for t, v in zip(times, thin["drawdown"])],
        "trades": list(result.trades[-MAX_TRADES:]),
    })
