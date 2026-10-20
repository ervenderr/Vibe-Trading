"""Pre-registered crypto timeframe study (scripts/PREREGISTRATION-crypto-timeframe-study.md).

Long/flat rules on closed bars; a signal at bar t is held over bar t+1, so there is no look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COST = 0.0035  # 25 bps fee + 10 bps slippage per side
BLOCK = 10  # bootstrap block length in bars
VOL_WINDOW = 20


def breakout(close: pd.Series, n: int) -> pd.Series:
    """Long after a close above the prior n-bar high, flat after a close below the prior n/2-bar low."""
    high, low = close.shift(1).rolling(n).max(), close.shift(1).rolling(max(n // 2, 1)).min()
    state, out = False, []
    for c, h, l in zip(close, high, low):
        if pd.notna(h) and c > h:
            state = True
        elif pd.notna(l) and c < l:
            state = False
        out.append(state)
    return pd.Series(out, index=close.index)


def sma_cross(close: pd.Series, fast: int, slow: int) -> pd.Series:
    return (close.rolling(fast).mean() > close.rolling(slow).mean()) & close.rolling(slow).mean().notna()


def momentum(close: pd.Series, lookback: int) -> pd.Series:
    return close.pct_change(lookback) > 0


def above_sma(close: pd.Series, n: int) -> pd.Series:
    return close > close.rolling(n).mean()


def vol_filtered(close: pd.Series, median_window: int) -> pd.Series:
    vol = close.pct_change().rolling(VOL_WINDOW).std()
    return above_sma(close, 100) & (vol < vol.rolling(median_window).median())


def vote(close: pd.Series) -> pd.Series:
    votes = breakout(close, 50).astype(int) + sma_cross(close, 50, 200).astype(int) + momentum(close, 90).astype(int)
    return votes >= 2


def variants(median_window: int) -> dict:
    """The fixed list of 13 candidates. Nothing is added after the pre-registration."""
    out = {f"breakout{n}": (lambda c, n=n: breakout(c, n)) for n in (20, 50, 100)}
    out |= {f"cross{f}_{s}": (lambda c, f=f, s=s: sma_cross(c, f, s)) for f, s in ((20, 100), (50, 200))}
    out |= {f"mom{n}": (lambda c, n=n: momentum(c, n)) for n in (30, 90, 180)}
    out |= {f"sma{n}": (lambda c, n=n: above_sma(c, n)) for n in (50, 100, 200)}
    out["volfilter"] = lambda c: vol_filtered(c, median_window)
    out["vote"] = vote
    return out


def strategy_returns(close: pd.Series, signal: pd.Series, cost: float = COST) -> pd.Series:
    position = signal.astype(float).shift(1).fillna(0.0)
    return close.pct_change().fillna(0.0) * position - position.diff().abs().fillna(position.abs()) * cost


def pooled(closes: dict[str, pd.Series], rule, cost: float = COST) -> pd.Series:
    return pd.concat([strategy_returns(c, rule(c), cost) for c in closes.values()], axis=1).mean(axis=1)


def hold(closes: dict[str, pd.Series], cost: float = COST) -> pd.Series:
    return pd.concat([strategy_returns(c, pd.Series(True, index=c.index), cost) for c in closes.values()], axis=1).mean(axis=1)


def round_trips(closes: dict[str, pd.Series], rule) -> int:
    return int(sum((rule(c).astype(int).diff() == -1).sum() for c in closes.values()))


def cagr(r: pd.Series, per_year: float) -> float:
    return float((1.0 + r).prod() ** (per_year / len(r)) - 1.0) if len(r) else 0.0


def sharpe(r: pd.Series, per_year: float) -> float:
    std = r.std(ddof=1)
    return float(r.mean() / std * np.sqrt(per_year)) if std > 0 else 0.0


def max_drawdown(r: pd.Series) -> float:
    equity = (1.0 + r).cumprod()
    return float((1.0 - equity / equity.cummax()).max())


def sharpe_lower(r: pd.Series, per_year: float, *, level: float = 0.975, n_boot: int = 2000, seed: int = 0) -> float:
    values, n = r.to_numpy(), len(r)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, -(-n // BLOCK)))
    sample = values[((starts[:, :, None] + np.arange(BLOCK)) % n).reshape(n_boot, -1)[:, :n]]
    std = sample.std(axis=1, ddof=1)
    s = np.where(std > 0, sample.mean(axis=1) / np.where(std > 0, std, 1.0) * np.sqrt(per_year), 0.0)
    return float(np.quantile(s, 1.0 - level))
