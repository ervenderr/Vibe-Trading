"""Malaysian SnR levels and a coded version of the trade rule. Research only; places no orders.

One documented interpretation of the commonly taught method (see scripts/PREREGISTRATION-malaysian-snr.md):
levels come from closing prices, and a level's freshness depends on whether a wick or a candle body touched it.
Nothing here looks at a bar after the one being decided.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ATR_PERIOD = 14
USABLE_LAG = 2       # a level is usable two bars after the bar it forms on (one bar to confirm, one to act)
SUPPORT_KINDS = ("V", "GAP_UP")
RESISTANCE_KINDS = ("A", "GAP_DOWN")


@dataclass(frozen=True)
class Level:
    kind: str       # "A" | "V" | "GAP_UP" | "GAP_DOWN"
    price: float
    formed: int     # bar index the level forms on
    usable_from: int


def _atr(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    true_range = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()


def find_levels(df: pd.DataFrame) -> list[Level]:
    """Every A, V and gap level in the history, each with the bar it becomes usable on."""
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    levels = []
    for i in range(1, len(c) - 1):
        if c[i] > c[i - 1] and c[i] > c[i + 1]:
            levels.append(Level("A", float(c[i]), i, i + USABLE_LAG))
        elif c[i] < c[i - 1] and c[i] < c[i + 1]:
            levels.append(Level("V", float(c[i]), i, i + USABLE_LAG))
    for i in range(len(c) - 1):
        if c[i] > o[i] and c[i + 1] > o[i + 1] and o[i + 1] > c[i]:
            levels.append(Level("GAP_UP", float(c[i]), i, i + USABLE_LAG))
        elif c[i] < o[i] and c[i + 1] < o[i + 1] and o[i + 1] < c[i]:
            levels.append(Level("GAP_DOWN", float(c[i]), i, i + USABLE_LAG))
    return sorted(levels, key=lambda lv: lv.usable_from)


def _touch(fresh: bool, price: float, o: float, h: float, low: float, c: float) -> bool:
    """New freshness after one bar: a body crossing the level resets it, a wick-only touch spends it."""
    if min(o, c) < price < max(o, c):
        return True
    if low <= price <= h:
        return False
    return fresh


def _run(df: pd.DataFrame, *, trend_window, touch_atr, stop_atr, target_atr, max_hold, max_age) -> tuple[pd.Series, list[dict]]:
    o, h, lo, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr = _atr(df).to_numpy()
    sma = df["close"].rolling(trend_window).mean().to_numpy() if trend_window else None
    pending = find_levels(df)
    cursor, active = 0, []             # active: [Level, fresh]
    signal = np.zeros(len(c), dtype=int)
    pos = None                          # (entry_bar, level_price, atr_at_entry, target)
    for t in range(len(c)):
        while cursor < len(pending) and pending[cursor].usable_from <= t:
            lv = pending[cursor]
            cursor += 1
            b = lv.formed + 1           # the confirming bar already happened: let it spend or reset the level
            active.append([lv, _touch(True, lv.price, o[b], h[b], lo[b], c[b])])
        active = [a for a in active if t - a[0].formed <= max_age]
        if pos is None:
            entry = _entry(t, o, h, lo, c, atr, sma, active, touch_atr)
            if entry is not None:
                price = entry
                above = [a[0].price for a in active if a[0].kind in RESISTANCE_KINDS and a[0].price > c[t]]
                target = min(above) if above else c[t] + target_atr * atr[t]
                pos = (t, price, atr[t], target)
                signal[t] = 1
        else:
            entry_bar, level, entry_atr, target = pos
            if c[t] < level - stop_atr * entry_atr or c[t] >= target or t - entry_bar >= max_hold:
                pos = None
            else:
                signal[t] = 1
        for a in active:                # the bar just seen changes freshness for the NEXT bar's decision
            a[1] = _touch(a[1], a[0].price, o[t], h[t], lo[t], c[t])
    now = [{"kind": lv.kind, "price": lv.price, "fresh": bool(fresh), "formed": lv.formed,
            "time": int(df.index[lv.formed].timestamp())} for lv, fresh in active]
    return pd.Series(signal, index=df.index), now


def _entry(t, o, h, lo, c, atr, sma, active, touch_atr):
    """Support price to trade at bar ``t``, or None. Uses freshness as of the start of the bar."""
    if t == 0 or np.isnan(atr[t]) or (sma is not None and (np.isnan(sma[t]) or c[t] <= sma[t])):
        return None
    below = [a[0].price for a in active if a[1] and a[0].kind in SUPPORT_KINDS and a[0].price < c[t - 1]]
    if not below:
        return None
    level = max(below)
    reached = lo[t] <= level + touch_atr * atr[t]
    return level if reached and c[t] > level and c[t] > o[t] else None


def snr_signal(df: pd.DataFrame, *, trend_window: int | None = 200, touch_atr: float = 0.25, stop_atr: float = 0.5,
               target_atr: float = 3.0, max_hold: int = 40, max_age: int = 250) -> pd.Series:
    """1 while the rule holds a long position after each bar's close, else 0."""
    return _run(df, trend_window=trend_window, touch_atr=touch_atr, stop_atr=stop_atr, target_atr=target_atr,
                max_hold=max_hold, max_age=max_age)[0]


def levels_asof(df: pd.DataFrame, max_age: int = 250) -> list[dict]:
    """Levels still active after the last bar, with their freshness, for drawing on a chart."""
    return _run(df, trend_window=None, touch_atr=0.25, stop_atr=0.5, target_atr=3.0, max_hold=40, max_age=max_age)[1]
