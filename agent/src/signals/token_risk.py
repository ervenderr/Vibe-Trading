"""Meme-coin risk gate: explicit, fail-closed rules over token facts.

Hard fails (honeypot, hidden owner, huge tax, thin liquidity, concentrated
holders, unverifiable data) force AVOID. Soft concerns, including Jev's reading
of scam-style promotional text, only lower the grade: a text judgment alone must
never veto or approve a token. No scoring here can *predict* a rug pull; it only
flags the known, checkable signs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from typesafe_sdk import Noul

HARD_FAIL_GRADE = "F"


@dataclass(frozen=True)
class TokenFacts:
    """Normalised facts about one token. ``None`` means unknown, never 'safe'."""

    chain: str
    address: str
    symbol: str = ""
    liquidity_usd: float | None = None
    age_hours: float | None = None
    volume_24h: float | None = None
    buys_1h: int | None = None
    sells_1h: int | None = None
    honeypot: bool | None = None
    buy_tax: float | None = None  # fraction, 0.05 = 5%
    sell_tax: float | None = None
    mintable: bool | None = None
    hidden_owner: bool | None = None
    owner_can_reclaim: bool | None = None
    top10_holder_pct: float | None = None  # fraction held by top-10 non-contract wallets
    lp_locked_pct: float | None = None  # fraction of liquidity locked
    rugged: bool = False
    danger_risks: tuple[str, ...] = ()
    warn_risks: tuple[str, ...] = ()
    scam_language: Mapping[str, float] | None = None
    security_checked: bool = False
    market_checked: bool = False


@dataclass(frozen=True)
class RiskThresholds:
    min_liquidity_usd: float = 25_000.0
    max_tax: float = 0.10
    max_top10_holders: float = 0.50
    young_hours: float = 168.0  # under a week old
    new_hours: float = 24.0
    established_hours: float = 2160.0  # 90 days survived: LP-lock risk no longer applies
    min_lp_locked: float = 0.50
    sell_pressure_ratio: float = 2.0
    min_trades_for_pressure: int = 20
    scam_language_warn: float = 0.85


@dataclass(frozen=True)
class RiskAssessment:
    grade: str  # A (cleanest) .. D, or F when any hard fail applies
    hard_fails: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def avoid(self) -> bool:
        return bool(self.hard_fails)

    @property
    def risk_flags(self) -> tuple[str, ...]:
        """Feed these to the verdict engine: any flag forces AVOID."""
        return self.hard_fails


def _known(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _hard_fails(f: TokenFacts, t: RiskThresholds) -> list[str]:
    fails: list[str] = []
    if not f.security_checked:
        fails.append("cannot verify contract security (data unavailable)")
    if not f.market_checked:
        fails.append("no market data: cannot verify liquidity or age")
    if f.rugged:
        fails.append("flagged as already rugged")
    if f.honeypot:
        fails.append("honeypot: buyers cannot sell")
    if _known(f.sell_tax) and f.sell_tax > t.max_tax:
        fails.append(f"sell tax {f.sell_tax:.0%} is too high")
    if _known(f.buy_tax) and f.buy_tax > t.max_tax:
        fails.append(f"buy tax {f.buy_tax:.0%} is too high")
    if f.hidden_owner:
        fails.append("hidden owner")
    if f.owner_can_reclaim:
        fails.append("owner can reclaim ownership")
    if f.market_checked and not _known(f.liquidity_usd):
        fails.append("liquidity unknown: cannot verify the token can be sold")
    elif _known(f.liquidity_usd) and f.liquidity_usd < t.min_liquidity_usd:
        fails.append(f"liquidity ${f.liquidity_usd:,.0f} is too thin to exit")
    if f.market_checked and not _known(f.age_hours):
        fails.append("token age unknown: cannot judge how new it is")
    if f.chain != "solana" and f.security_checked and f.honeypot is None:
        fails.append("honeypot status unknown: cannot verify buyers can sell")
    if _known(f.top10_holder_pct) and f.top10_holder_pct > t.max_top10_holders:
        fails.append(f"top holders control {f.top10_holder_pct:.0%} of supply")
    fails.extend(f"RugCheck danger: {name}" for name in f.danger_risks)
    young = f.age_hours is not None and f.age_hours < t.young_hours
    if young and f.lp_locked_pct is not None and f.lp_locked_pct < t.min_lp_locked:
        fails.append(f"LP mostly unlocked on a young token ({f.lp_locked_pct:.0%} locked)")
    return fails


def _warnings(f: TokenFacts, t: RiskThresholds) -> list[str]:
    warns: list[str] = []
    evm = f.chain != "solana"  # honeypot/tax checks only exist on EVM chains
    if evm and f.security_checked and f.honeypot is None:
        warns.append("honeypot status unknown")
    if evm and f.security_checked and (f.buy_tax is None or f.sell_tax is None):
        warns.append("buy/sell tax unknown")
    if f.mintable:
        warns.append("supply is mintable")
    if f.age_hours is not None and f.age_hours < t.new_hours:
        warns.append(f"very new token ({f.age_hours:.0f}h old)")
    young = f.age_hours is not None and f.age_hours < t.young_hours
    established = f.age_hours is not None and f.age_hours >= t.established_hours
    if not young and not established and f.lp_locked_pct is not None and f.lp_locked_pct < t.min_lp_locked:
        warns.append(f"LP mostly unlocked ({f.lp_locked_pct:.0%} locked)")
    if f.buys_1h is not None and f.sells_1h is not None:
        total = f.buys_1h + f.sells_1h
        if total >= t.min_trades_for_pressure and f.sells_1h > t.sell_pressure_ratio * max(f.buys_1h, 1):
            warns.append(f"heavy selling pressure ({f.sells_1h} sells vs {f.buys_1h} buys in 1h)")
    warns.extend(f"RugCheck warning: {name}" for name in f.warn_risks)
    flagged = sorted((n, p) for n, p in (f.scam_language or {}).items() if p >= t.scam_language_warn)
    if flagged:  # correlated text flags are one concern, not several
        warns.append("scam-style language: " + ", ".join(f"{n} ({p:.2f})" for n, p in flagged))
    return warns


def assess_token(facts: TokenFacts, thresholds: RiskThresholds | None = None) -> RiskAssessment:
    """Grade a token and list hard fails and warnings."""
    t = thresholds or RiskThresholds()
    fails = _hard_fails(facts, t)
    warns = _warnings(facts, t)
    if fails:
        grade = HARD_FAIL_GRADE
    else:
        grade = "A" if not warns else "B" if len(warns) == 1 else "C" if len(warns) <= 3 else "D"
    return RiskAssessment(grade, tuple(fails), tuple(warns))


def scam_questions() -> dict[str, Noul]:
    """Jev questions over ``{name, symbol, description, links_text}`` state."""
    return {
        "promises_returns": Noul(
            instructions=(
                "Does `description` promise or imply guaranteed profit, guaranteed returns, "
                "or a specific price multiple such as 100x or 1000x?"
            )
        ),
        "urgency_pressure": Noul(
            instructions=(
                "Does `description` pressure the reader to buy immediately, for example "
                "'last chance', 'buy before it is too late' or 'about to explode'?"
            )
        ),
        "unverified_claims": Noul(
            instructions=(
                "Does `description` claim a partnership, audit or endorsement by a named company "
                "or person without giving a link or other verifiable proof?"
            )
        ),
        "impersonation": Noul(
            instructions=(
                "Does `description` present this token as the official token of another "
                "well-known project, company or person?"
            )
        ),
        "no_substance": Noul(
            instructions=(
                "Does `description` lack any concrete explanation of what the token or project "
                "does, containing only hype or slogans?"
            )
        ),
    }


def scam_language_from_response(response) -> dict[str, float]:
    return {name: response.nouls[name].noul for name in scam_questions()}
