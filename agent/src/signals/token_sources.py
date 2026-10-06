"""Free token-data sources (DexScreener, GoPlus, RugCheck): parsers and fetching.

Parsers are pure and defensive: every field is validated, and anything missing or
malformed becomes ``None`` (unknown), never a default that looks safe.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from src.signals.text_safety import sanitize_text
from src.signals.token_risk import TokenFacts

logger = logging.getLogger(__name__)

DEXSCREENER_URL = "https://api.dexscreener.com/token-pairs/v1"
GOPLUS_URL = "https://api.gopluslabs.io/api/v1/token_security"
RUGCHECK_URL = "https://api.rugcheck.xyz/v1/tokens"
TIMEOUT_S = 10.0
TOP_HOLDERS = 10
MAX_SYMBOL_CHARS = 32
MAX_RISK_NAME_CHARS = 80

GOPLUS_CHAIN_IDS = {
    "ethereum": 1, "bsc": 56, "base": 8453, "arbitrum": 42161,
    "polygon": 137, "optimism": 10, "avalanche": 43114,
}
_EVM_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_SOL_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


def is_valid_address(chain: str, address: str) -> bool:
    if not isinstance(address, str):
        return False
    if chain == "solana":
        return _SOL_RE.fullmatch(address) is not None
    if chain in GOPLUS_CHAIN_IDS:
        return _EVM_RE.fullmatch(address) is not None
    return False


def _flag(value: Any) -> bool | None:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        return None  # unhashable or unexpected types are unknown, never an error
    return {"1": True, "0": False, 1: True, 0: False}.get(value)


def _fraction(value: Any) -> float | None:
    """Finite float or None. NaN/inf from a hostile feed must never pass a threshold."""
    try:
        number = float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
    return number if number is not None and math.isfinite(number) else None


def _unit(value: Any) -> float | None:
    """A finite fraction in [0, 1] or None (taxes, shares of supply)."""
    number = _fraction(value)
    return number if number is not None and 0.0 <= number <= 1.0 else None


def parse_dexscreener(payload: Any, address: str, now: datetime) -> dict[str, Any] | None:
    """Pick the deepest-liquidity pair of ``address`` and extract market facts."""
    pairs = payload.get("pairs") if isinstance(payload, dict) else payload
    if not isinstance(pairs, list):
        return None
    mine = [
        p for p in pairs
        if isinstance(p, dict) and str((p.get("baseToken") or {}).get("address", "")).lower() == address.lower()
    ]
    if not mine:
        return None
    best = max(mine, key=lambda p: _fraction((p.get("liquidity") or {}).get("usd")) or 0.0)
    h1 = (best.get("txns") or {}).get("h1") or {}
    created = best.get("pairCreatedAt")
    age = (now - datetime.fromtimestamp(created / 1000, tz=timezone.utc)).total_seconds() / 3600 if isinstance(created, (int, float)) else None
    return {
        "symbol": sanitize_text((best.get("baseToken") or {}).get("symbol"), MAX_SYMBOL_CHARS),
        "liquidity_usd": _fraction((best.get("liquidity") or {}).get("usd")),
        "volume_24h": _fraction((best.get("volume") or {}).get("h24")),
        "buys_1h": h1.get("buys") if isinstance(h1.get("buys"), int) else None,
        "sells_1h": h1.get("sells") if isinstance(h1.get("sells"), int) else None,
        "age_hours": age,
    }


def parse_goplus(payload: Any, address: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or payload.get("code") != 1:
        return None
    result = payload.get("result") or {}
    entry = next((v for k, v in result.items() if k.lower() == address.lower()), None)
    if not isinstance(entry, dict):
        return None
    buy_tax, sell_tax = _unit(entry.get("buy_tax")), _unit(entry.get("sell_tax"))
    honeypot = _flag(entry.get("is_honeypot"))
    if honeypot is None and buy_tax is None and sell_tax is None:
        return None  # GoPlus has not analysed this token: nothing was actually verified
    holders = [h for h in (entry.get("holders") or []) if isinstance(h, dict)]
    eligible = [h for h in holders[:TOP_HOLDERS] if _flag(h.get("is_contract")) is not True]
    shares = [_unit(h.get("percent")) for h in eligible]
    top10 = sum(shares) if holders and all(x is not None for x in shares) else None
    lps = [l for l in (entry.get("lp_holders") or []) if isinstance(l, dict)]
    lp_shares = [(_unit(l.get("percent")), _flag(l.get("is_locked"))) for l in lps]
    locked = (
        sum(p for p, lk in lp_shares if lk is True and p is not None)
        if lps and all(p is not None for p, _ in lp_shares)
        else None
    )
    return {
        "honeypot": honeypot,
        "buy_tax": buy_tax,
        "sell_tax": sell_tax,
        "mintable": _flag(entry.get("is_mintable")),
        "hidden_owner": _flag(entry.get("hidden_owner")),
        "owner_can_reclaim": _flag(entry.get("can_take_back_ownership")),
        "top10_holder_pct": top10,
        "lp_locked_pct": locked,
    }


def parse_rugcheck(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("risks"), list):
        return None
    risks = [r for r in payload["risks"] if isinstance(r, dict) and r.get("name")]
    lp = _fraction(payload.get("lpLockedPct"))
    return {
        "danger_risks": tuple(sanitize_text(r["name"], MAX_RISK_NAME_CHARS) for r in risks if r.get("level") == "danger"),
        "warn_risks": tuple(sanitize_text(r["name"], MAX_RISK_NAME_CHARS) for r in risks if r.get("level") == "warn"),
        "lp_locked_pct": _unit(lp / 100.0) if lp is not None else None,
    }


def parse_rugcheck_report(payload: Any) -> dict[str, Any] | None:
    """Solana facts from RugCheck's full report: authorities, fee, concentration."""
    if not isinstance(payload, dict) or ("topHolders" not in payload and "mintAuthority" not in payload):
        return None
    known = payload.get("knownAccounts") if isinstance(payload.get("knownAccounts"), dict) else {}
    top10 = None
    holders = payload.get("topHolders")
    if isinstance(holders, list):
        wallets = [
            h for h in holders[: TOP_HOLDERS * 2]
            if isinstance(h, dict) and h.get("owner") not in known and h.get("address") not in known
        ][:TOP_HOLDERS]
        shares = [_fraction(h.get("pct")) for h in wallets]
        if all(x is not None and 0.0 <= x <= 100.0 for x in shares):
            top10 = sum(shares) / 100.0
    fee_raw = payload.get("transferFee")
    fee = _fraction(fee_raw.get("pct")) if isinstance(fee_raw, dict) else None
    fee = _unit(fee / 100.0) if fee is not None else None
    return {
        "mintable": (payload["mintAuthority"] is not None) if "mintAuthority" in payload else None,
        "rugged": bool(payload.get("rugged")),
        "buy_tax": fee,
        "sell_tax": fee,
        "top10_holder_pct": top10,
    }


class TokenDataClient:
    """Fetches and merges token facts. Source failures become unverified, not safe."""

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(timeout=TIMEOUT_S, transport=transport)

    def _json(self, url: str, params: dict[str, str] | None = None) -> Any:
        try:
            response = self._http.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("token source request failed: %s", type(exc).__name__)
            return None

    def fetch_facts(self, chain: str, address: str, *, now: datetime | None = None) -> TokenFacts:
        if not is_valid_address(chain, address):
            raise ValueError(f"invalid {chain} address")
        now = now or datetime.now(timezone.utc)
        market = parse_dexscreener(self._json(f"{DEXSCREENER_URL}/{chain}/{address}"), address, now)
        if chain == "solana":
            security = parse_rugcheck(self._json(f"{RUGCHECK_URL}/{address}/report/summary"))
            report = parse_rugcheck_report(self._json(f"{RUGCHECK_URL}/{address}/report"))
            # Fail closed: concentration and mint authority live in the full report, so a
            # Solana token is only "verified" when BOTH RugCheck calls succeeded.
            security = {**report, **security} if security is not None and report is not None else None
        else:
            data = self._json(f"{GOPLUS_URL}/{GOPLUS_CHAIN_IDS[chain]}", {"contract_addresses": address})
            security = parse_goplus(data, address)
        facts: dict[str, Any] = {"chain": chain, "address": address}
        facts.update(market or {})
        facts["market_checked"] = market is not None
        facts.update(security or {})
        facts["security_checked"] = security is not None
        return TokenFacts(**facts)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "TokenDataClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
