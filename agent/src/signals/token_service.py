"""Token risk service: market/security facts + optional Jev scam-language check."""

from __future__ import annotations

import json
import logging
from argparse import Namespace
from dataclasses import replace
from typing import Any, Callable

from typesafe_sdk import TypeSafeError

from src.signals.token_risk import (
    RiskAssessment,
    TokenFacts,
    assess_token,
    scam_language_from_response,
    scam_questions,
)
from src.signals.token_sources import TokenDataClient, is_valid_address

logger = logging.getLogger(__name__)
MAX_DESCRIPTION_CHARS = 4000
DISCLAIMER = (
    "This flags known, checkable warning signs. It cannot predict rug pulls: a clean grade "
    "is not a guarantee, and new tokens can pass every check and still fail. Never risk "
    "money you cannot lose."
)


def assess_address(
    chain: str,
    address: str,
    *,
    description: str | None = None,
    data_client: Any,
    jev_client: Any,
    model: str,
) -> RiskAssessment:
    """Assess one token; the Jev text check runs only when a description is given."""
    assessment, _ = assess_with_facts(
        chain, address, description=description, data_client=data_client, jev_client=jev_client, model=model
    )
    return assessment


def assess_with_facts(
    chain: str,
    address: str,
    *,
    description: str | None = None,
    data_client: Any,
    jev_client: Any,
    model: str,
) -> tuple[RiskAssessment, TokenFacts]:
    if not is_valid_address(chain, address):
        raise ValueError(f"invalid {chain} address")
    facts = data_client.fetch_facts(chain, address)
    if description and jev_client is not None:
        scam = _scam_language(facts, description[:MAX_DESCRIPTION_CHARS], jev_client, model)
        if scam is not None:
            facts = replace(facts, scam_language=scam)
    return assess_token(facts), facts


def _scam_language(facts: TokenFacts, description: str, jev_client: Any, model: str) -> dict[str, float] | None:
    try:
        response = jev_client.system_one(
            {"symbol": facts.symbol, "description": description}, scam_questions(), model=model
        )
        return scam_language_from_response(response)
    except (TypeSafeError, KeyError, AttributeError, TypeError) as exc:
        logger.warning("Jev scam-language check unavailable: %s", type(exc).__name__)
        return None


def assessment_to_dict(facts: TokenFacts, assessment: RiskAssessment) -> dict[str, Any]:
    return {
        "chain": facts.chain,
        "address": facts.address,
        "symbol": facts.symbol,
        "grade": assessment.grade,
        "avoid": assessment.avoid,
        "hard_fails": list(assessment.hard_fails),
        "warnings": list(assessment.warnings),
        "facts": {
            "liquidity_usd": facts.liquidity_usd,
            "age_hours": facts.age_hours,
            "volume_24h": facts.volume_24h,
            "top10_holder_pct": facts.top10_holder_pct,
            "lp_locked_pct": facts.lp_locked_pct,
            "mintable": facts.mintable,
            "honeypot": facts.honeypot,
            "buy_tax": facts.buy_tax,
            "sell_tax": facts.sell_tax,
        },
        "disclaimer": DISCLAIMER,
    }


def assess_live(chain: str, address: str, description: str | None) -> tuple[TokenFacts, RiskAssessment]:
    """Assess with real data sources; Jev is used only if a key is configured."""
    from typesafe_sdk import TypeSafeClient

    from src.config.accessor import get_env_config

    cfg = get_env_config().data
    with TokenDataClient() as data:
        if description and cfg.typesafe_api_key:
            with TypeSafeClient(api_key=cfg.typesafe_api_key) as jev:
                assessment, facts = assess_with_facts(
                    chain, address, description=description, data_client=data, jev_client=jev, model=cfg.typesafe_model
                )
        else:
            assessment, facts = assess_with_facts(
                chain, address, data_client=data, jev_client=None, model=cfg.typesafe_model
            )
    return facts, assessment


def render_assessment(facts: TokenFacts, a: RiskAssessment) -> str:
    head = f"{facts.symbol or facts.address[:8]} ({facts.chain}): grade {a.grade}" + ("  -> AVOID" if a.avoid else "")
    lines = [head]
    lines.extend(f"  x {f}" for f in a.hard_fails)
    lines.extend(f"  ! {w}" for w in a.warnings)
    if not a.hard_fails and not a.warnings:
        lines.append("  no known warning signs found")
    lines.append(f"  {DISCLAIMER}")
    return "\n".join(lines)


def run_token_command(
    args: Namespace,
    *,
    assessor: Callable[[str, str, str | None], tuple[TokenFacts, RiskAssessment]] = assess_live,
    out: Callable[[str], None] = print,
) -> int:
    if not is_valid_address(args.chain, args.address):
        out(f"invalid {args.chain} address")
        return 1
    try:
        facts, assessment = assessor(args.chain, args.address, args.description)
    except Exception as exc:
        logger.exception("token assessment failed")
        out(f"assessment failed ({type(exc).__name__})")
        return 1
    out(json.dumps(assessment_to_dict(facts, assessment)) if args.json else render_assessment(facts, assessment))
    return 0
