"""Tests for the meme-coin risk gate: parsers (live payload shapes) and rules."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.signals.token_risk import (
    RiskThresholds,
    TokenFacts,
    assess_token,
    scam_language_from_response,
    scam_questions,
)
from src.signals.token_sources import (
    is_valid_address,
    parse_dexscreener,
    parse_goplus,
    parse_rugcheck,
)

NOW = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)
SOL = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
EVM = "0x6982508145454ce325ddbe47a25d4ec3d2311933"


def _safe(**kw) -> TokenFacts:
    base = dict(
        chain="solana", address=SOL, symbol="BONK", liquidity_usd=400_000.0, age_hours=24 * 900.0,
        volume_24h=700_000.0, buys_1h=268, sells_1h=300, honeypot=False, buy_tax=0.0, sell_tax=0.0,
        mintable=False, hidden_owner=False, owner_can_reclaim=False, top10_holder_pct=0.2,
        lp_locked_pct=0.9, danger_risks=(), warn_risks=(), security_checked=True, market_checked=True,
    )
    return TokenFacts(**{**base, **kw})


class TestAssess:
    def test_clean_established_token_is_grade_a(self) -> None:
        a = assess_token(_safe())
        assert a.grade == "A" and a.hard_fails == () and a.avoid is False

    @pytest.mark.parametrize(
        "kw,needle",
        [
            ({"honeypot": True}, "honeypot"),
            ({"sell_tax": 0.25}, "sell tax"),
            ({"buy_tax": 0.25}, "buy tax"),
            ({"hidden_owner": True}, "hidden owner"),
            ({"owner_can_reclaim": True}, "reclaim"),
            ({"liquidity_usd": 5_000.0}, "liquidity"),
            ({"top10_holder_pct": 0.8}, "holders"),
            ({"danger_risks": ("Freeze Authority still enabled",)}, "freeze"),
        ],
    )
    def test_hard_fails_force_avoid_and_grade_f(self, kw: dict, needle: str) -> None:
        a = assess_token(_safe(**kw))
        assert a.avoid is True and a.grade == "F"
        assert any(needle in f.lower() for f in a.hard_fails)

    def test_young_token_with_unlocked_lp_is_hard_fail(self) -> None:
        a = assess_token(_safe(age_hours=5.0, lp_locked_pct=0.0))
        assert a.avoid and any("lp" in f.lower() for f in a.hard_fails)

    def test_mid_age_token_with_unlocked_lp_is_only_a_warning(self) -> None:
        a = assess_token(_safe(age_hours=24.0 * 30, lp_locked_pct=0.1))
        assert a.avoid is False and any("lp" in w.lower() for w in a.warnings)

    def test_token_that_survived_months_is_not_warned_about_unlocked_lp(self) -> None:
        a = assess_token(_safe(age_hours=24.0 * 365, lp_locked_pct=0.0))
        assert a.avoid is False and a.warnings == () and a.grade == "A"

    def test_unverified_security_data_fails_closed(self) -> None:
        a = assess_token(_safe(security_checked=False, honeypot=None))
        assert a.avoid and any("cannot verify" in f.lower() for f in a.hard_fails)

    def test_missing_market_data_fails_closed(self) -> None:
        a = assess_token(_safe(market_checked=False, liquidity_usd=None))
        assert a.avoid

    def test_unknown_individual_fields_are_not_assumed_safe(self) -> None:
        a = assess_token(_safe(chain="ethereum", address=EVM, honeypot=None))
        assert a.avoid or any("honeypot" in w.lower() for w in a.warnings)

    def test_warnings_lower_the_grade_without_vetoing(self) -> None:
        a = assess_token(_safe(mintable=True, age_hours=10.0, sells_1h=2000, buys_1h=100))
        assert a.avoid is False and a.grade in {"C", "D"} and len(a.warnings) >= 3

    def test_scam_language_is_warning_only_never_a_veto(self) -> None:
        a = assess_token(_safe(scam_language={"promises_returns": 0.97, "impersonation": 0.9}))
        assert a.avoid is False and any("scam" in w.lower() or "promises" in w.lower() for w in a.warnings)

    def test_thresholds_are_configurable(self) -> None:
        strict = RiskThresholds(min_liquidity_usd=1_000_000.0)
        assert assess_token(_safe(), strict).avoid is True

    def test_risk_flags_property_feeds_the_verdict_engine(self) -> None:
        assert assess_token(_safe(honeypot=True)).risk_flags
        assert assess_token(_safe()).risk_flags == ()


class TestScamQuestions:
    def test_questions_exist_and_parse_back(self) -> None:
        qs = scam_questions()
        assert set(qs) == {"promises_returns", "urgency_pressure", "unverified_claims", "impersonation", "no_substance"}
        resp = SimpleNamespace(nouls={k: SimpleNamespace(noul=0.1) for k in qs})
        assert scam_language_from_response(resp) == {k: 0.1 for k in qs}


DEX_PAYLOAD = {"pairs": [
    {"chainId": "solana", "baseToken": {"address": SOL, "symbol": "Bonk"}, "liquidity": {"usd": 423433.03},
     "volume": {"h24": 761541.86}, "txns": {"h1": {"buys": 268, "sells": 385}},
     "pairCreatedAt": 1671980424000, "fdv": 3.3e8},
    {"chainId": "solana", "baseToken": {"address": SOL, "symbol": "Bonk"}, "liquidity": {"usd": 5000.0},
     "volume": {"h24": 10.0}, "txns": {"h1": {"buys": 1, "sells": 0}}, "pairCreatedAt": 1700000000000},
    {"chainId": "solana", "baseToken": {"address": "OTHER", "symbol": "X"}, "liquidity": {"usd": 9e9}},
]}


class TestParsers:
    def test_dexscreener_picks_deepest_pair_of_the_requested_token(self) -> None:
        m = parse_dexscreener(DEX_PAYLOAD, SOL, NOW)
        assert m["liquidity_usd"] == pytest.approx(423433.03)
        assert m["buys_1h"] == 268 and m["sells_1h"] == 385
        assert m["age_hours"] > 24 * 365

    def test_dexscreener_no_pairs_returns_none(self) -> None:
        assert parse_dexscreener({"pairs": None}, SOL, NOW) is None
        assert parse_dexscreener({"pairs": []}, SOL, NOW) is None

    def test_goplus_parses_flags_taxes_and_concentration(self) -> None:
        payload = {"code": 1, "result": {EVM: {
            "is_honeypot": "0", "buy_tax": "0.01", "sell_tax": "0.02", "is_mintable": "1",
            "hidden_owner": "0", "can_take_back_ownership": "1",
            "holders": [{"percent": "0.08", "is_contract": 0}, {"percent": "0.5", "is_contract": 1}],
            "lp_holders": [{"percent": "0.9", "is_locked": 1}, {"percent": "0.1", "is_locked": 0}],
        }}}
        s = parse_goplus(payload, EVM)
        assert s["honeypot"] is False and s["mintable"] is True
        assert s["buy_tax"] == pytest.approx(0.01) and s["sell_tax"] == pytest.approx(0.02)
        assert s["owner_can_reclaim"] is True
        assert s["top10_holder_pct"] == pytest.approx(0.08)  # contracts (pools/burn) excluded
        assert s["lp_locked_pct"] == pytest.approx(0.9)

    def test_goplus_unknown_token_returns_none(self) -> None:
        assert parse_goplus({"code": 1, "result": {}}, EVM) is None
        assert parse_goplus({"code": 2}, EVM) is None

    def test_goplus_empty_string_tax_is_unknown_not_zero(self) -> None:
        s = parse_goplus({"code": 1, "result": {EVM: {"is_honeypot": "0", "sell_tax": ""}}}, EVM)
        assert s["sell_tax"] is None

    def test_rugcheck_separates_danger_from_warning(self) -> None:
        payload = {"risks": [
            {"name": "Mutable metadata", "level": "warn"},
            {"name": "Freeze Authority still enabled", "level": "danger"},
        ], "lpLockedPct": 14.7}
        s = parse_rugcheck(payload)
        assert s["danger_risks"] == ("Freeze Authority still enabled",)
        assert s["warn_risks"] == ("Mutable metadata",)
        assert s["lp_locked_pct"] == pytest.approx(0.147)

    def test_rugcheck_garbage_returns_none(self) -> None:
        assert parse_rugcheck([]) is None
        assert parse_rugcheck({"error": "x"}) is None


class TestAddressValidation:
    @pytest.mark.parametrize("chain,addr,ok", [
        ("solana", SOL, True), ("ethereum", EVM, True), ("base", EVM, True),
        ("solana", EVM, False), ("ethereum", SOL, False), ("ethereum", "0x123", False),
        ("solana", "../etc/passwd", False), ("moonchain", EVM, False),
    ])
    def test_validation(self, chain: str, addr: str, ok: bool) -> None:
        assert is_valid_address(chain, addr) is ok


from src.signals.token_sources import parse_rugcheck_report

_REPORT = {
    "mintAuthority": None, "freezeAuthority": None, "rugged": False,
    "transferFee": {"pct": 0}, "knownAccounts": {"POOL1": {"name": "Orca", "type": "AMM"}},
    "topHolders": [
        {"address": "a1", "owner": "POOL1", "pct": 40.0, "insider": False},
        {"address": "a2", "owner": "w2", "pct": 8.0, "insider": False},
        {"address": "a3", "owner": "w3", "pct": 6.0, "insider": True},
    ],
}


class TestRugcheckReport:
    def test_parses_authorities_and_fee(self) -> None:
        r = parse_rugcheck_report(_REPORT)
        assert r["mintable"] is False and r["rugged"] is False
        assert r["buy_tax"] == 0.0 and r["sell_tax"] == 0.0

    def test_active_mint_authority_means_mintable(self) -> None:
        assert parse_rugcheck_report({**_REPORT, "mintAuthority": "Abc"})["mintable"] is True

    def test_top_holders_exclude_known_pools_and_use_fractions(self) -> None:
        assert parse_rugcheck_report(_REPORT)["top10_holder_pct"] == pytest.approx(0.14)

    def test_transfer_fee_becomes_both_taxes(self) -> None:
        r = parse_rugcheck_report({**_REPORT, "transferFee": {"pct": 12}})
        assert r["buy_tax"] == pytest.approx(0.12) and r["sell_tax"] == pytest.approx(0.12)

    def test_missing_holders_leaves_concentration_unknown(self) -> None:
        r = parse_rugcheck_report({k: v for k, v in _REPORT.items() if k != "topHolders"})
        assert r["top10_holder_pct"] is None

    def test_garbage_returns_none(self) -> None:
        assert parse_rugcheck_report("x") is None
        assert parse_rugcheck_report({"error": "nope"}) is None


class TestSolanaSpecifics:
    def test_rugged_token_is_a_hard_fail(self) -> None:
        a = assess_token(_safe(rugged=True))
        assert a.avoid and any("rugged" in f.lower() for f in a.hard_fails)

    def test_solana_does_not_warn_about_unknown_honeypot_or_tax(self) -> None:
        a = assess_token(_safe(honeypot=None, buy_tax=None, sell_tax=None))
        assert a.grade == "A"

    def test_evm_still_warns_about_unknown_honeypot(self) -> None:
        a = assess_token(_safe(chain="ethereum", address=EVM, honeypot=None))
        assert any("honeypot" in w.lower() for w in a.warnings)


class TestScamLanguageCountsOnce:
    def test_many_correlated_flags_are_one_warning(self) -> None:
        flags = {k: 0.99 for k in scam_questions()}
        a = assess_token(_safe(scam_language=flags))
        scam = [w for w in a.warnings if "scam" in w.lower()]
        assert len(scam) == 1 and a.grade == "B"
        assert all(name in scam[0] for name in ("promises_returns", "impersonation"))

    def test_below_threshold_flags_are_ignored(self) -> None:
        assert assess_token(_safe(scam_language={"promises_returns": 0.3})).grade == "A"
