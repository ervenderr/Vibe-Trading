"""Tests for the token risk service, agent tool and CLI wiring."""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from src.signals.token_risk import TokenFacts, scam_questions
from src.signals.token_service import assess_address, assessment_to_dict, run_token_command
from src.tools import token_risk_tool

SOL = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"


def _facts(**kw) -> TokenFacts:
    base = dict(chain="solana", address=SOL, symbol="BONK", liquidity_usd=400_000.0, age_hours=9000.0,
                honeypot=None, mintable=False, top10_holder_pct=0.2, lp_locked_pct=0.9,
                security_checked=True, market_checked=True)
    return TokenFacts(**{**base, **kw})


class FakeData:
    def __init__(self, facts: TokenFacts | Exception) -> None:
        self._facts = facts

    def fetch_facts(self, chain, address):
        if isinstance(self._facts, Exception):
            raise self._facts
        return self._facts


class FakeJev:
    def __init__(self, p: float) -> None:
        self.p, self.calls = p, []

    def system_one(self, state, questions, *, model=None, **_):
        self.calls.append(state)
        return SimpleNamespace(nouls={k: SimpleNamespace(noul=self.p) for k in scam_questions()})


class TestAssessAddress:
    def test_without_description_jev_is_not_called(self) -> None:
        jev = FakeJev(0.99)
        a = assess_address("solana", SOL, data_client=FakeData(_facts()), jev_client=jev, model="m")
        assert jev.calls == [] and a.grade == "A"

    def test_description_is_judged_and_warns(self) -> None:
        jev = FakeJev(0.95)
        a = assess_address("solana", SOL, description="GUARANTEED 1000x!!", data_client=FakeData(_facts()),
                           jev_client=jev, model="m")
        assert jev.calls[0]["description"] == "GUARANTEED 1000x!!"
        assert a.avoid is False and any("scam" in w.lower() for w in a.warnings)

    def test_jev_failure_does_not_break_the_assessment(self) -> None:
        from typesafe_sdk import TypeSafeError

        class Down(FakeJev):
            def system_one(self, *a, **k):
                raise TypeSafeError("down")

        a = assess_address("solana", SOL, description="x", data_client=FakeData(_facts()), jev_client=Down(0), model="m")
        assert a.grade == "A"

    def test_long_description_is_capped(self) -> None:
        jev = FakeJev(0.1)
        assess_address("solana", SOL, description="x" * 50_000, data_client=FakeData(_facts()), jev_client=jev, model="m")
        assert len(jev.calls[0]["description"]) <= 4000

    def test_invalid_address_rejected_before_any_fetch(self) -> None:
        with pytest.raises(ValueError):
            assess_address("solana", "../x", data_client=FakeData(_facts()), jev_client=None, model="m")


class TestToDict:
    def test_shape_and_disclaimer(self) -> None:
        a = assess_address("solana", SOL, data_client=FakeData(_facts(honeypot=None, liquidity_usd=100.0)), jev_client=None, model="m")
        d = assessment_to_dict(_facts(liquidity_usd=100.0), a)
        assert d["avoid"] is True and d["grade"] == "F" and d["hard_fails"]
        assert "cannot predict" in d["disclaimer"].lower()
        json.dumps(d)


class TestTool:
    def test_ok_json(self, monkeypatch) -> None:
        monkeypatch.setattr(token_risk_tool, "assess_live", lambda c, a, d: (_facts(), __import__("src.signals.token_risk", fromlist=["assess_token"]).assess_token(_facts())))
        out = json.loads(token_risk_tool.TokenRiskTool().execute(chain="solana", address=SOL))
        assert out["status"] == "ok" and out["grade"] in "ABCD"

    def test_bad_input_returns_error_json(self) -> None:
        out = json.loads(token_risk_tool.TokenRiskTool().execute(chain="solana", address="../x"))
        assert out["status"] == "error"
        out = json.loads(token_risk_tool.TokenRiskTool().execute(chain="moonchain", address=SOL))
        assert out["status"] == "error"

    def test_failure_does_not_leak_details(self, monkeypatch) -> None:
        def boom(c, a, d):
            raise RuntimeError("secret-xyz")

        monkeypatch.setattr(token_risk_tool, "assess_live", boom)
        raw = token_risk_tool.TokenRiskTool().execute(chain="solana", address=SOL)
        assert "secret-xyz" not in raw and json.loads(raw)["status"] == "error"

    def test_tool_metadata(self) -> None:
        t = token_risk_tool.TokenRiskTool()
        assert t.name == "token_risk" and t.is_readonly
        assert set(t.parameters["required"]) == {"chain", "address"}


class TestCommand:
    def test_prints_grade_and_reasons(self) -> None:
        out: list[str] = []
        code = run_token_command(Namespace(chain="solana", address=SOL, description=None, json=False),
                                 assessor=lambda c, a, d: (_facts(honeypot=None, liquidity_usd=50.0),
                                                           __import__("src.signals.token_risk", fromlist=["assess_token"]).assess_token(_facts(liquidity_usd=50.0))),
                                 out=out.append)
        text = "\n".join(out)
        assert code == 0 and "AVOID" in text and "liquidity" in text.lower()

    def test_invalid_address_exit_code_1(self) -> None:
        out: list[str] = []
        code = run_token_command(Namespace(chain="solana", address="../x", description=None, json=False),
                                 assessor=lambda c, a, d: None, out=out.append)
        assert code == 1
