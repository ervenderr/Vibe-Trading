"""TokenDataClient: merging sources and failing closed when a source is down."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from src.signals.token_risk import assess_token
from src.signals.token_sources import TokenDataClient

NOW = datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc)
SOL = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
EVM = "0x6982508145454ce325ddbe47a25d4ec3d2311933"

DEX = [{"chainId": "x", "baseToken": {"address": SOL, "symbol": "BONK"}, "liquidity": {"usd": 400000},
        "volume": {"h24": 1000}, "txns": {"h1": {"buys": 10, "sells": 12}}, "pairCreatedAt": 1671980424000}]
RUG_SUMMARY = {"risks": [{"name": "Mutable metadata", "level": "warn"}], "lpLockedPct": 15.0}
RUG_REPORT = {"mintAuthority": None, "rugged": False, "transferFee": {"pct": 0},
              "topHolders": [{"address": "a", "owner": "o", "pct": 8.0, "insider": False}]}
GOPLUS = {"code": 1, "result": {EVM: {"is_honeypot": "0", "buy_tax": "0", "sell_tax": "0", "is_mintable": "0",
                                      "hidden_owner": "0", "can_take_back_ownership": "0",
                                      "holders": [{"percent": "0.08", "is_contract": 0}],
                                      "lp_holders": [{"percent": "0.9", "is_locked": 1}]}}}


def _client(routes: dict[str, httpx.Response | Exception]) -> TokenDataClient:
    def handler(request: httpx.Request) -> httpx.Response:
        for needle, result in routes.items():
            if needle in str(request.url):
                if isinstance(result, Exception):
                    raise result
                return result
        return httpx.Response(404)

    return TokenDataClient(transport=httpx.MockTransport(handler))


def _json(data) -> httpx.Response:
    return httpx.Response(200, json=data)


class TestSolana:
    def test_merges_dexscreener_and_both_rugcheck_reports(self) -> None:
        c = _client({"dexscreener": _json(DEX), "report/summary": _json(RUG_SUMMARY), "/report": _json(RUG_REPORT)})
        f = c.fetch_facts("solana", SOL, now=NOW)
        assert f.market_checked and f.security_checked
        assert f.liquidity_usd == 400000 and f.top10_holder_pct == pytest.approx(0.08)
        assert f.lp_locked_pct == pytest.approx(0.15) and f.mintable is False
        assert f.warn_risks == ("Mutable metadata",)

    def test_rugcheck_down_means_unverified_and_avoid(self) -> None:
        c = _client({"dexscreener": _json(DEX), "rugcheck": httpx.Response(500)})
        f = c.fetch_facts("solana", SOL, now=NOW)
        assert f.security_checked is False
        assert assess_token(f).avoid is True

    def test_dexscreener_down_means_no_market_data_and_avoid(self) -> None:
        c = _client({"dexscreener": httpx.ConnectError("boom"), "report/summary": _json(RUG_SUMMARY),
                     "/report": _json(RUG_REPORT)})
        f = c.fetch_facts("solana", SOL, now=NOW)
        assert f.market_checked is False and assess_token(f).avoid is True

    def test_summary_ok_but_full_report_missing_is_unverified_and_avoid(self) -> None:
        # Holder concentration and mint authority live only in the full report.
        c = _client({"dexscreener": _json(DEX), "report/summary": _json(RUG_SUMMARY)})
        f = c.fetch_facts("solana", SOL, now=NOW)
        assert f.security_checked is False and assess_token(f).avoid is True


class TestEvm:
    def test_uses_goplus_for_evm_chains(self) -> None:
        dex = [{**DEX[0], "baseToken": {"address": EVM, "symbol": "PEPE"}}]
        c = _client({"dexscreener": _json(dex), "gopluslabs": _json(GOPLUS)})
        f = c.fetch_facts("ethereum", EVM, now=NOW)
        assert f.honeypot is False and f.security_checked and f.lp_locked_pct == pytest.approx(0.9)

    def test_goplus_non_json_body_is_unverified(self) -> None:
        dex = [{**DEX[0], "baseToken": {"address": EVM, "symbol": "PEPE"}}]
        c = _client({"dexscreener": _json(dex), "gopluslabs": httpx.Response(200, text="<html>")})
        assert c.fetch_facts("ethereum", EVM, now=NOW).security_checked is False


class TestValidation:
    def test_invalid_address_never_hits_the_network(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("network must not be used")

        c = TokenDataClient(transport=httpx.MockTransport(handler))
        with pytest.raises(ValueError):
            c.fetch_facts("solana", "../etc/passwd")
