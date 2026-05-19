from unittest.mock import MagicMock, patch

import anthropic
import pytest

from src.llm_verifier import LLMVerifier


def _make_tool_use_response(tool_input: dict):
    response = MagicMock()
    block = MagicMock()
    block.type = "tool_use"
    block.name = "report_match"
    block.input = tool_input
    response.content = [block]
    return response


class TestLLMVerifier:
    def test_parse_match_true(self):
        tool_input = {
            "match": True,
            "confidence": 0.95,
            "reason": "Exact match: Honda 10 kW gasoline generator",
            "matched_specs": ["brand: Honda", "power: 10 kW", "fuel: gasoline"],
            "mismatched_specs": [],
        }
        mock_response = _make_tool_use_response(tool_input)

        with patch("src.llm_verifier.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_cls.return_value = mock_client

            verifier = LLMVerifier()
            result = verifier.verify(
                query="Honda генератор 10 кВт",
                listing_id="1",
                listing_text="Honda gasoline generator 10 kW",
                embedding_score=0.88,
            )

        assert result.match is True
        assert result.confidence == 0.95
        assert result.listing_id == "1"
        assert len(result.matched_specs) == 3

    def test_parse_match_false_wrong_power(self):
        tool_input = {
            "match": False,
            "confidence": 0.92,
            "reason": "Power mismatch: query asks for 10 kW, listing is 15 kW (50% difference, exceeds 5% tolerance)",
            "matched_specs": ["brand: Honda", "fuel: gasoline"],
            "mismatched_specs": ["power: 10 kW vs 15 kW"],
        }
        mock_response = _make_tool_use_response(tool_input)

        with patch("src.llm_verifier.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_cls.return_value = mock_client

            verifier = LLMVerifier()
            result = verifier.verify(
                query="Honda генератор 10 кВт",
                listing_id="4",
                listing_text="Honda генератор 15 кВт",
                embedding_score=0.82,
            )

        assert result.match is False
        assert "15 kW" in result.reason or "15 кВт" in result.reason

    def test_retry_on_529(self):
        tool_input = {
            "match": True,
            "confidence": 0.9,
            "reason": "Match after retry",
        }
        mock_response = _make_tool_use_response(tool_input)

        overload_error = anthropic.APIStatusError(
            message="Overloaded",
            response=MagicMock(status_code=529, headers={}, content=b""),
            body={"error": {"message": "Overloaded", "type": "overloaded_error"}},
        )

        with patch("src.llm_verifier.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = [
                overload_error,
                mock_response,
            ]
            mock_cls.return_value = mock_client

            with patch("src.llm_verifier.time.sleep"):
                verifier = LLMVerifier()
                result = verifier.verify(
                    query="test",
                    listing_id="1",
                    listing_text="test listing",
                    embedding_score=0.7,
                )

        assert result.match is True
        assert mock_client.messages.create.call_count == 2

    def test_retry_exhausted_raises(self):
        overload_error = anthropic.APIStatusError(
            message="Overloaded",
            response=MagicMock(status_code=529, headers={}, content=b""),
            body={"error": {"message": "Overloaded", "type": "overloaded_error"}},
        )

        with patch("src.llm_verifier.anthropic.Anthropic") as mock_cls:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = overload_error
            mock_cls.return_value = mock_client

            with patch("src.llm_verifier.time.sleep"):
                verifier = LLMVerifier()
                with pytest.raises(anthropic.APIStatusError):
                    verifier.verify(
                        query="test",
                        listing_id="1",
                        listing_text="test listing",
                        embedding_score=0.7,
                    )

        assert mock_client.messages.create.call_count == 3
