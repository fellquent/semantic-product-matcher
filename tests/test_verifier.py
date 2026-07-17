from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from semantic_product_matcher import ClaudeVerifier, MatcherConfig
from semantic_product_matcher.verifier import build_system_prompt


def tool_response(verdict, input_tokens=300, output_tokens=80):
    block = SimpleNamespace(type="tool_use", name="report_match", input=verdict)
    return SimpleNamespace(
        content=[block], stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def verifier_with(*responses, **config):
    client = MagicMock()
    client.messages.create.side_effect = list(responses)
    return ClaudeVerifier(MatcherConfig(_env_file=None, **config), client=client), client


def test_parses_a_match():
    verifier, _ = verifier_with(tool_response({
        "match": True, "confidence": 0.95, "reason": "Same drive; 2000 GB = 2 TB",
        "matched_specs": ["brand: Samsung", "model: 990 PRO", "capacity: 2 TB"], "mismatched_specs": [],
    }))
    r = verifier.verify("Samsung 990 PRO NVMe SSD 2 TB", "1", "Samsung 990 Pro 2000 GB M.2 NVMe drive", 0.87)
    assert (r.match, r.confidence, r.listing_id, r.embedding_score) == (True, 0.95, "1", 0.87)
    assert r.matched_specs == ["brand: Samsung", "model: 990 PRO", "capacity: 2 TB"]


def test_spec_lists_sent_as_json_strings_are_decoded():
    verifier, _ = verifier_with(tool_response({
        "match": False, "confidence": 0.0, "reason": "Capacity differs",
        "mismatched_specs": '["capacity: 2 TB vs 1 TB"]',
    }))
    r = verifier.verify("Samsung 990 PRO 2 TB", "3", "Samsung 990 PRO 1TB", 0.91)
    assert r.match is False and r.mismatched_specs == ["capacity: 2 TB vs 1 TB"] and r.matched_specs == []


def test_request_forces_the_tool_by_default():
    verifier, client = verifier_with(tool_response({"match": True, "confidence": 0.9, "reason": "ok"}))
    verifier.verify("q", "1", "listing", 0.7)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "report_match"}
    assert kwargs["tools"][0]["name"] == "report_match"


def test_missing_tool_call_is_an_error():
    text_only = SimpleNamespace(content=[SimpleNamespace(type="text", text="I think so")], stop_reason="end_turn",
                                usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    verifier, _ = verifier_with(text_only)
    with pytest.raises(ValueError, match="report_match"):
        verifier.verify("q", "1", "listing", 0.7)


def test_verify_many_keeps_order_and_counts_usage():
    responses = [tool_response({"match": i % 2 == 0, "confidence": 0.5, "reason": str(i)}) for i in range(6)]
    verifier, client = verifier_with(*responses, llm_concurrency=1)
    results = verifier.verify_many("q", [(str(i), f"listing {i}", 0.6) for i in range(6)])
    assert [r.listing_id for r in results] == [str(i) for i in range(6)]
    assert (verifier.api_calls, verifier.input_tokens, verifier.output_tokens) == (6, 1800, 480)
    assert verifier.verify_many("q", []) == []


def test_reason_language_follows_config():
    prompt = build_system_prompt(MatcherConfig(_env_file=None, reason_language="German"))
    assert 'Write the "reason" field in German' in prompt


def test_client_uses_sdk_retries_and_env_credentials():
    with patch("semantic_product_matcher.verifier.anthropic.Anthropic") as anthropic_cls:
        ClaudeVerifier(MatcherConfig(_env_file=None, anthropic_api_key="", max_retries=7))
    # api_key=None lets the SDK resolve ANTHROPIC_API_KEY / an `ant auth login` profile itself
    anthropic_cls.assert_called_once_with(api_key=None, max_retries=7)
