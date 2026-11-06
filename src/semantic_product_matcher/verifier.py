"""Stage 2: an LLM checks each candidate against the query, spec by spec.

Embeddings rank "Samsung 990 PRO 2TB" and "Samsung 990 PRO 1TB" as near
duplicates; the verifier is where such numeric mismatches get rejected.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import anthropic

from .config import MatcherConfig
from .models import MatchResult

# (listing_id, listing_text, embedding_score)
Candidate = tuple[str, str, float]


class Verifier(Protocol):
    """Anything that can judge candidates; HybridMatcher accepts any implementation."""

    def verify_many(self, query: str, items: list[Candidate]) -> list[MatchResult]: ...


REPORT_MATCH_TOOL = {
    "name": "report_match",
    "description": "Report whether a listing matches the search query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "match": {"type": "boolean", "description": "True if the listing matches the query."},
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence score from 0.0 to 1.0.",
            },
            "reason": {"type": "string", "description": "Short explanation of the verdict."},
            "matched_specs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Query specs that the listing satisfies.",
            },
            "mismatched_specs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Query specs that the listing does NOT satisfy.",
            },
        },
        "required": ["match", "confidence", "reason"],
    },
}

SYSTEM_PROMPT = """\
You are a product matching expert. You compare a marketplace listing against a user's \
search query and determine whether they refer to the same product.

Step 1 - Extract required specs from the QUERY.
  Identify every meaningful attribute in the query: product type, numeric values \
  (kW, W, V, A, kg, L, mm, AWG, etc.), material, color, brand, model, fuel type, \
  and any other property explicitly stated.

Step 2 - Check each required spec against the LISTING.
  - Numeric specs: allowed tolerance ±{tolerance}%. Unit conversions count as equal \
    (10 kW = 10000 W). Example: 10 kW vs 10.4 kW is within ±5%; 10 kW vs 15 kW is a MISMATCH.
  - Text specs (color, material, brand, fuel type): must match in meaning. \
    Language does not matter - the query and the listing may be in different languages.
  - A spec present in the query but ABSENT or UNCLEAR in the listing is a MISMATCH.

Step 3 - Strict rejection.
  If ANY required spec does not match, reject the listing: match = false, \
  confidence = 0.0, and list every mismatched spec in mismatched_specs.

Step 4 - Extra specs in the listing are ignored.
  Attributes the query does not mention (extra colors, bundle size, certifications) \
  neither help nor hurt. Only query specs matter.

Step 5 - Accept only if ALL required specs match.
  Then match = true, with confidence between 0.7 and 1.0 depending on how precisely \
  the listing matches.

Write the "reason" field in {language}. Keep it short (1-2 sentences).

Always call the report_match tool with your verdict.\
"""


def build_system_prompt(config: MatcherConfig) -> str:
    tolerance = f"{config.numeric_tolerance * 100:g}"
    return SYSTEM_PROMPT.format(tolerance=tolerance, language=config.reason_language)


def _as_list(value) -> list[str]:
    """Spec lists occasionally come back as a JSON-encoded string."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except ValueError:
            return [value]
        return [str(v) for v in parsed] if isinstance(parsed, list) else [str(parsed)]
    return []


class ClaudeVerifier:
    def __init__(self, config: MatcherConfig | None = None, client: anthropic.Anthropic | None = None) -> None:
        self.config = config or MatcherConfig()
        self.client = client or anthropic.Anthropic(
            api_key=self.config.anthropic_api_key or None,
            max_retries=self.config.max_retries,
        )
        self.system_prompt = build_system_prompt(self.config)
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()

    def verify_many(self, query: str, items: list[Candidate]) -> list[MatchResult]:
        """Verify candidates concurrently. Results come back in the order of `items`."""
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=self.config.llm_concurrency) as pool:
            futures = [pool.submit(self.verify, query, *item) for item in items]
            return [f.result() for f in futures]

    def verify(self, query: str, listing_id: str, listing_text: str, embedding_score: float) -> MatchResult:
        verdict = self._report(
            f"Search query: {query}\n\n"
            f"Listing: {listing_text}\n\n"
            f"Embedding similarity score: {embedding_score:.3f}\n\n"
            "Does this listing match the search query? Call the report_match tool with your analysis."
        )
        return MatchResult(
            listing_id=listing_id,
            listing_text=listing_text,
            match=bool(verdict.get("match", False)),
            confidence=float(verdict.get("confidence", 0.0)),
            reason=str(verdict.get("reason", "")),
            matched_specs=_as_list(verdict.get("matched_specs")),
            mismatched_specs=_as_list(verdict.get("mismatched_specs")),
            embedding_score=embedding_score,
        )

    def _report(self, user_message: str) -> dict:
        tool_choice = (
            {"type": "tool", "name": REPORT_MATCH_TOOL["name"]}
            if self.config.force_tool_choice
            else {"type": "auto"}
        )
        response = self.client.messages.create(
            model=self.config.claude_model,
            max_tokens=self.config.max_tokens,
            system=self.system_prompt,
            tools=[REPORT_MATCH_TOOL],
            tool_choice=tool_choice,
            messages=[{"role": "user", "content": user_message}],
        )
        with self._lock:
            self.api_calls += 1
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
        for block in response.content:
            if block.type == "tool_use" and block.name == REPORT_MATCH_TOOL["name"]:
                return block.input
        raise ValueError(f"model did not call {REPORT_MATCH_TOOL['name']} (stop_reason={response.stop_reason})")
