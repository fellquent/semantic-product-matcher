from __future__ import annotations

import time

import anthropic

from src.config import settings
from src.models import MatchResult

REPORT_MATCH_TOOL = {
    "name": "report_match",
    "description": "Report whether a listing matches the search query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "match": {
                "type": "boolean",
                "description": "True if the listing matches the query.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence score from 0.0 to 1.0.",
            },
            "reason": {
                "type": "string",
                "description": "Short explanation of why the listing matches or not.",
            },
            "matched_specs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specs that match between query and listing.",
            },
            "mismatched_specs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specs that do NOT match between query and listing.",
            },
        },
        "required": ["match", "confidence", "reason"],
    },
}

SYSTEM_PROMPT = """\
You are a product matching expert. You compare a marketplace listing against a user's \
search query and determine whether they refer to the same product.

Rules:
1. Numeric specs (кВт/kW, V, kg, L, mm, A, Вт/W) MUST match within ±5% tolerance. \
   For example, 10 kW vs 10.4 kW is acceptable, but 10 kW vs 15 kW is NOT a match. \
   10 кВт and 10000 Вт ARE the same (unit conversion).
2. Brand and model must match exactly if both are specified in the query.
3. Fuel type (бензин/gasoline/diesel/electric/дизель/електричний) must match exactly.
4. Language does not matter — compare by meaning. Ukrainian, Russian, and English \
   descriptions of the same product should match.
5. If the listing is missing a spec that the query requires, set confidence below 0.5 \
   and explain what is missing.

Always call the report_match tool with your verdict.\
"""


class LLMVerifier:
    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.api_calls = 0

    def verify(
        self,
        query: str,
        listing_id: str,
        listing_text: str,
        embedding_score: float,
    ) -> MatchResult:
        user_message = (
            f"Search query: {query}\n\n"
            f"Listing: {listing_text}\n\n"
            f"Embedding similarity score: {embedding_score:.3f}\n\n"
            "Does this listing match the search query? "
            "Call the report_match tool with your analysis."
        )

        tool_input = self._call_with_retry(user_message)
        self.api_calls += 1

        return MatchResult(
            listing_id=listing_id,
            listing_text=listing_text,
            match=tool_input.get("match", False),
            confidence=tool_input.get("confidence", 0.0),
            reason=tool_input.get("reason", ""),
            matched_specs=tool_input.get("matched_specs", []),
            mismatched_specs=tool_input.get("mismatched_specs", []),
            embedding_score=embedding_score,
        )

    def _call_with_retry(self, user_message: str, max_retries: int = 3) -> dict:
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=settings.claude_model,
                    max_tokens=512,
                    system=SYSTEM_PROMPT,
                    tools=[REPORT_MATCH_TOOL],
                    tool_choice={"type": "tool", "name": "report_match"},
                    messages=[{"role": "user", "content": user_message}],
                )
                for block in response.content:
                    if block.type == "tool_use" and block.name == "report_match":
                        return block.input
                raise ValueError("No report_match tool_use block in response")
            except anthropic.APIStatusError as e:
                if e.status_code == 529 and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)
                    continue
                raise
        raise RuntimeError("Max retries exceeded")
