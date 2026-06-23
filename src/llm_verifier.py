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

Step 1 — Extract required specs from the QUERY.
  Identify every meaningful attribute in the query: product type, numeric values \
  (кВт/kW, V, kg, L, mm, A, Вт/W, AWG, etc.), material, color, brand, model, \
  fuel type, flexibility, and any other property explicitly stated.

Step 2 — Check each required spec against the LISTING.
  For every spec extracted in Step 1, check if the listing satisfies it:
  - Numeric specs: allowed tolerance ±5%. Example: 10 kW vs 10.4 kW → OK; \
    10 kW vs 15 kW → MISMATCH. Unit conversions count as equal (10 кВт = 10000 Вт).
  - Text specs (color, material, brand, fuel type): must match exactly in meaning. \
    Language does not matter — Ukrainian/Russian/English are equivalent.
  - If a spec is present in the query but ABSENT or UNCLEAR in the listing → MISMATCH.

Step 3 — Apply strict rejection rule.
  If ANY required spec from Step 1 does not match → the listing is REJECTED:
    match = false, confidence = 0.0
  List every mismatched spec in mismatched_specs.

Step 4 — Handle extra specs in the listing.
  The listing may contain additional specs not mentioned in the query (e.g. extra \
  colors, bundle size, certifications). These are IGNORED for matching purposes — \
  they do not make a listing worse or better. Only query specs matter.

Step 5 — Accept only if ALL required specs match.
  If every spec from Step 1 is satisfied: match = true, confidence between 0.7–1.0 \
  based on how precisely the listing matches (exact wording, additional detail, etc.).

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

        import json as _json

        def _to_list(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    parsed = _json.loads(val)
                    return parsed if isinstance(parsed, list) else [parsed]
                except Exception:
                    return [val] if val else []
            return []

        return MatchResult(
            listing_id=listing_id,
            listing_text=listing_text,
            match=tool_input.get("match", False),
            confidence=tool_input.get("confidence", 0.0),
            reason=tool_input.get("reason", ""),
            matched_specs=_to_list(tool_input.get("matched_specs", [])),
            mismatched_specs=_to_list(tool_input.get("mismatched_specs", [])),
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
