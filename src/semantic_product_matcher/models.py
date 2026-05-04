from pydantic import BaseModel, Field


class Listing(BaseModel):
    """One product offer to match against. `text` is what gets embedded and verified."""

    id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class MatchResult(BaseModel):
    listing_id: str
    listing_text: str
    match: bool
    confidence: float
    reason: str
    matched_specs: list[str] = Field(default_factory=list)
    mismatched_specs: list[str] = Field(default_factory=list)
    embedding_score: float
