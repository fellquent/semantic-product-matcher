from pydantic import BaseModel


class Listing(BaseModel):
    id: str
    text: str
    metadata: dict = {}


class MatchResult(BaseModel):
    listing_id: str
    listing_text: str
    match: bool
    confidence: float
    reason: str
    matched_specs: list[str] = []
    mismatched_specs: list[str] = []
    embedding_score: float
