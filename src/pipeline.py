from __future__ import annotations
from dataclasses import dataclass, field

from src.config import settings
from src.embedder import EmbeddingRetriever
from src.llm_verifier import LLMVerifier
from src.models import Listing, MatchResult


@dataclass
class SearchDebugInfo:
    total_listings: int = 0
    faiss_retrieved: int = 0
    above_threshold: int = 0
    sent_to_llm: int = 0
    below_threshold: list[tuple[str, float]] = field(default_factory=list)  # (text, score)
    skipped_llm_cap: list[tuple[str, float]] = field(default_factory=list)  # (text, score)
    llm_rejected: list[MatchResult] = field(default_factory=list)
    llm_accepted: list[MatchResult] = field(default_factory=list)


class HybridMatcher:
    def __init__(self, listings: list[Listing]) -> None:
        self.listings = listings
        self.retriever = EmbeddingRetriever(listings)
        self.verifier = LLMVerifier()
        self.debug: SearchDebugInfo = SearchDebugInfo()

    def search(self, query: str) -> list[MatchResult]:
        self.debug = SearchDebugInfo(total_listings=len(self.listings))

        all_candidates = self.retriever.retrieve(query, k=settings.faiss_top_k)
        self.debug.faiss_retrieved = len(all_candidates)

        above = [(idx, score) for idx, score in all_candidates if score >= settings.similarity_threshold]
        below = [(idx, score) for idx, score in all_candidates if score < settings.similarity_threshold]
        self.debug.above_threshold = len(above)
        self.debug.below_threshold = [(self.listings[idx].text, score) for idx, score in below]

        candidates = above
        self.debug.sent_to_llm = len(candidates)
        self.debug.skipped_llm_cap = []

        results: list[MatchResult] = []
        for idx, score in candidates:
            listing = self.listings[idx]
            result = self.verifier.verify(
                query=query,
                listing_id=listing.id,
                listing_text=listing.text,
                embedding_score=score,
            )
            if result.match:
                results.append(result)
                self.debug.llm_accepted.append(result)
            else:
                self.debug.llm_rejected.append(result)

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    @property
    def api_calls(self) -> int:
        return self.verifier.api_calls
