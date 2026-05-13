from __future__ import annotations

from src.config import settings
from src.embedder import EmbeddingRetriever
from src.llm_verifier import LLMVerifier
from src.models import Listing, MatchResult


class HybridMatcher:
    def __init__(self, listings: list[Listing]) -> None:
        self.listings = listings
        self.retriever = EmbeddingRetriever(listings)
        self.verifier = LLMVerifier()

    def search(self, query: str) -> list[MatchResult]:
        candidates = self.retriever.retrieve(query, k=settings.faiss_top_k)

        candidates = [
            (idx, score)
            for idx, score in candidates
            if score >= settings.similarity_threshold
        ]

        candidates = candidates[: settings.llm_top_k]

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

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    @property
    def api_calls(self) -> int:
        return self.verifier.api_calls
