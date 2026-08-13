from __future__ import annotations
import time
from dataclasses import dataclass, field

from .config import settings
from .embedder import EmbeddingRetriever
from .llm_verifier import LLMVerifier
from .models import Listing, MatchResult


@dataclass
class SearchDebugInfo:
    total_listings: int = 0
    faiss_retrieved: int = 0
    above_threshold: int = 0
    sent_to_llm: int = 0
    below_threshold: list[tuple[str, float]] = field(default_factory=list)  # (text, score)
    llm_rejected: list[MatchResult] = field(default_factory=list)
    llm_accepted: list[MatchResult] = field(default_factory=list)
    embedding_time: float = 0.0
    llm_time: float = 0.0


class HybridMatcher:
    def __init__(self, listings: list[Listing]) -> None:
        self.listings = listings
        self.retriever = EmbeddingRetriever(listings)
        self.verifier = LLMVerifier()
        self.debug: SearchDebugInfo = SearchDebugInfo()

    def search(self, query: str, similarity_threshold: float | None = None) -> list[MatchResult]:
        self.debug = SearchDebugInfo(total_listings=len(self.listings))
        threshold = similarity_threshold if similarity_threshold is not None else settings.similarity_threshold

        embed_start = time.time()
        all_candidates = self.retriever.retrieve(query)
        self.debug.embedding_time = time.time() - embed_start
        self.debug.faiss_retrieved = len(all_candidates)

        above = [(idx, score) for idx, score in all_candidates if score >= threshold]
        below = [(idx, score) for idx, score in all_candidates if score < threshold]
        self.debug.above_threshold = len(above)
        self.debug.below_threshold = [(self.listings[idx].text, score) for idx, score in below]

        candidates = above
        self.debug.sent_to_llm = len(candidates)

        llm_start = time.time()
        items = [(self.listings[idx].id, self.listings[idx].text, score) for idx, score in candidates]
        verified = self.verifier.verify_many(query, items)
        self.debug.llm_time = time.time() - llm_start

        results: list[MatchResult] = []
        for result in verified:
            if result.match:
                results.append(result)
                self.debug.llm_accepted.append(result)
            else:
                self.debug.llm_rejected.append(result)

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    @property
    def input_tokens(self) -> int:
        return self.verifier.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.verifier.output_tokens

    @property
    def api_calls(self) -> int:
        return self.verifier.api_calls
