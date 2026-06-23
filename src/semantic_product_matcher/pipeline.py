"""The hybrid matcher: embedding retrieval narrows the field, the verifier decides."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

from .config import MatcherConfig
from .embedder import EmbeddingRetriever
from .models import Listing, MatchResult
from .verifier import ClaudeVerifier, Verifier


@dataclass
class SearchDebugInfo:
    total_listings: int = 0
    scored: int = 0
    above_threshold: int = 0
    below_threshold: list[tuple[str, float]] = field(default_factory=list)  # (text, score)
    llm_accepted: list[MatchResult] = field(default_factory=list)
    llm_rejected: list[MatchResult] = field(default_factory=list)
    embedding_time: float = 0.0
    llm_time: float = 0.0


class HybridMatcher:
    """Build once per listing set, then call `search` for any number of queries.

    `retriever` and `verifier` can be swapped (a different embedding model, a
    rule-based or another LLM verifier); by default they come from `config`.
    """

    def __init__(
        self,
        listings: Sequence[Listing],
        config: MatcherConfig | None = None,
        *,
        retriever: EmbeddingRetriever | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self.config = config or MatcherConfig()
        self.listings = list(listings)
        self.retriever = retriever or EmbeddingRetriever(self.listings, model_name=self.config.embedding_model)
        self.verifier = verifier or ClaudeVerifier(self.config)
        self.debug = SearchDebugInfo()

    def search(self, query: str, similarity_threshold: float | None = None) -> list[MatchResult]:
        """Listings the verifier accepted, highest confidence first."""
        threshold = self.config.similarity_threshold if similarity_threshold is None else similarity_threshold
        self.debug = debug = SearchDebugInfo(total_listings=len(self.listings))

        start = time.time()
        scored = self.retriever.retrieve(query)
        debug.embedding_time = time.time() - start
        debug.scored = len(scored)

        candidates = [(i, s) for i, s in scored if s >= threshold]
        debug.above_threshold = len(candidates)
        debug.below_threshold = [(self.listings[i].text, s) for i, s in scored if s < threshold]

        start = time.time()
        verified = self.verifier.verify_many(
            query, [(self.listings[i].id, self.listings[i].text, s) for i, s in candidates]
        )
        debug.llm_time = time.time() - start

        for result in verified:
            (debug.llm_accepted if result.match else debug.llm_rejected).append(result)
        return sorted(debug.llm_accepted, key=lambda r: r.confidence, reverse=True)

    # Usage counters of the default ClaudeVerifier; 0 for verifiers that don't track them.
    @property
    def api_calls(self) -> int:
        return getattr(self.verifier, "api_calls", 0)

    @property
    def input_tokens(self) -> int:
        return getattr(self.verifier, "input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return getattr(self.verifier, "output_tokens", 0)
