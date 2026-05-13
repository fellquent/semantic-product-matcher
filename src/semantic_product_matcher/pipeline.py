"""The hybrid matcher: embedding retrieval narrows the field, the verifier decides."""

from __future__ import annotations

from typing import Sequence

from .config import MatcherConfig
from .embedder import EmbeddingRetriever
from .models import Listing, MatchResult
from .verifier import ClaudeVerifier, Verifier


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

    def search(self, query: str) -> list[MatchResult]:
        """Listings the verifier accepted, highest confidence first."""
        threshold = self.config.similarity_threshold
        candidates = [(i, s) for i, s in self.retriever.retrieve(query) if s >= threshold]
        verified = self.verifier.verify_many(
            query, [(self.listings[i].id, self.listings[i].text, s) for i, s in candidates]
        )
        return sorted((r for r in verified if r.match), key=lambda r: r.confidence, reverse=True)

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
