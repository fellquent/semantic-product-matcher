"""Hybrid product matching: embedding retrieval + LLM verification of specs."""

from .config import MatcherConfig
from .embedder import EmbeddingRetriever, sentence_transformer_encoder
from .models import Listing, MatchResult
from .pipeline import HybridMatcher
from .verifier import ClaudeVerifier, Verifier

__version__ = "0.1.0"

__all__ = [
    "ClaudeVerifier",
    "EmbeddingRetriever",
    "HybridMatcher",
    "Listing",
    "MatchResult",
    "MatcherConfig",
    "Verifier",
    "sentence_transformer_encoder",
]
