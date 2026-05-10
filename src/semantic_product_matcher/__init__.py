"""Hybrid product matching: embedding retrieval + LLM verification of specs."""

from .config import MatcherConfig
from .embedder import EmbeddingRetriever, sentence_transformer_encoder
from .models import Listing, MatchResult
from .verifier import ClaudeVerifier, Verifier

__version__ = "0.1.0"

__all__ = [
    "ClaudeVerifier",
    "EmbeddingRetriever",
    "Listing",
    "MatchResult",
    "MatcherConfig",
    "Verifier",
    "sentence_transformer_encoder",
]
