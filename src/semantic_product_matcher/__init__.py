"""Hybrid product matching: embedding retrieval + LLM verification of specs."""

from .config import MatcherConfig
from .embedder import EmbeddingRetriever, sentence_transformer_encoder
from .models import Listing, MatchResult

__version__ = "0.1.0"

__all__ = [
    "EmbeddingRetriever",
    "Listing",
    "MatchResult",
    "MatcherConfig",
    "sentence_transformer_encoder",
]
