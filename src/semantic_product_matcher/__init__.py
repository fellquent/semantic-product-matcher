"""Hybrid product matching: embedding retrieval + LLM verification of specs."""

from .config import MatcherConfig
from .models import Listing, MatchResult

__version__ = "0.1.0"

__all__ = [
    "Listing",
    "MatchResult",
    "MatcherConfig",
]
