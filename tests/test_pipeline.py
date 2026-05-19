from unittest.mock import MagicMock, patch

import pytest

from src.models import Listing, MatchResult
from src.pipeline import HybridMatcher


@pytest.fixture
def listings():
    return [
        Listing(id="1", text="Бензиновий генератор Honda 10 кВт"),
        Listing(id="2", text="Honda gasoline generator 10 kW"),
        Listing(id="3", text="Бензиновий генератор Honda 15 кВт"),
        Listing(id="4", text="Дриль Bosch 18В"),
    ]


def _make_verify_result(listing_id, listing_text, match, confidence, embedding_score):
    return MatchResult(
        listing_id=listing_id,
        listing_text=listing_text,
        match=match,
        confidence=confidence,
        reason="test reason",
        embedding_score=embedding_score,
    )


class TestHybridMatcher:
    @patch("src.pipeline.LLMVerifier")
    @patch("src.pipeline.EmbeddingRetriever")
    def test_returns_only_matches(self, mock_retriever_cls, mock_verifier_cls, listings):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            (0, 0.90),
            (1, 0.85),
            (2, 0.80),
            (3, 0.30),
        ]
        mock_retriever_cls.return_value = mock_retriever

        mock_verifier = MagicMock()
        mock_verifier.api_calls = 0

        def verify_side_effect(query, listing_id, listing_text, embedding_score):
            if listing_id in ("1", "2"):
                return _make_verify_result(listing_id, listing_text, True, 0.95, embedding_score)
            return _make_verify_result(listing_id, listing_text, False, 0.1, embedding_score)

        mock_verifier.verify.side_effect = verify_side_effect
        mock_verifier_cls.return_value = mock_verifier

        matcher = HybridMatcher(listings)
        results = matcher.search("Honda генератор 10 кВт")

        assert len(results) == 2
        assert all(r.match for r in results)

    @patch("src.pipeline.LLMVerifier")
    @patch("src.pipeline.EmbeddingRetriever")
    def test_sorted_by_confidence_desc(self, mock_retriever_cls, mock_verifier_cls, listings):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            (0, 0.90),
            (1, 0.85),
        ]
        mock_retriever_cls.return_value = mock_retriever

        mock_verifier = MagicMock()
        mock_verifier.api_calls = 0

        def verify_side_effect(query, listing_id, listing_text, embedding_score):
            conf = 0.80 if listing_id == "1" else 0.95
            return _make_verify_result(listing_id, listing_text, True, conf, embedding_score)

        mock_verifier.verify.side_effect = verify_side_effect
        mock_verifier_cls.return_value = mock_verifier

        matcher = HybridMatcher(listings)
        results = matcher.search("Honda генератор 10 кВт")

        assert results[0].confidence >= results[1].confidence

    @patch("src.pipeline.LLMVerifier")
    @patch("src.pipeline.EmbeddingRetriever")
    def test_filters_below_similarity_threshold(self, mock_retriever_cls, mock_verifier_cls, listings):
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            (0, 0.90),
            (3, 0.30),
        ]
        mock_retriever_cls.return_value = mock_retriever

        mock_verifier = MagicMock()
        mock_verifier.api_calls = 0
        mock_verifier.verify.return_value = _make_verify_result("1", "test", True, 0.9, 0.9)
        mock_verifier_cls.return_value = mock_verifier

        matcher = HybridMatcher(listings)
        results = matcher.search("Honda генератор 10 кВт")

        assert mock_verifier.verify.call_count == 1


@pytest.mark.integration
class TestHybridMatcherIntegration:
    def test_real_api_search(self, listings):
        matcher = HybridMatcher(listings)
        results = matcher.search("Honda генератор 10 кВт")
        assert isinstance(results, list)
