import pytest

from semantic_product_matcher import EmbeddingRetriever, HybridMatcher, MatchResult, MatcherConfig
from tests.conftest import bag_of_words_encoder


class FixedRetriever:
    """Returns preset (index, score) pairs regardless of the query."""

    def __init__(self, scored):
        self.scored = scored

    def retrieve(self, query, k=None):
        return self.scored


class RecordingVerifier:
    """Accepts listings whose id is in `accept`, with the given confidences."""

    def __init__(self, accept):
        self.accept = accept
        self.seen = []

    def verify_many(self, query, items):
        self.seen.extend(items)
        return [
            MatchResult(listing_id=lid, listing_text=text, match=lid in self.accept,
                        confidence=self.accept.get(lid, 0.0), reason="test", embedding_score=score)
            for lid, text, score in items
        ]


@pytest.fixture
def config():
    return MatcherConfig(_env_file=None, similarity_threshold=0.55)


def test_only_candidates_above_threshold_reach_the_verifier(ssd_listings, config):
    verifier = RecordingVerifier(accept={"1": 0.9})
    matcher = HybridMatcher(ssd_listings, config,
                            retriever=FixedRetriever([(0, 0.95), (2, 0.80), (3, 0.30)]), verifier=verifier)
    results = matcher.search("Samsung 990 PRO NVMe SSD 2 TB")
    assert [lid for lid, _, _ in verifier.seen] == ["1", "3"]
    assert [r.listing_id for r in results] == ["1"]
    d = matcher.debug
    assert (d.total_listings, d.scored, d.above_threshold) == (5, 3, 2)
    assert d.below_threshold == [("Logitech MX Master 3S wireless mouse", 0.30)]
    assert [r.listing_id for r in d.llm_rejected] == ["3"]


def test_results_sorted_by_confidence(ssd_listings, config):
    matcher = HybridMatcher(ssd_listings, config,
                            retriever=FixedRetriever([(0, 0.9), (1, 0.85), (4, 0.8)]),
                            verifier=RecordingVerifier(accept={"1": 0.75, "2": 0.95, "5": 0.85}))
    assert [r.listing_id for r in matcher.search("q")] == ["2", "5", "1"]


def test_threshold_override(ssd_listings, config):
    verifier = RecordingVerifier(accept={})
    matcher = HybridMatcher(ssd_listings, config,
                            retriever=FixedRetriever([(0, 0.9), (1, 0.6)]), verifier=verifier)
    matcher.search("q", similarity_threshold=0.7)
    assert [lid for lid, _, _ in verifier.seen] == ["1"]


def test_end_to_end_with_real_index(ssd_listings, config):
    verifier = RecordingVerifier(accept={"1": 0.9})
    matcher = HybridMatcher(ssd_listings, config,
                            retriever=EmbeddingRetriever(ssd_listings, encoder=bag_of_words_encoder),
                            verifier=verifier)
    results = matcher.search("Samsung 990 PRO NVMe SSD 2TB")
    assert [r.listing_id for r in results] == ["1"]
    assert all(score >= 0.55 for _, _, score in verifier.seen)


def test_usage_counters_default_to_zero_for_custom_verifiers(ssd_listings, config):
    matcher = HybridMatcher(ssd_listings, config, retriever=FixedRetriever([]),
                            verifier=RecordingVerifier(accept={}))
    assert matcher.search("q") == []
    assert (matcher.api_calls, matcher.input_tokens, matcher.output_tokens) == (0, 0, 0)
