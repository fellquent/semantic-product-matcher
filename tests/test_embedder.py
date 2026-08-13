import pytest

from semantic_product_matcher import EmbeddingRetriever, Listing
from tests.conftest import bag_of_words_encoder


@pytest.fixture
def retriever(ssd_listings):
    return EmbeddingRetriever(ssd_listings, encoder=bag_of_words_encoder)


def test_every_listing_is_indexed(retriever, ssd_listings):
    assert retriever._index.ntotal == len(ssd_listings)


def test_scores_all_listings_best_first(retriever):
    results = retriever.retrieve("Samsung 990 PRO NVMe SSD 2TB")
    assert len(results) == 5
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0] == (0, pytest.approx(1.0))


def test_k_caps_the_number_of_results(retriever):
    assert len(retriever.retrieve("Samsung", k=2)) == 2
    assert len(retriever.retrieve("Samsung", k=50)) == 5


def test_no_listings_no_results():
    assert EmbeddingRetriever([], encoder=bag_of_words_encoder).retrieve("anything") == []


def test_scores_every_listing_by_default():
    listings = [Listing(id=str(i), text=f"SSD model {i}") for i in range(80)]
    retriever = EmbeddingRetriever(listings, encoder=bag_of_words_encoder)
    assert len(retriever.retrieve("SSD")) == 80


@pytest.mark.integration
def test_real_model_separates_related_and_unrelated(ssd_listings):
    retriever = EmbeddingRetriever(ssd_listings)  # BAAI/bge-m3
    scores = {retriever.texts[i]: s for i, s in retriever.retrieve("Samsung 990 PRO NVMe SSD 2 TB")}
    assert scores["Samsung 990 PRO NVMe SSD 2TB"] > 0.9
    assert scores["Logitech MX Master 3S wireless mouse"] < 0.5
    # the reason stage 2 exists: a wrong capacity still looks like a near-duplicate
    assert scores["Samsung 990 PRO NVMe SSD 1TB"] > 0.8
