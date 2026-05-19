import pytest

from src.embedder import EmbeddingRetriever
from src.models import Listing


@pytest.fixture
def sample_listings():
    return [
        Listing(id="1", text="Бензиновий генератор Honda 10 кВт"),
        Listing(id="2", text="Honda gasoline generator 10 kW inverter"),
        Listing(id="3", text="Дизельний генератор Honda 15 кВт"),
        Listing(id="4", text="Дриль акумуляторна Bosch GSR 18V"),
        Listing(id="5", text="Генератор бензиновий 10000 Вт Honda"),
    ]


@pytest.fixture
def retriever(sample_listings):
    return EmbeddingRetriever(sample_listings)


def test_index_built_correctly(retriever, sample_listings):
    assert retriever._index.ntotal == len(sample_listings)


def test_retrieve_returns_tuples(retriever):
    results = retriever.retrieve("Honda генератор 10 кВт", k=3)
    assert len(results) > 0
    for idx, score in results:
        assert isinstance(idx, int)
        assert isinstance(score, float)


def test_identical_texts_high_score(retriever):
    results = retriever.retrieve("Бензиновий генератор Honda 10 кВт", k=5)
    top_idx, top_score = results[0]
    assert top_score > 0.85


def test_unrelated_product_low_score(retriever):
    results = retriever.retrieve("Honda генератор 10 кВт", k=5)
    for idx, score in results:
        if retriever.texts[idx] == "Дриль акумуляторна Bosch GSR 18V":
            assert score < 0.5
            return
    pytest.fail("Bosch drill listing not found in results")
