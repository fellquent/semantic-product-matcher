from __future__ import annotations

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.models import Listing


_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


class EmbeddingRetriever:
    def __init__(self, listings: list[Listing]) -> None:
        self.listings = listings
        self.texts = [l.text for l in listings]
        self._index = self._build_index()

    def _build_index(self) -> faiss.Index:
        model = _get_model()
        embeddings = model.encode(self.texts, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve(self, query: str, k: int | None = None) -> list[tuple[int, float]]:
        if k is None:
            k = settings.faiss_top_k
        k = min(k, len(self.listings))

        model = _get_model()
        query_vec = model.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype=np.float32)

        distances, indices = self._index.search(query_vec, k)

        results: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results
