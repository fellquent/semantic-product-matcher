"""Stage 1: dense retrieval.

Listings are embedded once into a FAISS inner-product index. Vectors are
L2-normalized, so the inner product is the cosine similarity.
"""

from __future__ import annotations

from typing import Callable, Sequence

import faiss
import numpy as np

from .models import Listing

# texts -> array of shape (len(texts), dim) with L2-normalized rows
Encoder = Callable[[list[str]], np.ndarray]

_models: dict[str, object] = {}


def sentence_transformer_encoder(model_name: str) -> Encoder:
    """Encoder backed by sentence-transformers. Each model is loaded once per process."""

    def encode(texts: list[str]) -> np.ndarray:
        model = _models.get(model_name)
        if model is None:
            # Imported here: pulling in torch is slow and not needed with a custom encoder.
            from sentence_transformers import SentenceTransformer

            model = _models[model_name] = SentenceTransformer(model_name)
        return model.encode(list(texts), normalize_embeddings=True)

    return encode


class EmbeddingRetriever:
    def __init__(
        self,
        listings: Sequence[Listing],
        encoder: Encoder | None = None,
        model_name: str = "BAAI/bge-m3",
    ) -> None:
        self.listings = list(listings)
        self.texts = [listing.text for listing in self.listings]
        self._encode = encoder or sentence_transformer_encoder(model_name)
        self._index = self._build_index() if self.listings else None

    def _vectors(self, texts: list[str]) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(self._encode(texts), dtype=np.float32))

    def _build_index(self) -> faiss.Index:
        vectors = self._vectors(self.texts)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def retrieve(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        """(listing index, cosine similarity) for the k most similar listings, best first."""
        if self._index is None:
            return []
        k = min(k, len(self.listings))
        scores, indices = self._index.search(self._vectors([query]), k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i != -1]
