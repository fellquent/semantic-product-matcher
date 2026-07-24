from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from .config import settings
from .models import Listing


_ONNX_MODEL_DIR = Path(__file__).resolve().parent.parent / settings.onnx_model_dir

_tokenizer: AutoTokenizer | None = None
_session: ort.InferenceSession | None = None


def _get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_ONNX_MODEL_DIR)
    return _tokenizer


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(str(_ONNX_MODEL_DIR / "model.onnx"))
    return _session


def _encode(texts: list[str]) -> np.ndarray:
    tokenizer = _get_tokenizer()
    session = _get_session()

    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
    onnx_inputs = {
        "input_ids": tokens["input_ids"].astype(np.int64),
        "attention_mask": tokens["attention_mask"].astype(np.int64),
    }
    # Graph outputs [token_embeddings, sentence_embedding]; the latter is
    # already mean-pooled and L2-normalized by the exported ONNX graph.
    _, sentence_embedding = session.run(None, onnx_inputs)
    return sentence_embedding.astype(np.float32)


class EmbeddingRetriever:
    def __init__(self, listings: list[Listing]) -> None:
        self.listings = listings
        self.texts = [l.text for l in listings]
        self._index = self._build_index()

    def _build_index(self) -> faiss.Index:
        embeddings = _encode(self.texts)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve(self, query: str, k: int | None = None) -> list[tuple[int, float]]:
        if k is None:
            k = settings.faiss_top_k
        k = min(k, len(self.listings))

        query_vec = _encode([query])

        distances, indices = self._index.search(query_vec, k)

        results: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append((int(idx), float(score)))
        return results
