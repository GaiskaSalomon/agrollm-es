"""Embedding model wrapper (multilingual, CPU-friendly).

Uses sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-dim), a small
multilingual model that handles Spanish well and runs on CPU.
"""
from __future__ import annotations

from functools import lru_cache

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]):
    """Return L2-normalized embeddings as a list of numpy arrays."""
    vecs = _model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    return vecs
