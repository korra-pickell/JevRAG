from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np

from ..base import Searchable, check_k
from ..types import Doc, Hit, SearchResult, corpus_hash

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
Encoder = Callable[[list[str]], np.ndarray]


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def sentence_transformer_encoder(model_name: str, device: str | None = None,
                                 batch_size: int = 64) -> Encoder:
    model = None

    def encode(texts: list[str]) -> np.ndarray:
        nonlocal model
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device=device)
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=len(texts) > 1000)

    return encode


class DenseRetriever(Searchable):
    """Exact cosine search over sentence-transformers embeddings (install `dense`)."""

    def __init__(self, docs: Sequence[Doc], *, model_name: str = "BAAI/bge-base-en-v1.5",
                 query_prefix: str = BGE_QUERY_PREFIX, encoder: Encoder | None = None,
                 cache_dir: str | Path | None = None, device: str | None = None,
                 batch_size: int = 64):
        if not docs:
            raise ValueError("docs must not be empty")
        self.docs = list(docs)
        self.query_prefix = query_prefix
        self._encode = encoder or sentence_transformer_encoder(model_name, device, batch_size)
        self._lock = threading.Lock()
        self.from_cache = False
        t0 = perf_counter()
        self.embeddings = self._load_or_embed(model_name, cache_dir)
        self.build_seconds = perf_counter() - t0

    def _embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return _normalize(np.asarray(self._encode(texts), dtype=np.float32))

    def _load_or_embed(self, model_name: str, cache_dir: str | Path | None) -> np.ndarray:
        path = None
        if cache_dir:
            slug = re.sub(r"[^A-Za-z0-9.-]+", "_", model_name)
            path = Path(cache_dir) / f"{slug}-{corpus_hash(self.docs)}.npy"
            if path.exists():
                self.from_cache = True
                return np.load(path)
        emb = self._embed([d.full_text for d in self.docs])
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb

    def _search(self, query: str, k: int) -> list[Hit]:
        q = self._embed([self.query_prefix + query])[0]
        scores = self.embeddings @ q
        k = min(k, len(self.docs))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [Hit(self.docs[i], float(scores[i]), {"dense": float(scores[i])}) for i in idx]

    async def asearch_traced(self, query: str, k: int) -> SearchResult:
        check_k(k)
        t0 = perf_counter()
        hits = await asyncio.to_thread(self._search, query, k)
        return SearchResult(hits, timings={"retrieve": perf_counter() - t0})
