from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence

from ..base import Reranker, check_k, rank_hits
from ..text import passage
from ..types import Hit

PairScorer = Callable[[list[tuple[str, str]]], Sequence[float]]


def _cross_encoder_scorer(model_name: str, device: str | None, batch_size: int) -> PairScorer:
    model = None

    def score(pairs: list[tuple[str, str]]) -> Sequence[float]:
        nonlocal model
        if model is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name, max_length=512, device=device)
        return model.predict(pairs, batch_size=batch_size, show_progress_bar=False).tolist()

    return score


class CrossEncoderReranker(Reranker):
    def __init__(self, *, model_name: str = "BAAI/bge-reranker-base",
                 max_passage_tokens: int = 512, batch_size: int = 32,
                 device: str | None = None, scorer: PairScorer | None = None):
        self.max_passage_tokens = max_passage_tokens
        self._score = scorer or _cross_encoder_scorer(model_name, device, batch_size)
        self._lock = threading.Lock()

    def _locked(self, pairs):
        with self._lock:
            return list(self._score(pairs))

    async def arerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        check_k(k)
        if not hits:
            return []
        pairs = [(query, passage(h.doc, self.max_passage_tokens)) for h in hits]
        scores = await asyncio.to_thread(self._locked, pairs)
        return rank_hits(hits, scores, stage="ce", k=k)
