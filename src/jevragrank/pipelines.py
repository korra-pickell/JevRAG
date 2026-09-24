from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from .base import Reranker, Searchable, check_k
from .jev_client import JevClient
from .rerankers.jev import JevReranker
from .retrievers.dense import DenseRetriever
from .types import Doc, SearchResult


class RetrieveThenRerank(Searchable):
    """Retrieve `candidates` hits, then rerank them down to k."""

    def __init__(self, retriever: Searchable, reranker: Reranker, *, candidates: int = 50):
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        self.retriever = retriever
        self.reranker = reranker
        self.candidates = candidates

    async def asearch_traced(self, query: str, k: int) -> SearchResult:
        check_k(k)
        if self.candidates < k:
            raise ValueError(f"candidates ({self.candidates}) must be >= k ({k})")
        t0 = perf_counter()
        retrieved = await self.retriever.asearch_traced(query, self.candidates)
        t1 = perf_counter()
        hits = await self.reranker.arerank(query, retrieved.hits, k)
        t2 = perf_counter()
        return SearchResult(hits, candidates=[h.doc.id for h in retrieved.hits],
                            timings={"retrieve": t1 - t0, "rerank": t2 - t1})


class JevRAGRank(RetrieveThenRerank):
    """Dense embedding recall, then Jev reranking."""

    def __init__(self, docs: Sequence[Doc], *, jev: JevClient, candidates: int = 50,
                 strategy: str = "score", retriever: Searchable | None = None,
                 dense_model: str = "BAAI/bge-base-en-v1.5",
                 cache_dir: str | Path | None = None, device: str | None = None,
                 **reranker_kwargs):
        retriever = retriever or DenseRetriever(docs, model_name=dense_model,
                                                cache_dir=cache_dir, device=device)
        super().__init__(retriever, JevReranker(jev, strategy=strategy, **reranker_kwargs),
                         candidates=candidates)
