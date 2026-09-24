from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ._sync import run_sync
from .types import Hit, SearchResult


def check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def rank_hits(hits: list[Hit], scores: Sequence[float], *, stage: str, k: int,
              degraded: Sequence[bool] | None = None) -> list[Hit]:
    """Sort hits by new score, breaking ties by the incoming score, then input order."""
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], -hits[i].score, i))
    out = []
    for i in order[:k]:
        h = hits[i]
        s = float(scores[i])
        out.append(Hit(h.doc, s, {**h.stage_scores, stage: s},
                       degraded=h.degraded or bool(degraded and degraded[i])))
    return out


class Searchable(ABC):
    @abstractmethod
    async def asearch_traced(self, query: str, k: int) -> SearchResult: ...

    async def asearch(self, query: str, k: int = 10) -> list[Hit]:
        return (await self.asearch_traced(query, k)).hits

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return run_sync(self.asearch(query, k))


class Reranker(ABC):
    @abstractmethod
    async def arerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...

    def rerank(self, query: str, hits: list[Hit], k: int = 10) -> list[Hit]:
        return run_sync(self.arerank(query, hits, k))
