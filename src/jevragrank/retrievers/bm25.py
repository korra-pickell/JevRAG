from __future__ import annotations

import asyncio
from collections.abc import Sequence
from time import perf_counter

from ..base import Searchable, check_k
from ..types import Doc, Hit, SearchResult


class BM25Retriever(Searchable):
    """Lexical BM25 via bm25s (install the `bm25` extra)."""

    def __init__(self, docs: Sequence[Doc], *, stopwords: str = "en", stem: bool = True):
        import bm25s
        import Stemmer

        if not docs:
            raise ValueError("docs must not be empty")
        self.docs = list(docs)
        self._bm25s = bm25s
        self._stopwords = stopwords
        self._stemmer = Stemmer.Stemmer("english") if stem else None
        t0 = perf_counter()
        tokens = bm25s.tokenize([d.full_text for d in self.docs], stopwords=stopwords,
                                stemmer=self._stemmer, show_progress=False)
        self._index = bm25s.BM25()
        self._index.index(tokens, show_progress=False)
        self.build_seconds = perf_counter() - t0

    def _search(self, query: str, k: int) -> list[Hit]:
        q = self._bm25s.tokenize([query], stopwords=self._stopwords, stemmer=self._stemmer,
                                 return_ids=False, show_progress=False)
        if not q[0]:
            return []
        idx, scores = self._index.retrieve(q, k=min(k, len(self.docs)), show_progress=False)
        return [Hit(self.docs[i], float(s), {"bm25": float(s)})
                for i, s in zip(idx[0].tolist(), scores[0].tolist(), strict=True) if s > 0]

    async def asearch_traced(self, query: str, k: int) -> SearchResult:
        check_k(k)
        t0 = perf_counter()
        hits = await asyncio.to_thread(self._search, query, k)
        return SearchResult(hits, timings={"retrieve": perf_counter() - t0})
