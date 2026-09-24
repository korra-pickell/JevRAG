from __future__ import annotations

import asyncio
from collections.abc import Sequence

from ..base import Reranker, check_k, rank_hits
from ..jev_answers import choice_probabilities, get_answer, noul_probability, score_fraction
from ..jev_client import JevClient, JevError
from ..text import estimate_tokens, passage
from ..types import Hit

DEFAULT_LEVELS = ("irrelevant", "same topic, doesn't help", "partially answers",
                  "mostly answers", "fully answers")
STRATEGIES = ("score", "noul", "choice")
DEFAULT_INSTRUCTIONS = {
    "score": "How well does this passage answer the query?",
    "noul": "Does this passage help answer the query?",
    "choice": "Which passage best answers the query?",
}


def query_state(query: str) -> str:
    return f"Query: {query}"


def _minmax(values: Sequence[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class JevReranker(Reranker):
    """Rerank candidates with typed Jev questions (score, noul or listwise choice)."""

    def __init__(self, jev: JevClient, *, strategy: str = "score", batch_size: int = 12,
                 max_passage_tokens: int = 512, blend: float = 0.0,
                 levels: Sequence[str] = DEFAULT_LEVELS, instructions: str | None = None,
                 on_error: str = "raise", max_choice_options: int = 255,
                 max_choice_tokens: int = 24_000):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if not 2 <= len(levels) <= 10:
            raise ValueError("a score scale needs 2-10 levels")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be within [0, 1]")
        if on_error not in ("raise", "fallback"):
            raise ValueError("on_error must be 'raise' or 'fallback'")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 2 <= max_choice_options <= 255:
            raise ValueError("max_choice_options must be within 2-255")
        self.jev = jev
        self.strategy = strategy
        self.batch_size = batch_size
        self.max_passage_tokens = max_passage_tokens
        self.blend = blend
        self.levels = list(levels)
        self.instructions = instructions or DEFAULT_INSTRUCTIONS[strategy]
        self.on_error = on_error
        self.max_choice_options = max_choice_options
        self.max_choice_tokens = max_choice_tokens

    async def arerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        check_k(k)
        if not hits:
            return []
        passages = [passage(h.doc, self.max_passage_tokens) for h in hits]
        if self.strategy == "choice":
            scores = await self._choice_scores(query, passages)
        else:
            scores = await self._pointwise_scores(query, passages)
        retrieval = _minmax([h.score for h in hits])
        failed = [s is None for s in scores]
        final = [retrieval[i] if failed[i]
                 else (1 - self.blend) * scores[i] + self.blend * retrieval[i]
                 for i in range(len(hits))]
        return rank_hits(hits, final, stage="jev", k=k, degraded=failed)

    def _question(self, text: str) -> dict:
        q = {"type": self.strategy, "instructions": f"{self.instructions}\n\nPassage:\n{text}"}
        if self.strategy == "score":
            q["criteria"] = self.levels
        return q

    def _parse(self, answer) -> float:
        if self.strategy == "score":
            return score_fraction(answer, self.levels)
        return noul_probability(answer)

    async def _guarded(self, coro) -> dict[int, float] | None:
        try:
            return await coro
        except JevError:
            if self.on_error == "raise":
                raise
            return None

    async def _pointwise_scores(self, query: str, passages: list[str]) -> list[float | None]:
        n = len(passages)
        size = self.batch_size
        batches = [list(range(i, min(i + size, n))) for i in range(0, n, size)]

        async def run(batch: list[int]) -> dict[int, float]:
            answers = await self.jev.ask(query_state(query),
                                         {f"c{j}": self._question(passages[j]) for j in batch})
            return {j: self._parse(get_answer(answers, f"c{j}")) for j in batch}

        scores: list[float | None] = [None] * n
        for result in await asyncio.gather(*(self._guarded(run(b)) for b in batches)):
            if result:
                for j, s in result.items():
                    scores[j] = s
        return scores

    def _choice_chunks(self, passages: list[str]) -> list[list[int]]:
        chunks: list[list[int]] = [[]]
        tokens = 0
        for j, p in enumerate(passages):
            t = estimate_tokens(p)
            full = len(chunks[-1]) >= self.max_choice_options
            if chunks[-1] and (full or tokens + t > self.max_choice_tokens):
                chunks.append([])
                tokens = 0
            chunks[-1].append(j)
            tokens += t
        return chunks

    async def _choice_scores(self, query: str, passages: list[str]) -> list[float | None]:
        chunks = self._choice_chunks(passages)

        async def run(chunk: list[int]) -> dict[int, float]:
            if len(chunk) == 1:
                return {chunk[0]: 1.0}
            criteria = {f"p{j}": passages[j] for j in chunk}
            answers = await self.jev.ask(query_state(query), {"pick": {
                "type": "choice", "instructions": self.instructions, "criteria": criteria}})
            probs = choice_probabilities(get_answer(answers, "pick"))
            top = max(probs.values(), default=0.0) if len(chunks) > 1 else 1.0
            top = top if top > 0 else 1.0
            return {j: probs.get(f"p{j}", 0.0) / top for j in chunk}

        scores: list[float | None] = [None] * len(passages)
        for result in await asyncio.gather(*(self._guarded(run(c)) for c in chunks)):
            if result:
                for j, s in result.items():
                    scores[j] = s
        return scores
