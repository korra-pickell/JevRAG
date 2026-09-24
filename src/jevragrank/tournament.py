from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from time import perf_counter

from .base import Reranker, Searchable, check_k
from .jev_answers import choice_probabilities, get_answer
from .jev_client import JevClient
from .rerankers.jev import JevReranker, query_state
from .text import truncate
from .types import Doc, Hit, SearchResult


class JevRank(Searchable):
    """Embedding-free retrieval: batched Jev `choice` rounds, then a Jev `score` final."""

    def __init__(self, docs: Sequence[Doc], *, jev: JevClient, group_size: int = 64,
                 preview_tokens: Sequence[int] = (96, 256), survivors: Sequence[int] = (3, 12),
                 final_pool: int = 50, final_reranker: Reranker | None = None,
                 instructions: str = "Which passage most likely answers the query?",
                 seed: int = 0):
        if not docs:
            raise ValueError("docs must not be empty")
        if not 2 <= group_size <= 255:
            raise ValueError("group_size must be within 2-255")
        if not survivors or len(survivors) != len(preview_tokens):
            raise ValueError("survivors and preview_tokens need one entry per round")
        if min(survivors) < 1 or final_pool < 1:
            raise ValueError("survivors and final_pool must be positive")
        self.docs = list(docs)
        self.jev = jev
        self.group_size = group_size
        self.survivors = list(survivors)
        self.final_pool = final_pool
        self.instructions = instructions
        self.seed = seed
        self.final_reranker = final_reranker or JevReranker(jev, strategy="score")
        self.previews = [[truncate(d.full_text, t) for d in self.docs] for t in preview_tokens]

    async def _judge(self, query: str, group: list[int], rnd: int) -> dict[int, float]:
        if len(group) == 1:
            return {group[0]: 1.0}
        criteria = {f"p{n}": self.previews[rnd][i] for n, i in enumerate(group)}
        answers = await self.jev.ask(query_state(query), {"pick": {
            "type": "choice", "instructions": self.instructions, "criteria": criteria}})
        probs = choice_probabilities(get_answer(answers, "pick"))
        return {i: probs.get(f"p{n}", 0.0) for n, i in enumerate(group)}

    async def asearch_traced(self, query: str, k: int) -> SearchResult:
        check_k(k)
        timings: dict[str, float] = {}
        pool = list(range(len(self.docs)))
        last_prob = {i: 1.0 for i in pool}
        eliminated: list[tuple[int, float, int]] = []  # (round reached, probability, doc index)
        candidates: list[str] | None = None
        rounds = len(self.survivors)
        for rnd, keep in enumerate(self.survivors):
            if len(pool) <= self.final_pool:
                break
            t0 = perf_counter()
            order = pool[:]
            random.Random(f"{self.seed}:{rnd}:{query}").shuffle(order)
            groups = [order[i:i + self.group_size] for i in range(0, len(order), self.group_size)]
            results = await asyncio.gather(*(self._judge(query, g, rnd) for g in groups))
            pool = []
            for group, probs in zip(groups, results, strict=True):
                ranked = sorted(group, key=lambda i: (-probs[i], i))
                pool.extend(ranked[:keep])
                eliminated.extend((rnd, probs[i], i) for i in ranked[keep:])
                last_prob.update(probs)
            timings[f"round{rnd + 1}"] = perf_counter() - t0
            if candidates is None:
                candidates = [self.docs[i].id for i in pool]
        if len(pool) > self.final_pool:
            pool.sort(key=lambda i: (-last_prob[i], i))
            eliminated.extend((rounds, last_prob[i], i) for i in pool[self.final_pool:])
            pool = pool[: self.final_pool]
        t0 = perf_counter()
        finalists = [Hit(self.docs[i], last_prob[i], {"tournament": last_prob[i]}) for i in pool]
        hits = await self.final_reranker.arerank(query, finalists, len(finalists))
        timings["final"] = perf_counter() - t0
        hits = hits[:k]
        if len(hits) < k:
            eliminated.sort(key=lambda e: (-e[0], -e[1], e[2]))
            for rnd, p, i in eliminated[: k - len(hits)]:
                score = (rnd - rounds) - 1 + 0.5 * p
                hits.append(Hit(self.docs[i], score, {"tournament": p, "round_reached": rnd}))
        if candidates is None:
            candidates = [d.id for d in self.docs]
        return SearchResult(hits, candidates=candidates, timings=timings)
