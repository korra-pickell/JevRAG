from __future__ import annotations

import json

import httpx
import pytest

from jevragrank.jev_client import JevClient


def keyword_relevance(query: str, text: str) -> float:
    q = set(query.lower().split())
    t = set(text.lower().replace("\n", " ").split())
    return len(q & t) / max(len(q), 1)


def _passage_of(instructions: str) -> str:
    return instructions.split("Passage:\n", 1)[1]


class FakeJev:
    """In-memory /v1/systemone server; relevance(query, text) -> [0, 1] drives answers."""

    def __init__(self, relevance=keyword_relevance):
        self.relevance = relevance
        self.bodies: list[dict] = []
        self.fail_next: list[int] = []
        self.models_hits = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.models_hits += 1
            return httpx.Response(200, json={"models": [{"name": "fake"}]})
        if self.fail_next:
            return httpx.Response(self.fail_next.pop(0), json={"detail": "boom"})
        body = json.loads(request.content)
        self.bodies.append(body)
        query = body["state"].removeprefix("Query: ")
        answers = {}
        for key, q in body["questions"].items():
            if q["type"] == "score":
                n = len(q["criteria"])
                level = round(self.relevance(query, _passage_of(q["instructions"])) * (n - 1))
                probs = [0.0] * n
                probs[level] = 1.0
                answers[key] = {"type": "score", "score": float(level), "probabilities": probs}
            elif q["type"] == "noul":
                p = self.relevance(query, _passage_of(q["instructions"]))
                answers[key] = {"type": "noul", "noul": p}
            else:
                raw = {o: self.relevance(query, t) + 1e-6 for o, t in q["criteria"].items()}
                total = sum(raw.values())
                probs = {o: v / total for o, v in raw.items()}
                best = max(probs, key=probs.get)
                answers[key] = {"type": "choice", "choice": best, "confidence": probs[best],
                                "probabilities": probs}
        return httpx.Response(200, json={"model": "fake", "answers": answers,
                                         "usage": {"input_tokens": 1, "output_tokens": 0}})


@pytest.fixture
def fake_jev() -> FakeJev:
    return FakeJev()


@pytest.fixture
def jev_client(fake_jev) -> JevClient:
    return JevClient("http://fake", transport=httpx.MockTransport(fake_jev.handler), backoff=0.0)
