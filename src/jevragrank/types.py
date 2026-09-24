from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    title: str = ""

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass
class Hit:
    doc: Doc
    score: float
    stage_scores: dict[str, float] = field(default_factory=dict)
    degraded: bool = False


@dataclass
class SearchResult:
    hits: list[Hit]
    candidates: list[str] | None = None
    timings: dict[str, float] = field(default_factory=dict)


def corpus_hash(docs: Iterable[Doc]) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update(d.id.encode())
        h.update(b"\x1f")
        h.update(d.full_text.encode())
        h.update(b"\x1e")
    return h.hexdigest()[:16]
