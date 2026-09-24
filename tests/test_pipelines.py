import numpy as np
import pytest

from jevragrank.pipelines import JevRAGRank, RetrieveThenRerank
from jevragrank.retrievers import DenseRetriever
from jevragrank.types import Doc

DOCS = [Doc(str(i), t) for i, t in enumerate(
    ["dogs bark loudly", "vitamin d reduces fractures", "cats purr", "vitamin pills",
     "bone fractures heal", "weather is sunny"])]
VOCAB = ["dogs", "vitamin", "fractures", "cats", "bone", "weather"]


def enc(texts):
    return np.array([[t.count(w) for w in VOCAB] for t in texts], dtype=np.float32) + 1e-3


def dense():
    return DenseRetriever(DOCS, encoder=enc)


def test_jevragrank_end_to_end(jev_client, fake_jev):
    rag = JevRAGRank(DOCS, jev=jev_client, candidates=4, retriever=dense())
    import asyncio
    res = asyncio.run(rag.asearch_traced("vitamin d fractures", 2))
    assert res.hits[0].doc.id == "1"
    assert len(res.candidates) == 4
    assert set(res.timings) == {"retrieve", "rerank"}
    total_questions = sum(len(b["questions"]) for b in fake_jev.bodies)
    assert total_questions == 4


def test_candidates_must_cover_k(jev_client):
    rag = JevRAGRank(DOCS, jev=jev_client, candidates=2, retriever=dense())
    with pytest.raises(ValueError, match="candidates"):
        rag.search("vitamin", k=3)


def test_retrieve_then_rerank_rejects_bad_candidates():
    with pytest.raises(ValueError):
        RetrieveThenRerank(dense(), None, candidates=0)
