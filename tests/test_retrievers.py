import numpy as np
import pytest

from jevragrank.retrievers import BGE_QUERY_PREFIX, BM25Retriever, DenseRetriever
from jevragrank.types import Doc

DOCS = [
    Doc("a", "cats purr loudly at night", "Cats"),
    Doc("b", "dogs bark at the mailman", "Dogs"),
    Doc("c", "vitamin d supplementation and bone fractures in adults", "Vitamin D"),
]
VOCAB = ["cats", "dogs", "vitamin", "fractures", "bark", "purr"]


def bow_encoder(texts):
    vecs = np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=np.float32)
    return vecs + 1e-3


def test_bm25_ranks_matching_doc_first():
    hits = BM25Retriever(DOCS).search("vitamin fracture risk", k=2)
    assert hits[0].doc.id == "c"
    assert hits[0].stage_scores["bm25"] == hits[0].score > 0


def test_bm25_drops_non_matching_docs():
    assert BM25Retriever(DOCS).search("zzzz qqqq", k=3) == []


def test_bm25_clamps_k_to_corpus():
    assert len(BM25Retriever(DOCS).search("cats dogs vitamin", k=50)) <= 3


def test_bm25_records_timing():
    import asyncio
    out = asyncio.run(BM25Retriever(DOCS).asearch_traced("dogs", 1))
    assert "retrieve" in out.timings and out.candidates is None


def test_dense_uses_encoder_and_prefix():
    seen = []

    def enc(texts):
        seen.extend(texts)
        return bow_encoder(texts)

    r = DenseRetriever(DOCS, encoder=enc)
    hits = r.search("dogs bark", k=1)
    assert hits[0].doc.id == "b"
    assert seen[-1] == BGE_QUERY_PREFIX + "dogs bark"
    assert r.embeddings.shape == (3, len(VOCAB))


def test_dense_scores_are_cosine():
    hits = DenseRetriever(DOCS, encoder=bow_encoder).search("cats purr", k=3)
    assert -1.0 <= hits[-1].score <= hits[0].score <= 1.0 + 1e-6


def test_dense_embedding_cache(tmp_path):
    calls = []

    def enc(texts):
        calls.append(len(texts))
        return bow_encoder(texts)

    DenseRetriever(DOCS, encoder=enc, cache_dir=tmp_path)
    r2 = DenseRetriever(DOCS, encoder=enc, cache_dir=tmp_path)
    assert calls == [3] and r2.from_cache


@pytest.mark.parametrize("cls", [BM25Retriever, DenseRetriever])
def test_empty_corpus_rejected(cls):
    with pytest.raises(ValueError):
        cls([], encoder=bow_encoder) if cls is DenseRetriever else cls([])
