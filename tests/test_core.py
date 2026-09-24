import asyncio

import pytest

from jevragrank.base import Reranker, Searchable, check_k, rank_hits
from jevragrank.text import CHARS_PER_TOKEN, estimate_tokens, passage, truncate
from jevragrank.types import Doc, Hit, SearchResult, corpus_hash


def test_full_text_joins_title_and_text():
    assert Doc("1", "body", "Title").full_text == "Title\nbody"
    assert Doc("1", "body").full_text == "body"


def test_corpus_hash_changes_with_content():
    a = [Doc("1", "x"), Doc("2", "y")]
    assert corpus_hash(a) == corpus_hash(list(a))
    assert corpus_hash(a) != corpus_hash([Doc("1", "x"), Doc("2", "z")])


def test_truncate_keeps_short_text():
    assert truncate("short text", 10) == "short text"


def test_truncate_cuts_at_word_boundary_and_marks():
    text = "word " * 100
    out = truncate(text, 10)
    assert len(out) <= 10 * CHARS_PER_TOKEN + 2
    assert out.endswith(" …")
    assert "wor …" not in out


def test_truncate_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        truncate("x", 0)


def test_estimate_tokens_rounds_up():
    assert estimate_tokens("abcde") == 2


def test_passage_uses_full_text():
    assert passage(Doc("1", "body", "T"), 100) == "T\nbody"


def test_check_k():
    check_k(1)
    with pytest.raises(ValueError):
        check_k(0)


def _hits():
    return [Hit(Doc(str(i), f"d{i}"), score=s, stage_scores={"dense": s})
            for i, s in enumerate([0.9, 0.8, 0.7])]


def test_rank_hits_orders_by_score_then_retrieval_score():
    ranked = rank_hits(_hits(), [0.5, 0.9, 0.5], stage="jev", k=3)
    assert [h.doc.id for h in ranked] == ["1", "0", "2"]
    assert ranked[0].score == 0.9
    assert ranked[0].stage_scores == {"dense": 0.8, "jev": 0.9}


def test_rank_hits_truncates_and_marks_degraded():
    ranked = rank_hits(_hits(), [0.1, 0.2, 0.3], stage="jev", k=2,
                       degraded=[False, False, True])
    assert [h.doc.id for h in ranked] == ["2", "1"]
    assert ranked[0].degraded and not ranked[1].degraded


class Echo(Searchable):
    async def asearch_traced(self, query, k):
        return SearchResult([Hit(Doc("1", query), 1.0)][:k])


def test_search_works_without_and_inside_running_loop():
    assert Echo().search("hello", 1)[0].doc.text == "hello"

    async def inner():
        return Echo().search("inside", 1)

    assert asyncio.run(inner())[0].doc.text == "inside"


class Reverse(Reranker):
    async def arerank(self, query, hits, k):
        return list(reversed(hits))[:k]


def test_reranker_sync_wrapper():
    assert [h.doc.id for h in Reverse().rerank("q", _hits(), 2)] == ["2", "1"]
