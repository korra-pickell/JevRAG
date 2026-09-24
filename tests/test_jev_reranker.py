import httpx
import pytest

from jevragrank.jev_client import JevClient, JevRequestError
from jevragrank.rerankers import DEFAULT_LEVELS, JevReranker
from jevragrank.types import Doc, Hit

TEXTS = ["dogs bark loudly", "vitamin d reduces fractures", "cats purr", "vitamin pills",
         "bone fractures heal"]
HITS = [Hit(Doc(str(i), t), score=1.0 - i / 10, stage_scores={"dense": 1.0 - i / 10})
        for i, t in enumerate(TEXTS)]
QUERY = "vitamin d fractures"


def test_score_strategy_batches_and_ranks(jev_client, fake_jev):
    out = JevReranker(jev_client, batch_size=2).rerank(QUERY, HITS, k=3)
    assert out[0].doc.id == "1"
    assert len(fake_jev.bodies) == 3
    first = fake_jev.bodies[0]
    assert first["state"] == f"Query: {QUERY}"
    assert list(first["questions"]) == ["c0", "c1"]
    q = first["questions"]["c0"]
    assert q["type"] == "score" and q["criteria"] == list(DEFAULT_LEVELS)
    assert q["instructions"].endswith("Passage:\ndogs bark loudly")
    assert out[0].stage_scores == {"dense": 0.9, "jev": 1.0}


def test_ties_broken_by_retrieval_score(jev_client):
    out = JevReranker(jev_client, strategy="noul").rerank("zzz", HITS, k=5)
    assert [h.doc.id for h in out] == ["0", "1", "2", "3", "4"]


def test_noul_strategy(jev_client, fake_jev):
    out = JevReranker(jev_client, strategy="noul").rerank(QUERY, HITS, k=1)
    assert out[0].doc.id == "1"
    assert fake_jev.bodies[0]["questions"]["c0"]["type"] == "noul"


def test_choice_strategy_single_listwise_question(jev_client, fake_jev):
    out = JevReranker(jev_client, strategy="choice").rerank(QUERY, HITS, k=2)
    assert out[0].doc.id == "1"
    assert len(fake_jev.bodies) == 1
    pick = fake_jev.bodies[0]["questions"]["pick"]
    assert pick["type"] == "choice" and set(pick["criteria"]) == {f"p{i}" for i in range(5)}


def test_choice_splits_when_over_option_limit(jev_client, fake_jev):
    JevReranker(jev_client, strategy="choice", max_choice_options=2).rerank(QUERY, HITS, k=5)
    assert [len(b["questions"]["pick"]["criteria"]) for b in fake_jev.bodies] == [2, 2]


def test_choice_splits_on_token_budget(jev_client, fake_jev):
    JevReranker(jev_client, strategy="choice", max_choice_tokens=10).rerank(QUERY, HITS, k=5)
    assert len(fake_jev.bodies) >= 2


def test_blend_mixes_in_retrieval_score(jev_client):
    out = JevReranker(jev_client, strategy="noul", blend=1.0).rerank(QUERY, HITS, k=5)
    assert [h.doc.id for h in out] == ["0", "1", "2", "3", "4"]


def test_error_raises_by_default(fake_jev):
    fake_jev.fail_next = [422]
    c = JevClient("http://fake", transport=httpx.MockTransport(fake_jev.handler), backoff=0)
    with pytest.raises(JevRequestError):
        JevReranker(c, batch_size=5).rerank(QUERY, HITS, k=3)


def test_fallback_keeps_retrieval_order_and_flags(fake_jev):
    fake_jev.fail_next = [422]
    c = JevClient("http://fake", transport=httpx.MockTransport(fake_jev.handler), backoff=0)
    out = JevReranker(c, batch_size=5, on_error="fallback").rerank(QUERY, HITS, k=5)
    assert [h.doc.id for h in out] == ["0", "1", "2", "3", "4"]
    assert all(h.degraded for h in out)


@pytest.mark.parametrize("kwargs", [{"strategy": "vote"}, {"levels": ("one",)}, {"blend": 2},
                                    {"on_error": "ignore"}, {"batch_size": 0},
                                    {"max_choice_options": 300}])
def test_invalid_config(jev_client, kwargs):
    with pytest.raises(ValueError):
        JevReranker(jev_client, **kwargs)
