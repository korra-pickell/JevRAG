import asyncio

import pytest

from jevragrank.tournament import JevRank
from jevragrank.types import Doc

RELEVANT = {"r1": "vitamin d fractures trial", "r2": "vitamin d bone fractures"}


def corpus(n_noise: int) -> list[Doc]:
    docs = [Doc(f"n{i}", f"noise document number {i} about weather") for i in range(n_noise)]
    docs[7:7] = [Doc(k, v) for k, v in RELEVANT.items()]
    return docs


def choice_bodies(fake):
    return [b for b in fake.bodies if "pick" in b["questions"]]


def test_tournament_finds_relevant_docs(jev_client, fake_jev):
    jr = JevRank(corpus(200), jev=jev_client, group_size=16, survivors=(2, 4),
                 preview_tokens=(20, 40), final_pool=10)
    res = asyncio.run(jr.asearch_traced("vitamin d fractures", 2))
    assert {h.doc.id for h in res.hits} == {"r1", "r2"}
    assert {"r1", "r2"} <= set(res.candidates)
    assert set(res.timings) == {"round1", "round2", "final"}


def test_request_counts_follow_group_math(jev_client, fake_jev):
    docs = corpus(126)  # 128 docs
    JevRank(docs, jev=jev_client, group_size=16, survivors=(2, 4), preview_tokens=(20, 40),
            final_pool=10).search("vitamin d fractures", 5)
    choices = choice_bodies(fake_jev)
    # round 1: 128/16 = 8 groups -> 16 survivors; round 2: 1 group of 16 -> 4 survivors
    assert [len(b["questions"]["pick"]["criteria"]) for b in choices] == [16] * 8 + [16]
    score_qs = sum(len(b["questions"]) for b in fake_jev.bodies if "pick" not in b["questions"])
    assert score_qs == 4


def test_previews_are_truncated_per_round(jev_client, fake_jev):
    long_docs = [Doc(str(i), "word " * 400) for i in range(40)]
    JevRank(long_docs, jev=jev_client, group_size=10, survivors=(2, 2), preview_tokens=(10, 30),
            final_pool=3).search("word", 1)
    r1 = choice_bodies(fake_jev)[0]["questions"]["pick"]["criteria"]
    assert max(len(t) for t in r1.values()) <= 10 * 4 + 2


def test_small_corpus_skips_rounds(jev_client, fake_jev):
    docs = corpus(3)
    res = asyncio.run(JevRank(docs, jev=jev_client, final_pool=50).asearch_traced("vitamin", 2))
    assert choice_bodies(fake_jev) == []
    assert res.candidates == [d.id for d in docs]
    assert set(res.timings) == {"final"}


def test_eliminated_docs_fill_ranking_below_finalists(jev_client):
    jr = JevRank(corpus(60), jev=jev_client, group_size=16, survivors=(1,), preview_tokens=(20,),
                 final_pool=4)
    hits = jr.search("vitamin d fractures", 10)
    assert len(hits) == 10
    finalists = [h for h in hits if "round_reached" not in h.stage_scores]
    assert len(finalists) == 4
    assert all(h.score < 0 for h in hits[4:])
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_shuffle_is_deterministic_per_query(jev_client, fake_jev):
    jr = JevRank(corpus(60), jev=jev_client, group_size=16, survivors=(2,), preview_tokens=(20,),
                 final_pool=10)
    jr.search("vitamin", 3)
    first = [b["questions"]["pick"]["criteria"] for b in choice_bodies(fake_jev)]
    fake_jev.bodies.clear()
    jr.search("vitamin", 3)
    assert [b["questions"]["pick"]["criteria"] for b in choice_bodies(fake_jev)] == first


@pytest.mark.parametrize("kwargs", [{"group_size": 1}, {"group_size": 256},
                                    {"survivors": (3,), "preview_tokens": (10, 20)},
                                    {"survivors": (0,), "preview_tokens": (10,)},
                                    {"final_pool": 0}])
def test_invalid_config(jev_client, kwargs):
    with pytest.raises(ValueError):
        JevRank(corpus(5), jev=jev_client, **kwargs)
