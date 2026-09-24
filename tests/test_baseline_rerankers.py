import math

import pytest

from jevragrank.rerankers import (
    RATING_PROMPT,
    CrossEncoderReranker,
    LLMReranker,
    digit_plan,
    expected_rating,
)
from jevragrank.types import Doc, Hit

HITS = [Hit(Doc(str(i), t), 1.0 - i / 10) for i, t in
        enumerate(["dogs bark", "vitamin d fractures", "cats purr"])]


def overlap(query, text):
    return len(set(query.split()) & set(text.split()))


def test_cross_encoder_ranks_by_scorer():
    ce = CrossEncoderReranker(scorer=lambda pairs: [overlap(q, p) for q, p in pairs])
    out = ce.rerank("vitamin d", HITS, k=2)
    assert [h.doc.id for h in out] == ["1", "0"]
    assert out[0].stage_scores["ce"] == 2


def test_cross_encoder_empty():
    assert CrossEncoderReranker(scorer=lambda p: []).rerank("q", [], 3) == []


def test_llm_reranker_builds_prompts_and_ranks():
    seen = []

    def scorer(prompts):
        seen.extend(prompts)
        return [overlap("vitamin d", p.split("Passage:")[1]) / 2 for p in prompts]

    out = LLMReranker(scorer=scorer).rerank("vitamin d", HITS, k=1)
    assert out[0].doc.id == "1"
    assert seen[0] == RATING_PROMPT.format(query="vitamin d", passage="dogs bark")
    assert seen[0].endswith("Rating:")


class SplitTok:
    """Tokenizer that splits on spaces but keeps ' ' as its own token before digits."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, piece):
        return self.vocab.setdefault(piece, len(self.vocab))

    def encode(self, text, add_special_tokens=False):
        return [self._id(p) for p in text.replace(" ", "\0 \0").split("\0") if p]


class MergeTok(SplitTok):
    """Tokenizer that merges the space into the digit token."""

    def encode(self, text, add_special_tokens=False):
        return [self._id(p) for p in text.replace(" ", "\0 ").split("\0") if p]


def test_digit_plan_with_separate_space_token():
    tok = SplitTok()
    suffix, digits = digit_plan(tok)
    assert suffix == tok.encode(" ")
    assert digits == [tok.encode(str(d))[0] for d in range(5)]


def test_digit_plan_with_merged_space_digit():
    tok = MergeTok()
    suffix, digits = digit_plan(tok)
    assert suffix == []
    assert digits == [tok.encode(f" {d}")[0] for d in range(5)]
    assert len(set(digits)) == 5


def test_expected_rating():
    assert expected_rating([0, 0, 0, 0, 50]) == pytest.approx(1.0)
    assert expected_rating([50, 0, 0, 0, 0]) == pytest.approx(0.0)
    assert expected_rating([1, 1, 1, 1, 1]) == pytest.approx(0.5)
    assert not math.isnan(expected_rating([-1e4] * 5))
