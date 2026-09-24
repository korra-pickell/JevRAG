import json
from pathlib import Path

import pytest

from jevragrank.jev_answers import (
    choice_probabilities,
    get_answer,
    noul_probability,
    score_fraction,
)
from jevragrank.jev_client import JevResponseError

LEVELS = ["irrelevant", "same topic, doesn't help", "partially answers", "mostly answers",
          "fully answers"]
FIX = Path(__file__).parent / "fixtures" / "decider"


def test_noul_variants():
    assert noul_probability({"type": "noul", "noul": 0.25}) == 0.25
    assert noul_probability(0.5) == 0.5
    with pytest.raises(JevResponseError):
        noul_probability({"type": "noul"})


def test_score_prefers_full_precision_probabilities():
    ans = {"score": 2.0, "probabilities": [0.0, 0.0, 0.5, 0.5, 0.0]}
    assert score_fraction(ans, LEVELS) == pytest.approx(2.5 / 4)


def test_score_probabilities_keyed_by_label():
    probs = {lvl: 0.0 for lvl in LEVELS}
    probs["fully answers"] = 1.0
    assert score_fraction({"probabilities": probs}, LEVELS) == 1.0


def test_score_falls_back_to_expected_level():
    assert score_fraction({"score": 3.0}, LEVELS) == 0.75


def test_score_is_clamped_and_validated():
    assert score_fraction({"score": 9.0}, LEVELS) == 1.0
    with pytest.raises(JevResponseError):
        score_fraction({"confidence": 0.3}, LEVELS)


def test_choice_probabilities():
    assert choice_probabilities({"probabilities": {"p0": 0.7, "p1": 0.3}}) == {"p0": 0.7,
                                                                                "p1": 0.3}
    with pytest.raises(JevResponseError):
        choice_probabilities({"choice": "p0"})


def test_get_answer_missing_key():
    with pytest.raises(JevResponseError, match="c3"):
        get_answer({"c1": {}}, "c3")


@pytest.mark.skipif(not FIX.exists(), reason="Task 0 fixtures not captured")
def test_parsers_accept_captured_decider_responses():
    for ans in json.loads((FIX / "score.json").read_text("utf-8"))["answers"].values():
        assert 0.0 <= score_fraction(ans, LEVELS) <= 1.0
    for ans in json.loads((FIX / "noul.json").read_text("utf-8"))["answers"].values():
        assert 0.0 <= noul_probability(ans) <= 1.0
    pick = json.loads((FIX / "choice.json").read_text("utf-8"))["answers"]["pick"]
    probs = choice_probabilities(pick)
    assert len(probs) == 64 and sum(probs.values()) == pytest.approx(1.0, abs=0.05)
