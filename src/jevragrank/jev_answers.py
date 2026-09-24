from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .jev_client import JevResponseError


def get_answer(answers: dict, key: str) -> Any:
    try:
        return answers[key]
    except KeyError:
        raise JevResponseError(f"answer for question {key!r} missing from response") from None


def noul_probability(answer: Any) -> float:
    if isinstance(answer, (int, float)):
        return float(answer)
    if isinstance(answer, dict) and "noul" in answer:
        return float(answer["noul"])
    raise JevResponseError(f"can't read a noul probability from {str(answer)[:200]}")


def score_fraction(answer: Any, levels: Sequence[str]) -> float:
    """Expected level of a score answer, normalized to 0-1."""
    n = len(levels)
    probs = answer.get("probabilities") if isinstance(answer, dict) else None
    vec = None
    if isinstance(probs, list) and len(probs) == n:
        vec = [float(p) for p in probs]
    elif isinstance(probs, dict) and all(lvl in probs for lvl in levels):
        vec = [float(probs[lvl]) for lvl in levels]
    elif isinstance(probs, dict) and all(str(i) in probs for i in range(n)):
        vec = [float(probs[str(i)]) for i in range(n)]
    if vec is not None and sum(vec) > 0:
        expected = sum(i * p for i, p in enumerate(vec)) / sum(vec)
    elif isinstance(answer, dict) and "score" in answer:
        expected = float(answer["score"])
    else:
        raise JevResponseError(f"can't read a score from {str(answer)[:200]}")
    return min(max(expected / (n - 1), 0.0), 1.0)


def choice_probabilities(answer: Any) -> dict[str, float]:
    probs = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(probs, dict):
        raise JevResponseError(f"can't read choice probabilities from {str(answer)[:200]}")
    return {str(k): float(v) for k, v in probs.items()}
