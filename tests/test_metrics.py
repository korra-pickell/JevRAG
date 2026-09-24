import numpy as np
import pytest

from bench.metrics import (
    bootstrap_ci,
    candidate_recall,
    latency_stats,
    per_query,
    randomization_test,
)

QRELS = {"q1": {"d1": 1, "d3": 2}, "q2": {"d9": 1}, "q3": {"d4": 1}}


def test_per_query_matches_trec_eval_linear_gain():
    run = {"q1": ["d1", "d2", "d3"], "q2": ["d5", "d9"]}
    pq = per_query(QRELS, run)
    assert pq["ndcg@10"]["q1"] == pytest.approx(2 / (2 + 1 / np.log2(3)))
    assert pq["mrr@10"]["q2"] == pytest.approx(0.5)
    assert pq["recall@10"]["q1"] == 1.0
    assert pq["ndcg@10"]["q3"] == 0.0


def test_bootstrap_ci_brackets_mean():
    v = np.random.default_rng(0).normal(0.5, 0.1, 300)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi and hi - lo < 0.05


def test_randomization_test():
    rng = np.random.default_rng(1)
    a = rng.normal(0.5, 0.1, 200)
    assert randomization_test(a, a + 0.05) < 0.01
    assert randomization_test(a, a + rng.normal(0, 0.001, 200)) > 0.05


def test_candidate_recall_counts_relevant_survivors():
    cands = {"q1": ["d1", "d7"], "q2": ["d9"]}
    assert candidate_recall(QRELS, cands) == pytest.approx((0.5 + 1.0) / 2)


def test_latency_stats():
    s = latency_stats([0.1] * 19 + [1.0])
    assert s["p50"] == pytest.approx(0.1) and s["p95"] >= 0.1 and s["mean"] > 0.1
