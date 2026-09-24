from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ranx import Qrels, Run, evaluate

METRICS = ["ndcg@10", "recall@10", "mrr@10"]


def per_query(qrels: dict[str, dict[str, int]], run: dict[str, list[str]],
              metrics: Sequence[str] = METRICS) -> dict[str, dict[str, float]]:
    """Per-query metric values; ranking order is preserved via rank-derived scores."""
    scored = {q: {d: float(len(ids) - r) for r, d in enumerate(ids)}
              for q, ids in run.items() if q in qrels and ids}
    out: dict[str, dict[str, float]] = {m: {q: 0.0 for q in qrels} for m in metrics}
    if scored:
        sub_qrels = Qrels({q: qrels[q] for q in scored})
        r = Run(scored)
        evaluate(sub_qrels, r, list(metrics))
        for m in metrics:
            for q, v in r.scores[m].items():
                out[m][q] = float(v)
    return out


def bootstrap_ci(values: Sequence[float], n: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def randomization_test(a: Sequence[float], b: Sequence[float], n: int = 10000,
                       seed: int = 0) -> float:
    """Two-sided paired Fisher randomization test on the mean difference."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    observed = abs(d.mean())
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    return float((np.sum(perm >= observed - 1e-12) + 1) / (n + 1))


def candidate_recall(qrels: dict[str, dict[str, int]],
                     candidates: dict[str, list[str]]) -> float:
    values = []
    for q, cands in candidates.items():
        rel = {d for d, s in qrels.get(q, {}).items() if s > 0}
        if rel:
            values.append(len(rel & set(cands)) / len(rel))
    return float(np.mean(values)) if values else float("nan")


def latency_stats(seconds: Sequence[float]) -> dict[str, float]:
    v = np.asarray(seconds, dtype=np.float64)
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "mean": float(v.mean())}
