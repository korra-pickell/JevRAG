import json

from bench.plots import render_all


def run(system, label="", ndcg=0.5, p50=0.1, candidates=50, corpus=5183, **extra):
    return {"system": system, "label": label, "options": {"candidates": candidates},
            "corpus_size": corpus, "note": "", "n_queries": 300,
            "metrics": {m: {"mean": ndcg, "ci": [ndcg - 0.02, ndcg + 0.02]}
                        for m in ("ndcg@10", "recall@10", "mrr@10")},
            "latency": {"p50": p50, "p95": p50 * 2, "mean": p50}, "p_vs_ce": 0.01,
            "candidate_recall": 0.9, "qps": 5.0, "jev_requests": 5, **extra}


def summary():
    runs = {"bm25": run("bm25", ndcg=0.66, p50=0.004), "dense": run("dense", ndcg=0.74, p50=0.02),
            "dense+ce": run("dense+ce", ndcg=0.76, p50=0.3),
            "dense+llm": run("dense+llm", ndcg=0.70, p50=2.0),
            "jevragrank": run("jevragrank", ndcg=0.78, p50=0.9),
            "jevrank": run("jevrank", ndcg=0.60, p50=25.0),
            "jevragrank@noul": run("jevragrank", "noul", 0.75, 0.8),
            "jevragrank@choice": run("jevragrank", "choice", 0.72, 0.4),
            "jevragrank@0.8b": run("jevragrank", "0.8b", 0.72, 0.5),
            "jevrank@0.8b": run("jevrank", "0.8b", 0.5, 15.0)}
    for k in (10, 20, 100):
        runs[f"jevragrank@k{k}"] = run("jevragrank", f"k{k}", 0.7 + k / 1000, k / 50, k)
        runs[f"dense+ce@k{k}"] = run("dense+ce", f"k{k}", 0.7, k / 150, k)
    for n in (500, 1000, 2000, 5183):
        for s, base in (("dense", 0.01), ("jevragrank", 0.8), ("jevrank", 5.0)):
            runs[f"{s}@n{n}"] = run(s, f"n{n}", 0.7, base * (n / 500 if s == "jevrank" else 1),
                                    corpus=n)
    return {"datasets": {"scifact": {"split": "test", "runs": runs},
                         "nfcorpus": {"split": "test", "runs": dict(runs)}}}


def test_render_all_writes_every_chart(tmp_path):
    paths = render_all(summary(), tmp_path)
    names = {p.name for p in paths}
    for chart in ("speed-accuracy", "accuracy", "scaling", "depth", "ablations"):
        for theme in ("light", "dark"):
            assert f"{chart}-{theme}.png" in names and f"{chart}-{theme}.svg" in names
    assert all(p.stat().st_size > 1000 for p in paths)


def test_render_all_tolerates_missing_ablations(tmp_path):
    s = json.loads(json.dumps(summary()))
    for ds in s["datasets"].values():
        ds["runs"] = {k: v for k, v in ds["runs"].items() if "@" not in k}
    assert render_all(s, tmp_path)
