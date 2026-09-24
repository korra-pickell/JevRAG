import json

from bench.datasets import Dataset
from bench.summary import markdown_table, summarize
from jevragrank.types import Doc

QRELS = {f"q{i}": {f"d{i}": 1} for i in range(20)}
DS = Dataset("toy", "test", [Doc(f"d{i}", "") for i in range(20)],
             {q: q for q in QRELS}, QRELS)


def write_run(root, name, hit_rate, latency, candidates=True):
    d = root / "toy"
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for i in range(20):
            first = f"d{i}" if i < hit_rate * 20 else "dx"
            f.write(json.dumps({"qid": f"q{i}", "ranking": [[first, 1.0]],
                                "candidates": [f"d{i}"] if candidates else None,
                                "timings": {}, "latency": latency, "jev_requests": 3}) + "\n")
    system, _, label = name.partition("@")
    (d / f"{name}.meta.json").write_text(json.dumps(
        {"system": system, "label": label, "options": {}, "split": "test", "note": "",
         "corpus_size": 20, "index_seconds": 1.0, "env": {"gpu": "fake"}}), "utf-8")


def test_summarize_computes_metrics_ci_and_significance(tmp_path, monkeypatch):
    monkeypatch.setattr("bench.summary.load_beir", lambda name, split, data_dir: DS)
    write_run(tmp_path, "dense+ce", 0.5, 0.2)
    write_run(tmp_path, "jevragrank", 1.0, 0.8)
    s = summarize(tmp_path, tmp_path / "data")
    runs = s["datasets"]["toy"]["runs"]
    jr = runs["jevragrank"]
    assert jr["metrics"]["ndcg@10"]["mean"] == 1.0
    assert jr["metrics"]["ndcg@10"]["ci"][0] <= 1.0
    assert jr["p_vs_ce"] < 0.05 and jr["latency"]["p50"] == 0.8
    assert jr["candidate_recall"] == 1.0 and jr["jev_requests"] == 3
    assert runs["dense+ce"]["p_vs_ce"] is None
    assert (tmp_path / "summary.json").exists()
    table = markdown_table(s, "toy")
    assert "| jevragrank |" in table and "1.000" in table
