import json

from bench.report import render_report, write_report

SUMMARY = {"generated": "2026-09-30T00:00:00+00:00", "env": {"gpu": "RTX 4070 Laptop"},
           "datasets": {"scifact": {"split": "test", "runs": {"dense": {
               "system": "dense", "label": "", "options": {}, "corpus_size": 5183, "note": "",
               "n_queries": 300,
               "metrics": {m: {"mean": 0.74, "ci": [0.7, 0.78]}
                           for m in ("ndcg@10", "recall@10", "mrr@10")},
               "latency": {"p50": 0.02, "p95": 0.03, "mean": 0.02}, "p_vs_ce": None,
               "candidate_recall": None, "qps": 50.0, "jev_requests": 0}}}}}


def test_standalone_page_embeds_data_and_title():
    html = render_report(SUMMARY, standalone=True)
    assert html.startswith("<!doctype html>")
    assert "<title>JevRAGRank Benchmarks</title>" in html
    assert json.dumps(SUMMARY["datasets"]["scifact"]["runs"]["dense"]["metrics"]) in html


def test_fragment_has_no_document_wrapper():
    html = render_report(SUMMARY, standalone=False)
    assert "<!doctype" not in html.lower() and "<body" not in html
    assert html.lstrip().startswith("<title>")


def test_write_report_writes_both_files(tmp_path):
    out = write_report(SUMMARY, tmp_path / "report.html")
    assert out.exists() and (tmp_path / "report.fragment.html").exists()


def test_data_cannot_break_out_of_script():
    s = json.loads(json.dumps(SUMMARY))
    s["env"]["gpu"] = "</script><b>x</b>"
    assert "</script><b>" not in render_report(s, standalone=True)
