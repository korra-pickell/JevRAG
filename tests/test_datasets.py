import json
import zipfile

import pytest

from bench.datasets import load_beir, sample_queries, subsample_corpus


def make_zip(tmp_path, name="toy"):
    docs = [{"_id": f"d{i}", "title": f"T{i}", "text": f"text {i}"} for i in range(10)]
    queries = [{"_id": "q1", "text": "first"}, {"_id": "q2", "text": "second"},
               {"_id": "q3", "text": "unjudged"}]
    qrels = "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t2\nq2\td5\t1\nq2\td6\t0\n"
    with zipfile.ZipFile(tmp_path / f"{name}.zip", "w") as z:
        z.writestr(f"{name}/corpus.jsonl", "\n".join(json.dumps(d) for d in docs))
        z.writestr(f"{name}/queries.jsonl", "\n".join(json.dumps(q) for q in queries))
        z.writestr(f"{name}/qrels/test.tsv", qrels)


def test_load_beir_from_cached_zip(tmp_path):
    make_zip(tmp_path)
    ds = load_beir("toy", "test", data_dir=tmp_path)
    assert len(ds.docs) == 10 and ds.docs[1].title == "T1"
    assert ds.queries == {"q1": "first", "q2": "second"}
    assert ds.qrels["q1"] == {"d1": 1, "d2": 2}
    assert ds.qrels["q2"] == {"d5": 1, "d6": 0}


def test_missing_split_is_a_clear_error(tmp_path):
    make_zip(tmp_path)
    with pytest.raises(ValueError, match="dev"):
        load_beir("toy", "dev", data_dir=tmp_path)


def test_sample_queries_is_deterministic(tmp_path):
    make_zip(tmp_path)
    ds = load_beir("toy", data_dir=tmp_path)
    a, b = sample_queries(ds, 1, seed=3), sample_queries(ds, 1, seed=3)
    assert a.queries == b.queries and len(a.queries) == 1
    assert set(a.qrels) == set(a.queries)
    assert "1 of 2 queries" in a.note


def test_subsample_keeps_relevant_docs(tmp_path):
    make_zip(tmp_path)
    ds = load_beir("toy", data_dir=tmp_path)
    sub = subsample_corpus(ds, 5, seed=1)
    ids = {d.id for d in sub.docs}
    assert len(ids) == 5 and {"d1", "d2", "d5"} <= ids
    assert [d.id for d in sub.docs] == [d.id for d in ds.docs if d.id in ids]
    with pytest.raises(ValueError):
        subsample_corpus(ds, 2)


@pytest.mark.slow
def test_real_scifact_shape(tmp_path):
    ds = load_beir("scifact", "test", data_dir="data")
    assert len(ds.docs) == 5183 and len(ds.queries) == 300
