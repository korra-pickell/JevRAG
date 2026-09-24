import asyncio
import json

import pytest

from bench.datasets import Dataset
from bench.runner import Paths, RunSpec, run_spec, throughput_spec
from bench.systems import Built
from jevragrank.base import Searchable
from jevragrank.types import Doc, Hit, SearchResult

DS = Dataset("toy", "test", [Doc(f"d{i}", f"text {i}") for i in range(5)],
             {"q1": "one", "q2": "two", "q3": "three"},
             {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}})


class Fake(Searchable):
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    async def asearch_traced(self, query, k):
        self.calls.append(query)
        if query in self.fail_on:
            raise RuntimeError("boom")
        return SearchResult([Hit(Doc("d1", ""), 1.0), Hit(Doc("d2", ""), 0.5)][:k],
                            candidates=["d1", "d2"], timings={"retrieve": 0.001})


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr("bench.runner.prepare", lambda spec, paths: DS)
    return Paths(results=tmp_path / "results", data=tmp_path / "data", cache=tmp_path / "cache")


def builder_for(system):
    return lambda name, ctx, overrides: Built(system, None, 0.01)


def lines(path):
    return [json.loads(x) for x in path.read_text("utf-8").splitlines()]


def test_run_writes_one_line_per_query_and_meta(paths):
    fake = Fake()
    out = asyncio.run(run_spec(RunSpec("toy", "bm25"), paths, warmup=1,
                               builder=builder_for(fake)))
    rows = lines(out)
    assert [r["qid"] for r in rows] == ["q1", "q2", "q3"]
    assert rows[0]["ranking"] == [["d1", 1.0], ["d2", 0.5]]
    assert rows[0]["candidates"] == ["d1", "d2"] and rows[0]["latency"] > 0
    assert fake.calls[0] == "one" and len(fake.calls) == 4  # 1 warm-up + 3
    meta = json.loads(out.with_suffix(".meta.json").read_text("utf-8"))
    assert meta["system"] == "bm25" and meta["corpus_size"] == 5


def test_run_resumes_and_retries_failures(paths):
    spec = RunSpec("toy", "bm25")
    asyncio.run(run_spec(spec, paths, warmup=0, builder=builder_for(Fake(fail_on={"two"}))))
    out = paths.results / "toy" / "bm25.jsonl"
    assert [r["qid"] for r in lines(out)] == ["q1", "q3"]
    assert "q2" in out.with_suffix(".errors.log").read_text("utf-8")
    again = Fake()
    asyncio.run(run_spec(spec, paths, warmup=0, builder=builder_for(again)))
    assert again.calls == ["two"]
    assert sorted(r["qid"] for r in lines(out)) == ["q1", "q2", "q3"]


def test_run_aborts_after_consecutive_failures(paths):
    with pytest.raises(SystemExit, match="consecutive"):
        asyncio.run(run_spec(RunSpec("toy", "bm25"), paths, warmup=0,
                             max_consecutive_failures=2,
                             builder=builder_for(Fake(fail_on={"one", "two", "three"}))))


def test_label_and_split_shape_paths():
    assert RunSpec("scifact", "jevragrank", label="noul").name == "jevragrank@noul"
    assert RunSpec("nfcorpus", "dense", split="dev").dir_name == "nfcorpus-dev"


def test_throughput_writes_qps(paths):
    res = asyncio.run(throughput_spec(RunSpec("toy", "bm25"), paths, n=3, concurrency=2,
                                      builder=builder_for(Fake())))
    assert res["qps"] > 0 and res["n"] == 3
    assert (paths.results / "toy" / "bm25.throughput.json").exists()


def test_completed_run_is_a_no_op(paths):
    spec = RunSpec("toy", "bm25")
    asyncio.run(run_spec(spec, paths, warmup=1, builder=builder_for(Fake())))
    out = paths.results / "toy" / "bm25.jsonl"
    before = out.read_text("utf-8")

    def must_not_build(name, ctx, overrides):
        raise AssertionError("a completed run must not build its system")

    asyncio.run(run_spec(spec, paths, warmup=1, builder=must_not_build))
    assert out.read_text("utf-8") == before


def test_importing_systems_does_not_import_torch():
    import subprocess
    import sys

    code = "import sys, bench.systems, bench.cli; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
