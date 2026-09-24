from __future__ import annotations

import asyncio
import json
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from .datasets import Dataset, load_beir, sample_queries, subsample_corpus
from .env import environment
from .systems import DEFAULTS, Context, build


@dataclass
class RunSpec:
    dataset: str
    system: str
    label: str = ""
    overrides: dict = field(default_factory=dict)
    split: str = "test"
    sample: int | None = None
    corpus_size: int | None = None
    seed: int = 0

    @property
    def name(self) -> str:
        return f"{self.system}@{self.label}" if self.label else self.system

    @property
    def dir_name(self) -> str:
        return self.dataset if self.split == "test" else f"{self.dataset}-{self.split}"


@dataclass
class Paths:
    results: Path = Path("results")
    data: Path = Path("data")
    cache: Path = Path(".cache")


def prepare(spec: RunSpec, paths: Paths) -> Dataset:
    ds = load_beir(spec.dataset, spec.split, paths.data)
    if spec.sample:
        ds = sample_queries(ds, spec.sample, spec.seed)
    if spec.corpus_size:
        ds = subsample_corpus(ds, spec.corpus_size, spec.seed)
    return ds


def _done_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["qid"] for line in path.read_text("utf-8").splitlines() if line}


def _log_error(out: Path, what: str) -> None:
    with out.with_suffix(".errors.log").open("a", encoding="utf-8") as log:
        log.write(f"{what}\n{traceback.format_exc()}\n")


async def run_spec(spec: RunSpec, paths: Paths, *, k: int = 10, warmup: int = 5,
                   jev_url: str = "http://127.0.0.1:8000", cache_mode: str = "write",
                   device: str | None = None, max_consecutive_failures: int = 3,
                   builder=build) -> Path:
    ds = prepare(spec, paths)
    out_dir = paths.results / spec.dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{spec.name}.jsonl"
    meta_path = out.with_suffix(".meta.json")
    qids = list(ds.queries)
    done = _done_qids(out)
    pending = [q for q in qids if q not in done]
    if not pending and meta_path.exists():
        return out  # complete: don't build the system, touch the GPU or call a Jev server
    ctx = Context(ds.docs, jev_url, paths.cache, cache_mode, device)
    built = builder(spec.system, ctx, spec.overrides)
    if built.jev is not None and pending:
        await built.jev.check()
    meta = {"system": spec.system, "label": spec.label,
            "options": {**DEFAULTS.get(spec.system, {}), **spec.overrides},
            "split": spec.split, "note": ds.note, "corpus_size": len(ds.docs),
            "n_queries": len(ds.queries), "index_seconds": built.index_seconds,
            "index_cached": bool(getattr(built.system, "from_cache", False)),
            "k": k, "spec": asdict(spec), "env": environment(jev_url if built.jev else None),
            "started": datetime.now(timezone.utc).isoformat()}
    meta_path.write_text(json.dumps(meta, indent=2), "utf-8")
    if not pending:
        return out
    for qid in qids[:warmup]:
        try:
            await built.system.asearch_traced(ds.queries[qid], k)
        except Exception:  # noqa: BLE001 - warm-up is untimed; real failures surface below
            _log_error(out, f"warm-up {qid}")
    failures = 0
    with out.open("a", encoding="utf-8") as f:
        for qid in pending:
            before = built.jev.stats.requests if built.jev else 0
            t0 = perf_counter()
            try:
                res = await built.system.asearch_traced(ds.queries[qid], k)
            except Exception:  # noqa: BLE001 - logged and retried on resume
                failures += 1
                _log_error(out, qid)
                if failures >= max_consecutive_failures:
                    raise SystemExit(f"{failures} consecutive failures in {spec.name}; see "
                                     f"{out.with_suffix('.errors.log')}") from None
                continue
            latency = perf_counter() - t0
            failures = 0
            row = {"qid": qid, "ranking": [[h.doc.id, h.score] for h in res.hits],
                   "candidates": res.candidates, "timings": res.timings, "latency": latency,
                   "jev_requests": (built.jev.stats.requests - before) if built.jev else 0,
                   "warm": True}
            f.write(json.dumps(row) + "\n")
            f.flush()
    return out


async def throughput_spec(spec: RunSpec, paths: Paths, *, n: int = 64, concurrency: int = 16,
                          k: int = 10, jev_url: str = "http://127.0.0.1:8000",
                          device: str | None = None, builder=build) -> dict:
    ds = prepare(spec, paths)
    built = builder(spec.system, Context(ds.docs, jev_url, paths.cache, "write", device),
                    spec.overrides)
    if built.jev is not None:
        await built.jev.check()
    queries = list(ds.queries.values())
    queries = (queries * (n // max(len(queries), 1) + 1))[:n]
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str) -> None:
        async with sem:
            await built.system.asearch_traced(q, k)

    await one(queries[0])  # warm-up
    t0 = perf_counter()
    await asyncio.gather(*(one(q) for q in queries))
    elapsed = perf_counter() - t0
    result = {"n": n, "concurrency": concurrency, "seconds": elapsed, "qps": n / elapsed}
    out_dir = paths.results / spec.dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{spec.name}.throughput.json").write_text(json.dumps(result, indent=2), "utf-8")
    return result
