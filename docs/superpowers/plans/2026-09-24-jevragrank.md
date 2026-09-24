# JevRAGRank Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `jevragrank`, an open-source library with two retrieval pipelines: JevRAGRank (dense recall → Jev rerank) and JevRank (Jev-only choice tournament). Add a BEIR benchmark with charts comparing both against BM25, dense, cross-encoder and same-size LLM reranking on a local decider-2b server.

**Architecture:**
- A thin async client speaks the Jev `/v1/systemone` wire API.
- Retrievers and rerankers implement two small interfaces (`Searchable.asearch_traced`, `Reranker.arerank`). Pipelines and baselines are compositions of them.
- A `bench/` package loads BEIR, runs systems with resumable per-query JSONL output, computes metrics/CIs/significance, and renders matplotlib charts plus a self-contained HTML report.

**Tech Stack:**
- Python ≥3.10 (dev on 3.12), uv, hatchling.
- `httpx`, `numpy`; optional `sentence-transformers`, `bm25s` + `PyStemmer`, `transformers` + `torch` (CUDA 12.6 wheels on Windows).
- `ranx`, `matplotlib`, `pytest` + `pytest-asyncio`, `ruff`.
- Report page: Observable Plot + d3 from jsdelivr.

**Spec:** `docs/superpowers/specs/2026-09-24-jevragrank-design.md`. Read it before starting any task.

## Global Constraints

- Package name `jevragrank`; CLI `jevragrank-bench`; license MIT; `requires-python = ">=3.10"`.
- Jev wire API: `POST /v1/systemone` with body `{state, questions, model?}`; response `{model, answers, usage}`. The health probe is `GET /v1/models`.
- Choice questions: 2–255 options. Score questions: 2–10 levels. decider context is 32k tokens.
- Token budgets are approximated as **4 characters per token** (`jevragrank.text.CHARS_PER_TOKEN`).
- Default Jev score scale (exact strings): `"irrelevant"`, `"same topic, doesn't help"`, `"partially answers"`, `"mostly answers"`, `"fully answers"`.
- Dense model `BAAI/bge-base-en-v1.5` with query prefix `"Represent this sentence for searching relevant passages: "`. Cross-encoder `BAAI/bge-reranker-base`. LLM baseline `Qwen/Qwen3.5-2B-Base`. Jev backend `Mapika/decider-2b` (ablation: `Mapika/decider-0.8b`).
- Benchmark datasets: BEIR `scifact` and `nfcorpus` from `https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip`. Test splits for reported numbers. Tuning only on `scifact` train / `nfcorpus` dev.
- Evaluation depth `k = 10`. Metrics: `ndcg@10`, `recall@10`, `mrr@10`. 95% bootstrap CIs (1,000 resamples, seed 0). Paired Fisher randomization test vs `dense+ce` (10,000 permutations, seed 0), significant at p < 0.05.
- Timing runs never read the Jev cache (`cache_mode="write"`). Five warm-up queries per session are excluded.
- Hardware: RTX 4070 Laptop, 8 GB VRAM, Windows 11. decider and Qwen3.5-2B must not be resident together.
- Chart colors (validated with the dataviz validator, all pairs, both modes):

  | role | light | dark |
  |---|---|---|
  | JevRAGRank | `#2a78d6` | `#3987e5` |
  | JevRank | `#eb6834` | `#d95926` |
  | Baselines | gray `#898781` + distinct marker shapes + direct labels | same |
  | Surface | `#fcfcfb` | `#1a1a19` |

  No dual axes.
- Git identity is repo-local: `Korra Pickell <ronanpickell@gmail.com>`. Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **After each task's final commit: `git push origin main`** (never force-push).
- All commands run from the repo root `C:\PRGM\JevRAG` in Git Bash unless marked PowerShell.

## File map

```
pyproject.toml, LICENSE, .gitignore, .gitattributes, README.md
.github/workflows/tests.yml
serving/            start-decider.ps1, start-decider.sh, probe.py, Dockerfile, docker-compose.yml, README.md
src/jevragrank/
  __init__.py       public exports
  types.py          Doc, Hit, SearchResult, corpus_hash
  text.py           CHARS_PER_TOKEN, truncate, estimate_tokens, passage
  _sync.py          run_sync
  base.py           Searchable, Reranker, rank_hits, check_k
  jev_client.py     JevClient, JevStats, JevError family
  jev_answers.py    noul_probability, score_fraction, choice_probabilities, get_answer
  retrievers/       __init__.py, bm25.py (BM25Retriever), dense.py (DenseRetriever)
  rerankers/        __init__.py, cross_encoder.py, llm.py, jev.py (JevReranker)
  pipelines.py      RetrieveThenRerank, JevRAGRank
  tournament.py     JevRank
bench/
  __init__.py
  datasets.py       Dataset, load_beir, sample_queries, subsample_corpus
  metrics.py        per_query, bootstrap_ci, randomization_test, candidate_recall, latency_stats
  env.py            environment()
  systems.py        DEFAULTS, Context, Built, build()
  runner.py         RunSpec, Paths, run_spec(), throughput_spec()
  summary.py        summarize(), markdown_table()
  plots.py          render_all()
  report.py         render_report()
  report_template.html
  cli.py            main()
tests/              conftest.py + test_*.py, fixtures/decider/*.json
results/            committed benchmark outputs
docs/charts/        committed charts; docs/report.html
```

---

### Task 0: Serve decider-2b and probe the wire format (feasibility spike)

This task has no TDD: it produces serving scripts, captured response fixtures and measured latencies. Spike-only code stays in the scratchpad. Only the serving scripts, probe, fixtures and notes are committed.

**Files:**
- Create: `serving/start-decider.ps1`, `serving/start-decider.sh`, `serving/probe.py`, `serving/Dockerfile`, `serving/docker-compose.yml`, `serving/README.md`
- Create: `tests/fixtures/decider/score.json`, `noul.json`, `choice.json` (captured)
- Create: `docs/spike-notes.md`
- Create: `.gitignore`

**Interfaces:**
- Produces: a decider server at `http://127.0.0.1:8000` (2b) and optionally `:8001` (0.8b). Fixtures whose `answers` objects later tests parse. Measured p50 latencies that decide the 100-query sample rule (spec §7.7).

- [ ] **Step 1: Write `.gitignore`**

```gitignore
.venv/
serving/.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.ruff_cache/
data/
.cache/
dist/
```

- [ ] **Step 2: Create the serving venv and install decider** (Git Bash)

```bash
uv venv serving/.venv --python 3.12
uv pip install --python serving/.venv "decider-ai[serve]" --torch-backend cu126
serving/.venv/Scripts/python -c "import torch, decider; print(torch.__version__, torch.cuda.is_available())"
```
Expected: a torch version string and `True`.

- [ ] **Step 3: Write the start scripts**

`serving/start-decider.ps1`:
```powershell
param([string]$Model = "Mapika/decider-2b", [int]$Port = 8000)
$env:DECIDER_MODEL = $Model
& "$PSScriptRoot\.venv\Scripts\uvicorn.exe" decider.serve:app --host 127.0.0.1 --port $Port
```

`serving/start-decider.sh`:
```bash
#!/usr/bin/env bash
# Usage: serving/start-decider.sh [model] [port]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
bin="$here/.venv/bin/uvicorn"; [ -x "$bin" ] || bin="$here/.venv/Scripts/uvicorn.exe"
DECIDER_MODEL="${1:-Mapika/decider-2b}" exec "$bin" decider.serve:app --host 127.0.0.1 --port "${2:-8000}"
```

- [ ] **Step 4: Start the server** (run in background; first start downloads ~4 GB of weights)

```bash
bash serving/start-decider.sh Mapika/decider-2b 8000
```
Then poll `curl -s http://127.0.0.1:8000/health` until it returns JSON with `"ok": true`.

**If it fails natively on Windows**, use Docker instead (Steps 4b–4c). Otherwise skip to Step 5.

- [ ] **Step 4b: Docker fallback files**

`serving/Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime
RUN pip install --no-cache-dir "decider-ai[serve]"
ENV DECIDER_MODEL=Mapika/decider-2b
EXPOSE 8000
CMD ["uvicorn", "decider.serve:app", "--host", "0.0.0.0", "--port", "8000"]
```

`serving/docker-compose.yml`:
```yaml
services:
  decider:
    build: .
    ports: ["8000:8000"]
    environment:
      DECIDER_MODEL: ${DECIDER_MODEL:-Mapika/decider-2b}
    volumes:
      - ${HOME:-~}/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices: [{ driver: nvidia, count: 1, capabilities: [gpu] }]
```

- [ ] **Step 4c: Run it** (only if Step 4 failed): `docker compose -f serving/docker-compose.yml up -d --build`, then poll `/health` as in Step 4.

- [ ] **Step 5: Write the probe**

`serving/probe.py`:
```python
"""Probe a Jev-compatible server: capture answer shapes and measure latency.

Usage: python serving/probe.py --url http://127.0.0.1:8000 --save tests/fixtures/decider
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

LEVELS = ["irrelevant", "same topic, doesn't help", "partially answers", "mostly answers",
          "fully answers"]
SENTENCE = ("In a randomized trial of 25,000 adults, daily vitamin D3 supplementation did not "
            "reduce total fractures compared with placebo over five years of follow-up. ")
PASSAGE = SENTENCE * 12          # ~2,000 chars, about 500 tokens
PREVIEW = SENTENCE * 2           # ~340 chars, about 85 tokens
STATE = "Query: does vitamin D reduce fracture risk?"


def score_body(n: int) -> dict:
    q = {"type": "score", "instructions": f"How well does this passage answer the query?\n\n"
         f"Passage:\n{PASSAGE}", "criteria": LEVELS}
    return {"state": STATE, "questions": {f"c{i}": q for i in range(n)}}


def noul_body(n: int) -> dict:
    q = {"type": "noul", "instructions": f"Does this passage help answer the query?\n\n"
         f"Passage:\n{PASSAGE}"}
    return {"state": STATE, "questions": {f"c{i}": q for i in range(n)}}


def choice_body(n: int) -> dict:
    criteria = {f"p{i}": f"{PREVIEW} (study {i})" for i in range(n)}
    return {"state": STATE, "questions": {"pick": {
        "type": "choice", "instructions": "Which passage most likely answers the query?",
        "criteria": criteria}}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--save", type=Path)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()
    with httpx.Client(base_url=args.url, timeout=120) as c:
        for path in ("/health", "/v1/models"):
            r = c.get(path)
            print(path, r.status_code, r.text[:300])
        cases = {"score": score_body(12), "noul": noul_body(12), "choice": choice_body(64)}
        for name, body in cases.items():
            r = c.post("/v1/systemone", json=body)
            print(f"\n{name}: HTTP {r.status_code}")
            r.raise_for_status()
            data = r.json()
            first = next(iter(data["answers"].values()))
            print(json.dumps(first, indent=2)[:800])
            if args.save:
                args.save.mkdir(parents=True, exist_ok=True)
                (args.save / f"{name}.json").write_text(json.dumps(data, indent=2), "utf-8")
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                c.post("/v1/systemone", json=body).raise_for_status()
                times.append(time.perf_counter() - t0)
            print(f"{name}: p50 {statistics.median(times) * 1000:.0f} ms over {args.repeats}")
        r = c.post("/v1/systemone", json=choice_body(255))
        print(f"\nchoice with 255 options: HTTP {r.status_code}")
        r = c.post("/v1/systemone", json=choice_body(256))
        print(f"choice with 256 options: HTTP {r.status_code} {r.text[:200]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the probe**

```bash
uv run --no-project --with httpx python serving/probe.py --url http://127.0.0.1:8000 --save tests/fixtures/decider
```
Expected: `/v1/models` 200, three HTTP 200 cases with printed answers, p50 lines, 255 options → 200 and 256 → 4xx.

**If an answer shape differs from spec §3.1** (e.g. noul is a bare float, or score `probabilities` is a dict keyed by label), note it. Task 2's parsers already accept these variants; if a *new* variant appears, add it to `jev_answers.py` in Task 2 with a test.

**If the request is rejected** (e.g. `criteria` must be a dict for score), fix the body in `probe.py`, record the accepted form in the notes, and use that form in Task 5's `JevReranker._question`.

- [ ] **Step 7: Probe decider-0.8b too** (for the model-size ablation)

```bash
bash serving/start-decider.sh Mapika/decider-0.8b 8001
uv run --no-project --with httpx python serving/probe.py --url http://127.0.0.1:8001 --repeats 5
```
Record p50s. Stop the 0.8b server afterwards.

- [ ] **Step 8: Write `docs/spike-notes.md` and `serving/README.md`**

`docs/spike-notes.md` records:
- how decider was served (native Windows or Docker) and the exact command;
- the verbatim answer shape for each type;
- p50 for 12×score, 12×noul, 64-option choice, for 2b and 0.8b;
- the max options result;
- the projected runtimes.

Compute the projections as follows:
- **Rerank projection per dataset** = `(50/12) × p50(score12) × queries`. Queries: 300 for SciFact, 323 for NFCorpus.
- **Tournament projection per query** ≈ `ceil(docs/64) × p50(choice64) / min(16, groups) × batching_factor + 4 × p50(score12)`.
  - The division by `min(16, groups)` optimistically assumes 16-way concurrency.
  - Measure the real `batching_factor` with one extra probe run of 16 concurrent choice requests via `python -c` in the scratchpad.
- **Decision:** if any single run projects over 6 h, write `JEVRANK_SAMPLE = 100` in the notes. Task 12 then uses `--sample 100` for `jevrank` runs.

`serving/README.md`: how to start decider natively (`start-decider.ps1` / `.sh`) or via Docker, the port convention (2b on 8000, 0.8b on 8001), and "stop decider before running `dense+llm`".

- [ ] **Step 9: Commit and push**

```bash
git add .gitignore serving tests/fixtures docs/spike-notes.md
git commit -m "Serve decider-2b and capture wire-format fixtures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 1: Project scaffold, core types and text helpers

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `.gitattributes`, `src/jevragrank/__init__.py`, `src/jevragrank/types.py`, `src/jevragrank/text.py`, `src/jevragrank/_sync.py`, `src/jevragrank/base.py`, `bench/__init__.py`
- Test: `tests/test_core.py`

**Interfaces:**
- Produces:
  - `Doc(id: str, text: str, title: str = "")` with `.full_text`.
  - `Hit(doc, score: float, stage_scores: dict[str, float] = {}, degraded: bool = False)`.
  - `SearchResult(hits: list[Hit], candidates: list[str] | None = None, timings: dict[str, float] = {})`.
  - `corpus_hash(docs) -> str`.
  - `truncate(text, max_tokens) -> str`, `estimate_tokens(text) -> int`, `passage(doc, max_tokens) -> str`.
  - `run_sync(coro)`.
  - `Searchable` (abstract `asearch_traced(query, k) -> SearchResult`; provides `asearch`, `search`).
  - `Reranker` (abstract `arerank(query, hits, k) -> list[Hit]`; provides `rerank`).
  - `rank_hits(hits, scores, *, stage, k, degraded=None) -> list[Hit]`, `check_k(k)`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "jevragrank"
version = "0.1.0"
description = "Typed-decision (Jev) reranking and retrieval for RAG, with open benchmarks."
readme = "README.md"
license = "MIT"
license-files = ["LICENSE"]
authors = [{ name = "Korra Pickell" }]
requires-python = ">=3.10"
dependencies = ["httpx>=0.27", "numpy>=1.26"]

[project.optional-dependencies]
dense = ["sentence-transformers>=3.0"]
bm25 = ["bm25s>=0.2.10", "PyStemmer>=2.2"]
baselines = ["sentence-transformers>=3.0", "transformers>=4.57", "torch>=2.5", "accelerate>=1.0"]
bench = ["jevragrank[dense,bm25,baselines]", "ranx>=0.3.20", "matplotlib>=3.9"]
dev = ["pytest>=8", "pytest-asyncio>=0.24", "ruff>=0.6", "bm25s>=0.2.10", "PyStemmer>=2.2",
       "ranx>=0.3.20", "matplotlib>=3.9"]

[project.scripts]
jevragrank-bench = "bench.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/jevragrank", "bench"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = ["slow: downloads models or datasets", "live: needs a running Jev server",
           "gpu: needs a CUDA GPU"]
addopts = "-m 'not slow and not live and not gpu'"
filterwarnings = ["ignore::SyntaxWarning"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]
```

- [ ] **Step 2: Write `LICENSE`** (MIT, "Copyright (c) 2026 Korra Pickell", standard MIT text) and `.gitattributes`:

```gitattributes
* text=auto eol=lf
*.png binary
```

- [ ] **Step 3: Create the dev environment**

```bash
uv sync --extra dev
```
Expected: `.venv` created, no errors.

- [ ] **Step 4: Write the failing tests** — `tests/test_core.py`

```python
import asyncio

import pytest

from jevragrank.base import Reranker, Searchable, check_k, rank_hits
from jevragrank.text import CHARS_PER_TOKEN, estimate_tokens, passage, truncate
from jevragrank.types import Doc, Hit, SearchResult, corpus_hash


def test_full_text_joins_title_and_text():
    assert Doc("1", "body", "Title").full_text == "Title\nbody"
    assert Doc("1", "body").full_text == "body"


def test_corpus_hash_changes_with_content():
    a = [Doc("1", "x"), Doc("2", "y")]
    assert corpus_hash(a) == corpus_hash(list(a))
    assert corpus_hash(a) != corpus_hash([Doc("1", "x"), Doc("2", "z")])


def test_truncate_keeps_short_text():
    assert truncate("short text", 10) == "short text"


def test_truncate_cuts_at_word_boundary_and_marks():
    text = "word " * 100
    out = truncate(text, 10)
    assert len(out) <= 10 * CHARS_PER_TOKEN + 2
    assert out.endswith(" …")
    assert "wor …" not in out


def test_truncate_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        truncate("x", 0)


def test_estimate_tokens_rounds_up():
    assert estimate_tokens("abcde") == 2


def test_passage_uses_full_text():
    assert passage(Doc("1", "body", "T"), 100) == "T\nbody"


def test_check_k():
    check_k(1)
    with pytest.raises(ValueError):
        check_k(0)


def _hits():
    return [Hit(Doc(str(i), f"d{i}"), score=s, stage_scores={"dense": s})
            for i, s in enumerate([0.9, 0.8, 0.7])]


def test_rank_hits_orders_by_score_then_retrieval_score():
    ranked = rank_hits(_hits(), [0.5, 0.9, 0.5], stage="jev", k=3)
    assert [h.doc.id for h in ranked] == ["1", "0", "2"]
    assert ranked[0].score == 0.9
    assert ranked[0].stage_scores == {"dense": 0.8, "jev": 0.9}


def test_rank_hits_truncates_and_marks_degraded():
    ranked = rank_hits(_hits(), [0.1, 0.2, 0.3], stage="jev", k=2,
                       degraded=[False, False, True])
    assert [h.doc.id for h in ranked] == ["2", "1"]
    assert ranked[0].degraded and not ranked[1].degraded


class Echo(Searchable):
    async def asearch_traced(self, query, k):
        return SearchResult([Hit(Doc("1", query), 1.0)][:k])


def test_search_works_without_and_inside_running_loop():
    assert Echo().search("hello", 1)[0].doc.text == "hello"

    async def inner():
        return Echo().search("inside", 1)

    assert asyncio.run(inner())[0].doc.text == "inside"


class Reverse(Reranker):
    async def arerank(self, query, hits, k):
        return list(reversed(hits))[:k]


def test_reranker_sync_wrapper():
    assert [h.doc.id for h in Reverse().rerank("q", _hits(), 2)] == ["2", "1"]
```

- [ ] **Step 5: Run and confirm failure**

Run: `uv run pytest tests/test_core.py -q`
Expected: collection errors (`ModuleNotFoundError: jevragrank.base`).

- [ ] **Step 6: Implement**

`src/jevragrank/types.py`:
```python
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    title: str = ""

    @property
    def full_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


@dataclass
class Hit:
    doc: Doc
    score: float
    stage_scores: dict[str, float] = field(default_factory=dict)
    degraded: bool = False


@dataclass
class SearchResult:
    hits: list[Hit]
    candidates: list[str] | None = None
    timings: dict[str, float] = field(default_factory=dict)


def corpus_hash(docs: Iterable[Doc]) -> str:
    h = hashlib.sha256()
    for d in docs:
        h.update(d.id.encode())
        h.update(b"\x1f")
        h.update(d.full_text.encode())
        h.update(b"\x1e")
    return h.hexdigest()[:16]
```

`src/jevragrank/text.py`:
```python
from __future__ import annotations

from .types import Doc

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


def truncate(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    limit = max_tokens * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.8:
        cut = cut[:space]
    return cut.rstrip() + " …"


def passage(doc: Doc, max_tokens: int) -> str:
    return truncate(doc.full_text, max_tokens)
```

`src/jevragrank/_sync.py`:
```python
from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from sync code, even when an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
```

`src/jevragrank/base.py`:
```python
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ._sync import run_sync
from .types import Hit, SearchResult


def check_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def rank_hits(hits: list[Hit], scores: Sequence[float], *, stage: str, k: int,
              degraded: Sequence[bool] | None = None) -> list[Hit]:
    """Sort hits by new score, breaking ties by the incoming score, then input order."""
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], -hits[i].score, i))
    out = []
    for i in order[:k]:
        h = hits[i]
        s = float(scores[i])
        out.append(Hit(h.doc, s, {**h.stage_scores, stage: s},
                       degraded=h.degraded or bool(degraded and degraded[i])))
    return out


class Searchable(ABC):
    @abstractmethod
    async def asearch_traced(self, query: str, k: int) -> SearchResult: ...

    async def asearch(self, query: str, k: int = 10) -> list[Hit]:
        return (await self.asearch_traced(query, k)).hits

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return run_sync(self.asearch(query, k))


class Reranker(ABC):
    @abstractmethod
    async def arerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...

    def rerank(self, query: str, hits: list[Hit], k: int = 10) -> list[Hit]:
        return run_sync(self.arerank(query, hits, k))
```

`src/jevragrank/__init__.py`:
```python
from .types import Doc, Hit, SearchResult

__all__ = ["Doc", "Hit", "SearchResult"]
__version__ = "0.1.0"
```

`bench/__init__.py`: empty file.

- [ ] **Step 7: Run tests and lint**

Run: `uv run pytest tests/test_core.py -q && uv run ruff check .`
Expected: all pass, `All checks passed!`

- [ ] **Step 8: Commit and push**

```bash
git add pyproject.toml uv.lock LICENSE .gitattributes src bench tests/test_core.py
git commit -m "Scaffold package with core types and text helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 2: Jev client and answer parsing

**Files:**
- Create: `src/jevragrank/jev_client.py`, `src/jevragrank/jev_answers.py`, `tests/conftest.py`
- Modify: `src/jevragrank/__init__.py`
- Test: `tests/test_jev_client.py`, `tests/test_jev_answers.py`

**Interfaces:**
- Consumes: nothing from earlier tasks except the fixtures from Task 0.
- Produces:
  - `JevClient(base_url="http://localhost:8000", *, model=None, api_key=None, timeout=60.0, max_retries=3, concurrency=16, cache_dir=None, cache_mode="readwrite", transport=None, backoff=0.5)`.
    - `await ask(state, questions: dict) -> dict` returns the `answers` object.
    - `await check()` probes the server; `await aclose()` closes it; `.stats: JevStats(requests, cache_hits, retries)`.
    - `cache_mode ∈ {"readwrite", "write", "off"}`.
  - Errors: `JevError` → `JevConnectionError`, `JevRequestError(status, message)` (`.status`), `JevResponseError`.
  - Parsers: `get_answer(answers, key) -> Any`, `noul_probability(answer) -> float`, `score_fraction(answer, levels) -> float` (0–1), `choice_probabilities(answer) -> dict[str, float]`.
  - Test fixtures (conftest): `FakeJev` class, `keyword_relevance(query, text) -> float`, fixtures `fake_jev`, `jev_client`.

- [ ] **Step 1: Write `tests/conftest.py`** (fake Jev server used by every later test)

```python
from __future__ import annotations

import json

import httpx
import pytest

from jevragrank.jev_client import JevClient


def keyword_relevance(query: str, text: str) -> float:
    q = set(query.lower().split())
    t = set(text.lower().replace("\n", " ").split())
    return len(q & t) / max(len(q), 1)


def _passage_of(instructions: str) -> str:
    return instructions.split("Passage:\n", 1)[1]


class FakeJev:
    """In-memory /v1/systemone server; relevance(query, text) -> [0, 1] drives answers."""

    def __init__(self, relevance=keyword_relevance):
        self.relevance = relevance
        self.bodies: list[dict] = []
        self.fail_next: list[int] = []
        self.models_hits = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            self.models_hits += 1
            return httpx.Response(200, json={"models": [{"name": "fake"}]})
        if self.fail_next:
            return httpx.Response(self.fail_next.pop(0), json={"detail": "boom"})
        body = json.loads(request.content)
        self.bodies.append(body)
        query = body["state"].removeprefix("Query: ")
        answers = {}
        for key, q in body["questions"].items():
            if q["type"] == "score":
                n = len(q["criteria"])
                level = round(self.relevance(query, _passage_of(q["instructions"])) * (n - 1))
                probs = [0.0] * n
                probs[level] = 1.0
                answers[key] = {"type": "score", "score": float(level), "probabilities": probs}
            elif q["type"] == "noul":
                p = self.relevance(query, _passage_of(q["instructions"]))
                answers[key] = {"type": "noul", "noul": p}
            else:
                raw = {o: self.relevance(query, t) + 1e-6 for o, t in q["criteria"].items()}
                total = sum(raw.values())
                probs = {o: v / total for o, v in raw.items()}
                best = max(probs, key=probs.get)
                answers[key] = {"type": "choice", "choice": best, "confidence": probs[best],
                                "probabilities": probs}
        return httpx.Response(200, json={"model": "fake", "answers": answers,
                                         "usage": {"input_tokens": 1, "output_tokens": 0}})


@pytest.fixture
def fake_jev() -> FakeJev:
    return FakeJev()


@pytest.fixture
def jev_client(fake_jev) -> JevClient:
    return JevClient("http://fake", transport=httpx.MockTransport(fake_jev.handler), backoff=0.0)
```

- [ ] **Step 2: Write the failing client tests** — `tests/test_jev_client.py`

```python
import asyncio

import httpx
import pytest

from jevragrank.jev_client import (JevClient, JevConnectionError, JevRequestError,
                                   JevResponseError)

NOUL = {"a": {"type": "noul", "instructions": "Is it?\n\nPassage:\nvitamin d"}}


def client(fake, **kw):
    return JevClient("http://fake", transport=httpx.MockTransport(fake.handler), backoff=0.0, **kw)


async def test_ask_posts_state_and_questions(fake_jev):
    answers = await client(fake_jev).ask("Query: vitamin d", NOUL)
    assert answers["a"]["noul"] == 1.0
    assert fake_jev.bodies == [{"state": "Query: vitamin d", "questions": NOUL}]
    assert fake_jev.models_hits == 1


async def test_model_is_sent_when_set(fake_jev):
    await client(fake_jev, model="decider-2b").ask("Query: x", NOUL)
    assert fake_jev.bodies[0]["model"] == "decider-2b"


async def test_readwrite_cache_serves_repeat_requests(fake_jev, tmp_path):
    c = client(fake_jev, cache_dir=tmp_path)
    first = await c.ask("Query: x", NOUL)
    second = await c.ask("Query: x", NOUL)
    assert first == second
    assert len(fake_jev.bodies) == 1
    assert c.stats.cache_hits == 1


async def test_write_mode_never_reads_cache(fake_jev, tmp_path):
    c = client(fake_jev, cache_dir=tmp_path, cache_mode="write")
    await c.ask("Query: x", NOUL)
    await c.ask("Query: x", NOUL)
    assert len(fake_jev.bodies) == 2
    assert any(tmp_path.rglob("*.json"))


async def test_cache_is_namespaced_by_server(fake_jev, tmp_path):
    await client(fake_jev, cache_dir=tmp_path).ask("Query: x", NOUL)
    other = JevClient("http://other", transport=httpx.MockTransport(fake_jev.handler),
                      cache_dir=tmp_path, backoff=0.0)
    await other.ask("Query: x", NOUL)
    assert len(fake_jev.bodies) == 2


async def test_retries_transient_errors(fake_jev):
    fake_jev.fail_next = [503, 429]
    c = client(fake_jev)
    await c.ask("Query: x", NOUL)
    assert c.stats.retries == 2


async def test_gives_up_after_max_retries(fake_jev):
    fake_jev.fail_next = [503] * 4
    with pytest.raises(JevRequestError) as e:
        await client(fake_jev, max_retries=3).ask("Query: x", NOUL)
    assert e.value.status == 503


async def test_client_errors_are_not_retried(fake_jev):
    fake_jev.fail_next = [422]
    c = client(fake_jev)
    with pytest.raises(JevRequestError, match="boom"):
        await c.ask("Query: x", NOUL)
    assert c.stats.retries == 0


async def test_unreachable_server_raises_connection_error():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    c = JevClient("http://nowhere", transport=httpx.MockTransport(refuse), backoff=0.0)
    with pytest.raises(JevConnectionError, match="http://nowhere"):
        await c.ask("Query: x", NOUL)


async def test_response_without_answers_is_rejected():
    def bad(request):
        if request.method == "GET":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"model": "x"})

    c = JevClient("http://x", transport=httpx.MockTransport(bad), backoff=0.0)
    with pytest.raises(JevResponseError):
        await c.ask("Query: x", NOUL)


def test_client_survives_separate_event_loops(fake_jev):
    c = client(fake_jev)
    asyncio.run(c.ask("Query: x", NOUL))
    asyncio.run(c.ask("Query: y", NOUL))
    assert len(fake_jev.bodies) == 2


def test_invalid_cache_mode():
    with pytest.raises(ValueError):
        JevClient(cache_mode="sometimes")
```

- [ ] **Step 3: Write the failing parser tests** — `tests/test_jev_answers.py`

```python
import json
from pathlib import Path

import pytest

from jevragrank.jev_answers import (choice_probabilities, get_answer, noul_probability,
                                    score_fraction)
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
```

- [ ] **Step 4: Run and confirm failure**

Run: `uv run pytest tests/test_jev_client.py tests/test_jev_answers.py -q`
Expected: `ModuleNotFoundError: jevragrank.jev_client`.

- [ ] **Step 5: Implement `src/jevragrank/jev_client.py`**

```python
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

RETRY_STATUS = {429, 500, 502, 503, 504}
CACHE_MODES = ("readwrite", "write", "off")


class JevError(Exception):
    """Base class for Jev client errors."""


class JevConnectionError(JevError):
    """The server could not be reached."""


class JevResponseError(JevError):
    """The server answered, but not in the expected shape."""


class JevRequestError(JevError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Jev server rejected the request (HTTP {status}): {message}")
        self.status = status


@dataclass
class JevStats:
    requests: int = 0
    cache_hits: int = 0
    retries: int = 0


def _error_message(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return resp.text[:300]
    if isinstance(data, dict):
        return str(data.get("detail") or data.get("error") or data)[:300]
    return str(data)[:300]


class JevClient:
    """Async client for any Jev-compatible `POST /v1/systemone` server."""

    def __init__(self, base_url: str = "http://localhost:8000", *, model: str | None = None,
                 api_key: str | None = None, timeout: float = 60.0, max_retries: int = 3,
                 concurrency: int = 16, cache_dir: str | Path | None = None,
                 cache_mode: str = "readwrite", transport: httpx.AsyncBaseTransport | None = None,
                 backoff: float = 0.5):
        if cache_mode not in CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {CACHE_MODES}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.concurrency = concurrency
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_mode = cache_mode
        self.backoff = backoff
        self.stats = JevStats()
        self._transport = transport
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: httpx.AsyncClient | None = None
        self._sem: asyncio.Semaphore | None = None
        self._checked = False

    def _loop_state(self) -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout,
                                             headers=headers, transport=self._transport)
            self._sem = asyncio.Semaphore(self.concurrency)
            self._loop = loop
            self._checked = False
        return self._client, self._sem

    async def check(self) -> None:
        client, _ = self._loop_state()
        try:
            await client.get("/v1/models")
        except httpx.TransportError as e:
            raise JevConnectionError(
                f"Can't reach a Jev server at {self.base_url} ({e.__class__.__name__}). "
                "Start one (see serving/README.md) or pass the right base_url.") from e
        self._checked = True

    def _cache_path(self, body: dict) -> Path | None:
        if self.cache_dir is None or self.cache_mode == "off":
            return None
        ns = self.model or self.base_url
        raw = json.dumps({"ns": ns, "body": body}, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(raw.encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    async def ask(self, state: Any, questions: dict) -> dict:
        body: dict[str, Any] = {"state": state, "questions": questions}
        if self.model:
            body["model"] = self.model
        path = self._cache_path(body)
        if path is not None and self.cache_mode == "readwrite" and path.exists():
            self.stats.cache_hits += 1
            return json.loads(path.read_text("utf-8"))
        client, sem = self._loop_state()
        if not self._checked:
            await self.check()
        async with sem:
            answers = await self._post(client, body)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(answers), "utf-8")
            os.replace(tmp, path)
        return answers

    async def _post(self, client: httpx.AsyncClient, body: dict) -> dict:
        for attempt in range(self.max_retries + 1):
            self.stats.requests += 1
            try:
                resp = await client.post("/v1/systemone", json=body)
            except httpx.TransportError as e:
                err: JevError = JevConnectionError(f"Request to {self.base_url} failed: {e!r}")
            else:
                if resp.status_code < 400:
                    data = resp.json()
                    answers = data.get("answers") if isinstance(data, dict) else None
                    if not isinstance(answers, dict):
                        raise JevResponseError(f"Response has no 'answers' object: {str(data)[:200]}")
                    return answers
                err = JevRequestError(resp.status_code, _error_message(resp))
                if resp.status_code not in RETRY_STATUS:
                    raise err
            if attempt == self.max_retries:
                raise err
            self.stats.retries += 1
            await asyncio.sleep(self.backoff * 2**attempt * (1 + random.random()))
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
```

- [ ] **Step 6: Implement `src/jevragrank/jev_answers.py`**

```python
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
```

- [ ] **Step 7: Export from `src/jevragrank/__init__.py`**

```python
from .jev_client import (JevClient, JevConnectionError, JevError, JevRequestError,
                         JevResponseError)
from .types import Doc, Hit, SearchResult

__all__ = ["Doc", "Hit", "JevClient", "JevConnectionError", "JevError", "JevRequestError",
           "JevResponseError", "SearchResult"]
__version__ = "0.1.0"
```

- [ ] **Step 8: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all pass (the fixture test passes if Task 0 captured fixtures, otherwise it is skipped).

- [ ] **Step 9: Commit and push**

```bash
git add src tests
git commit -m "Add Jev client with cache, retries and answer parsers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 3: BM25 and dense retrievers

**Files:**
- Create: `src/jevragrank/retrievers/__init__.py`, `bm25.py`, `dense.py`
- Test: `tests/test_retrievers.py`

**Interfaces:**
- Consumes: `Searchable`, `check_k` (Task 1), `Doc`, `Hit`, `SearchResult`, `corpus_hash`.
- Produces:
  - `BM25Retriever(docs, *, stopwords="en", stem=True)`, a `Searchable`. Stage key `"bm25"`, drops zero-score hits, has `.build_seconds`.
  - `DenseRetriever(docs, *, model_name="BAAI/bge-base-en-v1.5", query_prefix=BGE_QUERY_PREFIX, encoder=None, cache_dir=None, device=None, batch_size=64)`, a `Searchable`. Stage key `"dense"`.
    - `encoder: Callable[[list[str]], np.ndarray]`.
    - Attributes `.embeddings`, `.build_seconds`, `.from_cache`.
  - `BGE_QUERY_PREFIX`.

- [ ] **Step 1: Write the failing tests** — `tests/test_retrievers.py`

```python
import numpy as np
import pytest

from jevragrank.retrievers import BGE_QUERY_PREFIX, BM25Retriever, DenseRetriever
from jevragrank.types import Doc

DOCS = [
    Doc("a", "cats purr loudly at night", "Cats"),
    Doc("b", "dogs bark at the mailman", "Dogs"),
    Doc("c", "vitamin d supplementation and bone fractures in adults", "Vitamin D"),
]
VOCAB = ["cats", "dogs", "vitamin", "fractures", "bark", "purr"]


def bow_encoder(texts):
    vecs = np.array([[t.lower().count(w) for w in VOCAB] for t in texts], dtype=np.float32)
    return vecs + 1e-3


def test_bm25_ranks_matching_doc_first():
    hits = BM25Retriever(DOCS).search("vitamin fracture risk", k=2)
    assert hits[0].doc.id == "c"
    assert hits[0].stage_scores["bm25"] == hits[0].score > 0


def test_bm25_drops_non_matching_docs():
    assert BM25Retriever(DOCS).search("zzzz qqqq", k=3) == []


def test_bm25_clamps_k_to_corpus():
    assert len(BM25Retriever(DOCS).search("cats dogs vitamin", k=50)) <= 3


def test_bm25_records_timing():
    import asyncio
    out = asyncio.run(BM25Retriever(DOCS).asearch_traced("dogs", 1))
    assert "retrieve" in out.timings and out.candidates is None


def test_dense_uses_encoder_and_prefix():
    seen = []

    def enc(texts):
        seen.extend(texts)
        return bow_encoder(texts)

    r = DenseRetriever(DOCS, encoder=enc)
    hits = r.search("dogs bark", k=1)
    assert hits[0].doc.id == "b"
    assert seen[-1] == BGE_QUERY_PREFIX + "dogs bark"
    assert r.embeddings.shape == (3, len(VOCAB))


def test_dense_scores_are_cosine():
    hits = DenseRetriever(DOCS, encoder=bow_encoder).search("cats purr", k=3)
    assert -1.0 <= hits[-1].score <= hits[0].score <= 1.0 + 1e-6


def test_dense_embedding_cache(tmp_path):
    calls = []

    def enc(texts):
        calls.append(len(texts))
        return bow_encoder(texts)

    DenseRetriever(DOCS, encoder=enc, cache_dir=tmp_path)
    r2 = DenseRetriever(DOCS, encoder=enc, cache_dir=tmp_path)
    assert calls == [3] and r2.from_cache


@pytest.mark.parametrize("cls", [BM25Retriever, DenseRetriever])
def test_empty_corpus_rejected(cls):
    with pytest.raises(ValueError):
        cls([], encoder=bow_encoder) if cls is DenseRetriever else cls([])
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_retrievers.py -q`
Expected: `ModuleNotFoundError: jevragrank.retrievers`.

- [ ] **Step 3: Implement**

`src/jevragrank/retrievers/__init__.py`:
```python
from .bm25 import BM25Retriever
from .dense import BGE_QUERY_PREFIX, DenseRetriever

__all__ = ["BGE_QUERY_PREFIX", "BM25Retriever", "DenseRetriever"]
```

`src/jevragrank/retrievers/bm25.py`:
```python
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from time import perf_counter

from ..base import Searchable, check_k
from ..types import Doc, Hit, SearchResult


class BM25Retriever(Searchable):
    """Lexical BM25 via bm25s (install the `bm25` extra)."""

    def __init__(self, docs: Sequence[Doc], *, stopwords: str = "en", stem: bool = True):
        import bm25s
        import Stemmer

        if not docs:
            raise ValueError("docs must not be empty")
        self.docs = list(docs)
        self._bm25s = bm25s
        self._stopwords = stopwords
        self._stemmer = Stemmer.Stemmer("english") if stem else None
        t0 = perf_counter()
        tokens = bm25s.tokenize([d.full_text for d in self.docs], stopwords=stopwords,
                                stemmer=self._stemmer, show_progress=False)
        self._index = bm25s.BM25()
        self._index.index(tokens, show_progress=False)
        self.build_seconds = perf_counter() - t0

    def _search(self, query: str, k: int) -> list[Hit]:
        q = self._bm25s.tokenize([query], stopwords=self._stopwords, stemmer=self._stemmer,
                                 return_ids=False, show_progress=False)
        if not q[0]:
            return []
        idx, scores = self._index.retrieve(q, k=min(k, len(self.docs)), show_progress=False)
        return [Hit(self.docs[i], float(s), {"bm25": float(s)})
                for i, s in zip(idx[0].tolist(), scores[0].tolist()) if s > 0]

    async def asearch_traced(self, query: str, k: int) -> SearchResult:
        check_k(k)
        t0 = perf_counter()
        hits = await asyncio.to_thread(self._search, query, k)
        return SearchResult(hits, timings={"retrieve": perf_counter() - t0})
```

`src/jevragrank/retrievers/dense.py`:
```python
from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np

from ..base import Searchable, check_k
from ..types import Doc, Hit, SearchResult, corpus_hash

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
Encoder = Callable[[list[str]], np.ndarray]


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def sentence_transformer_encoder(model_name: str, device: str | None = None,
                                 batch_size: int = 64) -> Encoder:
    model = None

    def encode(texts: list[str]) -> np.ndarray:
        nonlocal model
        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name, device=device)
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=len(texts) > 1000)

    return encode


class DenseRetriever(Searchable):
    """Exact cosine search over sentence-transformers embeddings (install `dense`)."""

    def __init__(self, docs: Sequence[Doc], *, model_name: str = "BAAI/bge-base-en-v1.5",
                 query_prefix: str = BGE_QUERY_PREFIX, encoder: Encoder | None = None,
                 cache_dir: str | Path | None = None, device: str | None = None,
                 batch_size: int = 64):
        if not docs:
            raise ValueError("docs must not be empty")
        self.docs = list(docs)
        self.query_prefix = query_prefix
        self._encode = encoder or sentence_transformer_encoder(model_name, device, batch_size)
        self._lock = threading.Lock()
        self.from_cache = False
        t0 = perf_counter()
        self.embeddings = self._load_or_embed(model_name, cache_dir)
        self.build_seconds = perf_counter() - t0

    def _embed(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            return _normalize(np.asarray(self._encode(texts), dtype=np.float32))

    def _load_or_embed(self, model_name: str, cache_dir: str | Path | None) -> np.ndarray:
        path = None
        if cache_dir:
            slug = re.sub(r"[^A-Za-z0-9.-]+", "_", model_name)
            path = Path(cache_dir) / f"{slug}-{corpus_hash(self.docs)}.npy"
            if path.exists():
                self.from_cache = True
                return np.load(path)
        emb = self._embed([d.full_text for d in self.docs])
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        return emb

    def _search(self, query: str, k: int) -> list[Hit]:
        q = self._embed([self.query_prefix + query])[0]
        scores = self.embeddings @ q
        k = min(k, len(self.docs))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        return [Hit(self.docs[i], float(scores[i]), {"dense": float(scores[i])}) for i in idx]

    async def asearch_traced(self, query: str, k: int) -> SearchResult:
        check_k(k)
        t0 = perf_counter()
        hits = await asyncio.to_thread(self._search, query, k)
        return SearchResult(hits, timings={"retrieve": perf_counter() - t0})
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_retrievers.py -q && uv run ruff check .`
Expected: all pass.

- [ ] **Step 5: Commit and push**

```bash
git add src/jevragrank/retrievers tests/test_retrievers.py
git commit -m "Add BM25 and dense retrievers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 4: Baseline rerankers (cross-encoder, same-size LLM)

**Files:**
- Create: `src/jevragrank/rerankers/__init__.py`, `cross_encoder.py`, `llm.py`
- Test: `tests/test_baseline_rerankers.py`, `tests/test_llm_gpu.py`

**Interfaces:**
- Consumes: `Reranker`, `rank_hits`, `check_k` (Task 1); `passage` (Task 1).
- Produces:
  - `CrossEncoderReranker(*, model_name="BAAI/bge-reranker-base", max_passage_tokens=512, batch_size=32, device=None, scorer=None)`. Stage `"ce"`; `scorer: Callable[[list[tuple[str, str]]], Sequence[float]]`.
  - `LLMReranker(*, model_name="Qwen/Qwen3.5-2B-Base", max_passage_tokens=512, batch_size=8, device="cuda", scorer=None)`. Stage `"llm"`; `scorer: Callable[[list[str]], Sequence[float]]` takes prompts and returns expected ratings in 0–1.
  - `RATING_PROMPT`, `digit_plan(tokenizer, cue="Rating:", n=5) -> tuple[list[int], list[int]]`, `expected_rating(digit_logits) -> float`.

- [ ] **Step 1: Write the failing tests** — `tests/test_baseline_rerankers.py`

```python
import math

import pytest

from jevragrank.rerankers import (RATING_PROMPT, CrossEncoderReranker, LLMReranker, digit_plan,
                                  expected_rating)
from jevragrank.types import Doc, Hit

HITS = [Hit(Doc(str(i), t), 1.0 - i / 10) for i, t in
        enumerate(["dogs bark", "vitamin d fractures", "cats purr"])]


def overlap(query, text):
    return len(set(query.split()) & set(text.split()))


def test_cross_encoder_ranks_by_scorer():
    ce = CrossEncoderReranker(scorer=lambda pairs: [overlap(q, p) for q, p in pairs])
    out = ce.rerank("vitamin d", HITS, k=2)
    assert [h.doc.id for h in out] == ["1", "0"]
    assert out[0].stage_scores["ce"] == 2


def test_cross_encoder_empty():
    assert CrossEncoderReranker(scorer=lambda p: []).rerank("q", [], 3) == []


def test_llm_reranker_builds_prompts_and_ranks():
    seen = []

    def scorer(prompts):
        seen.extend(prompts)
        return [overlap("vitamin d", p.split("Passage:")[1]) / 2 for p in prompts]

    out = LLMReranker(scorer=scorer).rerank("vitamin d", HITS, k=1)
    assert out[0].doc.id == "1"
    assert seen[0] == RATING_PROMPT.format(query="vitamin d", passage="dogs bark")
    assert seen[0].endswith("Rating:")


class SplitTok:
    """Tokenizer that splits on spaces but keeps ' ' as its own token before digits."""

    vocab: dict = {}

    def encode(self, text, add_special_tokens=False):
        out = []
        for part in text.replace(" ", "\0 \0").split("\0"):
            if part:
                out.append(self.vocab.setdefault(part, len(self.vocab)))
        return out


class MergeTok(SplitTok):
    """Tokenizer that merges the space into the digit token."""

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(p, len(self.vocab)) for p in text.replace(" ", "\0 ").split("\0") if p]


def test_digit_plan_with_separate_space_token():
    tok = SplitTok()
    suffix, digits = digit_plan(tok)
    assert suffix == tok.encode(" ")
    assert digits == [tok.encode(str(d))[0] for d in range(5)]


def test_digit_plan_with_merged_space_digit():
    tok = MergeTok()
    suffix, digits = digit_plan(tok)
    assert suffix == []
    assert len(set(digits)) == 5


def test_expected_rating():
    assert expected_rating([0, 0, 0, 0, 50]) == pytest.approx(1.0)
    assert expected_rating([50, 0, 0, 0, 0]) == pytest.approx(0.0)
    assert expected_rating([1, 1, 1, 1, 1]) == pytest.approx(0.5)
    assert not math.isnan(expected_rating([-1e4] * 5))
```

`tests/test_llm_gpu.py`:
```python
import pytest

from jevragrank.rerankers import LLMReranker
from jevragrank.types import Doc, Hit


@pytest.mark.gpu
@pytest.mark.slow
def test_qwen_base_prefers_relevant_passage():
    hits = [Hit(Doc("off", "The Eiffel Tower is in Paris and was finished in 1889."), 0.5),
            Hit(Doc("on", "Daily vitamin D3 did not reduce fracture risk in a large "
                          "randomized trial of older adults."), 0.5)]
    out = LLMReranker().rerank("does vitamin D reduce fracture risk?", hits, k=2)
    assert out[0].doc.id == "on"
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_baseline_rerankers.py -q`
Expected: `ModuleNotFoundError: jevragrank.rerankers`.

- [ ] **Step 3: Implement**

`src/jevragrank/rerankers/cross_encoder.py`:
```python
from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence

from ..base import Reranker, check_k, rank_hits
from ..text import passage
from ..types import Hit

PairScorer = Callable[[list[tuple[str, str]]], Sequence[float]]


def _cross_encoder_scorer(model_name: str, device: str | None, batch_size: int) -> PairScorer:
    model = None

    def score(pairs: list[tuple[str, str]]) -> Sequence[float]:
        nonlocal model
        if model is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(model_name, max_length=512, device=device)
        return model.predict(pairs, batch_size=batch_size, show_progress_bar=False).tolist()

    return score


class CrossEncoderReranker(Reranker):
    def __init__(self, *, model_name: str = "BAAI/bge-reranker-base",
                 max_passage_tokens: int = 512, batch_size: int = 32,
                 device: str | None = None, scorer: PairScorer | None = None):
        self.max_passage_tokens = max_passage_tokens
        self._score = scorer or _cross_encoder_scorer(model_name, device, batch_size)
        self._lock = threading.Lock()

    def _locked(self, pairs):
        with self._lock:
            return list(self._score(pairs))

    async def arerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        check_k(k)
        if not hits:
            return []
        pairs = [(query, passage(h.doc, self.max_passage_tokens)) for h in hits]
        scores = await asyncio.to_thread(self._locked, pairs)
        return rank_hits(hits, scores, stage="ce", k=k)
```

`src/jevragrank/rerankers/llm.py`:
```python
from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Sequence

import numpy as np

from ..base import Reranker, check_k, rank_hits
from ..text import passage
from ..types import Hit

RATING_PROMPT = """Rate how well a passage answers a search query.

Query: {query}

Passage: {passage}

Scale:
0 = irrelevant
1 = same topic, doesn't help
2 = partially answers
3 = mostly answers
4 = fully answers

Rating:"""

PromptScorer = Callable[[list[str]], Sequence[float]]


def digit_plan(tokenizer, cue: str = "Rating:", n: int = 5) -> tuple[list[int], list[int]]:
    """Tokens to append after the cue, and the token id of each rating digit."""
    base = tokenizer.encode(cue, add_special_tokens=False)
    seqs = [tokenizer.encode(f"{cue} {d}", add_special_tokens=False) for d in range(n)]
    prefix = seqs[0][:-1]
    if any(s[:-1] != prefix for s in seqs) or prefix[: len(base)] != base:
        raise ValueError("tokenizer splits rating digits unexpectedly")
    return prefix[len(base):], [s[-1] for s in seqs]


def expected_rating(digit_logits: Sequence[float]) -> float:
    x = np.asarray(digit_logits, dtype=np.float64)
    p = np.exp(x - x.max())
    p /= p.sum()
    return float((p * np.arange(len(p))).sum() / (len(p) - 1))


def _hf_scorer(model_name: str, device: str, batch_size: int) -> PromptScorer:
    state: dict = {}

    def load() -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
        suffix, digits = digit_plan(tok)
        state.update(torch=torch, tok=tok, model=model.to(device).eval(), suffix=suffix,
                     digits=digits)

    def score(prompts: list[str]) -> list[float]:
        if not state:
            load()
        torch, tok, model = state["torch"], state["tok"], state["model"]
        out: list[float] = []
        for i in range(0, len(prompts), batch_size):
            seqs = [tok.encode(p, add_special_tokens=False) + state["suffix"]
                    for p in prompts[i:i + batch_size]]
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, s in enumerate(seqs):
                ids[r, width - len(s):] = torch.tensor(s)
                mask[r, width - len(s):] = 1
            with torch.inference_mode():
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            sel = logits[:, -1, state["digits"]].float().cpu().numpy()
            out.extend(expected_rating(row) for row in sel)
        return out

    return score


class LLMReranker(Reranker):
    """Baseline: prompt a causal LM for a 0-4 rating and read digit probabilities."""

    def __init__(self, *, model_name: str = "Qwen/Qwen3.5-2B-Base", max_passage_tokens: int = 512,
                 batch_size: int = 8, device: str = "cuda", scorer: PromptScorer | None = None):
        self.max_passage_tokens = max_passage_tokens
        self._score = scorer or _hf_scorer(model_name, device, batch_size)
        self._lock = threading.Lock()

    def _locked(self, prompts):
        with self._lock:
            return list(self._score(prompts))

    async def arerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        check_k(k)
        if not hits:
            return []
        prompts = [RATING_PROMPT.format(query=query, passage=passage(h.doc, self.max_passage_tokens))
                   for h in hits]
        scores = await asyncio.to_thread(self._locked, prompts)
        return rank_hits(hits, scores, stage="llm", k=k)
```

`src/jevragrank/rerankers/__init__.py`:
```python
from .cross_encoder import CrossEncoderReranker
from .llm import RATING_PROMPT, LLMReranker, digit_plan, expected_rating

__all__ = ["RATING_PROMPT", "CrossEncoderReranker", "LLMReranker", "digit_plan",
           "expected_rating"]
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_baseline_rerankers.py -q && uv run ruff check .`
Expected: all pass.

- [ ] **Step 5: Verify the real Qwen baseline on the GPU** (stop any decider server first)

```bash
uv sync --extra bench --extra dev
uv run pytest tests/test_llm_gpu.py -m "gpu" -q
```
Expected: PASS. If `digit_plan` raises for Qwen's tokenizer, print `tok.encode("Rating: 3")` and adapt `digit_plan`, then add a unit test mirroring that tokenization. If loading fails because transformers lacks the architecture, raise the floor in `pyproject.toml` (`transformers>=` the version whose release notes add Qwen3.5), re-sync and re-run.

- [ ] **Step 6: Commit and push**

```bash
git add src/jevragrank/rerankers tests/test_baseline_rerankers.py tests/test_llm_gpu.py pyproject.toml uv.lock
git commit -m "Add cross-encoder and same-size LLM baseline rerankers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 5: JevReranker and the JevRAGRank pipeline

**Files:**
- Create: `src/jevragrank/rerankers/jev.py`, `src/jevragrank/pipelines.py`
- Modify: `src/jevragrank/rerankers/__init__.py`, `src/jevragrank/__init__.py`
- Test: `tests/test_jev_reranker.py`, `tests/test_pipelines.py`

**Interfaces:**
- Consumes: `JevClient`, `JevError`, the parsers (Task 2); `Reranker`, `Searchable`, `rank_hits`, `check_k`, `passage`, `estimate_tokens` (Task 1); `DenseRetriever` (Task 3).
- Produces:
  - Constants: `DEFAULT_LEVELS`, `query_state(query) -> str` (returns `"Query: {query}"`), `STRATEGIES`.
  - `JevReranker(jev, *, strategy="score", batch_size=12, max_passage_tokens=512, blend=0.0, levels=DEFAULT_LEVELS, instructions=None, on_error="raise", max_choice_options=255, max_choice_tokens=24000)`. Stage key `"jev"`.
  - `RetrieveThenRerank(retriever, reranker, *, candidates=50)`. `SearchResult.candidates` holds the retrieved ids; timings are `retrieve`, `rerank`.
  - `JevRAGRank(docs, *, jev, candidates=50, strategy="score", retriever=None, dense_model="BAAI/bge-base-en-v1.5", cache_dir=None, device=None, **reranker_kwargs)`.

- [ ] **Step 1: Write the failing reranker tests** — `tests/test_jev_reranker.py`

```python
import httpx
import pytest

from jevragrank.jev_client import JevClient, JevRequestError
from jevragrank.rerankers import DEFAULT_LEVELS, JevReranker
from jevragrank.types import Doc, Hit

TEXTS = ["dogs bark loudly", "vitamin d reduces fractures", "cats purr", "vitamin pills",
         "bone fractures heal"]
HITS = [Hit(Doc(str(i), t), score=1.0 - i / 10, stage_scores={"dense": 1.0 - i / 10})
        for i, t in enumerate(TEXTS)]
QUERY = "vitamin d fractures"


def test_score_strategy_batches_and_ranks(jev_client, fake_jev):
    out = JevReranker(jev_client, batch_size=2).rerank(QUERY, HITS, k=3)
    assert out[0].doc.id == "1"
    assert len(fake_jev.bodies) == 3
    first = fake_jev.bodies[0]
    assert first["state"] == f"Query: {QUERY}"
    assert list(first["questions"]) == ["c0", "c1"]
    q = first["questions"]["c0"]
    assert q["type"] == "score" and q["criteria"] == list(DEFAULT_LEVELS)
    assert q["instructions"].endswith("Passage:\ndogs bark loudly")
    assert out[0].stage_scores == {"dense": 0.9, "jev": 1.0}


def test_ties_broken_by_retrieval_score(jev_client):
    out = JevReranker(jev_client, strategy="noul").rerank("zzz", HITS, k=5)
    assert [h.doc.id for h in out] == ["0", "1", "2", "3", "4"]


def test_noul_strategy(jev_client, fake_jev):
    out = JevReranker(jev_client, strategy="noul").rerank(QUERY, HITS, k=1)
    assert out[0].doc.id == "1"
    assert fake_jev.bodies[0]["questions"]["c0"]["type"] == "noul"


def test_choice_strategy_single_listwise_question(jev_client, fake_jev):
    out = JevReranker(jev_client, strategy="choice").rerank(QUERY, HITS, k=2)
    assert out[0].doc.id == "1"
    assert len(fake_jev.bodies) == 1
    pick = fake_jev.bodies[0]["questions"]["pick"]
    assert pick["type"] == "choice" and set(pick["criteria"]) == {f"p{i}" for i in range(5)}


def test_choice_splits_when_over_option_limit(jev_client, fake_jev):
    JevReranker(jev_client, strategy="choice", max_choice_options=2).rerank(QUERY, HITS, k=5)
    assert [len(b["questions"]["pick"]["criteria"]) for b in fake_jev.bodies] == [2, 2]


def test_choice_splits_on_token_budget(jev_client, fake_jev):
    JevReranker(jev_client, strategy="choice", max_choice_tokens=10).rerank(QUERY, HITS, k=5)
    assert len(fake_jev.bodies) >= 2


def test_blend_mixes_in_retrieval_score(jev_client):
    out = JevReranker(jev_client, strategy="noul", blend=1.0).rerank(QUERY, HITS, k=5)
    assert [h.doc.id for h in out] == ["0", "1", "2", "3", "4"]


def test_error_raises_by_default(fake_jev):
    fake_jev.fail_next = [422]
    c = JevClient("http://fake", transport=httpx.MockTransport(fake_jev.handler), backoff=0)
    with pytest.raises(JevRequestError):
        JevReranker(c, batch_size=5).rerank(QUERY, HITS, k=3)


def test_fallback_keeps_retrieval_order_and_flags(fake_jev):
    fake_jev.fail_next = [422]
    c = JevClient("http://fake", transport=httpx.MockTransport(fake_jev.handler), backoff=0)
    out = JevReranker(c, batch_size=5, on_error="fallback").rerank(QUERY, HITS, k=5)
    assert [h.doc.id for h in out] == ["0", "1", "2", "3", "4"]
    assert all(h.degraded for h in out)


@pytest.mark.parametrize("kwargs", [{"strategy": "vote"}, {"levels": ("one",)}, {"blend": 2},
                                    {"on_error": "ignore"}, {"batch_size": 0},
                                    {"max_choice_options": 300}])
def test_invalid_config(jev_client, kwargs):
    with pytest.raises(ValueError):
        JevReranker(jev_client, **kwargs)
```

- [ ] **Step 2: Write the failing pipeline tests** — `tests/test_pipelines.py`

```python
import numpy as np
import pytest

from jevragrank.pipelines import JevRAGRank, RetrieveThenRerank
from jevragrank.retrievers import DenseRetriever
from jevragrank.types import Doc

DOCS = [Doc(str(i), t) for i, t in enumerate(
    ["dogs bark loudly", "vitamin d reduces fractures", "cats purr", "vitamin pills",
     "bone fractures heal", "weather is sunny"])]
VOCAB = ["dogs", "vitamin", "fractures", "cats", "bone", "weather"]


def enc(texts):
    return np.array([[t.count(w) for w in VOCAB] for t in texts], dtype=np.float32) + 1e-3


def dense():
    return DenseRetriever(DOCS, encoder=enc)


def test_jevragrank_end_to_end(jev_client, fake_jev):
    rag = JevRAGRank(DOCS, jev=jev_client, candidates=4, retriever=dense())
    import asyncio
    res = asyncio.run(rag.asearch_traced("vitamin d fractures", 2))
    assert res.hits[0].doc.id == "1"
    assert len(res.candidates) == 4
    assert set(res.timings) == {"retrieve", "rerank"}
    total_questions = sum(len(b["questions"]) for b in fake_jev.bodies)
    assert total_questions == 4


def test_candidates_must_cover_k(jev_client):
    rag = JevRAGRank(DOCS, jev=jev_client, candidates=2, retriever=dense())
    with pytest.raises(ValueError, match="candidates"):
        rag.search("vitamin", k=3)


def test_retrieve_then_rerank_rejects_bad_candidates():
    with pytest.raises(ValueError):
        RetrieveThenRerank(dense(), None, candidates=0)
```

- [ ] **Step 3: Run and confirm failure**

Run: `uv run pytest tests/test_jev_reranker.py tests/test_pipelines.py -q`
Expected: `ImportError` for `JevReranker` / `jevragrank.pipelines`.

- [ ] **Step 4: Implement `src/jevragrank/rerankers/jev.py`**

If Task 0's notes recorded a different accepted request form, use it in `_question` and the choice body.

```python
from __future__ import annotations

import asyncio
from collections.abc import Sequence

from ..base import Reranker, check_k, rank_hits
from ..jev_answers import choice_probabilities, get_answer, noul_probability, score_fraction
from ..jev_client import JevClient, JevError
from ..text import estimate_tokens, passage
from ..types import Hit

DEFAULT_LEVELS = ("irrelevant", "same topic, doesn't help", "partially answers",
                  "mostly answers", "fully answers")
STRATEGIES = ("score", "noul", "choice")
DEFAULT_INSTRUCTIONS = {
    "score": "How well does this passage answer the query?",
    "noul": "Does this passage help answer the query?",
    "choice": "Which passage best answers the query?",
}


def query_state(query: str) -> str:
    return f"Query: {query}"


def _minmax(values: Sequence[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class JevReranker(Reranker):
    """Rerank candidates with typed Jev questions (score, noul or listwise choice)."""

    def __init__(self, jev: JevClient, *, strategy: str = "score", batch_size: int = 12,
                 max_passage_tokens: int = 512, blend: float = 0.0,
                 levels: Sequence[str] = DEFAULT_LEVELS, instructions: str | None = None,
                 on_error: str = "raise", max_choice_options: int = 255,
                 max_choice_tokens: int = 24_000):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if not 2 <= len(levels) <= 10:
            raise ValueError("a score scale needs 2-10 levels")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be within [0, 1]")
        if on_error not in ("raise", "fallback"):
            raise ValueError("on_error must be 'raise' or 'fallback'")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 2 <= max_choice_options <= 255:
            raise ValueError("max_choice_options must be within 2-255")
        self.jev = jev
        self.strategy = strategy
        self.batch_size = batch_size
        self.max_passage_tokens = max_passage_tokens
        self.blend = blend
        self.levels = list(levels)
        self.instructions = instructions or DEFAULT_INSTRUCTIONS[strategy]
        self.on_error = on_error
        self.max_choice_options = max_choice_options
        self.max_choice_tokens = max_choice_tokens

    async def arerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        check_k(k)
        if not hits:
            return []
        passages = [passage(h.doc, self.max_passage_tokens) for h in hits]
        if self.strategy == "choice":
            scores = await self._choice_scores(query, passages)
        else:
            scores = await self._pointwise_scores(query, passages)
        retrieval = _minmax([h.score for h in hits])
        failed = [s is None for s in scores]
        final = [retrieval[i] if failed[i]
                 else (1 - self.blend) * scores[i] + self.blend * retrieval[i]
                 for i in range(len(hits))]
        return rank_hits(hits, final, stage="jev", k=k, degraded=failed)

    def _question(self, text: str) -> dict:
        q = {"type": self.strategy, "instructions": f"{self.instructions}\n\nPassage:\n{text}"}
        if self.strategy == "score":
            q["criteria"] = self.levels
        return q

    def _parse(self, answer) -> float:
        if self.strategy == "score":
            return score_fraction(answer, self.levels)
        return noul_probability(answer)

    async def _guarded(self, coro) -> dict[int, float] | None:
        try:
            return await coro
        except JevError:
            if self.on_error == "raise":
                raise
            return None

    async def _pointwise_scores(self, query: str, passages: list[str]) -> list[float | None]:
        n = len(passages)
        batches = [list(range(i, min(i + self.batch_size, n))) for i in range(0, n, self.batch_size)]

        async def run(batch: list[int]) -> dict[int, float]:
            answers = await self.jev.ask(query_state(query),
                                         {f"c{j}": self._question(passages[j]) for j in batch})
            return {j: self._parse(get_answer(answers, f"c{j}")) for j in batch}

        scores: list[float | None] = [None] * n
        for result in await asyncio.gather(*(self._guarded(run(b)) for b in batches)):
            if result:
                for j, s in result.items():
                    scores[j] = s
        return scores

    def _choice_chunks(self, passages: list[str]) -> list[list[int]]:
        chunks: list[list[int]] = [[]]
        tokens = 0
        for j, p in enumerate(passages):
            t = estimate_tokens(p)
            full = len(chunks[-1]) >= self.max_choice_options
            if chunks[-1] and (full or tokens + t > self.max_choice_tokens):
                chunks.append([])
                tokens = 0
            chunks[-1].append(j)
            tokens += t
        return chunks

    async def _choice_scores(self, query: str, passages: list[str]) -> list[float | None]:
        chunks = self._choice_chunks(passages)

        async def run(chunk: list[int]) -> dict[int, float]:
            if len(chunk) == 1:
                return {chunk[0]: 1.0}
            criteria = {f"p{j}": passages[j] for j in chunk}
            answers = await self.jev.ask(query_state(query), {"pick": {
                "type": "choice", "instructions": self.instructions, "criteria": criteria}})
            probs = choice_probabilities(get_answer(answers, "pick"))
            top = max(probs.values(), default=0.0) if len(chunks) > 1 else 1.0
            top = top if top > 0 else 1.0
            return {j: probs.get(f"p{j}", 0.0) / top for j in chunk}

        scores: list[float | None] = [None] * len(passages)
        for result in await asyncio.gather(*(self._guarded(run(c)) for c in chunks)):
            if result:
                for j, s in result.items():
                    scores[j] = s
        return scores
```

- [ ] **Step 5: Implement `src/jevragrank/pipelines.py`**

```python
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from .base import Reranker, Searchable, check_k
from .jev_client import JevClient
from .rerankers.jev import JevReranker
from .retrievers.dense import DenseRetriever
from .types import Doc, SearchResult


class RetrieveThenRerank(Searchable):
    """Retrieve `candidates` hits, then rerank them down to k."""

    def __init__(self, retriever: Searchable, reranker: Reranker, *, candidates: int = 50):
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        self.retriever = retriever
        self.reranker = reranker
        self.candidates = candidates

    async def asearch_traced(self, query: str, k: int) -> SearchResult:
        check_k(k)
        if self.candidates < k:
            raise ValueError(f"candidates ({self.candidates}) must be >= k ({k})")
        t0 = perf_counter()
        retrieved = await self.retriever.asearch_traced(query, self.candidates)
        t1 = perf_counter()
        hits = await self.reranker.arerank(query, retrieved.hits, k)
        t2 = perf_counter()
        return SearchResult(hits, candidates=[h.doc.id for h in retrieved.hits],
                            timings={"retrieve": t1 - t0, "rerank": t2 - t1})


class JevRAGRank(RetrieveThenRerank):
    """Dense embedding recall, then Jev reranking."""

    def __init__(self, docs: Sequence[Doc], *, jev: JevClient, candidates: int = 50,
                 strategy: str = "score", retriever: Searchable | None = None,
                 dense_model: str = "BAAI/bge-base-en-v1.5",
                 cache_dir: str | Path | None = None, device: str | None = None,
                 **reranker_kwargs):
        retriever = retriever or DenseRetriever(docs, model_name=dense_model,
                                                cache_dir=cache_dir, device=device)
        super().__init__(retriever, JevReranker(jev, strategy=strategy, **reranker_kwargs),
                         candidates=candidates)
```

- [ ] **Step 6: Update exports**

`src/jevragrank/rerankers/__init__.py`: add `from .jev import DEFAULT_LEVELS, STRATEGIES, JevReranker, query_state` and add those names to `__all__`.

`src/jevragrank/__init__.py`: add `from .pipelines import JevRAGRank, RetrieveThenRerank` and `from .rerankers.jev import JevReranker`, and add the names to `__all__`.

- [ ] **Step 7: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all pass.

- [ ] **Step 8: Commit and push**

```bash
git add src tests/test_jev_reranker.py tests/test_pipelines.py
git commit -m "Add JevReranker (score/noul/choice) and JevRAGRank pipeline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 6: JevRank — the Jev-only choice tournament

**Files:**
- Create: `src/jevragrank/tournament.py`
- Modify: `src/jevragrank/__init__.py`
- Test: `tests/test_tournament.py`

**Interfaces:**
- Consumes: `JevClient`, parsers (Task 2); `Searchable`, `check_k`, `truncate` (Task 1); `JevReranker`, `query_state` (Task 5).
- Produces: `JevRank(docs, *, jev, group_size=64, preview_tokens=(96, 256), survivors=(3, 12), final_pool=50, final_reranker=None, instructions="Which passage most likely answers the query?", seed=0)`, a `Searchable`.
  - `SearchResult.candidates` holds the round-1 survivor ids (all ids if round 1 did not run).
  - Timings are `round1`, `round2`, …, `final`.
  - Finalist hits carry stage keys `"tournament"` and `"jev"`.
  - Eliminated hits carry `"tournament"` and `"round_reached"`, with negative scores ordered by round, then probability.

- [ ] **Step 1: Write the failing tests** — `tests/test_tournament.py`

```python
import asyncio

import pytest

from jevragrank.tournament import JevRank
from jevragrank.types import Doc

RELEVANT = {"r1": "vitamin d fractures trial", "r2": "vitamin d bone fractures"}


def corpus(n_noise: int) -> list[Doc]:
    docs = [Doc(f"n{i}", f"noise document number {i} about weather") for i in range(n_noise)]
    docs[7:7] = [Doc(k, v) for k, v in RELEVANT.items()]
    return docs


def choice_bodies(fake):
    return [b for b in fake.bodies if "pick" in b["questions"]]


def test_tournament_finds_relevant_docs(jev_client, fake_jev):
    jr = JevRank(corpus(200), jev=jev_client, group_size=16, survivors=(2, 4),
                 preview_tokens=(20, 40), final_pool=10)
    res = asyncio.run(jr.asearch_traced("vitamin d fractures", 2))
    assert {h.doc.id for h in res.hits} == {"r1", "r2"}
    assert {"r1", "r2"} <= set(res.candidates)
    assert set(res.timings) == {"round1", "round2", "final"}


def test_request_counts_follow_group_math(jev_client, fake_jev):
    docs = corpus(126)  # 128 docs
    JevRank(docs, jev=jev_client, group_size=16, survivors=(2, 4), preview_tokens=(20, 40),
            final_pool=10).search("vitamin d fractures", 5)
    choices = choice_bodies(fake_jev)
    # round 1: 128/16 = 8 groups -> 16 survivors; round 2: 1 group of 16 -> 4 survivors
    assert [len(b["questions"]["pick"]["criteria"]) for b in choices] == [16] * 8 + [16]
    score_qs = sum(len(b["questions"]) for b in fake_jev.bodies if "pick" not in b["questions"])
    assert score_qs == 4


def test_previews_are_truncated_per_round(jev_client, fake_jev):
    long_docs = [Doc(str(i), "word " * 400) for i in range(40)]
    JevRank(long_docs, jev=jev_client, group_size=10, survivors=(2, 2), preview_tokens=(10, 30),
            final_pool=3).search("word", 1)
    r1 = choice_bodies(fake_jev)[0]["questions"]["pick"]["criteria"]
    assert max(len(t) for t in r1.values()) <= 10 * 4 + 2


def test_small_corpus_skips_rounds(jev_client, fake_jev):
    docs = corpus(3)
    res = asyncio.run(JevRank(docs, jev=jev_client, final_pool=50).asearch_traced("vitamin", 2))
    assert choice_bodies(fake_jev) == []
    assert res.candidates == [d.id for d in docs]
    assert set(res.timings) == {"final"}


def test_eliminated_docs_fill_ranking_below_finalists(jev_client):
    jr = JevRank(corpus(60), jev=jev_client, group_size=16, survivors=(1,), preview_tokens=(20,),
                 final_pool=4)
    hits = jr.search("vitamin d fractures", 10)
    assert len(hits) == 10
    finalists = [h for h in hits if "round_reached" not in h.stage_scores]
    assert len(finalists) == 4
    assert all(h.score < 0 for h in hits[4:])
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_shuffle_is_deterministic_per_query(jev_client, fake_jev):
    jr = JevRank(corpus(60), jev=jev_client, group_size=16, survivors=(2,), preview_tokens=(20,),
                 final_pool=10)
    jr.search("vitamin", 3)
    first = [b["questions"]["pick"]["criteria"] for b in choice_bodies(fake_jev)]
    fake_jev.bodies.clear()
    jr.search("vitamin", 3)
    assert [b["questions"]["pick"]["criteria"] for b in choice_bodies(fake_jev)] == first


@pytest.mark.parametrize("kwargs", [{"group_size": 1}, {"group_size": 256},
                                    {"survivors": (3,), "preview_tokens": (10, 20)},
                                    {"survivors": (0,), "preview_tokens": (10,)},
                                    {"final_pool": 0}])
def test_invalid_config(jev_client, kwargs):
    with pytest.raises(ValueError):
        JevRank(corpus(5), jev=jev_client, **kwargs)
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_tournament.py -q`
Expected: `ModuleNotFoundError: jevragrank.tournament`.

- [ ] **Step 3: Implement `src/jevragrank/tournament.py`**

```python
from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from time import perf_counter

from .base import Reranker, Searchable, check_k
from .jev_answers import choice_probabilities, get_answer
from .jev_client import JevClient
from .rerankers.jev import JevReranker, query_state
from .text import truncate
from .types import Doc, Hit, SearchResult


class JevRank(Searchable):
    """Embedding-free retrieval: batched Jev `choice` rounds, then a Jev `score` final."""

    def __init__(self, docs: Sequence[Doc], *, jev: JevClient, group_size: int = 64,
                 preview_tokens: Sequence[int] = (96, 256), survivors: Sequence[int] = (3, 12),
                 final_pool: int = 50, final_reranker: Reranker | None = None,
                 instructions: str = "Which passage most likely answers the query?",
                 seed: int = 0):
        if not docs:
            raise ValueError("docs must not be empty")
        if not 2 <= group_size <= 255:
            raise ValueError("group_size must be within 2-255")
        if not survivors or len(survivors) != len(preview_tokens):
            raise ValueError("survivors and preview_tokens need one entry per round")
        if min(survivors) < 1 or final_pool < 1:
            raise ValueError("survivors and final_pool must be positive")
        self.docs = list(docs)
        self.jev = jev
        self.group_size = group_size
        self.survivors = list(survivors)
        self.final_pool = final_pool
        self.instructions = instructions
        self.seed = seed
        self.final_reranker = final_reranker or JevReranker(jev, strategy="score")
        self.previews = [[truncate(d.full_text, t) for d in self.docs] for t in preview_tokens]

    async def _judge(self, query: str, group: list[int], rnd: int) -> dict[int, float]:
        if len(group) == 1:
            return {group[0]: 1.0}
        criteria = {f"p{n}": self.previews[rnd][i] for n, i in enumerate(group)}
        answers = await self.jev.ask(query_state(query), {"pick": {
            "type": "choice", "instructions": self.instructions, "criteria": criteria}})
        probs = choice_probabilities(get_answer(answers, "pick"))
        return {i: probs.get(f"p{n}", 0.0) for n, i in enumerate(group)}

    async def asearch_traced(self, query: str, k: int) -> SearchResult:
        check_k(k)
        timings: dict[str, float] = {}
        pool = list(range(len(self.docs)))
        last_prob = {i: 1.0 for i in pool}
        eliminated: list[tuple[int, float, int]] = []  # (round reached, probability, doc index)
        candidates: list[str] | None = None
        rounds = len(self.survivors)
        for rnd, keep in enumerate(self.survivors):
            if len(pool) <= self.final_pool:
                break
            t0 = perf_counter()
            order = pool[:]
            random.Random(f"{self.seed}:{rnd}:{query}").shuffle(order)
            groups = [order[i:i + self.group_size] for i in range(0, len(order), self.group_size)]
            results = await asyncio.gather(*(self._judge(query, g, rnd) for g in groups))
            pool = []
            for group, probs in zip(groups, results):
                ranked = sorted(group, key=lambda i: (-probs[i], i))
                pool.extend(ranked[:keep])
                eliminated.extend((rnd, probs[i], i) for i in ranked[keep:])
                last_prob.update(probs)
            timings[f"round{rnd + 1}"] = perf_counter() - t0
            if candidates is None:
                candidates = [self.docs[i].id for i in pool]
        if len(pool) > self.final_pool:
            pool.sort(key=lambda i: (-last_prob[i], i))
            eliminated.extend((rounds, last_prob[i], i) for i in pool[self.final_pool:])
            pool = pool[: self.final_pool]
        t0 = perf_counter()
        finalists = [Hit(self.docs[i], last_prob[i], {"tournament": last_prob[i]}) for i in pool]
        hits = await self.final_reranker.arerank(query, finalists, len(finalists))
        timings["final"] = perf_counter() - t0
        hits = hits[:k]
        if len(hits) < k:
            eliminated.sort(key=lambda e: (-e[0], -e[1], e[2]))
            for rnd, p, i in eliminated[: k - len(hits)]:
                score = (rnd - rounds) - 1 + 0.5 * p
                hits.append(Hit(self.docs[i], score, {"tournament": p, "round_reached": rnd}))
        if candidates is None:
            candidates = [d.id for d in self.docs]
        return SearchResult(hits, candidates=candidates, timings=timings)
```

- [ ] **Step 4: Export.** In `src/jevragrank/__init__.py` add `from .tournament import JevRank` and `"JevRank"` to `__all__`.

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all pass.

- [ ] **Step 6: Commit and push**

```bash
git add src tests/test_tournament.py
git commit -m "Add JevRank embedding-free choice tournament

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 7: BEIR dataset loading

**Files:**
- Create: `bench/datasets.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: `Doc` (Task 1).
- Produces:
  - `Dataset(name, split, docs: list[Doc], queries: dict[str, str], qrels: dict[str, dict[str, int]], note: str = "")`.
  - `load_beir(name, split="test", data_dir="data") -> Dataset`.
  - `sample_queries(ds, n, seed=0) -> Dataset`.
  - `subsample_corpus(ds, n_docs, seed=0) -> Dataset`.
  - `BEIR_URL`.

- [ ] **Step 1: Write the failing tests** — `tests/test_datasets.py`

```python
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
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_datasets.py -q`
Expected: `ModuleNotFoundError: bench.datasets`.

- [ ] **Step 3: Implement `bench/datasets.py`**

```python
from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

from jevragrank.types import Doc

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


@dataclass
class Dataset:
    name: str
    split: str
    docs: list[Doc]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    note: str = ""


def _download(name: str, data_dir: Path) -> Path:
    path = data_dir / f"{name}.zip"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(BEIR_URL.format(name=name), tmp)
        tmp.replace(path)
    return path


def load_beir(name: str, split: str = "test", data_dir: str | Path = "data") -> Dataset:
    with zipfile.ZipFile(_download(name, Path(data_dir))) as z:
        qrels_name = f"{name}/qrels/{split}.tsv"
        if qrels_name not in z.namelist():
            splits = sorted(n.rsplit("/", 1)[1][:-4] for n in z.namelist()
                            if n.startswith(f"{name}/qrels/") and n.endswith(".tsv"))
            raise ValueError(f"{name} has no {split!r} split; available: {splits}")
        docs = []
        for line in z.read(f"{name}/corpus.jsonl").decode("utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                docs.append(Doc(str(o["_id"]), o.get("text") or "", o.get("title") or ""))
        all_queries = {}
        for line in z.read(f"{name}/queries.jsonl").decode("utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                all_queries[str(o["_id"])] = o["text"]
        qrels: dict[str, dict[str, int]] = {}
        for line in z.read(qrels_name).decode("utf-8").splitlines()[1:]:
            if line.strip():
                qid, did, score = line.split("\t")[:3]
                qrels.setdefault(qid, {})[did] = int(score)
    qrels = {q: r for q, r in qrels.items() if q in all_queries and any(s > 0 for s in r.values())}
    queries = {q: all_queries[q] for q in sorted(qrels)}
    return Dataset(name, split, docs, queries, {q: qrels[q] for q in queries})


def sample_queries(ds: Dataset, n: int, seed: int = 0) -> Dataset:
    if n >= len(ds.queries):
        return ds
    qids = sorted(random.Random(seed).sample(sorted(ds.queries), n))
    note = f"{n} of {len(ds.queries)} queries (seed {seed})"
    return replace(ds, queries={q: ds.queries[q] for q in qids},
                   qrels={q: ds.qrels[q] for q in qids}, note=note)


def subsample_corpus(ds: Dataset, n_docs: int, seed: int = 0) -> Dataset:
    relevant = {d for rels in ds.qrels.values() for d, s in rels.items() if s > 0}
    must = [d for d in ds.docs if d.id in relevant]
    if n_docs < len(must):
        raise ValueError(f"n_docs={n_docs} is below the {len(must)} relevant docs")
    rest = [d for d in ds.docs if d.id not in relevant]
    keep = {d.id for d in must} | {d.id for d in random.Random(seed).sample(rest, n_docs - len(must))}
    note = "; ".join(filter(None, [ds.note, f"corpus {n_docs} of {len(ds.docs)} docs"]))
    return replace(ds, docs=[d for d in ds.docs if d.id in keep], note=note)
```

- [ ] **Step 4: Run tests and verify real data**

Run: `uv run pytest tests/test_datasets.py -q && uv run pytest tests/test_datasets.py -m slow -q`
Expected: all pass. The slow test downloads SciFact to `data/` and confirms 5,183 docs and 300 queries. If the zip layout differs, fix the paths in `load_beir` and add a matching case to `make_zip`.

- [ ] **Step 5: Commit and push**

```bash
git add bench/datasets.py tests/test_datasets.py
git commit -m "Add BEIR loader with query sampling and corpus subsampling

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 8: Metrics, confidence intervals and significance

**Files:**
- Create: `bench/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `METRICS = ["ndcg@10", "recall@10", "mrr@10"]`.
  - `per_query(qrels, run: dict[str, list[str]]) -> dict[str, dict[str, float]]`. The run maps qid → ranked doc ids; the result maps metric → qid → value, covering every qid in qrels (0 if missing from the run).
  - `bootstrap_ci(values, n=1000, seed=0, alpha=0.05) -> tuple[float, float]`.
  - `randomization_test(a, b, n=10000, seed=0) -> float`.
  - `candidate_recall(qrels, candidates: dict[str, list[str]]) -> float`.
  - `latency_stats(seconds: list[float]) -> dict` with keys `p50`, `p95`, `mean` (seconds).

- [ ] **Step 1: Write the failing tests** — `tests/test_metrics.py`

```python
import numpy as np
import pytest

from bench.metrics import (bootstrap_ci, candidate_recall, latency_stats, per_query,
                           randomization_test)

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
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_metrics.py -q`
Expected: `ModuleNotFoundError: bench.metrics`.

- [ ] **Step 3: Implement `bench/metrics.py`**

```python
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
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_metrics.py -q && uv run ruff check .`
Expected: all pass.

- [ ] **Step 5: Commit and push**

```bash
git add bench/metrics.py tests/test_metrics.py
git commit -m "Add metrics, bootstrap CIs and randomization test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 9: Systems registry, resumable runner, summary and CLI

**Files:**
- Create: `bench/env.py`, `bench/systems.py`, `bench/runner.py`, `bench/summary.py`, `bench/cli.py`
- Test: `tests/test_runner.py`, `tests/test_summary.py`

**Interfaces:**
- Consumes: everything in `jevragrank`; `bench.datasets`; `bench.metrics`.
- Produces:
  - `environment(jev_url: str | None) -> dict`.
  - `DEFAULTS: dict[str, dict]`; `SYSTEMS = tuple(DEFAULTS)`.
  - `Context(docs, jev_url, cache_dir: Path, jev_cache_mode: str, device: str | None)`.
  - `Built(system: Searchable, jev: JevClient | None, index_seconds: float)`; `build(name, ctx, overrides) -> Built`.
  - `RunSpec(dataset, system, label="", overrides={}, split="test", sample=None, corpus_size=None, seed=0)` with `.name` and `.dir_name`.
  - `Paths(results=Path("results"), data=Path("data"), cache=Path(".cache"))`.
  - `prepare(spec, paths) -> Dataset`.
  - `async run_spec(spec, paths, *, k=10, warmup=5, jev_url="http://127.0.0.1:8000", cache_mode="write", device=None, max_consecutive_failures=3, builder=build) -> Path`.
  - `async throughput_spec(spec, paths, *, n=64, concurrency=16, k=10, jev_url=..., device=None, builder=build) -> dict`.
  - `summarize(paths, data_dir) -> dict`, which also writes `results/summary.json`.
  - `markdown_table(summary, dataset) -> str`.
  - `cli.main(argv=None)`.
- **Results layout:**
  - `results/{dir_name}/{name}.jsonl`: one line per query, `{"qid", "ranking": [[doc_id, score]...], "candidates", "timings", "latency", "jev_requests", "warm": true}`.
  - `{name}.meta.json` and `{name}.throughput.json` sit alongside.
  - `{name}.errors.log` records failures.

- [ ] **Step 1: Write the failing runner tests** — `tests/test_runner.py`

```python
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
```

- [ ] **Step 2: Write the failing summary tests** — `tests/test_summary.py`

```python
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
```

- [ ] **Step 3: Run and confirm failure**

Run: `uv run pytest tests/test_runner.py tests/test_summary.py -q`
Expected: `ModuleNotFoundError: bench.runner`.

- [ ] **Step 4: Implement `bench/env.py`**

```python
from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version

import httpx

PACKAGES = ["jevragrank", "torch", "transformers", "sentence-transformers", "bm25s", "ranx",
            "httpx", "numpy"]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment(jev_url: str | None = None) -> dict:
    info: dict = {"python": platform.python_version(), "platform": platform.platform(),
                  "packages": {}}
    for p in PACKAGES:
        try:
            info["packages"][p] = version(p)
        except PackageNotFoundError:
            pass
    gpu = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader"])
    if gpu:
        name, driver, mem = (x.strip() for x in gpu.splitlines()[0].split(","))
        info.update(gpu=name, driver=driver, gpu_memory=mem)
    try:
        import torch
        info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    info["git_commit"] = _run(["git", "rev-parse", "HEAD"])
    if jev_url:
        for path in ("/health", "/v1/models"):
            try:
                info["jev_server"] = httpx.get(jev_url + path, timeout=5).json()
                break
            except (httpx.HTTPError, ValueError):
                continue
    return info
```

- [ ] **Step 5: Implement `bench/systems.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from jevragrank.base import Searchable
from jevragrank.jev_client import JevClient
from jevragrank.pipelines import JevRAGRank, RetrieveThenRerank
from jevragrank.rerankers import CrossEncoderReranker, JevReranker, LLMReranker
from jevragrank.retrievers import BM25Retriever, DenseRetriever
from jevragrank.tournament import JevRank
from jevragrank.types import Doc

# Frozen after tuning on dev splits (Task 11). Test-split runs must not change these.
DEFAULTS: dict[str, dict] = {
    "bm25": {},
    "dense": {},
    "dense+ce": {"candidates": 50},
    "dense+llm": {"candidates": 50},
    "jevragrank": {"candidates": 50, "strategy": "score", "batch_size": 12,
                   "max_passage_tokens": 512},
    "jevrank": {"group_size": 64, "preview_tokens": [96, 256], "survivors": [3, 12],
                "final_pool": 50, "batch_size": 12, "max_passage_tokens": 512},
}
SYSTEMS = tuple(DEFAULTS)
LLM_VRAM_GB = 6.0


@dataclass
class Context:
    docs: list[Doc]
    jev_url: str
    cache_dir: Path
    jev_cache_mode: str = "write"
    device: str | None = None


@dataclass
class Built:
    system: Searchable
    jev: JevClient | None
    index_seconds: float


def _require_free_vram(gb: float) -> None:
    import torch
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free < gb * 1e9:
            raise SystemExit(f"Only {free / 1e9:.1f} GB of GPU memory is free; dense+llm needs "
                             f"about {gb:.0f} GB. Stop the decider server and run again.")


def build(name: str, ctx: Context, overrides: dict) -> Built:
    if name not in DEFAULTS:
        raise SystemExit(f"Unknown system {name!r}; choose from {', '.join(SYSTEMS)}")
    opts = {**DEFAULTS[name], **overrides}
    jev_url = opts.pop("jev_url", ctx.jev_url)
    emb_cache = ctx.cache_dir / "embeddings"
    jev = None
    if name in ("jevragrank", "jevrank"):
        jev = JevClient(jev_url, cache_dir=ctx.cache_dir / "jev", cache_mode=ctx.jev_cache_mode)
    if name == "dense+llm":
        _require_free_vram(LLM_VRAM_GB)
    t0 = perf_counter()
    if name == "bm25":
        system: Searchable = BM25Retriever(ctx.docs)
    elif name == "dense":
        system = DenseRetriever(ctx.docs, cache_dir=emb_cache, device=ctx.device)
    elif name == "dense+ce":
        system = RetrieveThenRerank(DenseRetriever(ctx.docs, cache_dir=emb_cache, device=ctx.device),
                                    CrossEncoderReranker(device=ctx.device),
                                    candidates=opts["candidates"])
    elif name == "dense+llm":
        system = RetrieveThenRerank(DenseRetriever(ctx.docs, cache_dir=emb_cache, device=ctx.device),
                                    LLMReranker(device=ctx.device or "cuda"),
                                    candidates=opts["candidates"])
    elif name == "jevragrank":
        retriever = DenseRetriever(ctx.docs, cache_dir=emb_cache, device=ctx.device)
        rerank_opts = {k: v for k, v in opts.items() if k != "candidates"}
        system = JevRAGRank(ctx.docs, jev=jev, candidates=opts["candidates"],
                            retriever=retriever, **rerank_opts)
    else:
        final = JevReranker(jev, strategy="score", batch_size=opts["batch_size"],
                            max_passage_tokens=opts["max_passage_tokens"])
        system = JevRank(ctx.docs, jev=jev, group_size=opts["group_size"],
                         preview_tokens=tuple(opts["preview_tokens"]),
                         survivors=tuple(opts["survivors"]), final_pool=opts["final_pool"],
                         final_reranker=final)
    return Built(system, jev, perf_counter() - t0)
```

- [ ] **Step 6: Implement `bench/runner.py`**

```python
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


async def run_spec(spec: RunSpec, paths: Paths, *, k: int = 10, warmup: int = 5,
                   jev_url: str = "http://127.0.0.1:8000", cache_mode: str = "write",
                   device: str | None = None, max_consecutive_failures: int = 3,
                   builder=build) -> Path:
    ds = prepare(spec, paths)
    out_dir = paths.results / spec.dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{spec.name}.jsonl"
    ctx = Context(ds.docs, jev_url, paths.cache, cache_mode, device)
    built = builder(spec.system, ctx, spec.overrides)
    if built.jev is not None:
        await built.jev.check()
    meta = {"system": spec.system, "label": spec.label,
            "options": {**DEFAULTS.get(spec.system, {}), **spec.overrides},
            "split": spec.split, "note": ds.note, "corpus_size": len(ds.docs),
            "n_queries": len(ds.queries), "index_seconds": built.index_seconds,
            "index_cached": bool(getattr(built.system, "from_cache", False)),
            "k": k, "spec": asdict(spec), "env": environment(jev_url if built.jev else None),
            "started": datetime.now(timezone.utc).isoformat()}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    qids = list(ds.queries)
    for qid in qids[:warmup]:
        await built.system.asearch_traced(ds.queries[qid], k)
    done = _done_qids(out)
    failures = 0
    with out.open("a", encoding="utf-8") as f:
        for qid in (q for q in qids if q not in done):
            before = built.jev.stats.requests if built.jev else 0
            t0 = perf_counter()
            try:
                res = await built.system.asearch_traced(ds.queries[qid], k)
            except Exception:  # noqa: BLE001 - logged and retried on resume
                failures += 1
                with out.with_suffix(".errors.log").open("a", encoding="utf-8") as log:
                    log.write(f"{qid}\n{traceback.format_exc()}\n")
                if failures >= max_consecutive_failures:
                    raise SystemExit(f"{failures} consecutive failures in {spec.name}; see "
                                     f"{out.with_suffix('.errors.log')}") from None
                continue
            failures = 0
            row = {"qid": qid, "ranking": [[h.doc.id, h.score] for h in res.hits],
                   "candidates": res.candidates, "timings": res.timings,
                   "latency": perf_counter() - t0,
                   "jev_requests": (built.jev.stats.requests - before) if built.jev else 0}
            f.write(json.dumps(row) + "\n")
            f.flush()
    return out


async def throughput_spec(spec: RunSpec, paths: Paths, *, n: int = 64, concurrency: int = 16,
                          k: int = 10, jev_url: str = "http://127.0.0.1:8000",
                          device: str | None = None, builder=build) -> dict:
    ds = prepare(spec, paths)
    built = builder(spec.system, Context(ds.docs, jev_url, paths.cache, "write", device),
                    spec.overrides)
    queries = list(ds.queries.values())
    queries = (queries * (n // max(len(queries), 1) + 1))[:n]
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str) -> None:
        async with sem:
            await built.system.asearch_traced(q, k)

    await one(queries[0])
    t0 = perf_counter()
    await asyncio.gather(*(one(q) for q in queries))
    elapsed = perf_counter() - t0
    result = {"n": n, "concurrency": concurrency, "seconds": elapsed, "qps": n / elapsed}
    out_dir = paths.results / spec.dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{spec.name}.throughput.json").write_text(json.dumps(result, indent=2), "utf-8")
    return result
```

- [ ] **Step 7: Implement `bench/summary.py`**

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .datasets import load_beir
from .metrics import (METRICS, bootstrap_ci, candidate_recall, latency_stats, per_query,
                      randomization_test)

REFERENCE = "dense+ce"


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def summarize(results_dir: Path, data_dir: Path) -> dict:
    results_dir = Path(results_dir)
    out: dict = {"generated": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    for ds_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        name, _, split = ds_dir.name.partition("-")
        split = split or "test"
        ds = None
        pq_by_run: dict[str, dict] = {}
        runs: dict[str, dict] = {}
        for path in sorted(ds_dir.glob("*.jsonl")):
            run_name = path.stem
            rows = _rows(path)
            if not rows:
                continue
            meta_path = path.with_suffix(".meta.json")
            meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
            if ds is None:
                ds = load_beir(name, split, data_dir)
            qids = [r["qid"] for r in rows]
            qrels = {q: ds.qrels[q] for q in qids if q in ds.qrels}
            pq = per_query(qrels, {r["qid"]: [d for d, _ in r["ranking"]] for r in rows})
            pq_by_run[run_name] = pq
            metrics = {}
            for m in METRICS:
                vals = np.array([pq[m][q] for q in qrels])
                lo, hi = bootstrap_ci(vals)
                metrics[m] = {"mean": float(vals.mean()), "ci": [lo, hi]}
            cands = {r["qid"]: r["candidates"] for r in rows if r.get("candidates") is not None}
            tp_path = path.with_suffix(".throughput.json")
            runs[run_name] = {
                "system": meta.get("system", run_name.split("@")[0]),
                "label": meta.get("label", ""), "options": meta.get("options", {}),
                "note": meta.get("note", ""), "corpus_size": meta.get("corpus_size"),
                "n_queries": len(qrels), "metrics": metrics,
                "latency": latency_stats([r["latency"] for r in rows]),
                "candidate_recall": candidate_recall(qrels, cands) if cands else None,
                "jev_requests": float(np.mean([r.get("jev_requests", 0) for r in rows])),
                "index_seconds": meta.get("index_seconds"),
                "qps": json.loads(tp_path.read_text("utf-8"))["qps"] if tp_path.exists() else None,
                "p_vs_ce": None,
            }
            out.setdefault("env", meta.get("env"))
        ref = pq_by_run.get(REFERENCE)
        for run_name, pq in pq_by_run.items():
            if ref is None or run_name == REFERENCE:
                continue
            common = sorted(set(pq["ndcg@10"]) & set(ref["ndcg@10"]))
            if len(common) >= 2:
                runs[run_name]["p_vs_ce"] = randomization_test(
                    [pq["ndcg@10"][q] for q in common], [ref["ndcg@10"][q] for q in common])
        if runs:
            out["datasets"][ds_dir.name] = {"split": split, "runs": runs}
    (results_dir / "summary.json").write_text(json.dumps(out, indent=2), "utf-8")
    return out


def markdown_table(summary: dict, dataset: str) -> str:
    runs = summary["datasets"][dataset]["runs"]
    lines = ["| system | nDCG@10 (95% CI) | Recall@10 | MRR@10 | p50 latency | vs cross-encoder |",
             "|---|---|---|---|---|---|"]
    for name, r in sorted(runs.items(), key=lambda kv: -kv[1]["metrics"]["ndcg@10"]["mean"]):
        n = r["metrics"]["ndcg@10"]
        p = r["p_vs_ce"]
        vs = "—" if p is None else (("better" if n["mean"] > runs.get("dense+ce", r)["metrics"]
                                     ["ndcg@10"]["mean"] else "worse") if p < 0.05 else "tie")
        lines.append(f"| {name} | {n['mean']:.3f} ({n['ci'][0]:.3f}–{n['ci'][1]:.3f}) | "
                     f"{r['metrics']['recall@10']['mean']:.3f} | {r['metrics']['mrr@10']['mean']:.3f} | "
                     f"{r['latency']['p50'] * 1000:.0f} ms | {vs} |")
    return "\n".join(lines)
```

- [ ] **Step 8: Implement `bench/cli.py`**

The `all` matrix is the one Task 12 runs. The plots and report commands exist here but their modules land in Tasks 13–14; they import lazily.

```python
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .runner import Paths, RunSpec, run_spec, throughput_spec
from .summary import markdown_table, summarize

JEV_2B = "http://127.0.0.1:8000"
JEV_08B = "http://127.0.0.1:8001"


def _parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        try:
            out[key] = json.loads(value)
        except json.JSONDecodeError:
            out[key] = value
    return out


def matrix(jevrank_sample: int | None) -> list[tuple[str, RunSpec]]:
    """(kind, spec) in execution order; kind is 'run' or 'throughput'."""
    items: list[tuple[str, RunSpec]] = []
    for ds in ("scifact", "nfcorpus"):
        for system in ("bm25", "dense", "dense+ce", "jevragrank"):
            items.append(("run", RunSpec(ds, system)))
        items.append(("run", RunSpec(ds, "jevrank", sample=jevrank_sample)))
        for strat in ("noul", "choice"):
            items.append(("run", RunSpec(ds, "jevragrank", label=strat,
                                         overrides={"strategy": strat})))
        for k in (10, 20, 100):
            for system in ("jevragrank", "dense+ce"):
                items.append(("run", RunSpec(ds, system, label=f"k{k}",
                                             overrides={"candidates": k})))
        for system in ("bm25", "dense", "dense+ce", "jevragrank", "jevrank"):
            items.append(("throughput", RunSpec(ds, system)))
    for n in (500, 1000, 2000, 5183):
        for system in ("dense", "jevragrank", "jevrank"):
            items.append(("run", RunSpec("scifact", system, label=f"n{n}", sample=50,
                                         corpus_size=None if n == 5183 else n)))
    for ds in ("scifact", "nfcorpus"):
        items.append(("run", RunSpec(ds, "jevragrank", label="0.8b",
                                     overrides={"jev_url": JEV_08B})))
        items.append(("run", RunSpec(ds, "jevrank", label="0.8b", sample=jevrank_sample,
                                     overrides={"jev_url": JEV_08B})))
    for ds in ("scifact", "nfcorpus"):
        items.append(("run", RunSpec(ds, "dense+llm")))
        items.append(("throughput", RunSpec(ds, "dense+llm")))
    return items


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="jevragrank-bench")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--cache", type=Path, default=Path(".cache"))
    ap.add_argument("--jev-url", default=JEV_2B)
    ap.add_argument("--device", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "throughput"):
        p = sub.add_parser(cmd)
        p.add_argument("--dataset", required=True)
        p.add_argument("--system", required=True)
        p.add_argument("--label", default="")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
        p.add_argument("--split", default="test")
        p.add_argument("--sample", type=int)
        p.add_argument("--corpus-size", type=int)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--cache-mode", default="write", choices=["write", "readwrite", "off"])
        p.add_argument("--n", type=int, default=64)
        p.add_argument("--concurrency", type=int, default=16)
    sub.add_parser("summarize")
    sub.add_parser("plots")
    sub.add_parser("report")
    p_all = sub.add_parser("all")
    p_all.add_argument("--jevrank-sample", type=int)
    args = ap.parse_args(argv)
    paths = Paths(args.results, args.data, args.cache)

    if args.cmd in ("run", "throughput"):
        spec = RunSpec(args.dataset, args.system, args.label, _parse_set(args.set), args.split,
                       args.sample, args.corpus_size, args.seed)
        if args.cmd == "run":
            out = asyncio.run(run_spec(spec, paths, jev_url=args.jev_url,
                                       cache_mode=args.cache_mode, device=args.device))
            print(f"wrote {out}")
        else:
            print(asyncio.run(throughput_spec(spec, paths, n=args.n, concurrency=args.concurrency,
                                              jev_url=args.jev_url, device=args.device)))
    elif args.cmd == "summarize":
        s = summarize(paths.results, paths.data)
        for ds in s["datasets"]:
            print(f"\n## {ds}\n\n{markdown_table(s, ds)}")
    elif args.cmd == "plots":
        from .plots import render_all
        render_all(json.loads((paths.results / "summary.json").read_text("utf-8")),
                   Path("docs/charts"))
    elif args.cmd == "report":
        from .report import write_report
        write_report(json.loads((paths.results / "summary.json").read_text("utf-8")),
                     Path("docs/report.html"))
    else:
        for kind, spec in matrix(args.jevrank_sample):
            print(f"== {kind} {spec.dir_name}/{spec.name}", flush=True)
            if kind == "run":
                asyncio.run(run_spec(spec, paths, jev_url=args.jev_url, device=args.device))
            elif not (paths.results / spec.dir_name / f"{spec.name}.throughput.json").exists():
                asyncio.run(throughput_spec(spec, paths, jev_url=args.jev_url, device=args.device))
        main(["--results", str(paths.results), "--data", str(paths.data), "summarize"])
```

- [ ] **Step 9: Run tests and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all pass.

- [ ] **Step 10: Smoke-test the CLI on real data** (no GPU models needed)

```bash
uv run jevragrank-bench --results .cache/smoke run --dataset scifact --system bm25 --sample 20
uv run jevragrank-bench --results .cache/smoke summarize
```
Expected: a table with a `bm25` row and nDCG@10 roughly 0.5–0.8 (20 queries, wide CI).

- [ ] **Step 11: Commit and push**

```bash
git add bench tests/test_runner.py tests/test_summary.py
git commit -m "Add benchmark runner, systems registry, summary and CLI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 10: Baseline sanity gate and live smoke test

**Files:**
- Create: `tests/test_live.py`
- Create (generated): `results/scifact/bm25.*`, `results/scifact/dense.*`, `results/nfcorpus/bm25.*`, `results/nfcorpus/dense.*`

**Interfaces:**
- Consumes: the CLI (Task 9); the decider server (Task 0).
- Produces: baseline results that pass the gate. Nothing downstream proceeds if the gate fails.

- [ ] **Step 1: Write the live smoke test** — `tests/test_live.py`

```python
import os

import pytest

from bench.datasets import load_beir, sample_queries
from jevragrank import JevClient, JevRAGRank, JevRank

URL = os.environ.get("JEV_URL", "http://127.0.0.1:8000")


@pytest.mark.live
@pytest.mark.slow
def test_both_pipelines_end_to_end():
    ds = sample_queries(load_beir("scifact", "test", "data"), 3, seed=0)
    jev = JevClient(URL, cache_mode="off")
    rag = JevRAGRank(ds.docs, jev=jev, candidates=20, cache_dir=".cache/embeddings")
    pure = JevRank(ds.docs, jev=jev)
    for qid, q in ds.queries.items():
        for system in (rag, pure):
            hits = system.search(q, k=10)
            assert len(hits) == 10
            assert all(not h.degraded for h in hits)
```

- [ ] **Step 2: Run the baseline gate** (dense needs no decider; the embeddings are cached for later runs)

```bash
for ds in scifact nfcorpus; do
  uv run jevragrank-bench run --dataset $ds --system bm25
  uv run jevragrank-bench run --dataset $ds --system dense
done
uv run jevragrank-bench summarize
```

Gate:

| run | expected nDCG@10 | allowed range |
|---|---|---|
| scifact bm25 | 0.665 | 0.635–0.720 |
| scifact dense | 0.740 | 0.710–0.770 |
| nfcorpus bm25 | 0.325 | 0.295–0.355 |
| nfcorpus dense | 0.373 | 0.343–0.403 |

The bm25s range is widened upward because bm25s with stemming reports about 0.686 on SciFact.

**If any run is outside its range, STOP.** Use superpowers:systematic-debugging on the loader, metrics or retriever. Delete the bad `results/*/*.jsonl` files and re-run. Do not start Jev runs until all four pass.

- [ ] **Step 3: Run the live smoke test** (decider running on :8000)

Run: `uv run pytest tests/test_live.py -m live -q`
Expected: PASS.

- [ ] **Step 4: Commit and push**

```bash
git add tests/test_live.py results
git commit -m "Pass BEIR baseline sanity gate; add live smoke test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 11: Tune Jev settings on dev splits and freeze them

**Files:**
- Modify: `bench/systems.py` (the `DEFAULTS` for `jevragrank` and `jevrank`, and `DEFAULT_INSTRUCTIONS`/`DEFAULT_LEVELS` only if wording wins)
- Create: `docs/tuning.md`

**Interfaces:**
- Produces: frozen defaults used by every test-split run in Task 12.

- [ ] **Step 1: Run the JevRAGRank grid** on 50 dev queries per dataset. Run 2 candidates each for strategy and batch size; each takes about 2–4 minutes. The Jev cache is on because timing doesn't matter here.

```bash
for ds_split in "scifact train" "nfcorpus dev"; do set -- $ds_split
  for strat in score noul choice; do
    uv run jevragrank-bench run --dataset $1 --split $2 --sample 50 --cache-mode readwrite \
      --system jevragrank --label $strat --set strategy=$strat
  done
  for bs in 6 24; do
    uv run jevragrank-bench run --dataset $1 --split $2 --sample 50 --cache-mode readwrite \
      --system jevragrank --label bs$bs --set batch_size=$bs
  done
done
```

- [ ] **Step 2: Run the tournament grid** (group size × round-1 survivors), on 30 dev queries:

```bash
for ds_split in "scifact train" "nfcorpus dev"; do set -- $ds_split
  for cfg in "32 [3,12]" "64 [3,12]" "64 [5,12]" "128 [6,12]"; do set -- $1 $2 $cfg
    uv run jevragrank-bench run --dataset $1 --split $2 --sample 30 --cache-mode readwrite \
      --system jevrank --label g$3-s$(echo $4 | tr -d '[],') --set group_size=$3 --set survivors=$4
  done
done
uv run jevragrank-bench summarize
```

- [ ] **Step 3: Pick and freeze the settings.** For each system, choose the config with the best mean nDCG@10 averaged over the two dev sets.
  - **Ties:** a tie is a difference under 0.01. Break ties with the lower p50 latency.
  - **Tournament rule:** reject any tournament config whose p50 latency projects the full test run over 6 hours, unless the 100-query sample rule from Task 0 already applies.
  - **Freeze:** write the winners into `DEFAULTS` in `bench/systems.py`. If `choice` wins the JevRAGRank strategy, the main `jevragrank` runs use it, and the Task 12 strategy ablation still runs all three.

- [ ] **Step 4: Write `docs/tuning.md`**: the markdown tables printed by `summarize` for `scifact-train` and `nfcorpus-dev`, the chosen settings, and one sentence per choice explaining why.

- [ ] **Step 5: Commit and push** (dev-split results are committed too, for transparency)

```bash
git add bench/systems.py docs/tuning.md results/scifact-train results/nfcorpus-dev results/summary.json
git commit -m "Freeze Jev settings tuned on dev splits

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 12: Full benchmark runs and ablations

**Files:**
- Create (generated): `results/scifact/*`, `results/nfcorpus/*`, `results/summary.json`

- [ ] **Step 1: Keep the laptop awake.** Plug in, and disable sleep for the duration. The runs are resumable if it sleeps anyway.

- [ ] **Step 2: Start decider-2b on :8000 and decider-0.8b on :8001.** If both don't fit in 8 GB (check `nvidia-smi` after both load), start 0.8b only when the matrix reaches the `0.8b` items. The runner aborts with a connection error at that point; start it and re-run the same command.

- [ ] **Step 3: Run the matrix** (in background; hours). Use `--jevrank-sample 100` only if `docs/spike-notes.md` set `JEVRANK_SAMPLE = 100`.

```bash
uv run jevragrank-bench all --jevrank-sample 100
```

- [ ] **Step 4: Handle the Qwen baseline.** When the matrix reaches `dense+llm` it exits with "Stop the decider server and run again". Stop both decider servers, then re-run the same `all` command. Completed items are skipped because the runs resume.

- [ ] **Step 5: Check completeness.** Every `.jsonl` should have as many rows as its `n_queries` in the `.meta.json`, and `*.errors.log` files should contain only failures that were retried successfully.

```bash
uv run python -c "
import json, pathlib
for m in sorted(pathlib.Path('results').glob('*/*.meta.json')):
    rows = sum(1 for _ in m.with_suffix('').with_suffix('.jsonl').open())
    n = json.loads(m.read_text())['n_queries']
    print(('OK  ' if rows == n else 'MISSING ') + f'{m.parent.name}/{m.stem[:-5]} {rows}/{n}')
"
```
Expected: every line starts with `OK`. Re-run `all` for any `MISSING`.

- [ ] **Step 6: Commit and push**

```bash
uv run jevragrank-bench summarize
git add results
git commit -m "Add full benchmark results and ablations

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 13: Charts

**Files:**
- Create: `bench/plots.py`
- Test: `tests/test_plots.py`
- Create (generated): `docs/charts/*.png`, `docs/charts/*.svg`

**Interfaces:**
- Consumes: the `summary.json` schema from Task 9. For each run it uses `system`, `label`, `options`, `corpus_size`, `metrics[m].mean/ci`, `latency.p50`, `p_vs_ce`.
- Produces: `render_all(summary: dict, out_dir: Path) -> list[Path]`. Each chart is written as `{chart}-{light|dark}.{png,svg}` for `speed-accuracy`, `accuracy`, `scaling`, `depth`, `ablations`.

Before writing this task's code, the implementer must load the `dataviz` skill and follow its procedure. The code below already applies its rules:
- emphasis form, with the Jev systems in the two validated hues and the baselines in gray with marker shapes;
- 2px lines, markers of 8px or more with a 2px surface ring;
- hairline solid grids and no dual axes;
- a legend for 2+ series plus selective direct labels;
- text in ink tokens.

- [ ] **Step 1: Write the failing test** — `tests/test_plots.py`

```python
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
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_plots.py -q`
Expected: `ModuleNotFoundError: bench.plots`.

- [ ] **Step 3: Implement `bench/plots.py`**

```python
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "jevragrank": "#2a78d6",
              "jevrank": "#eb6834", "base": "#898781"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "jevragrank": "#3987e5",
             "jevrank": "#d95926", "base": "#898781"},
}
LABELS = {"bm25": "BM25", "dense": "Dense (bge-base)", "dense+ce": "Dense → cross-encoder",
          "dense+llm": "Dense → Qwen3.5-2B prompt", "jevragrank": "JevRAGRank",
          "jevrank": "JevRank (no embeddings)"}
MARKERS = {"bm25": "s", "dense": "D", "dense+ce": "^", "dense+llm": "v",
           "jevragrank": "o", "jevrank": "o"}
MAIN = ["bm25", "dense", "dense+ce", "dense+llm", "jevragrank", "jevrank"]
DATASET_TITLES = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}


def _color(t: dict, system: str) -> str:
    return t[system] if system in ("jevragrank", "jevrank") else t["base"]


def _mix(hex_a: str, hex_b: str, w: float) -> tuple:
    a, b = to_rgb(hex_a), to_rgb(hex_b)
    return tuple(w * x + (1 - w) * y for x, y in zip(a, b))


def _style(ax, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=9, length=0)
    ax.grid(True, color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    ax.title.set_color(t["ink"])


def _figure(t: dict, ncols: int, title: str, subtitle: str, height: float = 4.2):
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, height), squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold", color=t["ink"])
    fig.text(0.01, 0.905, subtitle, ha="left", fontsize=9, color=t["ink2"])
    for ax in axes[0]:
        _style(ax, t)
    return fig, list(axes[0])


def _save(fig, out_dir: Path, name: str, theme: str) -> list[Path]:
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    paths = []
    for ext in ("png", "svg"):
        p = out_dir / f"{name}-{theme}.{ext}"
        fig.savefig(p, dpi=200, facecolor=fig.get_facecolor())
        paths.append(p)
    plt.close(fig)
    return paths


def _runs(summary: dict):
    for ds, block in summary["datasets"].items():
        if "-" not in ds:
            yield ds, block["runs"]


def _legend(ax, t: dict) -> None:
    leg = ax.legend(frameon=False, fontsize=8, loc="best")
    for text in leg.get_texts():
        text.set_color(t["ink2"])


def speed_accuracy(summary: dict, t: dict) -> plt.Figure:
    data = list(_runs(summary))
    fig, axes = _figure(t, len(data), "Speed vs accuracy",
                        "p50 latency per query (log scale) against nDCG@10, 95% CI bars. "
                        "Up and to the left is better.")
    for ax, (ds, runs) in zip(axes, data):
        pts = []
        for s in MAIN:
            r = runs.get(s)
            if not r:
                continue
            x, n = r["latency"]["p50"] * 1000, r["metrics"]["ndcg@10"]
            pts.append((x, n["mean"]))
            ax.errorbar(x, n["mean"], yerr=[[n["mean"] - n["ci"][0]], [n["ci"][1] - n["mean"]]],
                        fmt="none", ecolor=_color(t, s), elinewidth=1.5, capsize=0, zorder=2)
            ax.plot(x, n["mean"], MARKERS[s], ms=9, color=_color(t, s), mec=t["surface"],
                    mew=2, zorder=3, label=LABELS[s])
            ax.annotate(LABELS[s], (x, n["mean"]), xytext=(7, 5), textcoords="offset points",
                        fontsize=8, color=t["ink2"])
        frontier, best = [], -1.0
        for x, y in sorted(pts):
            if y > best:
                frontier.append((x, y))
                best = y
        if len(frontier) > 1:
            ax.plot(*zip(*frontier), color=t["muted"], lw=1, zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("p50 latency (ms)")
        ax.set_ylabel("nDCG@10")
        ax.set_title(DATASET_TITLES.get(ds, ds), loc="left", fontsize=11)
    _legend(axes[-1], t)
    return fig


def accuracy(summary: dict, t: dict) -> plt.Figure:
    data = list(_runs(summary))
    fig, axes = _figure(t, len(data), "Retrieval accuracy",
                        "nDCG@10 with 95% bootstrap CI. Marker vs cross-encoder: ▲ better, "
                        "= tie, ▼ worse (paired randomization test, p < 0.05).")
    for ax, (ds, runs) in zip(axes, data):
        present = [s for s in MAIN if s in runs]
        present.sort(key=lambda s: runs[s]["metrics"]["ndcg@10"]["mean"])
        ce = runs.get("dense+ce", {}).get("metrics", {}).get("ndcg@10", {}).get("mean")
        for y, s in enumerate(present):
            n = runs[s]["metrics"]["ndcg@10"]
            ax.barh(y, n["mean"], height=0.5, color=_color(t, s), edgecolor=t["surface"],
                    linewidth=2)
            ax.plot(n["ci"], [y, y], color=t["ink2"], lw=1.2)
            p = runs[s].get("p_vs_ce")
            mark = "" if p is None else ("=" if p >= 0.05 else ("▲" if n["mean"] > ce else "▼"))
            ax.text(n["ci"][1] + 0.01, y, f"{n['mean']:.3f} {mark}".strip(), va="center",
                    fontsize=8, color=t["ink2"])
        ax.set_yticks(range(len(present)), [LABELS[s] for s in present])
        ax.set_xlim(0, 1.0)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("nDCG@10")
        ax.set_title(DATASET_TITLES.get(ds, ds), loc="left", fontsize=11)
    return fig


def _labelled(runs: dict, system: str, prefix: str) -> list[dict]:
    return [r for r in runs.values() if r["system"] == system and r["label"].startswith(prefix)
            and r["label"][len(prefix):].isdigit()]


def scaling(summary: dict, t: dict) -> plt.Figure:
    runs = summary["datasets"].get("scifact", {}).get("runs", {})
    fig, (ax,) = _figure(t, 1, "Latency as the corpus grows",
                         "SciFact subsets, 50 queries, p50 latency per query (log–log).")
    for s in ("jevrank", "jevragrank", "dense"):
        pts = sorted((r["corpus_size"], r["latency"]["p50"] * 1000) for r in _labelled(runs, s, "n"))
        if not pts:
            continue
        ax.plot(*zip(*pts), "-o", color=_color(t, s), lw=2, ms=8, mec=t["surface"], mew=2,
                label=LABELS[s])
        ax.annotate(f"{pts[-1][1]:,.0f} ms", pts[-1], xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=t["ink2"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("documents in corpus")
    ax.set_ylabel("p50 latency (ms)")
    if ax.lines:
        _legend(ax, t)
    return fig


def depth(summary: dict, t: dict) -> plt.Figure:
    runs = summary["datasets"].get("scifact", {}).get("runs", {})
    fig, axes = _figure(t, 2, "How many candidates to rerank",
                        "SciFact. Rerank depth K (dense candidates) vs accuracy and latency.")
    for s in ("jevragrank", "dense+ce"):
        rs = _labelled(runs, s, "k") + ([runs[s]] if s in runs else [])
        pts = sorted((r["options"].get("candidates", 50), r) for r in rs)
        if not pts:
            continue
        ks = [k for k, _ in pts]
        style = dict(color=_color(t, s), lw=2, ms=8, mec=t["surface"], mew=2, label=LABELS[s])
        axes[0].plot(ks, [r["metrics"]["ndcg@10"]["mean"] for _, r in pts], "-o", **style)
        axes[1].plot(ks, [r["latency"]["p50"] * 1000 for _, r in pts], "-o", **style)
    axes[0].set_ylabel("nDCG@10")
    axes[1].set_ylabel("p50 latency (ms)")
    for ax in axes:
        ax.set_xlabel("candidates reranked (K)")
        ax.set_xscale("log")
        ax.set_xticks([10, 20, 50, 100], ["10", "20", "50", "100"])
        if ax.lines:
            _legend(ax, t)
    return fig


def ablations(summary: dict, t: dict) -> plt.Figure:
    data = list(_runs(summary))
    fig, axes = _figure(t, 2, "Ablations",
                        "Left: Jev question type for JevRAGRank. Right: decider model size. "
                        "nDCG@10 averaged over datasets.")
    strategies = [("score", ""), ("noul", "noul"), ("choice", "choice")]
    vals = []
    for name, label in strategies:
        key = "jevragrank" + (f"@{label}" if label else "")
        xs = [runs[key]["metrics"]["ndcg@10"]["mean"] for _, runs in data if key in runs]
        if xs:
            vals.append((name, sum(xs) / len(xs)))
    for i, (name, v) in enumerate(vals):
        axes[0].bar(i, v, width=0.5, color=t["jevragrank"], edgecolor=t["surface"], linewidth=2)
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, color=t["ink2"])
    axes[0].set_xticks(range(len(vals)), [n for n, _ in vals])
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="x", visible=False)
    axes[0].set_ylabel("nDCG@10")
    for i, s in enumerate(("jevragrank", "jevrank")):
        for j, (label, key) in enumerate((("0.8B", f"{s}@0.8b"), ("2B", s))):
            xs = [runs[key]["metrics"]["ndcg@10"]["mean"] for _, runs in data if key in runs]
            if not xs:
                continue
            v = sum(xs) / len(xs)
            color = _mix(t[s], t["surface"], 0.45) if label == "0.8B" else t[s]
            x = i + (j - 0.5) * 0.3
            axes[1].bar(x, v, width=0.28, color=color, edgecolor=t["surface"], linewidth=2,
                        label=f"{LABELS[s]} · {label}")
            axes[1].text(x, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, color=t["ink2"])
    axes[1].set_xticks([0, 1], [LABELS["jevragrank"], LABELS["jevrank"]])
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="x", visible=False)
    if axes[1].patches:
        _legend(axes[1], t)
    return fig


CHARTS = {"speed-accuracy": speed_accuracy, "accuracy": accuracy, "scaling": scaling,
          "depth": depth, "ablations": ablations}


def render_all(summary: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
                         "svg.fonttype": "none"})
    paths: list[Path] = []
    for theme, t in THEMES.items():
        for name, fn in CHARTS.items():
            paths += _save(fn(summary, t), out_dir, name, theme)
    return paths
```

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest tests/test_plots.py -q && uv run ruff check .`
Expected: all pass.

- [ ] **Step 5: Render the real charts and look at them**

```bash
uv run jevragrank-bench plots
```
Open `docs/charts/speed-accuracy-light.png`, `accuracy-dark.png`, `scaling-light.png`, `depth-light.png` and `ablations-dark.png` with the Read tool. Check against the dataviz anti-patterns:
- no colliding direct labels (if they collide, drop the direct label on the lower-priority baseline and rely on the legend);
- nothing clipped;
- the dark theme is readable.

Fix, then re-render once.

- [ ] **Step 6: Commit and push**

```bash
git add bench/plots.py tests/test_plots.py docs/charts
git commit -m "Add benchmark charts (light and dark)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

---

### Task 14: Interactive HTML report

**Files:**
- Create: `bench/report_template.html`, `bench/report.py`
- Test: `tests/test_report.py`
- Create (generated): `docs/report.html`

**Interfaces:**
- Consumes: `summary.json` (Task 9 schema).
- Produces:
  - `render_report(summary: dict, *, standalone: bool) -> str`. With `standalone=False` it returns an Artifact fragment without doctype/html/head/body tags.
  - `write_report(summary, path: Path) -> Path` writes the standalone page and a sibling `report.fragment.html` for Artifact publishing.

Before building, load the `artifact-design` skill; the template below follows its page contract. The design plan:
- **Palette:** a cool ink-on-paper neutral with a slight blue bias. Light: ground `#f6f7f9`, surface `#fcfcfb`, ink `#10151c`, secondary `#4a5361`, hairline `#dde1e6`. Dark: ground `#0e1116`, surface `#1a1a19`, ink `#f2f4f7`, secondary `#aab2bd`, hairline `#2a3038`. The chart hues are the validated pair.
- **Type:** "IBM Plex Sans" for body, "IBM Plex Mono" for figures and tables (tabular), both from Google Fonts with system fallbacks.
- **Layout:** a single 960px column. A summary line up top, then one filter row (dataset toggle and system checkboxes) scoping every chart. After that the speed-vs-accuracy chart as the headline, the results table as the table-view twin, the ablation charts in a 2-up grid that stacks at phone width, and a methodology section.

- [ ] **Step 1: Write the failing test** — `tests/test_report.py`

```python
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
```

- [ ] **Step 2: Run and confirm failure**

Run: `uv run pytest tests/test_report.py -q`
Expected: `ModuleNotFoundError: bench.report`.

- [ ] **Step 3: Implement `bench/report.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("report_template.html")


def render_report(summary: dict, *, standalone: bool) -> str:
    data = json.dumps(summary).replace("</", "<\\/")
    body = TEMPLATE.read_text("utf-8").replace("__SUMMARY_JSON__", data)
    if not standalone:
        return body
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, '
            'viewport-fit=cover">\n</head>\n<body>\n' + body + "\n</body>\n</html>\n")


def write_report(summary: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(summary, standalone=True), "utf-8")
    path.with_name(path.stem + ".fragment.html").write_text(
        render_report(summary, standalone=False), "utf-8")
    return path
```

`json.dumps` escapes `"` inside strings. The `</` → `<\/` replacement keeps `</script>` from terminating the script, and JSON parsers read `<\/` as `</`.

- [ ] **Step 4: Write `bench/report_template.html`**

```html
<title>JevRAGRank Benchmarks</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --ground: #f6f7f9; --surface: #fcfcfb; --ink: #10151c; --ink2: #4a5361; --muted: #7b8490;
  --hair: #dde1e6; --grid: #e1e0d9; --jevragrank: #2a78d6; --jevrank: #eb6834; --base: #898781;
  --focus: #2a78d6;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --ground: #0e1116; --surface: #1a1a19; --ink: #f2f4f7; --ink2: #aab2bd; --muted: #7d8591;
    --hair: #2a3038; --grid: #2c2c2a; --jevragrank: #3987e5; --jevrank: #d95926; --focus: #6aa8ef;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --ground: #0e1116; --surface: #1a1a19; --ink: #f2f4f7; --ink2: #aab2bd; --muted: #7d8591;
  --hair: #2a3038; --grid: #2c2c2a; --jevragrank: #3987e5; --jevrank: #d95926; --focus: #6aa8ef;
}
body { background: var(--ground); color: var(--ink); font: 15px/1.55 var(--sans); }
.page { max-width: 960px; margin: 0 auto; padding-inline: 16px; padding-block: 32px 64px;
        display: grid; gap: 28px; }
h1 { font-size: 30px; line-height: 1.15; margin: 0; font-weight: 600; text-wrap: balance; }
h2 { font-size: 18px; margin: 0 0 4px; font-weight: 600; text-wrap: balance; }
p { margin: 0; max-width: 65ch; color: var(--ink2); }
.eyebrow { font: 500 12px/1 var(--mono); letter-spacing: .06em; text-transform: uppercase;
           color: var(--muted); }
header { display: grid; gap: 10px; }
.filters { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center;
           padding-block: 12px; border-block: 1px solid var(--hair); }
.filters fieldset { border: 0; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 6px 14px; }
.filters legend { font: 500 12px var(--mono); color: var(--muted); margin-bottom: 4px; }
.seg button { font: 500 13px var(--sans); color: var(--ink2); background: transparent;
              border: 1px solid var(--hair); padding: 5px 12px; border-radius: 6px; cursor: pointer; }
.seg button[aria-pressed="true"] { color: var(--ink); background: var(--surface); border-color: var(--ink2); }
label.sys { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--ink2); cursor: pointer; }
.swatch { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
button:focus-visible, input:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
.card { background: var(--surface); border: 1px solid var(--hair); border-radius: 10px; padding: 18px; }
.chart { width: 100%; overflow-x: auto; }
.chart svg { max-width: 100%; height: auto; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 16px; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font: 13px var(--mono); font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 8px 10px; border-bottom: 1px solid var(--hair); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; font-family: var(--sans); }
th { color: var(--muted); font-weight: 500; }
.pill { font: 500 11px var(--mono); padding: 2px 7px; border-radius: 99px; border: 1px solid var(--hair); color: var(--ink2); }
.method { display: grid; gap: 10px; }
.method li { color: var(--ink2); max-width: 70ch; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="page">
  <header>
    <span class="eyebrow">Open benchmark · BEIR test splits</span>
    <h1>Typed-decision reranking against standard RAG</h1>
    <p id="lede"></p>
  </header>

  <div class="filters" role="group" aria-label="Filters for every chart below">
    <fieldset class="seg"><legend>Dataset</legend><div id="dataset-seg"></div></fieldset>
    <fieldset><legend>Systems</legend><div id="system-boxes" style="display:flex;flex-wrap:wrap;gap:6px 14px"></div></fieldset>
  </div>

  <section class="card">
    <h2>Speed vs accuracy</h2>
    <p>Each point is one system. p50 latency per query on a log scale against nDCG@10, with 95% bootstrap intervals. Up and to the left is better.</p>
    <div class="chart" id="chart-scatter"></div>
  </section>

  <section class="card">
    <h2>All results</h2>
    <div class="table-wrap"><table id="results-table"></table></div>
  </section>

  <div class="grid2">
    <section class="card"><h2>Rerank depth</h2><p>Accuracy as more dense candidates are reranked.</p><div class="chart" id="chart-depth"></div></section>
    <section class="card"><h2>Question type</h2><p>JevRAGRank with each Jev primitive.</p><div class="chart" id="chart-strategy"></div></section>
    <section class="card"><h2>Corpus size</h2><p>SciFact subsets, 50 queries: latency grows with the corpus only for the embedding-free tournament.</p><div class="chart" id="chart-scaling"></div></section>
    <section class="card"><h2>Model size</h2><p>decider-0.8B vs decider-2B.</p><div class="chart" id="chart-model"></div></section>
  </div>

  <section class="card method">
    <h2>Method</h2>
    <ul>
      <li>JevRAGRank: bge-base-en-v1.5 dense retrieval selects candidates; decider-2b answers one typed <code>score</code> question per candidate (five levels from "irrelevant" to "fully answers").</li>
      <li>JevRank: no embeddings. Batched <code>choice</code> questions over groups of documents keep the likeliest few per group, round after round, then a <code>score</code> round ranks the finalists.</li>
      <li>Baselines: BM25 (bm25s), dense only, dense → bge-reranker-base, dense → Qwen3.5-2B-Base prompted for a 0–4 rating (the model decider-2b was fine-tuned from).</li>
      <li>Jev settings were tuned on SciFact train and NFCorpus dev; test splits were run once. Latency is measured one query at a time after five warm-up queries, with no response cache.</li>
      <li id="env-line"></li>
    </ul>
  </section>
</div>

<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.17/dist/plot.umd.min.js"></script>
<script>
const SUMMARY = __SUMMARY_JSON__;
const LABELS = {"bm25": "BM25", "dense": "Dense", "dense+ce": "Dense → cross-encoder",
  "dense+llm": "Dense → Qwen prompt", "jevragrank": "JevRAGRank", "jevrank": "JevRank"};
const MAIN = ["bm25", "dense", "dense+ce", "dense+llm", "jevragrank", "jevrank"];
const SYMBOL = {"bm25": "square", "dense": "diamond2", "dense+ce": "triangle",
  "dense+llm": "triangle2", "jevragrank": "circle", "jevrank": "circle"};
const datasets = Object.keys(SUMMARY.datasets).filter(d => !d.includes("-"));
let state = { dataset: datasets[0], systems: new Set(MAIN) };
try { const saved = localStorage.getItem("jrr-dataset"); if (datasets.includes(saved)) state.dataset = saved; } catch (e) {}

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const colorOf = s => s === "jevragrank" ? css("--jevragrank") : s === "jevrank" ? css("--jevrank") : css("--base");
const runs = () => SUMMARY.datasets[state.dataset].runs;
const fmt = (v, d = 3) => v == null || Number.isNaN(v) ? "—" : v.toFixed(d);
const ms = s => s * 1000 < 10 ? (s * 1000).toFixed(1) : Math.round(s * 1000).toLocaleString();

function plotStyle() {
  return { background: "transparent", color: css("--ink2"), fontFamily: css("--sans"), fontSize: "12px" };
}

function buildFilters() {
  const seg = document.getElementById("dataset-seg");
  seg.innerHTML = "";
  for (const d of datasets) {
    const b = document.createElement("button");
    b.type = "button"; b.id = "ds-" + d; b.textContent = d === "scifact" ? "SciFact" : d === "nfcorpus" ? "NFCorpus" : d;
    b.setAttribute("aria-pressed", String(d === state.dataset));
    b.onclick = () => { state.dataset = d; try { localStorage.setItem("jrr-dataset", d); } catch (e) {} render(); };
    seg.appendChild(b);
  }
  const boxes = document.getElementById("system-boxes");
  boxes.innerHTML = "";
  for (const s of MAIN) {
    const id = "sys-" + s.replace(/\W/g, "_");
    const l = document.createElement("label"); l.className = "sys"; l.htmlFor = id;
    l.innerHTML = `<input type="checkbox" id="${id}" ${state.systems.has(s) ? "checked" : ""}><span class="swatch" style="background:${colorOf(s)}"></span>${LABELS[s]}`;
    l.querySelector("input").onchange = e => { e.target.checked ? state.systems.add(s) : state.systems.delete(s); render(); };
    boxes.appendChild(l);
  }
}

function scatter() {
  const rows = MAIN.filter(s => runs()[s] && state.systems.has(s)).map(s => {
    const r = runs()[s], n = r.metrics["ndcg@10"];
    return { s, name: LABELS[s], x: r.latency.p50 * 1000, y: n.mean, lo: n.ci[0], hi: n.ci[1] };
  });
  return Plot.plot({
    width: 880, height: 380, marginRight: 150, style: plotStyle(),
    x: { type: "log", label: "p50 latency (ms) →", grid: true },
    y: { label: "↑ nDCG@10", grid: true },
    marks: [
      Plot.ruleX(rows, { x: "x", y1: "lo", y2: "hi", stroke: d => colorOf(d.s), strokeWidth: 1.5 }),
      Plot.dot(rows, { x: "x", y: "y", r: 6, fill: d => colorOf(d.s), stroke: css("--surface"), strokeWidth: 2, symbol: d => SYMBOL[d.s] }),
      Plot.text(rows, { x: "x", y: "y", text: "name", dx: 12, textAnchor: "start", fill: css("--ink2") }),
      Plot.tip(rows, Plot.pointer({ x: "x", y: "y", title: d => `${d.name}\nnDCG@10 ${fmt(d.y)} (${fmt(d.lo)}–${fmt(d.hi)})\np50 ${ms(d.x / 1000)} ms` }))
    ]
  });
}

function table() {
  const r = runs();
  const names = Object.keys(r).filter(n => state.systems.has(r[n].system))
    .sort((a, b) => r[b].metrics["ndcg@10"].mean - r[a].metrics["ndcg@10"].mean);
  const ce = r["dense+ce"]?.metrics["ndcg@10"].mean;
  const head = "<tr><th>Run</th><th>nDCG@10</th><th>95% CI</th><th>Recall@10</th><th>MRR@10</th><th>p50 ms</th><th>p95 ms</th><th>q/s @16</th><th>Cand. recall</th><th>vs CE</th></tr>";
  const body = names.map(n => {
    const x = r[n], m = x.metrics;
    const vs = x.p_vs_ce == null ? "—" : x.p_vs_ce >= 0.05 ? "tie" : (m["ndcg@10"].mean > ce ? "better" : "worse");
    const label = (LABELS[x.system] || x.system) + (x.label ? ` <span class="pill">${x.label}</span>` : "");
    return `<tr><td>${label}</td><td>${fmt(m["ndcg@10"].mean)}</td><td>${fmt(m["ndcg@10"].ci[0])}–${fmt(m["ndcg@10"].ci[1])}</td><td>${fmt(m["recall@10"].mean)}</td><td>${fmt(m["mrr@10"].mean)}</td><td>${ms(x.latency.p50)}</td><td>${ms(x.latency.p95)}</td><td>${x.qps == null ? "—" : x.qps.toFixed(1)}</td><td>${fmt(x.candidate_recall)}</td><td>${vs}</td></tr>`;
  }).join("");
  document.getElementById("results-table").innerHTML = `<thead>${head}</thead><tbody>${body}</tbody>`;
}

function labelled(system, prefix) {
  return Object.values(runs()).filter(x => x.system === system && x.label.startsWith(prefix) && /^\d+$/.test(x.label.slice(prefix.length)));
}

function lineChart(series, xLabel, yLabel, xType, yType) {
  const rows = series.flatMap(({ s, pts }) => pts.map(([x, y]) => ({ s, name: LABELS[s], x, y })));
  if (!rows.length) { const p = document.createElement("p"); p.textContent = "Not run for this dataset."; return p; }
  return Plot.plot({
    width: 420, height: 260, style: plotStyle(), marginRight: 20,
    x: { type: xType, label: xLabel, grid: true }, y: { type: yType, label: yLabel, grid: true },
    color: { domain: series.map(d => LABELS[d.s]), range: series.map(d => colorOf(d.s)), legend: true },
    marks: [
      Plot.line(rows, { x: "x", y: "y", stroke: "name", strokeWidth: 2 }),
      Plot.dot(rows, { x: "x", y: "y", fill: "name", r: 4.5, stroke: css("--surface"), strokeWidth: 2 }),
      Plot.tip(rows, Plot.pointer({ x: "x", y: "y", title: d => `${d.name}\n${xLabel.replace(/ →/, "")}: ${d.x.toLocaleString()}\n${yLabel.replace(/↑ /, "")}: ${d.y < 10 ? d.y.toFixed(3) : Math.round(d.y).toLocaleString()}` }))
    ]
  });
}

function barChart(rows) {
  if (!rows.length) { const p = document.createElement("p"); p.textContent = "Not run for this dataset."; return p; }
  return Plot.plot({
    width: 420, height: 240, style: plotStyle(), marginBottom: 36,
    x: { label: null, domain: rows.map(d => d.name), padding: 0.45 }, y: { label: "↑ nDCG@10", grid: true, domain: [0, 1] },
    marks: [
      Plot.barY(rows, { x: "name", y: "y", fill: d => d.fill, rx1: 4 }),
      Plot.text(rows, { x: "name", y: "y", text: d => fmt(d.y), dy: -8, fill: css("--ink2") }),
      Plot.tip(rows, Plot.pointer({ x: "name", y: "y", title: d => `${d.name}\nnDCG@10 ${fmt(d.y)}` }))
    ]
  });
}

function depthChart() {
  const series = ["jevragrank", "dense+ce"].filter(s => state.systems.has(s)).map(s => {
    const rs = labelled(s, "k").concat(runs()[s] ? [runs()[s]] : []);
    return { s, pts: rs.map(r => [r.options.candidates ?? 50, r.metrics["ndcg@10"].mean]).sort((a, b) => a[0] - b[0]) };
  });
  return lineChart(series, "candidates reranked →", "↑ nDCG@10", "log", "linear");
}

function scalingChart() {
  const sci = SUMMARY.datasets.scifact?.runs || {};
  const series = ["jevrank", "jevragrank", "dense"].filter(s => state.systems.has(s)).map(s => ({
    s, pts: Object.values(sci).filter(x => x.system === s && /^n\d+$/.test(x.label)).map(r => [r.corpus_size, r.latency.p50 * 1000]).sort((a, b) => a[0] - b[0])
  }));
  return lineChart(series, "documents →", "↑ p50 latency (ms)", "log", "log");
}

function strategyChart() {
  const r = runs();
  return barChart([["score", "jevragrank"], ["noul", "jevragrank@noul"], ["choice", "jevragrank@choice"]]
    .filter(([, k]) => r[k]).map(([name, k]) => ({ name, y: r[k].metrics["ndcg@10"].mean, fill: colorOf("jevragrank") })));
}

function modelChart() {
  const r = runs(), rows = [];
  for (const s of ["jevragrank", "jevrank"]) {
    if (r[`${s}@0.8b`]) rows.push({ name: `${LABELS[s]} 0.8B`, y: r[`${s}@0.8b`].metrics["ndcg@10"].mean, fill: `color-mix(in srgb, ${colorOf(s)} 45%, ${css("--surface")})` });
    if (r[s]) rows.push({ name: `${LABELS[s]} 2B`, y: r[s].metrics["ndcg@10"].mean, fill: colorOf(s) });
  }
  return barChart(rows);
}

function mount(id, node) { const el = document.getElementById(id); el.replaceChildren(node); }

function render() {
  buildFilters();
  const r = runs(), best = MAIN.filter(s => r[s]).sort((a, b) => r[b].metrics["ndcg@10"].mean - r[a].metrics["ndcg@10"].mean)[0];
  const jr = r.jevragrank, ce = r["dense+ce"];
  document.getElementById("lede").textContent = jr && ce
    ? `On ${state.dataset === "scifact" ? "SciFact" : "NFCorpus"}, JevRAGRank scores nDCG@10 ${fmt(jr.metrics["ndcg@10"].mean)} at ${ms(jr.latency.p50)} ms per query, against ${fmt(ce.metrics["ndcg@10"].mean)} at ${ms(ce.latency.p50)} ms for a bge cross-encoder. Best overall: ${LABELS[best]}.`
    : "Benchmark results for JevRAGRank and JevRank against standard retrieval baselines.";
  if (typeof Plot === "undefined") { document.getElementById("chart-scatter").textContent = "Charts need the Plot library, which didn't load; the table below has every number."; table(); return; }
  mount("chart-scatter", scatter());
  mount("chart-depth", depthChart());
  mount("chart-strategy", strategyChart());
  mount("chart-scaling", scalingChart());
  mount("chart-model", modelChart());
  table();
}

const env = SUMMARY.env || {};
document.getElementById("env-line").textContent = `Hardware: ${env.gpu || "unknown GPU"}${env.driver ? ", driver " + env.driver : ""}. Generated ${SUMMARY.generated ? SUMMARY.generated.slice(0, 10) : ""}.`;
render();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
</script>
```

- [ ] **Step 5: Run tests, lint, and generate**

Run: `uv run pytest tests/test_report.py -q && uv run ruff check . && uv run jevragrank-bench report`
Expected: tests pass; `docs/report.html` and `docs/report.fragment.html` written.

- [ ] **Step 6: Look once, then publish.** Take one screenshot of `docs/report.html` via the browser pane, spending the look on the charts. Fix what it shows in one pass. Then publish `docs/report.fragment.html` with the Artifact tool:
  - `icon: "chart"`
  - `description: "Speed and accuracy of JevRAGRank and JevRank against BM25, dense, cross-encoder and LLM reranking on BEIR SciFact and NFCorpus."`

  Put the resulting link in the README (Task 15).

- [ ] **Step 7: Commit and push**

```bash
git add bench/report.py bench/report_template.html tests/test_report.py docs/report.html
git commit -m "Add interactive HTML benchmark report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

(`docs/report.fragment.html` is regenerated on demand; add `docs/report.fragment.html` to `.gitignore` in this commit.)

---

### Task 15: README, CI and final review

**Files:**
- Create: `README.md`, `.github/workflows/tests.yml`
- Modify: `.gitignore`

- [ ] **Step 1: Write `.github/workflows/tests.yml`**

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: ["3.10", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: ${{ matrix.python }}
      - run: uv sync --extra dev
      - run: uv run ruff check .
      - run: uv run pytest -q
```

- [ ] **Step 2: Write `README.md`** with these sections, in order and with real content:
  1. **Title and one-paragraph pitch.** Headline numbers are pulled from `results/summary.json`: JevRAGRank's and the cross-encoder's nDCG@10 and p50 latency on each dataset.
  2. **Headline chart**, using light/dark switching:
     ```html
     <picture>
       <source media="(prefers-color-scheme: dark)" srcset="docs/charts/speed-accuracy-dark.svg">
       <img alt="Speed vs accuracy on SciFact and NFCorpus" src="docs/charts/speed-accuracy-light.svg">
     </picture>
     ```
     Add the link to the interactive report from Task 14.
  3. **What it is:** two pipelines in 5 lines each, and why typed decisions are an interesting reranker.
  4. **Quickstart:**
     ```bash
     pip install "jevragrank[dense] @ git+https://github.com/korra-pickell/JevRAG"
     ```
     Then start decider (link `serving/README.md`), then:
     ```python
     from jevragrank import Doc, JevClient, JevRAGRank, JevRank

     docs = [Doc("1", "Daily vitamin D3 did not reduce fractures in a 25,000-person trial.", "VITAL"),
             Doc("2", "The Eiffel Tower was completed in 1889.", "Paris")]
     jev = JevClient("http://localhost:8000")
     rag = JevRAGRank(docs, jev=jev, candidates=2)
     print(rag.search("does vitamin D prevent fractures?", k=1)[0].doc.title)
     print(JevRank(docs, jev=jev).search("does vitamin D prevent fractures?", k=1)[0].doc.title)
     ```
     Mention that hosted Jev works by passing its base URL and `TYPESAFE_API_KEY`.
  5. **API:**
     - constructor parameters for `JevRAGRank` (`candidates`, `strategy`, `batch_size`, `blend`, `on_error`, `retriever`) and `JevRank` (`group_size`, `preview_tokens`, `survivors`, `final_pool`);
     - `JevClient` (`cache_dir`, `cache_mode`, `concurrency`).
  6. **Results:** one table per dataset from `uv run jevragrank-bench summarize`, pasted verbatim; the `accuracy`, `depth`, `scaling` and `ablations` charts (dark/light `<picture>`); and a short honest reading of each.
  7. **Reproduce:** `uv sync --extra bench --extra dev`, start decider, `uv run jevragrank-bench all [--jevrank-sample 100]`, the dense+llm server-swap note, and the expected runtime taken from `docs/spike-notes.md`.
  8. **Methodology:** tuning on dev splits (link `docs/tuning.md`), latency protocol, significance test, candidate recall.
  9. **Limitations:**
     - a 2B judge on a laptop GPU;
     - two English datasets;
     - tournament cost grows linearly with the corpus;
     - any sampled runs, named explicitly.
  10. **Credits and license:** decider (Apache-2.0), OpenJev projects, BEIR, bge; MIT.

- [ ] **Step 3: Run the full verification** (superpowers:verification-before-completion)

```bash
uv run ruff check . && uv run pytest -q
uv run jevragrank-bench summarize > /dev/null && uv run jevragrank-bench plots && uv run jevragrank-bench report
git status --short
```
Expected: clean lint, all tests pass, and regenerated charts/report show no diff (the results are unchanged).

- [ ] **Step 4: Request code review** with superpowers:requesting-code-review over the whole repo. Address the findings.

- [ ] **Step 5: Commit, push, and check CI**

```bash
git add README.md .github .gitignore
git commit -m "Add README with results, and CI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```
Open `https://github.com/korra-pickell/JevRAG/actions` and confirm the `tests` workflow is green on both Python versions.
