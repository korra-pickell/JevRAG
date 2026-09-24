# JevRAGRank — Design Spec

- **Date:** 2026-09-24
- **Status:** Approved in brainstorming, pending written-spec review
- **Repo:** `jevragrank` (MIT), local at `C:\PRGM\JevRAG`, to be published on GitHub after explicit approval

## 1. Summary

JevRAGRank is an open-source Python library and benchmark that uses **Jev-style typed decisions** (via the open `POST /v1/systemone` wire API) for retrieval-augmented generation:

1. **JevRAGRank** — dense-embedding retrieval selects the top-K candidates; Jev reranks them.
2. **JevRank** — a Jev-only retriever with **no embeddings at all**: a batched `choice` tournament over the whole corpus.

A reproducible benchmark compares both against standard RAG baselines (BM25, dense, cross-encoder rerank, same-size LLM rerank) on BEIR SciFact and NFCorpus, for accuracy and speed. The results are published as charts in the README and as an interactive HTML report.

## 2. Goals and non-goals

**Goals**
- A small, backend-agnostic library: it works with any Jev-compatible server (local decider, razorback16/openjev, or hosted TypeSafe Jev) by changing only the base URL.
- Honest, reproducible benchmarks on consumer hardware: an 8 GB laptop GPU, with a one-command reproduction.
- Clear charts covering the speed/accuracy trade-off, where the Jev variants win or lose, and why.

**Non-goals (this version)**
- LangChain/LlamaIndex integrations, a PyPI release.
- Hybrid BM25+dense baseline; FiQA/ArguAna datasets.
- Jev-built topic-index retriever (researched, rejected as too risky for v1).
- Running the benchmark against hosted Jev or the 26B OpenJev. Users can do this themselves via the base URL.

## 3. Background and constraints

### 3.1 The Jev wire API (`POST /v1/systemone`)

Request: `{model, state, questions}`, where `questions` is a dict of named, typed questions:

```json
{
  "model": "decider-2b",
  "state": "Query: does vitamin D reduce fracture risk?",
  "questions": {
    "d0": {"type": "score", "instructions": "How well does this passage answer the query?\n\nPassage: ...",
           "criteria": ["irrelevant", "same topic, doesn't help", "partially answers", "mostly answers", "fully answers"]},
    "d1": {"type": "noul", "instructions": "Does this passage help answer the query?\n\nPassage: ..."},
    "pick": {"type": "choice", "instructions": "Which passage most likely answers the query?",
             "criteria": {"p0": "passage text...", "p1": "passage text..."}}
  }
}
```

Response: `{model, answers, usage}`. The answer shapes are:
- `noul` → `P(yes)`.
- `choice` → `{choice, probabilities, confidence}`.
- `score` → `{score = Σ i·pᵢ, legend, probabilities, confidence}`.

### 3.2 Backend chosen for the benchmarks

- **Hardware:** NVIDIA RTX 4070 Laptop, **8 GB VRAM**, Windows 11, Docker Desktop 28, Python 3.12, uv 0.11.
- **razorback16/openjev** (DiffusionGemma 26B) needs ≥24 GB VRAM. It is excluded from local runs.
- **Chosen:** [Mapika/decider](https://github.com/Mapika/decider) **decider-2b**, as used by [SiliconLabAI/OpenJev](https://github.com/SiliconLabAI/OpenJev). Its properties:
  - Fine-tuned from Qwen3.5 2B, Apache-2.0, about 4 GB VRAM.
  - 32k context, `choice` with 2–255 options, `score` with 2–10 levels.
  - A continuous-batching HTTP server.
- **Also available:** decider-0.8B, used for the model-size ablation.
- **Serving risk:** decider's fast path uses `torch.compile` and CUDA graphs, which are unlikely to work natively on Windows. It will be served from **WSL2 or Docker (GPU)**, verified in the Phase 0 spike (§10).

### 3.3 Prior art (differentiation)

- [laguagu/jev-rerank-bench](https://github.com/laguagu/jev-rerank-bench): Jev reranking vs Voyage and others on Finnish legal corpora. It uses hosted Jev, reranking only.
- [kashyaprparmar/jev-rankkit](https://github.com/kashyaprparmar/jev-rankkit): a general reranking library with hosted Jev as an optional backend.
- **What is new here:**
  - A Jev-only retriever.
  - Open, local, laptop-GPU backends.
  - A same-size "Jev vs prompted base model" comparison.
  - An ablation of the three Jev primitives, run on standard BEIR sets.

## 4. Architecture

```
jevragrank/
├─ pyproject.toml            uv/hatchling; CLI entry point `jevragrank-bench`
├─ src/jevragrank/
│  ├─ __init__.py            exports JevClient, JevRAGRank, JevRank, Doc, Hit
│  ├─ types.py               Doc(id, title, text); Hit(doc, score, stage_scores: dict, degraded: bool)
│  ├─ jev_client.py          async client for /v1/systemone
│  ├─ text.py                truncation/preview helpers (token budget ≈ 4 chars/token)
│  ├─ retrievers/
│  │   ├─ base.py            Retriever protocol
│  │   ├─ dense.py           sentence-transformers BAAI/bge-base-en-v1.5 + numpy cosine
│  │   └─ bm25.py            bm25s
│  ├─ rerankers/
│  │   ├─ base.py            Reranker protocol
│  │   ├─ jev.py             JevReranker(strategy="score"|"noul"|"choice")
│  │   ├─ cross_encoder.py   BAAI/bge-reranker-base                  (baseline)
│  │   └─ llm.py             Qwen3.5-2B base, prompted 0–4, logprobs (baseline)
│  └─ pipelines.py           JevRAGRank, JevRank, RetrieveThenRerank (generic composer)
├─ bench/
│  ├─ datasets.py            BEIR loaders (HF: BeIR/scifact, BeIR/nfcorpus + qrels)
│  ├─ systems.py             registry: system name → constructed pipeline
│  ├─ run.py                 CLI: run systems × datasets; resumable JSONL output
│  ├─ metrics.py             ranx metrics, bootstrap CIs, paired randomization test
│  ├─ plots.py               matplotlib charts (PNG+SVG, light+dark)
│  └─ report.py              interactive HTML report
├─ serving/                  decider server config (docker-compose.yml and/or WSL2 script)
├─ tests/
├─ results/                  benchmark outputs (committed: summary + per-query JSONL)
└─ docs/
   ├─ charts/                generated charts
   └─ superpowers/specs/     this spec
```

### 4.1 Interfaces

```python
class Retriever(Protocol):
    async def asearch(self, query: str, k: int) -> list[Hit]: ...

class Reranker(Protocol):
    async def arerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...
```

- Every public object has a sync wrapper (`search`, `rerank`) that runs the async version. Async is used internally so Jev batches run concurrently.
- `Hit.stage_scores` records every stage's score (e.g. `{"dense": 0.81, "jev": 0.92}`) for diagnostics and blending.

### 4.2 Public API

```python
from jevragrank import JevClient, JevRAGRank, JevRank

jev  = JevClient("http://localhost:8000", model="decider-2b")
rag  = JevRAGRank(docs, jev=jev, candidates=50, strategy="score")   # dense → Jev
pure = JevRank(docs, jev=jev)                                        # Jev only
hits = rag.search("does vitamin D reduce fracture risk?", k=10)
```

### 4.3 Packaging extras

- **Core install:** `httpx`, `numpy` only. JevRank needs nothing else.
- `[dense]`: `sentence-transformers`.
- `[bm25]`: `bm25s`.
- `[baselines]`: `transformers`, `torch`.
- `[bench]`: all of the above plus `datasets`, `ranx`, `matplotlib`.
- `[dev]`: `pytest`, `pytest-asyncio`, `ruff`.

## 5. Components

### 5.1 JevClient
- An async `httpx` client, `ask(state, questions) -> answers`, with a concurrency semaphore (default 16 in-flight requests) so the server's continuous batching stays busy.
- **Response cache:** optional on-disk cache (`cache_dir=`), keyed by the SHA-256 of the canonical request JSON plus the model name. It has hit/miss counters and `bypass_cache=True` for timing runs.
- **Health check:** runs lazily on first use (`GET /v1/models`) and raises `JevConnectionError` with the URL and a hint if the server is unreachable.
- **Errors:** described in §8.

### 5.2 Retrievers (baselines and candidate generation)
- `DenseRetriever`:
  - `BAAI/bge-base-en-v1.5` with that model's query instruction prefix.
  - Normalized embeddings with a numpy dot product (exact search; corpora are ≤5.2k docs).
  - Embeddings are cached to disk per corpus hash.
- `BM25Retriever`: `bm25s` with its default English stemming and stopwords.

### 5.3 Rerankers
- `JevReranker`: see §6.1.
- `CrossEncoderReranker`: `BAAI/bge-reranker-base` via sentence-transformers `CrossEncoder`, batch 32, passages truncated to 512 tokens.
- `LLMReranker` (baseline): **the Qwen3.5 2B base checkpoint that decider-2b was fine-tuned from** (the exact HF id is confirmed in Phase 0 from decider's model card).
  - It is prompted to rate relevance 0–4 (the same five labels as the Jev scale).
  - It reads next-token logprobs over `"0".."4"`, and the score is the expected value, the same aggregation as Jev's `score`.
  - This makes it a fair same-size, same-scale comparison: "typed-decision fine-tune plus API" vs "prompting the base model".

### 5.4 Pipelines
- `RetrieveThenRerank(retriever, reranker, candidates)`: the generic composer. It builds baselines 3 and 4 and JevRAGRank.
- `JevRAGRank`: `RetrieveThenRerank(DenseRetriever, JevReranker, candidates=50)` with friendly defaults.
- `JevRank`: the tournament of §6.2.

## 6. Algorithms

### 6.1 JevReranker (JevRAGRank's second stage)

The query goes in `state`. Candidates are grouped into requests of `batch_size` (default 12) questions, one per candidate, and requests run concurrently.

| strategy | per-candidate question | rank score |
|---|---|---|
| `score` (default) | "How well does this passage answer the query?" 5-level scale: *irrelevant* / *same topic, doesn't help* / *partially answers* / *mostly answers* / *fully answers* | `score / (n_levels − 1)` (expected level normalized to 0–1) |
| `noul` | "Does this passage help answer the query?" | `P(yes)` |
| `choice` | a single listwise question: all candidates as options `p0..pN`, passages as option text | option probability |

- **Passage format:** `"{title}\n{text}"` truncated to `max_passage_tokens` (default 512).
- **`choice` with more than 255 candidates:** split into chunks of ≤255. Each chunk's probabilities are rescaled by the chunk's maximum probability, then merged. This is documented as approximate.
- **Tie-break:** by the retrieval score.
- **`blend=α`** (default 0.0): `final = (1−α)·jev + α·minmax(retrieval score)`.
- **Configurable:** `instructions`, the scale labels, and `strategy`. The defaults are fixed after tuning on dev splits (§7.6).

### 6.2 JevRank (Jev-only choice tournament)

```
corpus → Round 1: groups of G=64, preview = title + first ~96 tokens
                  one `choice` per group ("Which passage most likely answers the query?")
                  keep top S1=3 per group by probability               (~250 survivors on SciFact)
       → Round 2: groups of 64, preview ~256 tokens, keep top S2=12      (~50 survivors)
       → Final:   JevReranker(strategy="score") on full passages       → ranked top-k
```

- **Configurable:** `group_size`, preview lengths per round, survivors per round, and number of rounds. Rounds 2+ run only while survivors exceed `final_pool` (default 50).
- **Position-bias control:** option order within each group is shuffled with a seed derived from `hash(query)`, making it deterministic per query.
- **Groups with too few documents:** a group of one document (possible for the final chunk) passes through with no request, which matches server behaviour.
- **Indexing:** building the index only precomputes previews (no model), so build time is about 0.
- **Expected cost:** about 90 requests per query on SciFact (5,183 docs) and about 65 on NFCorpus (3,633 docs).
- **Ranking depth:** docs eliminated before the final round are ranked below all finalists, ordered by the round they reached, then by their probability in that round. This gives a full ranking for metrics.
- **Final-round ties** are broken by the doc's probability in the last `choice` round. `blend` does not apply to JevRank, because there is no retrieval score.

### 6.3 Candidate recall (diagnostic)

- **Definition:** for each pipeline, the fraction of relevant documents that survive its hard filter. That filter is the dense top-K for JevRAGRank and round 1 for JevRank.
- **Reporting:** it is reported next to nDCG@10 to separate "the judge was wrong" from "the judge never saw the document".

## 7. Benchmark

### 7.1 Systems

| # | name | pipeline |
|---|---|---|
| 1 | `bm25` | BM25Retriever |
| 2 | `dense` | DenseRetriever (bge-base-en-v1.5) |
| 3 | `dense+ce` | dense top-50 → bge-reranker-base |
| 4 | `dense+llm` | dense top-50 → Qwen3.5-2B prompted |
| 5 | `jevragrank` | dense top-50 → decider-2b, `score` |
| 6 | `jevrank` | decider-2b tournament |

### 7.2 Ablations
- **Strategy:** `jevragrank` with `score` / `noul` / `choice`.
- **Rerank depth:** K ∈ {10, 20, 50, 100}, for `jevragrank` vs `dense+ce`.
- **Model size:** decider-0.8B vs decider-2b, for `jevragrank` and `jevrank`.
- **Tournament scaling:** SciFact subsets of 500, 1k, 2k docs and the full 5,183.
  - Every doc relevant to a query in the sampled query set is always included.
  - The rest are filled randomly with a fixed seed.
  - Measured for `jevrank`, `jevragrank`, and `dense`.

### 7.3 Datasets
BEIR test splits via Hugging Face (`BeIR/scifact`, `BeIR/scifact-qrels`, `BeIR/nfcorpus`, `BeIR/nfcorpus-qrels`):
- **SciFact:** 300 queries, 5,183 docs.
- **NFCorpus:** 323 queries, 3,633 docs, graded relevance.

### 7.4 Accuracy metrics
- **Metrics:** nDCG@10 (headline), Recall@10, MRR@10, candidate recall (§6.3), computed with `ranx`.
- **Confidence intervals:** 95% bootstrap CIs over queries (1,000 resamples, fixed seed).
- **Significance:** a paired Fisher randomization test of each system vs `dense+ce` (p < 0.05), marked on the charts.

### 7.5 Speed methodology
- **Latency p50/p95 per query:**
  - Queries run one after another.
  - Concurrency inside a query is allowed.
  - The first 5 queries are warm-up and excluded.
  - Time is recorded per stage (retrieve / rerank / per tournament round).
- **Throughput:** queries/sec with 16 concurrent queries.
- **Index build time:** corpus embedding / BM25 index / tournament previews.
- **Timing runs always use `bypass_cache=True`.** The cache is only for accuracy reruns and chart regeneration.
- **GPU sharing:** the 8 GB can't hold decider and Qwen3.5-2B at once, so systems run one at a time. The decider server is stopped before `dense+llm` runs. The harness checks free VRAM before a run and fails fast with a clear message.
- **Environment record:** every results file records GPU name, driver, CUDA, torch and package versions, backend model id, and git commit.

### 7.6 Tuning protocol (no test-set leakage)
- Jev question wording, scale labels, `batch_size`, and tournament settings are tuned on **SciFact train** and **NFCorpus dev** (about 50 queries each).
- The final settings are frozen in `bench/systems.py` before any test-split run.

### 7.7 Runtime budget
Estimated for the 4070 Laptop:
- **Rerank configurations:** about 10–15 minutes per dataset.
- **`jevrank`:** about 20–35 s per query, which is roughly 2–3 hours per dataset.

**Rule:** if Phase 0 measurements project any single run above **6 hours**, that run uses a **fixed-seed 100-query sample**. The sample size is stated in the results file, chart subtitles, and README.

### 7.8 Outputs and resumability
- **Per-query output:** `results/{dataset}/{system}.jsonl`, one line per query with the ranked doc ids and scores, stage timings, and any errors.
- **Aggregated output:** `results/summary.json` with metrics, CIs, significance, latency, throughput, and the environment record.
- **Resuming:** runs skip queries already present in the JSONL. Failed queries are not written, so they re-run on resume.
- **CLI:**
  - `jevragrank-bench run --dataset scifact --systems jevragrank,dense+ce`
  - `jevragrank-bench report`
  - `jevragrank-bench all`

## 8. Error handling
- **JevClient:**
  - Per-request timeout (default 60 s).
  - Up to 3 retries with exponential backoff and jitter on connection errors, timeouts, 429 and 5xx.
  - Other 4xx errors (e.g. too many options) raise `JevRequestError` at once, carrying the server's message.
  - An unreachable server raises `JevConnectionError` (§5.1).
- **Rerankers:**
  - `on_error="raise"` (default).
  - `on_error="fallback"` keeps the retrieval order for a failed batch and sets `Hit.degraded=True`, so production RAG apps degrade instead of crashing.
- **Benchmark:** always `raise`. Failed queries are logged to `results/{dataset}/{system}.errors.log` and retried on resume, never silently dropped. The summary reports the count of completed vs attempted queries.
- **Input validation:**
  - Empty corpus, `k ≤ 0`, and `candidates < k` raise `ValueError`.
  - Score scales must have 2–10 levels.
  - Group size must be 2–255.

## 9. Charts, report, README

### 9.1 Charts
- **Format:** matplotlib, output as PNG and SVG, in light and dark variants embedded with `<picture>` + `prefers-color-scheme`.
- **Colour:** the two Jev systems get saturated accent colours; baselines are muted.
- **Process:** the `dataviz` skill is loaded before chart code is written.

The charts:
1. **Speed vs accuracy (headline):** x = p50 latency (log), y = nDCG@10 with CI bars, one panel per dataset, Pareto frontier drawn.
2. **Accuracy bars:** nDCG@10 per system with CIs; significance vs `dense+ce` marked.
3. **Tournament scaling:** latency vs corpus size (log–log).
4. **Rerank depth:** two panels, nDCG@10 vs K and latency vs K, with no dual axes.
5. **Strategy ablation** and **model-size ablation** (small multiples).

### 9.2 Interactive report
- A self-contained HTML page with hover values and system toggles.
- It is published as a private Artifact link the user can share.

### 9.3 README
- **Contents:**
  - What JevRAGRank is and why.
  - Quickstart: install, start decider via `serving/`, 10-line example.
  - API reference for the two pipelines.
  - Results table and charts, methodology summary, one-command reproduction.
- **Limitations:**
  - A 2B backend on a laptop GPU.
  - Two English BEIR sets.
  - Tournament cost grows linearly with corpus size.
  - Results on larger backends may differ.
- **Credits:** OpenJev, decider, BEIR, bge.
- **License:** MIT.

## 10. Implementation phases (overview; the detailed plan comes next)
0. **Spike (throwaway):**
   - Serve decider-2b (and 0.8B) in WSL2 or Docker.
   - Confirm `/v1/systemone` request/response shapes and the choice option limit.
   - Measure latency for a 12-question score request and a 64-option choice request.
   - Confirm the Qwen3.5-2B base checkpoint id.
   - Update §7.7 projections.
1. **Scaffold:** package, types, JevClient with fake-server tests.
2. Retrievers and baseline rerankers.
3. JevReranker (all strategies) and JevRAGRank.
4. JevRank tournament.
5. Benchmark harness: datasets, runner, metrics, baseline sanity check.
6. Tuning on dev splits, then full test runs and ablations.
7. Charts, HTML report, README.
8. CI (GitHub Actions: ruff + unit tests), final review, publish.

## 11. Testing
- **Unit tests (no GPU, run in CI):** use a fake Jev server (`httpx.MockTransport`) returning deterministic probabilities. They cover:
  - request shapes per strategy and batching into `batch_size` questions;
  - `score`→rank maths; tie-break; blend;
  - the choice-chunking merge;
  - tournament group/survivor logic and exact request counts;
  - cache hit/miss/bypass;
  - retries and backoff, and error types;
  - `on_error="fallback"`;
  - input validation.
- **Metric sanity (marked `slow`):**
  - `bm25` and `dense` on SciFact must be within ±0.03 nDCG@10 of published BEIR numbers (about 0.67 BM25, about 0.74 bge-base).
  - Failure means the harness is wrong and blocks further benchmarking.
- **Live smoke test (marked `live`):** 3 queries end to end through `jevragrank` and `jevrank` against a running decider server.
- **Process:** TDD for library code.

## 12. Publishing and repo hygiene
- **Git:** initialized in `C:\PRGM\JevRAG` (branch `main`), with a repo-local identity *Korra Pickell &lt;korrapickell@gmail.com&gt;*. The user's global gitconfig has stray `user.name/email = "="` duplicates; this is left untouched but flagged. Commits are made as the work proceeds.
- **Ignored and committed files:**
  - `.gitignore` excludes the model caches, embedding caches, the Jev response cache, and `.venv`.
  - `results/` (small JSONL and summary) and `docs/charts/` are committed.
- **Publishing:**
  - `gh` is installed at the end, and the user runs `gh auth login`.
  - The public GitHub repo `jevragrank` is created and pushed **only after the user explicitly approves the final contents**.
  - Before the first push, the user may switch the commit email to their GitHub no-reply address.

## 13. Risks
| risk | mitigation |
|---|---|
| decider won't serve on Windows | WSL2 or Docker GPU; CPU eager mode as a last resort for smoke tests only |
| decider-2b too slow for the full tournament | 100-query sample rule (§7.7); tournament knobs (group size, previews) |
| decider-2b's context/option limits differ from docs | Phase 0 verifies; `JevReranker`/`JevRank` read limits from config |
| 2B judge is weak, so Jev loses to the cross-encoder | Report honestly: the benchmark's value is the comparison, not a guaranteed win; the backend-agnostic design lets users rerun with larger backends |
| Baseline numbers off vs BEIR | Sanity gate (§11) blocks further runs until fixed |
