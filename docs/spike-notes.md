# Phase 0 spike notes: serving decider and probing the wire format

Date: 2026-09-24. Hardware: RTX 4070 Laptop GPU (8 GB, driver 566.36, WDDM), Windows 11. Stack: `decider-ai` 1.3.0, torch 2.14.0+cu126, transformers 5.17.0, triton-windows 3.8.0.post28, flash-linear-attention 0.5.2.

## How decider is served

decider runs **natively on Windows** and needs no Docker. The Docker files in `serving/` exist but were not needed and are untested.

```bash
uv venv serving/.venv --python cpython-3.12.13 --managed-python
uv pip install --python serving/.venv "decider-ai[serve]" --torch-backend cu126
uv pip install --python serving/.venv triton-windows
bash serving/start-decider.sh                         # Mapika/decider-2b on 127.0.0.1:8000
bash serving/start-decider.sh Mapika/decider-0.8b 8001
```

`start-decider.sh` / `.ps1` export these defaults. Each can be overridden by env:

```
DECIDER_GRAPH_TOKEN_BUDGET=8192
DECIDER_T_BUCKETS=256,512,768,1024,1536,2048,3072,4096,6144,8192
DECIDER_B_BUCKETS=1,4,16
```

With these defaults, start-up takes about 1.5–2 min: weights load, then 18 CUDA graphs are captured in about 100 s, including the first-run triton JIT. decider-2b then sits at about 5.5 GB, peaking at about 7.5 GB after a 255-option request, with no spill to shared memory.

What happened on the way there:

1. **Anaconda base interpreter.**
   - The first venv was built on anaconda's Python 3.12.4.
   - `import torch` failed with `OSError: [WinError 1114] ... c10.dll`. Anaconda's bundled old `vcruntime140.dll` is loaded first.
   - Fix: build the venv on a **uv-managed CPython** (`--managed-python`).
2. **Default CUDA-graph grid overflows 8 GB.**
   - Defaults are `GRAPH_TOKEN_BUDGET=32768`, 17 T buckets × B 1..32, 89 graphs.
   - VRAM grows with every captured graph (warm-up blocks cached per side-stream plus the graph pool). It hit 7.9 GB dedicated and then spilled 2–6 GB into shared system memory (Windows sysmem fallback). Capture slowed to a crawl: 40/89 graphs after 5 min.
   - Budget 16384 still spilled.
   - Budget 8192 with the full T ladder (66 graphs) finished in 517 s but left about 1.6 GB spilled.
   - The trimmed grid above: 18 graphs, 53 s, 7.0 GB, no spill. Latency was the same as with 66 graphs.
   - Another agent's `jevragrank-bench run --system dense` was holding about 2 GB of VRAM during the first attempt. **Nothing else may use the GPU while decider runs.**
3. **Reference kernels.**
   - transformers logged that `chunk_gated_delta_rule` and `causal_conv1d_fn` were falling back to reference PyTorch.
   - `flash-linear-attention` is installed by `decider-ai[serve]` but needs `triton`, which has no Windows wheel. Installing **`triton-windows`** enables it.
   - Effect on decider-2b:

     | | reference kernels | with fla |
     |---|---|---|
     | score×12 | 6.06 s | 4.45 s |
     | 128-option choice (eager, beyond the 8192 bucket) | 28.7 s | 1.65 s |
     | 255-option choice | 44.6 s | 3.4 s |
     | VRAM | 7.0 GB | 5.5 GB |

     The long-choice eager path under the reference kernels also spilled about 2 GB.
   - Answers agree to about 0.01–0.02 between the two kernel paths.
   - `causal_conv1d` has no Windows build and stays on the reference path.
4. `DECIDER_COMPILE` already defaults to 0 in `decider.serve` (1.3.0), so torch.compile was never involved.

## Wire format (as served)

The probe's request shapes (`serving/probe.py`, the plan's shapes) were **accepted verbatim**:

- `score`: `criteria` is a list of level strings.
- `noul`: `instructions` only.
- `choice`: `criteria` is a `{name: description}` map.
- `state` is a plain string.

No change to `probe.py` was needed. The server also accepts:

- choice `criteria` as a list of bare names;
- score `criteria` as a `{"0": ..., "1": ...}` map;
- noul `criteria` as `{"true": ..., "false": ...}`, which is optional;
- `independent` (default `true`) and `layout` as top-level request fields.

The request `model` field is accepted and ignored.

Response: `{"model": "decider-v10", "answers": {id: answer}, "usage": {"input_tokens": N, "output_tokens": 0}}`. The `model` value is the **served model's internal name**, not the HF id: `"decider-v10"` for 2b and `"decider-0.8b-v1"` for 0.8b. `/v1/models` returns `{"models": [{"name": "decider-v10", "description": ..., "release_date": "2026-09-19"}]}`. `/health` returns `{"ok": true, "model": "Mapika/decider-2b", "device": "cuda", "layout": "plain"}`.

Verbatim answer shapes, from `tests/fixtures/decider/*.json`:

```json
"noul":   {"type": "noul", "noul": 0.2487}
```

```json
"choice": {"type": "choice", "choice": "p0", "confidence": 0.398, "x_p_max": 0.4074, "certainty": 0.3103,
           "probabilities": {"p0": 0.4074, "p1": 0.0875, "p2": 0.0304, "...": "...", "p63": 0.0016}}
```

```json
"score":  {"type": "score", "score": 1.91, "confidence": 0.0, "x_p_max": 0.2964, "certainty": 0.0283,
           "legend": {"0": "irrelevant", "1": "same topic, doesn't help", "2": "partially answers",
                      "3": "mostly answers", "4": "fully answers"},
           "probabilities": {"0": 0.2964, "1": 0.1397, "2": 0.1721, "3": 0.1457, "4": 0.2461},
           "level_fit": {"0": 0.2487, "1": 0.1172, "2": 0.1444, "3": 0.1223, "4": 0.2065},
           "fit_mass": 0.8391}
```

Differences from spec §3.1:

- **noul** is an object `{"type": "noul", "noul": P(yes)}`, not a bare float.
- **score `probabilities`** is a dict keyed by level index strings `"0".."n-1"`, not a list and not keyed by label. `legend` is a dict of the same keys. `score` = Σ i·pᵢ, rounded to 2 dp.
- **choice `probabilities`** is a dict keyed by option name, in request order.
- Every answer carries `type` and extra fields:
  - `x_p_max` is the max probability.
  - `certainty` is 1 − normalised entropy.
  - `confidence` is TypeSafe's rescaled confidence, not p_max.
  - score only: `level_fit` and `fit_mass`.
- Probabilities are rounded to 4 dp.

Cost model:

- decider-2b runs with **isolated score levels** (`isolated_levels: true`). Each score question is expanded into one yes/no row per level. A 5-level score therefore costs about **5 noul rows**, which is why score×12 is about 4× noul×12.
- Every question is its own row (`independent: true`).
- The shared-prefix fork only kicks in when every row is at least 768 tokens (`DECIDER_SHARED_MIN_TOKENS`).

Errors:

- Invalid questions return **HTTP 422** with `{"detail": "..."}`.
- Oversized requests return 413. The limits: `max_rows` 1024 rows per request (score levels count as rows), `max_row_tokens` 36864, `max_request_tokens` 1048576.
- Queue overload returns 503 (`max_queue_rows` 4096).

Sanity check with contrasting passages for "does vitamin D reduce fracture risk?" on 2b:

| passage | noul | score | choice P |
|---|---|---|---|
| cooking passage (irrelevant) | 0.005 | 0.40 | 0.004 |
| null-result vitamin D trial | 0.58 | 2.37 | 0.069 |
| positive meta-analysis | 0.81 | 3.02 | 0.927 |

## Max options

- `choice` with 255 options: HTTP 200. It is about 21.5k tokens, runs eager, and has a p50 of about 3.4 s on 2b.
- 256 options: HTTP 422 `{"detail":"choice criteria: a map of 2..255 options"}`.
- A 128-option choice (10.7k tokens) has a p50 of 1.65 s.

## Measured latency (p50, final configuration, fla on, GPU otherwise idle)

Probe payloads:

- score and noul: 12 questions, each with an about 450-token passage. Every row pads to the T=512 bucket.
- choice: 64 options of about 85 tokens each, one 5.3k-token row, which falls in the T=6144 bucket.

| model | score×12 | noul×12 | choice×64 | 16 concurrent choice×64 (wall) | batching_factor |
|---|---|---|---|---|---|
| decider-2b | **4452 ms** | **1120 ms** | **931 ms** | 15253 ms | **16.2** |
| decider-0.8b | 2579 ms | 645 ms | 573 ms | 9274 ms | 16.0 |

Before `triton-windows`, with the reference kernels:

| model | score×12 | noul×12 | choice×64 | batching_factor |
|---|---|---|---|---|
| decider-2b | 6055 ms | 1533 ms | 1276 ms | 16.1 |
| decider-0.8b | 4163 ms | 1032 ms | 859 ms | 16.0 |

`batching_factor` = wall time of 16 concurrent choice×64 requests ÷ single-request p50. It was measured with 16 distinct states, three rounds, median. **A factor of about 16 means concurrency buys nothing.** The GPU is compute-saturated by one 5.3k-token row, and the T=6144 bucket only has B=1. Tournament groups effectively run serially.

## Projected runtimes

Rerank, per dataset: `(50/12) × p50(score12) × queries`.

| model | SciFact (300 q) | NFCorpus (323 q) |
|---|---|---|
| 2b | 5,565 s = **1.55 h** | 5,991 s = **1.66 h** |
| 0.8b | 0.90 h | 0.96 h |

These are all under 6 h, but well above the spec's 10–15 min estimate. For comparison, a noul-based rerank on 2b would take `(50/12) × 1.12 s × 300` = about 23 min on SciFact.

Tournament, per query: `ceil(docs/64) × p50(choice64) / min(16, groups) × batching_factor + 4 × p50(score12)`.

| model | dataset | groups | round 1 | final | per query | full split |
|---|---|---|---|---|---|---|
| 2b | SciFact (5,183 docs) | 81 | 81 × 0.931 / 16 × 16.22 = 76.4 s | 4 × 4.452 = 17.8 s | **94.2 s** | × 300 = **7.85 h** |
| 2b | NFCorpus (3,633 docs) | 57 | 53.8 s | 17.8 s | **71.6 s** | × 323 = **6.42 h** |
| 0.8b | SciFact | 81 | 46.4 s | 10.3 s | 56.7 s | 4.73 h |
| 0.8b | NFCorpus | 57 | 32.7 s | 10.3 s | 43.0 s | 3.86 h |

The formula leaves out round 2. That round has about 4 groups on SciFact (3 on NFCorpus) of 64 × 256-token previews, about 17k tokens each. Each runs eager at about 2.5–3 s, adding roughly 8–11 s per query.

Both 2b `jevrank` full-split runs project over 6 h. Decision:

**JEVRANK_SAMPLE = 100**

Task 12 uses `--sample 100` for `jevrank` runs. That projects to about 2.6 h on SciFact and 2.0 h on NFCorpus with 2b. The rerank configurations run on the full test splits.

Implications for later tasks:

- Tournament throughput cannot be improved by more client concurrency, because batching_factor ≈ 16. Fewer or shorter rows are the only lever.
- Keep `choice` rows at or under 8192 tokens where possible, so they replay a captured graph.
- Score costs about 4× noul, a relevant trade-off for Task 11 tuning.
- Never run `dense+llm`, or any other GPU job, while decider is up.
