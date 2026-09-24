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
    import torch  # lazy: only dense+llm needs torch

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

    def dense() -> DenseRetriever:
        return DenseRetriever(ctx.docs, cache_dir=emb_cache, device=ctx.device)

    t0 = perf_counter()
    if name == "bm25":
        system: Searchable = BM25Retriever(ctx.docs)
    elif name == "dense":
        system = dense()
    elif name == "dense+ce":
        system = RetrieveThenRerank(dense(), CrossEncoderReranker(device=ctx.device),
                                    candidates=opts["candidates"])
    elif name == "dense+llm":
        system = RetrieveThenRerank(dense(), LLMReranker(device=ctx.device or "cuda"),
                                    candidates=opts["candidates"])
    elif name == "jevragrank":
        rerank_opts = {k: v for k, v in opts.items() if k != "candidates"}
        system = JevRAGRank(ctx.docs, jev=jev, candidates=opts["candidates"],
                            retriever=dense(), **rerank_opts)
    else:
        final = JevReranker(jev, strategy="score", batch_size=opts["batch_size"],
                            max_passage_tokens=opts["max_passage_tokens"])
        system = JevRank(ctx.docs, jev=jev, group_size=opts["group_size"],
                         preview_tokens=tuple(opts["preview_tokens"]),
                         survivors=tuple(opts["survivors"]), final_pool=opts["final_pool"],
                         final_reranker=final)
    return Built(system, jev, perf_counter() - t0)
