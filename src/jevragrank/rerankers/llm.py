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
        prompts = [RATING_PROMPT.format(query=query,
                                        passage=passage(h.doc, self.max_passage_tokens))
                   for h in hits]
        scores = await asyncio.to_thread(self._locked, prompts)
        return rank_hits(hits, scores, stage="llm", k=k)
