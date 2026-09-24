from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .datasets import load_beir
from .metrics import (
    METRICS,
    bootstrap_ci,
    candidate_recall,
    latency_stats,
    per_query,
    randomization_test,
)

REFERENCE = "dense+ce"
NON_TEST_SPLITS = ("train", "dev")


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def _dataset_and_split(dir_name: str) -> tuple[str, str]:
    """Invert RunSpec.dir_name; dataset names may contain hyphens (e.g. trec-covid)."""
    name, _, split = dir_name.rpartition("-")
    if name and split in NON_TEST_SPLITS:
        return name, split
    return dir_name, "test"


def summarize(results_dir: Path, data_dir: Path) -> dict:
    results_dir = Path(results_dir)
    out: dict = {"generated": datetime.now(timezone.utc).isoformat(), "datasets": {}}
    for ds_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        name, split = _dataset_and_split(ds_dir.name)
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
            if not qrels:
                continue
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
            if meta.get("env") and not out.get("env"):
                out["env"] = meta["env"]
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


def _vs_reference(run: dict, ref: dict | None) -> str:
    """'better'/'worse' only when significant (p < 0.05) against the dense+ce mean."""
    p = run["p_vs_ce"]
    if p is None or ref is None:
        return "—"
    if p >= 0.05:
        return "tie"
    mean, ref_mean = run["metrics"]["ndcg@10"]["mean"], ref["metrics"]["ndcg@10"]["mean"]
    return "better" if mean > ref_mean else "worse"


def markdown_table(summary: dict, dataset: str) -> str:
    runs = summary["datasets"][dataset]["runs"]
    ref = runs.get(REFERENCE)
    lines = ["| system | nDCG@10 (95% CI) | Recall@10 | MRR@10 | p50 latency | vs cross-encoder |",
             "|---|---|---|---|---|---|"]
    for name, r in sorted(runs.items(), key=lambda kv: -kv[1]["metrics"]["ndcg@10"]["mean"]):
        n = r["metrics"]["ndcg@10"]
        lines.append(f"| {name} | {n['mean']:.3f} ({n['ci'][0]:.3f}–{n['ci'][1]:.3f}) | "
                     f"{r['metrics']['recall@10']['mean']:.3f} | "
                     f"{r['metrics']['mrr@10']['mean']:.3f} | "
                     f"{r['latency']['p50'] * 1000:.0f} ms | {_vs_reference(r, ref)} |")
    return "\n".join(lines)
