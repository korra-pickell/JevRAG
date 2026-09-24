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
    extra = random.Random(seed).sample(rest, n_docs - len(must))
    keep = {d.id for d in must} | {d.id for d in extra}
    note = "; ".join(filter(None, [ds.note, f"corpus {n_docs} of {len(ds.docs)} docs"]))
    return replace(ds, docs=[d for d in ds.docs if d.id in keep], note=note)
