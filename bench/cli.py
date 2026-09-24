from __future__ import annotations

import argparse
import asyncio
import json
import sys
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
    if hasattr(sys.stdout, "reconfigure"):  # tables use dashes; Windows pipes default to cp1252
        sys.stdout.reconfigure(encoding="utf-8")
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
