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
