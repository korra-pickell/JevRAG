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
    for q in ds.queries.values():
        for system in (rag, pure):
            hits = system.search(q, k=10)
            assert len(hits) == 10
            assert all(not h.degraded for h in hits)
