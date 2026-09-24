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
