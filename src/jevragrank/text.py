from __future__ import annotations

from .types import Doc

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


def truncate(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    limit = max_tokens * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.8:
        cut = cut[:space]
    return cut.rstrip() + " …"


def passage(doc: Doc, max_tokens: int) -> str:
    return truncate(doc.full_text, max_tokens)
