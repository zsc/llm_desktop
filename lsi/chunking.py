from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int


def chunk_text(text: str, *, chunk_chars: int, overlap_chars: int) -> Iterable[TextChunk]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be < chunk_chars")

    n = len(text)
    if n == 0:
        return

    step = chunk_chars - overlap_chars
    start = 0
    while start < n:
        end = min(start + chunk_chars, n)
        yield TextChunk(text=text[start:end], start=start, end=end)
        if end >= n:
            break
        start += step
