from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    def embed_images(self, images) -> np.ndarray: ...  # images: list[PIL.Image.Image]


@dataclass(frozen=True)
class TranscriptSegment:
    start_ms: int
    end_ms: int
    text: str

