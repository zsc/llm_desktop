from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _hash_bytes(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little", signed=False)


@dataclass
class HashEmbedder:
    dim: int = 384
    name: str = "hash"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = TOKEN_RE.findall(text.lower())
            if not tokens:
                # fallback to whole-text hash
                idx = _hash_bytes(text.encode("utf-8", errors="ignore")) % self.dim
                vecs[i, idx] = 1.0
                continue
            for tok in tokens:
                idx = _hash_bytes(tok.encode("utf-8")) % self.dim
                vecs[i, idx] += 1.0
            vecs[i] = np.log1p(vecs[i])
        return _l2_normalize(vecs)

    def embed_images(self, images: list[Any]) -> np.ndarray:
        vecs = np.zeros((len(images), self.dim), dtype=np.float32)
        for i, im in enumerate(images):
            try:
                rgb = im.convert("RGB")
                arr = np.asarray(rgb, dtype=np.uint8)
                h = _hash_bytes(arr.tobytes())
            except Exception:
                h = _hash_bytes(repr(im).encode("utf-8"))
            # Spread bits into a sparse-ish vector.
            for j in range(8):
                idx = (h >> (j * 8)) & 0xFF
                vecs[i, idx % self.dim] += 1.0
        return _l2_normalize(vecs)

