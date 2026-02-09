from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


VectorKind = Literal["text", "image"]


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        denom = np.linalg.norm(x) + 1e-12
        return x / denom
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


class _NumpyIndex:
    def __init__(self, path: Path, dim: int):
        self._path = path
        self._dim = dim
        self._lock = threading.RLock()
        self._ids = np.zeros((0,), dtype=np.int64)
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._load()

    @property
    def dim(self) -> int:
        return self._dim

    def _load(self) -> None:
        if not self._path.exists():
            return
        data = np.load(str(self._path), allow_pickle=False)
        self._ids = data["ids"].astype(np.int64)
        self._vectors = data["vectors"].astype(np.float32)
        if self._vectors.shape[1] != self._dim:
            raise ValueError(f"Vector dim mismatch: {self._vectors.shape[1]} != {self._dim}")

    def save(self) -> None:
        with self._lock:
            tmp = self._path.with_name(f"{self._path.stem}.tmp{self._path.suffix}")
            np.savez_compressed(str(tmp), ids=self._ids, vectors=self._vectors)
            tmp.replace(self._path)

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vectors = vectors.astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError("vectors must be shape (n, dim)")
        vectors = _l2_normalize(vectors)
        with self._lock:
            self._ids = np.concatenate([self._ids, np.asarray(ids, dtype=np.int64)])
            self._vectors = np.concatenate([self._vectors, vectors], axis=0)

    def knn(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        query = _l2_normalize(query.astype(np.float32).reshape(1, -1))
        with self._lock:
            if self._vectors.shape[0] == 0:
                return []
            sims = (self._vectors @ query.T).reshape(-1)
            topk = int(min(k, sims.shape[0]))
            idx = np.argpartition(-sims, topk - 1)[:topk]
            idx = idx[np.argsort(-sims[idx])]
            return [(int(self._ids[i]), float(sims[i])) for i in idx]


class _HnswIndex:
    def __init__(self, path: Path, dim: int, space: str = "cosine"):
        try:
            import hnswlib  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("hnswlib is not installed; install extras: local-semantic-indexer[vectors]") from e

        self._hnswlib = hnswlib
        self._path = path
        self._meta_path = path.with_suffix(path.suffix + ".meta.json")
        self._dim = dim
        self._space = space
        self._lock = threading.RLock()
        self._index = self._hnswlib.Index(space=space, dim=dim)
        self._count = 0
        self._capacity = 0
        self._load_or_init()

    @property
    def dim(self) -> int:
        return self._dim

    def _load_or_init(self) -> None:
        if self._path.exists() and self._meta_path.exists():
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            if int(meta.get("dim")) != self._dim or meta.get("space") != self._space:
                raise ValueError("HNSW index metadata mismatch; run `lsi index --rescan`.")
            self._capacity = int(meta.get("capacity", 0)) or 1
            self._count = int(meta.get("count", 0))
            self._index.load_index(str(self._path), max_elements=self._capacity)
            self._index.set_ef(int(meta.get("ef", 64)))
            return
        self._capacity = 1024
        self._count = 0
        self._index.init_index(max_elements=self._capacity, ef_construction=200, M=16)
        self._index.set_ef(64)

    def _ensure_capacity(self, need: int) -> None:
        if need <= self._capacity:
            return
        new_cap = self._capacity
        while new_cap < need:
            new_cap *= 2
        self._index.resize_index(new_cap)
        self._capacity = new_cap

    def save(self) -> None:
        with self._lock:
            self._index.save_index(str(self._path))
            self._meta_path.write_text(
                json.dumps(
                    {
                        "dim": self._dim,
                        "space": self._space,
                        "count": self._count,
                        "capacity": self._capacity,
                        "ef": 64,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def add(self, ids: list[int], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vectors = vectors.astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError("vectors must be shape (n, dim)")
        with self._lock:
            self._ensure_capacity(self._count + len(ids) + 1)
            self._index.add_items(vectors, np.asarray(ids, dtype=np.int64))
            self._count += len(ids)

    def knn(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        q = query.astype(np.float32).reshape(1, -1)
        with self._lock:
            if self._count == 0:
                return []
            labels, dists = self._index.knn_query(q, k=min(k, self._count))
        labels = labels.reshape(-1)
        dists = dists.reshape(-1)
        # For cosine space, hnswlib returns distance = 1 - cosine_similarity.
        sims = 1.0 - dists
        return [(int(l), float(s)) for l, s in zip(labels.tolist(), sims.tolist())]


@dataclass
class VectorStore:
    root_dir: Path
    dim: int
    backend: Literal["auto", "numpy", "hnsw"] = "auto"
    text_dim: int | None = None  # Optional separate dim for text embeddings

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.text_index = self._make_index(self.root_dir / "clip_text.npz", self.root_dir / "clip_text.hnsw", self.dim)
        self.image_index = self._make_index(self.root_dir / "clip_image.npz", self.root_dir / "clip_image.hnsw", self.dim)
        # Optional separate text index for sentence-transformers
        self._text_dim = self.text_dim or self.dim
        self._separate_text_index = self.text_dim is not None and self.text_dim != self.dim
        if self._separate_text_index:
            self.st_text_index = self._make_index(
                self.root_dir / "st_text.npz", self.root_dir / "st_text.hnsw", self._text_dim
            )
        else:
            self.st_text_index = self.text_index

    def _make_index(self, numpy_path: Path, hnsw_path: Path, dim: int):
        if self.backend in ("auto", "hnsw"):
            try:
                return _HnswIndex(hnsw_path, dim=dim)
            except Exception:
                if self.backend == "hnsw":
                    raise
        return _NumpyIndex(numpy_path, dim=dim)

    def add(self, kind: VectorKind, ids: list[int], vectors: np.ndarray, *, use_st: bool = False) -> None:
        if kind == "text" and use_st and self._separate_text_index:
            self.st_text_index.add(ids, vectors)
        elif kind == "text":
            self.text_index.add(ids, vectors)
        else:
            self.image_index.add(ids, vectors)

    def knn(self, kind: VectorKind, query: np.ndarray, k: int, *, use_st: bool = False) -> list[tuple[int, float]]:
        if kind == "text" and use_st and self._separate_text_index:
            return self.st_text_index.knn(query, k)
        elif kind == "text":
            return self.text_index.knn(query, k)
        else:
            return self.image_index.knn(query, k)

    def save(self) -> None:
        with self._lock:
            self.text_index.save()
            self.image_index.save()
            if self._separate_text_index and hasattr(self, 'st_text_index') and self.st_text_index is not self.text_index:
                self.st_text_index.save()
