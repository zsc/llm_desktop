from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ..storage.db import Database
from ..storage.vectors import VectorStore


@dataclass
class SearchResult:
    file_id: int
    path: str
    type: str
    score: float
    snippet: str = ""
    timestamp_ms: int | None = None
    sources: set[str] = field(default_factory=set)
    explanation: str | None = None


def _parse_since(s: str) -> int | None:
    if not s:
        return None
    # Ignore placeholder/help text
    if s in ("YYYY-MM-DD", "--since must be YYYY-MM-DD"):
        return None
    # Accept YYYY-MM-DD.
    try:
        dt = datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        raise ValueError("--since must be YYYY-MM-DD")
    return int(dt.timestamp() * 1e9)


def hybrid_search(
    *,
    db: Database,
    vectors: VectorStore,
    embedder: Any,
    query: str,
    top: int,
    type_filter: str | None = None,
    path_glob: str | None = None,
    since: str | None = None,
    vector_k: int = 50,
    fts_k: int = 50,
    w_vec: float = 0.7,
    w_fts: float = 0.3,
    text_embedder: Any | None = None,
) -> list[SearchResult]:
    since_ns = _parse_since(since or "") if since else None

    # Determine which embedder to use for text search
    use_st_text = text_embedder is not None
    text_embed = text_embedder if use_st_text else embedder
    
    # Vector recall (text->text + text->image).
    q_vec_text = text_embed.embed_texts([query]).astype(np.float32)
    text_hits = vectors.knn("text", q_vec_text[0], vector_k, use_st=use_st_text)
    
    # For image search, always use the main embedder (CLIP)
    q_vec_image = embedder.embed_texts([query]).astype(np.float32)
    image_hits = vectors.knn("image", q_vec_image[0], vector_k)

    vec_scores: dict[int, float] = {}
    vec_snippet: dict[int, tuple[str, int | None]] = {}
    vec_sources: dict[int, set[str]] = {}

    if text_hits:
        rows = db.lookup_chunk_by_vector_ids([vid for vid, _ in text_hits])
        by_vec = {int(r["vector_id"]): r for r in rows if r["vector_id"] is not None}
        for vid, sim in text_hits:
            r = by_vec.get(int(vid))
            if r is None:
                continue
            fid = int(r["file_id"])
            vec_scores[fid] = max(vec_scores.get(fid, 0.0), float(sim))
            vec_sources.setdefault(fid, set()).add("vector:text")
            if fid not in vec_snippet:
                vec_snippet[fid] = (str(r["snippet"] or ""), int(r["start_time_ms"]) if r["start_time_ms"] is not None else None)

    if image_hits:
        rows = db.lookup_frame_by_vector_ids([vid for vid, _ in image_hits])
        by_vec = {int(r["vector_id"]): r for r in rows if r["vector_id"] is not None}
        for vid, sim in image_hits:
            r = by_vec.get(int(vid))
            if r is None:
                continue
            fid = int(r["file_id"])
            vec_scores[fid] = max(vec_scores.get(fid, 0.0), float(sim))
            vec_sources.setdefault(fid, set()).add("vector:image")
            if fid not in vec_snippet:
                vec_snippet[fid] = ("", int(r["timestamp_ms"]) if r["timestamp_ms"] is not None else None)

    # Lexical recall (FTS5).
    fts_rows = db.fts_search(query, limit=fts_k) if query.strip() else []
    fts_scores: dict[int, float] = {}
    fts_snippet: dict[int, str] = {}
    for r in fts_rows:
        fid = int(r["file_id"])
        rank = float(r["rank"])
        score = 1.0 / (1.0 + abs(rank))
        fts_scores[fid] = max(fts_scores.get(fid, 0.0), score)
        if fid not in fts_snippet:
            fts_snippet[fid] = str(r["snippet"] or "")

    # Merge + filter.
    all_ids = set(vec_scores) | set(fts_scores)
    files = db.get_files_by_ids(list(all_ids))
    results: list[SearchResult] = []
    for fid in all_ids:
        fr = files.get(fid)
        if fr is None:
            continue
        ftype = str(fr["type"])
        fpath = str(fr["path"])
        if type_filter and ftype != type_filter:
            continue
        if path_glob and not fnmatch.fnmatchcase(fpath, os.path.expanduser(path_glob)):
            continue
        if since_ns is not None and int(fr["mtime_ns"]) < since_ns:
            continue

        score = w_vec * vec_scores.get(fid, 0.0) + w_fts * fts_scores.get(fid, 0.0)
        snippet = fts_snippet.get(fid, "")
        ts = None
        if fid in vec_snippet:
            s, ts = vec_snippet[fid]
            if s:
                snippet = s
        src = set()
        if fid in vec_sources:
            src |= vec_sources[fid]
        if fid in fts_scores:
            src.add("fts")

        results.append(SearchResult(file_id=fid, path=fpath, type=ftype, score=float(score), snippet=snippet, timestamp_ms=ts, sources=src))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[: int(top)]
