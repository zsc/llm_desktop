from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .hybrid import SearchResult


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def llm_parse_query(ollama: Any, query: str) -> dict[str, Any] | None:
    system = "Return strict JSON only. No prose."
    prompt = (
        "You are a query parser for a local file search engine.\n"
        "Convert the user's query into structured filters.\n\n"
        f"User query: {query!r}\n\n"
        "Return JSON with keys:\n"
        '- query: string (rewritten/expanded query for retrieval)\n'
        '- type: null or one of ["text","pdf","image","audio","video","archive"]\n'
        '- path_glob: null or string\n'
        '- since: null or YYYY-MM-DD\n'
        "- keywords: optional list of strings\n"
    )
    out = ollama.generate(prompt, system=system)
    return _extract_json(out)


@dataclass(frozen=True)
class RerankOutput:
    results: list[SearchResult]


def llm_rerank(ollama: Any, query: str, results: list[SearchResult], *, max_items: int = 30) -> RerankOutput:
    items = results[: max(0, int(max_items))]
    if not items:
        return RerankOutput(results=results)

    system = "Return strict JSON only. No prose."
    lines = []
    for i, r in enumerate(items):
        snippet = (r.snippet or "").replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "…"
        lines.append(f"{i}. path={r.path} type={r.type} snippet={snippet!r}")

    prompt = (
        "You are a reranker for local file search.\n"
        f"Query: {query!r}\n\n"
        "Candidates:\n"
        + "\n".join(lines)
        + "\n\nReturn JSON:\n"
        '{"order":[int,...], "explanations":{"0":"short reason", ...}}\n'
    )

    out = ollama.generate(prompt, system=system)
    data = _extract_json(out) or {}
    order = data.get("order")
    explanations = data.get("explanations") or {}

    if not isinstance(order, list) or not order:
        return RerankOutput(results=results)

    new_items: list[SearchResult] = []
    used = set()
    for idx in order:
        if not isinstance(idx, int):
            continue
        if idx < 0 or idx >= len(items):
            continue
        if idx in used:
            continue
        used.add(idx)
        r = items[idx]
        exp = explanations.get(str(idx)) if isinstance(explanations, dict) else None
        if isinstance(exp, str) and exp.strip():
            r.explanation = exp.strip()
        new_items.append(r)

    # Append any leftover items (stable).
    for i, r in enumerate(items):
        if i not in used:
            new_items.append(r)

    # Keep items beyond max_items unchanged.
    return RerankOutput(results=new_items + results[len(items) :])

