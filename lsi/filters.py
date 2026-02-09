from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path


def _norm_path(p: str) -> str:
    p = os.path.expanduser(p)
    try:
        return str(Path(p).resolve())
    except Exception:
        return str(Path(p))


def _norm_for_match(p: str) -> str:
    return _norm_path(p).replace(os.sep, "/")


def is_within_any_root(path: str, roots: list[str]) -> bool:
    if not roots:
        return True
    real = Path(_norm_path(path))
    for root in roots:
        try:
            root_p = Path(_norm_path(root))
            real.relative_to(root_p)
            return True
        except Exception:
            continue
    return False


def matches_any_glob(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    p = _norm_for_match(path)
    for pat in patterns:
        pat_norm = pat.replace(os.sep, "/")
        if fnmatch.fnmatchcase(p, pat_norm):
            return True
        # Treat patterns ending with "/**" as matching the directory itself too.
        if pat_norm.endswith("/**"):
            base = pat_norm[: -len("/**")]
            if base and fnmatch.fnmatchcase(p, base):
                return True
    return False


@dataclass(frozen=True)
class PathFilter:
    ignore_patterns: list[str]
    allow_roots: list[str]

    def allowed(self, path: str) -> bool:
        if not is_within_any_root(path, self.allow_roots):
            return False
        if matches_any_glob(path, self.ignore_patterns):
            return False
        return True
