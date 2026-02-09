from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .filters import PathFilter


@dataclass(frozen=True)
class ScanItem:
    path: Path


def scan_roots(roots: list[str], *, path_filter: PathFilter, follow_symlinks: bool = False) -> Iterator[ScanItem]:
    for root in roots:
        root_path = Path(os.path.expanduser(root))
        if not root_path.exists():
            continue
        if root_path.is_file():
            if path_filter.allowed(str(root_path)):
                yield ScanItem(path=root_path)
            continue

        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=follow_symlinks):
            dirpath_p = Path(dirpath)
            # Prune ignored directories early.
            pruned: list[str] = []
            for d in list(dirnames):
                p = dirpath_p / d
                if not path_filter.allowed(str(p)):
                    pruned.append(d)
            for d in pruned:
                dirnames.remove(d)

            for name in filenames:
                p = dirpath_p / name
                if not path_filter.allowed(str(p)):
                    continue
                if not follow_symlinks and p.is_symlink():
                    continue
                if p.is_file():
                    yield ScanItem(path=p)

