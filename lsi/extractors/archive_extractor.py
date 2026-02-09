from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class ArchiveEntry:
    inner_path: str
    size_bytes: int | None
    mtime_ns: int | None


@dataclass(frozen=True)
class ArchivePeek:
    entries: list[ArchiveEntry]


class ZipSlipError(RuntimeError):
    pass


def _safe_join(root: Path, inner: str) -> Path:
    # zip/tar entries always use forward slashes
    inner = inner.lstrip("/").replace("\\", "/")
    dest = (root / inner).resolve()
    root_resolved = root.resolve()
    try:
        dest.relative_to(root_resolved)
    except Exception as e:
        raise ZipSlipError(f"Blocked ZipSlip path: {inner!r}") from e
    return dest


def _rm_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


class ArchiveExtractor:
    def __init__(self, *, cache_dir: Path, large_threshold_bytes: int, enable_extract_small: bool):
        self._cache_dir = cache_dir
        self._large_threshold_bytes = int(large_threshold_bytes)
        self._enable_extract_small = bool(enable_extract_small)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def is_large(self, size_bytes: int) -> bool:
        return int(size_bytes) > self._large_threshold_bytes

    def peek(self, path: Path) -> ArchivePeek:
        suffix = path.name.lower()
        if suffix.endswith(".zip"):
            return ArchivePeek(entries=list(self._peek_zip(path)))
        if suffix.endswith(".tar") or suffix.endswith(".tgz") or suffix.endswith(".tar.gz"):
            return ArchivePeek(entries=list(self._peek_tar(path)))
        if suffix.endswith(".7z"):
            return ArchivePeek(entries=list(self._peek_7z(path)))
        raise RuntimeError(f"Unsupported archive: {path}")

    def extract_to_cache(self, path: Path) -> Path:
        if not self._enable_extract_small:
            raise RuntimeError("Archive extraction disabled by config.")
        h = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
        dest_root = self._cache_dir / "extract" / h
        _rm_tree(dest_root)
        dest_root.mkdir(parents=True, exist_ok=True)

        suffix = path.name.lower()
        try:
            if suffix.endswith(".zip"):
                self._extract_zip(path, dest_root)
            elif suffix.endswith(".tar") or suffix.endswith(".tgz") or suffix.endswith(".tar.gz"):
                self._extract_tar(path, dest_root)
            elif suffix.endswith(".7z"):
                self._extract_7z(path, dest_root)
            else:
                raise RuntimeError(f"Unsupported archive: {path}")
        except Exception:
            _rm_tree(dest_root)
            raise
        return dest_root

    def cleanup_extract(self, extract_dir: Path) -> None:
        _rm_tree(extract_dir)

    def _peek_zip(self, path: Path) -> Iterator[ArchiveEntry]:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                inner = info.filename
                # Zip stores date_time as localtime tuple.
                mtime_ns = None
                try:
                    import datetime as _dt

                    mtime_ns = int(_dt.datetime(*info.date_time).timestamp() * 1e9)
                except Exception:
                    pass
                yield ArchiveEntry(inner_path=inner, size_bytes=int(info.file_size), mtime_ns=mtime_ns)

    def _extract_zip(self, path: Path, dest_root: Path) -> None:
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                dest = _safe_join(dest_root, info.filename)
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info, "r") as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)

    def _peek_tar(self, path: Path) -> Iterator[ArchiveEntry]:
        mode = "r:gz" if path.name.lower().endswith((".tgz", ".tar.gz")) else "r:"
        with tarfile.open(path, mode) as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                inner = m.name
                mtime_ns = int(m.mtime * 1e9) if m.mtime else None
                yield ArchiveEntry(inner_path=inner, size_bytes=int(m.size), mtime_ns=mtime_ns)

    def _extract_tar(self, path: Path, dest_root: Path) -> None:
        mode = "r:gz" if path.name.lower().endswith((".tgz", ".tar.gz")) else "r:"
        with tarfile.open(path, mode) as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                dest = _safe_join(dest_root, m.name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    continue
                with src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)

    def _peek_7z(self, path: Path) -> Iterator[ArchiveEntry]:
        try:
            import py7zr  # type: ignore
        except Exception as e:
            raise RuntimeError("7z support requires `py7zr` (install extras: local-semantic-indexer[archives]).") from e

        with py7zr.SevenZipFile(path, mode="r") as z:
            for name, info in (z.list() or {}).items():  # type: ignore[union-attr]
                size = getattr(info, "uncompressed", None)
                yield ArchiveEntry(inner_path=str(name), size_bytes=int(size) if size is not None else None, mtime_ns=None)

    def _extract_7z(self, path: Path, dest_root: Path) -> None:
        try:
            import py7zr  # type: ignore
        except Exception as e:
            raise RuntimeError("7z support requires `py7zr` (install extras: local-semantic-indexer[archives]).") from e

        # py7zr does not provide a per-file safe extraction hook; extract then validate.
        with py7zr.SevenZipFile(path, mode="r") as z:
            z.extractall(path=dest_root)
        # Validate no ZipSlip-like paths.
        root_resolved = dest_root.resolve()
        for p in dest_root.rglob("*"):
            if not p.is_file():
                continue
            try:
                p.resolve().relative_to(root_resolved)
            except Exception as e:
                raise ZipSlipError(f"Blocked ZipSlip path after extraction: {p}") from e

