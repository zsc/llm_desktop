from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TEXT_EXTS = {
    ".txt",
    ".md",
    ".rst",
    ".log",
    ".csv",
    ".tsv",
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".js",
    ".ts",
    ".go",
    ".java",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".rb",
    ".php",
    ".html",
    ".css",
    ".sh",
    ".zsh",
    ".toml",
    ".ini",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
ARCHIVE_EXTS = {".zip", ".tar", ".tgz", ".tar.gz", ".7z"}


@dataclass(frozen=True)
class DetectedType:
    type: str  # text/pdf/image/audio/video/archive


def detect_type(path: Path) -> DetectedType | None:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTS:
        return DetectedType("text")
    if suffix in PDF_EXTS:
        return DetectedType("pdf")
    if suffix in IMAGE_EXTS:
        return DetectedType("image")
    if suffix in AUDIO_EXTS:
        return DetectedType("audio")
    if suffix in VIDEO_EXTS:
        return DetectedType("video")
    if name.endswith(".tar.gz"):
        return DetectedType("archive")
    if suffix in ARCHIVE_EXTS:
        return DetectedType("archive")
    return None

