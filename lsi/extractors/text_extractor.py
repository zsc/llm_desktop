from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TextExtraction:
    text: str


class TextExtractor:
    def __init__(self, *, max_bytes: int = 20 * 1024 * 1024):
        self._max_bytes = int(max_bytes)

    def extract(self, path: Path) -> TextExtraction:
        data = path.read_bytes()
        if len(data) > self._max_bytes:
            data = data[: self._max_bytes]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="ignore")
        return TextExtraction(text=text)

