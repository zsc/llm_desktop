from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PdfExtraction:
    pages: list[str]


class PdfExtractor:
    def extract(self, path: Path) -> PdfExtraction:
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception as e:
            raise RuntimeError("PDF support requires `pypdf` (install extras: local-semantic-indexer[pdf]).") from e

        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return PdfExtraction(pages=pages)

