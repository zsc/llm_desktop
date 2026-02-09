from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class ImageExtraction:
    image: Image.Image
    width: int
    height: int


class ImageExtractor:
    def extract(self, path: Path) -> ImageExtraction:
        with Image.open(path) as im:
            im = im.convert("RGB")
            width, height = im.size
            return ImageExtraction(image=im.copy(), width=width, height=height)

