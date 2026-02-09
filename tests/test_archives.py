import tempfile
import unittest
import zipfile
from pathlib import Path

from lsi.extractors.archive_extractor import ArchiveExtractor, ZipSlipError


class TestArchives(unittest.TestCase):
    def test_zip_slip_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            zpath = tmp_path / "bad.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("../evil.txt", "nope")

            ae = ArchiveExtractor(cache_dir=tmp_path / "cache", large_threshold_bytes=10**9, enable_extract_small=True)
            with self.assertRaises(ZipSlipError):
                ae.extract_to_cache(zpath)

    def test_archive_large_peek_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            zpath = tmp_path / "a.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("docs/readme.txt", "hello")

            ae = ArchiveExtractor(cache_dir=tmp_path / "cache", large_threshold_bytes=1, enable_extract_small=True)
            self.assertTrue(ae.is_large(zpath.stat().st_size))
            peek = ae.peek(zpath)
            self.assertTrue(any(e.inner_path == "docs/readme.txt" for e in peek.entries))
