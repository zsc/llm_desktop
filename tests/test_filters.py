import tempfile
import unittest
from pathlib import Path

from lsi.filters import PathFilter


class TestFilters(unittest.TestCase):
    def test_sensitive_paths_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "home"
            ssh_dir = root / ".ssh"
            ssh_dir.mkdir(parents=True)
            f = ssh_dir / "id_rsa"
            f.write_text("secret", encoding="utf-8")

            pf = PathFilter(ignore_patterns=["**/.ssh/**", "**/*id_rsa*"], allow_roots=[str(root)])
            self.assertFalse(pf.allowed(str(f)))
