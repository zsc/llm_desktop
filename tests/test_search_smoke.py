import os
import tempfile
import unittest
from pathlib import Path

from lsi.config import Config, LoadSheddingConfig
from lsi.indexer import IndexingService, build_embedder
from lsi.paths import ensure_app_dirs, get_db_path, get_vectors_dir
from lsi.search.hybrid import hybrid_search
from lsi.storage.db import Database
from lsi.storage.vectors import VectorStore


class TestSearchSmoke(unittest.TestCase):
    def test_index_and_search_text(self) -> None:
        with tempfile.TemporaryDirectory() as app_dir, tempfile.TemporaryDirectory() as root_dir:
            os.environ["LSI_APP_DIR"] = app_dir
            try:
                root = Path(root_dir)
                f1 = root / "meeting.txt"
                f2 = root / "random.txt"
                f1.write_text("客户 会议 录音 报价 讨论", encoding="utf-8")
                f2.write_text("unrelated content", encoding="utf-8")

                cfg = Config(
                    roots=[str(root)],
                    allow_roots=[str(root)],
                )
                cfg.indexing.max_workers_io = 1
                cfg.indexing.max_workers_model = 1
                cfg.load_shedding = LoadSheddingConfig(cpu_pause_percent=1000, mem_available_min_gb=0, pause_after_s=999, resume_after_s=999)

                ensure_app_dirs()
                db = Database(get_db_path())
                db.init_schema()
                embedder = build_embedder(cfg)
                vectors = VectorStore(get_vectors_dir(), dim=int(embedder.dim))

                svc = IndexingService(
                    cfg=cfg,
                    roots=[str(root)],
                    db=db,
                    vectors=vectors,
                    embedder=embedder,
                    transcriber=None,
                    rescan=True,
                    dry_run=False,
                )
                svc.run_once()

                results = hybrid_search(db=db, vectors=vectors, embedder=embedder, query="报价", top=5)
                self.assertTrue(results, "expected at least one result")
                self.assertTrue(results[0].path.endswith("meeting.txt"))
            finally:
                os.environ.pop("LSI_APP_DIR", None)

