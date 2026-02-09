from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass(frozen=True)
class FileRecord:
    file_id: int
    path: str
    real_path: str
    type: str
    size_bytes: int
    mtime_ns: int
    status: str


class Database:
    def __init__(self, path: Path):
        self._path = path
        self._local = threading.local()
        self._write_lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def init_schema(self) -> None:
        with self._write_lock:
            conn = self.connect()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    real_path TEXT NOT NULL,
                    type TEXT NOT NULL,
                    mime TEXT,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    ctime_ns INTEGER,
                    hash TEXT,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
                    chunk_type TEXT NOT NULL,
                    start_offset INTEGER,
                    end_offset INTEGER,
                    start_time_ms INTEGER,
                    end_time_ms INTEGER,
                    snippet TEXT,
                    vector_id INTEGER,
                    created_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_vector_id ON chunks(vector_id);

                CREATE TABLE IF NOT EXISTS frames (
                    frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
                    timestamp_ms INTEGER,
                    width INTEGER,
                    height INTEGER,
                    vector_id INTEGER,
                    created_at INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_frames_file_id ON frames(file_id);
                CREATE INDEX IF NOT EXISTS idx_frames_vector_id ON frames(vector_id);

                CREATE TABLE IF NOT EXISTS archive_entries (
                    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
                    inner_path TEXT NOT NULL,
                    size_bytes INTEGER,
                    mtime_ns INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_archive_entries_file_id ON archive_entries(file_id);

                CREATE TABLE IF NOT EXISTS index_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS fts_files USING fts5(
                    path,
                    basename,
                    snippet,
                    archive_inner_paths,
                    content=''
                );
                """
            )
            conn.commit()

    def get_state(self, key: str, default: Any | None = None) -> Any:
        conn = self.connect()
        row = conn.execute("SELECT value FROM index_state WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return row["value"]

    def set_state(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False)
        with self._write_lock:
            conn = self.connect()
            conn.execute(
                """
                INSERT INTO index_state(key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, payload, now_ms()),
            )
            conn.commit()

    def allocate_vector_ids(self, n: int) -> list[int]:
        if n <= 0:
            return []
        with self._write_lock:
            conn = self.connect()
            row = conn.execute("SELECT value FROM index_state WHERE key='next_vector_id'").fetchone()
            next_id = int(json.loads(row["value"])) if row else 1
            ids = list(range(next_id, next_id + n))
            conn.execute(
                """
                INSERT INTO index_state(key, value, updated_at)
                VALUES ('next_vector_id', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (json.dumps(next_id + n), now_ms()),
            )
            conn.commit()
        return ids

    def upsert_file(
        self,
        *,
        path: str,
        real_path: str,
        type: str,
        size_bytes: int,
        mtime_ns: int,
        status: str,
        error_message: str | None = None,
        ctime_ns: int | None = None,
        mime: str | None = None,
        content_hash: str | None = None,
    ) -> int:
        ts = now_ms()
        with self._write_lock:
            conn = self.connect()
            conn.execute(
                """
                INSERT INTO files(path, real_path, type, mime, size_bytes, mtime_ns, ctime_ns, hash, status, error_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    real_path=excluded.real_path,
                    type=excluded.type,
                    mime=excluded.mime,
                    size_bytes=excluded.size_bytes,
                    mtime_ns=excluded.mtime_ns,
                    ctime_ns=excluded.ctime_ns,
                    hash=excluded.hash,
                    status=excluded.status,
                    error_message=excluded.error_message,
                    updated_at=excluded.updated_at
                """,
                (
                    path,
                    real_path,
                    type,
                    mime,
                    int(size_bytes),
                    int(mtime_ns),
                    int(ctime_ns) if ctime_ns is not None else None,
                    content_hash,
                    status,
                    error_message,
                    ts,
                    ts,
                ),
            )
            row = conn.execute("SELECT file_id FROM files WHERE path=?", (path,)).fetchone()
            conn.commit()
        assert row is not None
        return int(row["file_id"])

    def get_file(self, path: str) -> sqlite3.Row | None:
        conn = self.connect()
        return conn.execute("SELECT * FROM files WHERE path=?", (path,)).fetchone()

    def file_changed(self, path: str, *, size_bytes: int, mtime_ns: int) -> bool:
        row = self.get_file(path)
        if row is None:
            return True
        return int(row["size_bytes"]) != int(size_bytes) or int(row["mtime_ns"]) != int(mtime_ns)

    def clear_file_children(self, file_id: int) -> None:
        with self._write_lock:
            conn = self.connect()
            conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
            conn.execute("DELETE FROM frames WHERE file_id=?", (file_id,))
            conn.execute("DELETE FROM archive_entries WHERE file_id=?", (file_id,))
            conn.commit()

    def insert_chunks(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self._write_lock:
            conn = self.connect()
            conn.executemany(
                """
                INSERT INTO chunks(file_id, chunk_type, start_offset, end_offset, start_time_ms, end_time_ms, snippet, vector_id, created_at)
                VALUES (:file_id, :chunk_type, :start_offset, :end_offset, :start_time_ms, :end_time_ms, :snippet, :vector_id, :created_at)
                """,
                rows,
            )
            conn.commit()

    def insert_frames(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self._write_lock:
            conn = self.connect()
            conn.executemany(
                """
                INSERT INTO frames(file_id, timestamp_ms, width, height, vector_id, created_at)
                VALUES (:file_id, :timestamp_ms, :width, :height, :vector_id, :created_at)
                """,
                rows,
            )
            conn.commit()

    def insert_archive_entries(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self._write_lock:
            conn = self.connect()
            conn.executemany(
                """
                INSERT INTO archive_entries(file_id, inner_path, size_bytes, mtime_ns)
                VALUES (:file_id, :inner_path, :size_bytes, :mtime_ns)
                """,
                rows,
            )
            conn.commit()

    def upsert_fts_file(
        self, file_id: int, *, path: str, basename: str, snippet: str, archive_inner_paths: str
    ) -> None:
        with self._write_lock:
            conn = self.connect()
            conn.execute("DELETE FROM fts_files WHERE rowid=?", (file_id,))
            conn.execute(
                "INSERT INTO fts_files(rowid, path, basename, snippet, archive_inner_paths) VALUES (?, ?, ?, ?, ?)",
                (file_id, path, basename, snippet, archive_inner_paths),
            )
            conn.commit()

    def fts_search(self, query: str, *, limit: int) -> list[sqlite3.Row]:
        conn = self.connect()
        sql = """
            SELECT rowid AS file_id, path, basename, snippet, archive_inner_paths, bm25(fts_files) AS rank
            FROM fts_files
            WHERE fts_files MATCH ?
            ORDER BY rank
            LIMIT ?
        """
        try:
            return list(conn.execute(sql, (query, int(limit))).fetchall())
        except sqlite3.OperationalError:
            # Fallback for queries with special FTS syntax: treat as a quoted phrase.
            safe = " ".join(query.replace('"', " ").split())
            if not safe:
                return []
            return list(conn.execute(sql, (f'"{safe}"', int(limit))).fetchall())

    def get_files_by_ids(self, file_ids: list[int]) -> dict[int, sqlite3.Row]:
        if not file_ids:
            return {}
        conn = self.connect()
        q = ",".join("?" for _ in file_ids)
        rows = conn.execute(f"SELECT * FROM files WHERE file_id IN ({q})", tuple(file_ids)).fetchall()
        return {int(r["file_id"]): r for r in rows}

    def lookup_chunk_by_vector_ids(self, vector_ids: list[int]) -> list[sqlite3.Row]:
        if not vector_ids:
            return []
        conn = self.connect()
        q = ",".join("?" for _ in vector_ids)
        return list(conn.execute(f"SELECT * FROM chunks WHERE vector_id IN ({q})", tuple(vector_ids)).fetchall())

    def lookup_frame_by_vector_ids(self, vector_ids: list[int]) -> list[sqlite3.Row]:
        if not vector_ids:
            return []
        conn = self.connect()
        q = ",".join("?" for _ in vector_ids)
        return list(conn.execute(f"SELECT * FROM frames WHERE vector_id IN ({q})", tuple(vector_ids)).fetchall())

    def delete_virtual_files_for_archive(self, archive_path: str) -> None:
        """
        Remove records for files extracted from an archive (where real_path is the archive_path),
        excluding the archive container itself. Vector entries are not reclaimed.
        """
        with self._write_lock:
            conn = self.connect()
            rows = conn.execute(
                "SELECT file_id FROM files WHERE real_path=? AND path!=?",
                (archive_path, archive_path),
            ).fetchall()
            file_ids = [int(r["file_id"]) for r in rows]
            if file_ids:
                q = ",".join("?" for _ in file_ids)
                conn.execute(f"DELETE FROM fts_files WHERE rowid IN ({q})", tuple(file_ids))
            conn.execute("DELETE FROM files WHERE real_path=? AND path!=?", (archive_path, archive_path))
            conn.commit()
