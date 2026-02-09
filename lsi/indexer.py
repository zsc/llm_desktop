from __future__ import annotations

import hashlib
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunking import chunk_text
from .config import Config
from .extractors.archive_extractor import ArchiveExtractor
from .extractors.image_extractor import ImageExtractor
from .extractors.pdf_extractor import PdfExtractor
from .extractors.text_extractor import TextExtractor
from .extractors.video_extractor import VideoExtractor
from .filetypes import detect_type
from .filters import PathFilter, matches_any_glob
from .paths import get_cache_dir
from .scheduler.load_monitor import LoadMonitor
from .storage.db import Database, now_ms
from .storage.vectors import VectorStore


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _abs_path(p: Path) -> str:
    return str(p.expanduser().resolve())


@dataclass
class IndexStats:
    enqueued: int = 0
    processed: int = 0
    skipped: int = 0
    errors: int = 0


class SkipFile(RuntimeError):
    pass


class IndexingService:
    def __init__(
        self,
        *,
        cfg: Config,
        roots: list[str],
        db: Database,
        vectors: VectorStore,
        embedder: Any,
        transcriber: Any | None,
        rescan: bool,
        dry_run: bool,
        text_embedder: Any | None = None,
    ):
        self.cfg = cfg
        self.roots = [os.path.expanduser(r) for r in roots]
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.text_embedder = text_embedder if text_embedder is not None else embedder
        self.use_st_text = text_embedder is not None
        self.transcriber = transcriber
        self.rescan = bool(rescan)
        self.dry_run = bool(dry_run)

        self.stats = IndexStats()
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=1000)
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._model_sem = threading.Semaphore(max(1, int(cfg.indexing.max_workers_model)))
        self._inflight: set[str] = set()
        self._inflight_lock = threading.RLock()

        self._status_thread: threading.Thread | None = None
        self._last_status_ms: int = 0

        self._path_filter = PathFilter(ignore_patterns=cfg.ignore_patterns, allow_roots=cfg.allow_roots or self.roots)
        self._text_extractor = TextExtractor()
        self._image_extractor = ImageExtractor()
        self._pdf_extractor = PdfExtractor()
        self._archive_extractor = ArchiveExtractor(
            cache_dir=get_cache_dir(),
            large_threshold_bytes=cfg.archive.large_threshold_bytes,
            enable_extract_small=cfg.archive.enable_extract_small,
        )
        self._video_extractor = VideoExtractor(
            cache_dir=get_cache_dir(),
            frame_every_seconds=cfg.video.frame_every_seconds,
            max_frames_per_video=cfg.video.max_frames_per_video,
        )

        self._workers: list[threading.Thread] = []
        self._load_monitor = LoadMonitor(
            cpu_pause_percent=cfg.load_shedding.cpu_pause_percent,
            mem_available_min_gb=cfg.load_shedding.mem_available_min_gb,
            pause_after_s=cfg.load_shedding.pause_after_s,
            resume_after_s=cfg.load_shedding.resume_after_s,
        )
        self._load_monitor.attach_pause_event(self._pause)

    def start_workers(self) -> None:
        n = max(1, int(self.cfg.indexing.max_workers_io))
        self._stop.clear()
        self._load_monitor.start()
        self._status_thread = threading.Thread(target=self._status_loop, name="lsi-status-loop", daemon=True)
        self._status_thread.start()
        for i in range(n):
            t = threading.Thread(target=self._worker_loop, name=f"lsi-worker-{i}", daemon=True)
            self._workers.append(t)
            t.start()

    def stop_workers(self) -> None:
        self._stop.set()
        for _ in self._workers:
            self._queue.put(None)
        for t in self._workers:
            t.join(timeout=5)
        self._workers.clear()
        self._load_monitor.stop()
        if self._status_thread is not None:
            self._status_thread.join(timeout=2)
        self._status_thread = None
        self._flush_status()

    def enqueue_path(self, path: str) -> None:
        with self._inflight_lock:
            if path in self._inflight:
                return
            self._inflight.add(path)
        self._queue.put(path)
        self.stats.enqueued += 1

    def run_once(self) -> None:
        self.start_workers()
        try:
            self._scan_and_enqueue()
            self._queue.join()
            self.vectors.save()
        finally:
            self.stop_workers()

    def run_daemon(self) -> None:
        self.start_workers()
        try:
            while not self._stop.is_set():
                self._scan_and_enqueue()
                self.db.set_state("last_scan_ms", now_ms())
                time.sleep(float(self.cfg.indexing.rescan_interval_s))
        finally:
            self.stop_workers()

    def _status_loop(self) -> None:
        while not self._stop.is_set():
            self._flush_status()
            time.sleep(1.0)

    def _scan_and_enqueue(self) -> None:
        from .scanner import scan_roots

        self.db.set_state("roots", self.roots)
        self.db.set_state("daemon_pid", os.getpid())
        self.db.set_state("paused", self._pause.is_set())
        if self._pause.is_set():
            self.db.set_state("pause_reason", self._load_monitor.pause_reason)

        for item in scan_roots(self.roots, path_filter=self._path_filter):
            p = item.path
            dt = detect_type(p)
            if dt is None:
                continue
            abspath = _abs_path(p)
            if self.dry_run:
                print(abspath)
                continue
            try:
                st = p.stat()
            except Exception:
                continue
            if not self.rescan and not self.db.file_changed(abspath, size_bytes=st.st_size, mtime_ns=st.st_mtime_ns):
                self.stats.skipped += 1
                continue
            self.enqueue_path(abspath)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.2)
                continue
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if item is None:
                    return
                self._process_path(item)
            finally:
                if item is not None:
                    with self._inflight_lock:
                        self._inflight.discard(item)
                self._queue.task_done()

    def _flush_status(self) -> None:
        now = now_ms()
        # Throttle DB writes.
        if now - self._last_status_ms < 300:
            return
        self._last_status_ms = now
        self.db.set_state(
            "status",
            {
                "queue_length": int(self._queue.qsize()),
                "enqueued": self.stats.enqueued,
                "processed": self.stats.processed,
                "skipped": self.stats.skipped,
                "errors": self.stats.errors,
                "paused": self._pause.is_set(),
                "pause_reason": self._load_monitor.pause_reason,
                "paused_for_s": self._load_monitor.paused_for_s,
                "updated_at_ms": now,
            },
        )

    def _process_path(self, abspath: str) -> None:
        p = Path(abspath)
        try:
            st = p.stat()
        except Exception as e:
            self.stats.errors += 1
            self.db.upsert_file(
                path=abspath,
                real_path=abspath,
                type="unknown",
                size_bytes=0,
                mtime_ns=0,
                status="error",
                error_message=str(e),
            )
            return

        dt = detect_type(p)
        if dt is None:
            self.stats.skipped += 1
            return

        file_id = self.db.upsert_file(
            path=abspath,
            real_path=abspath,
            type=dt.type,
            size_bytes=st.st_size,
            mtime_ns=st.st_mtime_ns,
            ctime_ns=getattr(st, "st_ctime_ns", None),
            status="indexing",
        )
        self.db.clear_file_children(file_id)
        # Ensure the file is at least searchable by name/path even if extraction fails.
        self.db.upsert_fts_file(file_id, path=abspath, basename=p.name, snippet="", archive_inner_paths="")

        try:
            if dt.type == "text":
                self._index_text(file_id, p, abspath)
            elif dt.type == "pdf":
                self._index_pdf(file_id, p, abspath)
            elif dt.type == "image":
                self._index_image(file_id, p, abspath)
            elif dt.type == "audio":
                self._index_audio(file_id, p, abspath)
            elif dt.type == "video":
                self._index_video(file_id, p, abspath)
            elif dt.type == "archive":
                self._index_archive(file_id, p, abspath, size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)
            else:
                self.stats.skipped += 1
                self.db.upsert_file(
                    path=abspath,
                    real_path=abspath,
                    type=dt.type,
                    size_bytes=st.st_size,
                    mtime_ns=st.st_mtime_ns,
                    status="skipped",
                    error_message="unsupported type",
                )
                return

            self.db.upsert_file(
                path=abspath,
                real_path=abspath,
                type=dt.type,
                size_bytes=st.st_size,
                mtime_ns=st.st_mtime_ns,
                status="indexed",
                error_message=None,
            )
            self.stats.processed += 1
        except SkipFile as e:
            self.stats.skipped += 1
            self.db.upsert_file(
                path=abspath,
                real_path=abspath,
                type=dt.type,
                size_bytes=st.st_size,
                mtime_ns=st.st_mtime_ns,
                status="skipped",
                error_message=str(e),
            )
        except Exception as e:
            self.stats.errors += 1
            self.db.upsert_file(
                path=abspath,
                real_path=abspath,
                type=dt.type,
                size_bytes=st.st_size,
                mtime_ns=st.st_mtime_ns,
                status="error",
                error_message=str(e),
            )

    def _index_text(self, file_id: int, path: Path, abspath: str) -> None:
        extracted = self._text_extractor.extract(path)
        chunks = list(chunk_text(extracted.text, chunk_chars=self.cfg.indexing.chunk_chars, overlap_chars=self.cfg.indexing.chunk_overlap_chars))
        if not chunks:
            self.db.upsert_fts_file(file_id, path=abspath, basename=path.name, snippet="", archive_inner_paths="")
            return

        texts = [c.text for c in chunks]
        with self._model_sem:
            vecs = self.text_embedder.embed_texts(texts)
        vec_ids = self.db.allocate_vector_ids(len(chunks))
        self.vectors.add("text", vec_ids, vecs, use_st=self.use_st_text)

        created = now_ms()
        self.db.insert_chunks(
            {
                "file_id": file_id,
                "chunk_type": "text",
                "start_offset": c.start,
                "end_offset": c.end,
                "start_time_ms": None,
                "end_time_ms": None,
                "snippet": (c.text[:512] if c.text else ""),
                "vector_id": vid,
                "created_at": created,
            }
            for c, vid in zip(chunks, vec_ids)
        )
        self.db.upsert_fts_file(
            file_id,
            path=abspath,
            basename=path.name,
            snippet=(chunks[0].text[:512] if chunks else ""),
            archive_inner_paths="",
        )

    def _index_pdf(self, file_id: int, path: Path, abspath: str) -> None:
        try:
            pdf = self._pdf_extractor.extract(path)
        except Exception as e:
            raise SkipFile(str(e)) from e
        all_chunks = []
        for page_num, page_text in enumerate(pdf.pages):
            for c in chunk_text(page_text or "", chunk_chars=self.cfg.indexing.chunk_chars, overlap_chars=self.cfg.indexing.chunk_overlap_chars):
                all_chunks.append((page_num, c))

        if not all_chunks:
            self.db.upsert_fts_file(file_id, path=abspath, basename=path.name, snippet="", archive_inner_paths="")
            return

        texts = [c.text for _, c in all_chunks]
        with self._model_sem:
            vecs = self.text_embedder.embed_texts(texts)
        vec_ids = self.db.allocate_vector_ids(len(all_chunks))
        self.vectors.add("text", vec_ids, vecs, use_st=self.use_st_text)
        created = now_ms()
        self.db.insert_chunks(
            {
                "file_id": file_id,
                "chunk_type": "pdf_page",
                "start_offset": page_num,
                "end_offset": page_num,
                "start_time_ms": None,
                "end_time_ms": None,
                "snippet": (c.text[:512] if c.text else ""),
                "vector_id": vid,
                "created_at": created,
            }
            for (page_num, c), vid in zip(all_chunks, vec_ids)
        )
        self.db.upsert_fts_file(
            file_id,
            path=abspath,
            basename=path.name,
            snippet=(all_chunks[0][1].text[:512] if all_chunks else ""),
            archive_inner_paths="",
        )

    def _index_image(self, file_id: int, path: Path, abspath: str) -> None:
        img = self._image_extractor.extract(path)
        with self._model_sem:
            vec = self.embedder.embed_images([img.image])
        vec_id = self.db.allocate_vector_ids(1)[0]
        self.vectors.add("image", [vec_id], vec)
        created = now_ms()
        self.db.insert_frames(
            [
                {
                    "file_id": file_id,
                    "timestamp_ms": None,
                    "width": img.width,
                    "height": img.height,
                    "vector_id": vec_id,
                    "created_at": created,
                }
            ]
        )
        self.db.upsert_fts_file(file_id, path=abspath, basename=path.name, snippet="", archive_inner_paths="")

    def _index_audio(self, file_id: int, path: Path, abspath: str) -> None:
        if self.transcriber is None:
            raise SkipFile("whisper disabled/unavailable")
        with self._model_sem:
            segs = self.transcriber.transcribe(path)
        if not segs:
            self.db.upsert_fts_file(file_id, path=abspath, basename=path.name, snippet="", archive_inner_paths="")
            return
        texts = [s.text for s in segs]
        with self._model_sem:
            vecs = self.text_embedder.embed_texts(texts)
        vec_ids = self.db.allocate_vector_ids(len(segs))
        self.vectors.add("text", vec_ids, vecs, use_st=self.use_st_text)
        created = now_ms()
        self.db.insert_chunks(
            {
                "file_id": file_id,
                "chunk_type": "audio_segment",
                "start_offset": None,
                "end_offset": None,
                "start_time_ms": s.start_ms,
                "end_time_ms": s.end_ms,
                "snippet": s.text[:512],
                "vector_id": vid,
                "created_at": created,
            }
            for s, vid in zip(segs, vec_ids)
        )
        self.db.upsert_fts_file(
            file_id,
            path=abspath,
            basename=path.name,
            snippet=segs[0].text[:512],
            archive_inner_paths="",
        )

    def _index_video(self, file_id: int, path: Path, abspath: str) -> None:
        key = _sha256_hex(abspath + str(path.stat().st_mtime_ns))[:16]
        extracted = self._video_extractor.extract(path, key=key)

        # Frames
        frame_rows = []
        frame_imgs = []
        frame_meta = []
        for fr in extracted.frames:
            try:
                img = self._image_extractor.extract(fr.path)
            except Exception:
                continue
            frame_imgs.append(img.image)
            frame_meta.append((fr.timestamp_ms, img.width, img.height))
        if frame_imgs:
            with self._model_sem:
                vecs = self.embedder.embed_images(frame_imgs)
            vec_ids = self.db.allocate_vector_ids(len(frame_imgs))
            self.vectors.add("image", vec_ids, vecs)
            created = now_ms()
            frame_rows = [
                {
                    "file_id": file_id,
                    "timestamp_ms": ts,
                    "width": w,
                    "height": h,
                    "vector_id": vid,
                    "created_at": created,
                }
                for (ts, w, h), vid in zip(frame_meta, vec_ids)
            ]
            self.db.insert_frames(frame_rows)

        # Transcript
        snippet = ""
        if extracted.audio_wav and extracted.audio_wav.exists() and self.transcriber is not None:
            with self._model_sem:
                segs = self.transcriber.transcribe(extracted.audio_wav)
            if segs:
                texts = [s.text for s in segs]
                with self._model_sem:
                    vecs = self.text_embedder.embed_texts(texts)
                vec_ids = self.db.allocate_vector_ids(len(segs))
                self.vectors.add("text", vec_ids, vecs, use_st=self.use_st_text)
                created = now_ms()
                self.db.insert_chunks(
                    {
                        "file_id": file_id,
                        "chunk_type": "video_transcript",
                        "start_offset": None,
                        "end_offset": None,
                        "start_time_ms": s.start_ms,
                        "end_time_ms": s.end_ms,
                        "snippet": s.text[:512],
                        "vector_id": vid,
                        "created_at": created,
                    }
                    for s, vid in zip(segs, vec_ids)
                )
                snippet = segs[0].text[:512]

        self.db.upsert_fts_file(file_id, path=abspath, basename=path.name, snippet=snippet, archive_inner_paths="")

    def _index_archive(self, file_id: int, path: Path, abspath: str, *, size_bytes: int, mtime_ns: int) -> None:
        # Always index the archive container itself.
        if self._archive_extractor.is_large(size_bytes):
            peek = self._archive_extractor.peek(path)
            rows = [
                {
                    "file_id": file_id,
                    "inner_path": e.inner_path,
                    "size_bytes": e.size_bytes,
                    "mtime_ns": e.mtime_ns,
                }
                for e in peek.entries
            ]
            self.db.insert_archive_entries(rows)
            inner_text = "\n".join(e.inner_path for e in peek.entries[:5000])
            self.db.upsert_fts_file(file_id, path=abspath, basename=path.name, snippet="", archive_inner_paths=inner_text)
            return

        extract_dir = self._archive_extractor.extract_to_cache(path)
        try:
            # Clear previous extracted children from this archive.
            self.db.delete_virtual_files_for_archive(abspath)

            inner_paths: list[str] = []
            for inner in sorted(extract_dir.rglob("*")):
                if not inner.is_file():
                    continue
                rel = inner.relative_to(extract_dir).as_posix()
                if matches_any_glob("/" + rel, self.cfg.ignore_patterns):
                    continue
                vpath = f"{abspath}::{rel}"
                inner_paths.append(rel)
                self._index_virtual_file(vpath=vpath, real_archive=abspath, extracted_path=inner)
            self.db.upsert_fts_file(
                file_id,
                path=abspath,
                basename=path.name,
                snippet="",
                archive_inner_paths="\n".join(inner_paths[:5000]),
            )
        finally:
            self._archive_extractor.cleanup_extract(extract_dir)

    def _index_virtual_file(self, *, vpath: str, real_archive: str, extracted_path: Path) -> None:
        dt = detect_type(extracted_path)
        if dt is None:
            return
        try:
            st = extracted_path.stat()
        except Exception:
            return
        if matches_any_glob(vpath, self.cfg.ignore_patterns):
            return
        file_id = self.db.upsert_file(
            path=vpath,
            real_path=real_archive,
            type=dt.type,
            size_bytes=st.st_size,
            mtime_ns=st.st_mtime_ns,
            status="indexing",
        )
        self.db.clear_file_children(file_id)
        if dt.type == "text":
            extracted = self._text_extractor.extract(extracted_path)
            chunks = list(
                chunk_text(
                    extracted.text,
                    chunk_chars=self.cfg.indexing.chunk_chars,
                    overlap_chars=self.cfg.indexing.chunk_overlap_chars,
                )
            )
            if not chunks:
                self.db.upsert_fts_file(file_id, path=vpath, basename=Path(vpath).name, snippet="", archive_inner_paths="")
            else:
                texts = [c.text for c in chunks]
                with self._model_sem:
                    vecs = self.text_embedder.embed_texts(texts)
                vec_ids = self.db.allocate_vector_ids(len(chunks))
                self.vectors.add("text", vec_ids, vecs, use_st=self.use_st_text)
                created = now_ms()
                self.db.insert_chunks(
                    {
                        "file_id": file_id,
                        "chunk_type": "text",
                        "start_offset": c.start,
                        "end_offset": c.end,
                        "start_time_ms": None,
                        "end_time_ms": None,
                        "snippet": c.text[:512],
                        "vector_id": vid,
                        "created_at": created,
                    }
                    for c, vid in zip(chunks, vec_ids)
                )
                self.db.upsert_fts_file(
                    file_id, path=vpath, basename=Path(vpath).name, snippet=chunks[0].text[:512], archive_inner_paths=""
                )
        elif dt.type == "image":
            img = self._image_extractor.extract(extracted_path)
            with self._model_sem:
                vec = self.embedder.embed_images([img.image])
            vec_id = self.db.allocate_vector_ids(1)[0]
            self.vectors.add("image", [vec_id], vec)
            self.db.insert_frames(
                [
                    {
                        "file_id": file_id,
                        "timestamp_ms": None,
                        "width": img.width,
                        "height": img.height,
                        "vector_id": vec_id,
                        "created_at": now_ms(),
                    }
                ]
            )
            self.db.upsert_fts_file(file_id, path=vpath, basename=Path(vpath).name, snippet="", archive_inner_paths="")
        else:
            # For other types inside archives, keep metadata only for now.
            self.db.upsert_fts_file(file_id, path=vpath, basename=Path(vpath).name, snippet="", archive_inner_paths="")

        self.db.upsert_file(
            path=vpath,
            real_path=real_archive,
            type=dt.type,
            size_bytes=st.st_size,
            mtime_ns=st.st_mtime_ns,
            status="indexed",
            error_message=None,
        )


def build_embedder(cfg: Config):
    backend = (cfg.models.clip.backend or "auto").lower()
    if backend == "hash":
        from .embedding.hash_embedder import HashEmbedder

        return HashEmbedder()
    if backend == "open_clip":
        from .embedding.clip_embedder import OpenClipEmbedder

        # Support both checkpoint_path and pretrained tag loading
        checkpoint_path = getattr(cfg.models.clip, 'checkpoint_path', None)
        pretrained = getattr(cfg.models.clip, 'pretrained', 'openai')
        
        kwargs = {
            "model_name": cfg.models.clip.name,
            "device_prefer": cfg.models.clip.device_prefer,
            "batch_size": cfg.models.clip.batch_size,
        }
        if checkpoint_path:
            kwargs["checkpoint_path"] = checkpoint_path
        else:
            kwargs["pretrained"] = pretrained
        
        return OpenClipEmbedder(**kwargs)
    if backend == "auto":
        checkpoint_path = getattr(cfg.models.clip, 'checkpoint_path', None)
        pretrained = getattr(cfg.models.clip, 'pretrained', 'openai')
        
        if checkpoint_path or hasattr(cfg.models.clip, 'pretrained'):
            try:
                from .embedding.clip_embedder import OpenClipEmbedder

                kwargs = {
                    "model_name": cfg.models.clip.name,
                    "device_prefer": cfg.models.clip.device_prefer,
                    "batch_size": cfg.models.clip.batch_size,
                }
                if checkpoint_path:
                    kwargs["checkpoint_path"] = checkpoint_path
                else:
                    kwargs["pretrained"] = pretrained
                
                return OpenClipEmbedder(**kwargs)
            except Exception:
                pass
        from .embedding.hash_embedder import HashEmbedder

        return HashEmbedder()
    raise RuntimeError(f"Unknown models.clip.backend: {cfg.models.clip.backend}")


def build_transcriber(cfg: Config):
    if not cfg.models.whisper.enabled:
        return None
    try:
        from .embedding.whisper_transcriber import WhisperTranscriber

        return WhisperTranscriber(
            model_size=cfg.models.whisper.model_size,
            device_prefer=cfg.models.whisper.device_prefer,
            download_root=cfg.models.whisper.download_root,
        )
    except Exception:
        return None


def build_text_embedder(cfg: Config):
    """Build text embedder for semantic text search."""
    backend = (cfg.models.text_embedder.backend or "clip").lower()
    
    if backend == "sentence_transformers":
        try:
            from .embedding.text_embedder import SentenceTransformersEmbedder

            return SentenceTransformersEmbedder(
                model_name=cfg.models.text_embedder.model_name,
                device_prefer=cfg.models.text_embedder.device_prefer,
                batch_size=cfg.models.text_embedder.batch_size,
            )
        except Exception as e:
            print(f"Warning: Failed to load sentence-transformers embedder: {e}")
            # Fall back to CLIP for text
            backend = "clip"
    
    if backend == "clip":
        # Use the main embedder (CLIP) for text
        return None  # Signal to use the main embedder
    
    raise RuntimeError(f"Unknown models.text_embedder.backend: {cfg.models.text_embedder.backend}")
