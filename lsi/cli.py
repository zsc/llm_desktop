from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .config import Config, load_config, save_default_config
from .indexer import IndexingService, build_embedder, build_text_embedder, build_transcriber
from .paths import (
    ensure_app_dirs,
    get_app_dir,
    get_cache_dir,
    get_config_path,
    get_db_path,
    get_logs_dir,
    get_vectors_dir,
)
from .search.hybrid import hybrid_search
from .search.rerank import llm_parse_query, llm_rerank
from .security import disallow_network
from .storage.db import Database
from .storage.vectors import VectorStore
from .filters import is_within_any_root


def _cmd_init(_: argparse.Namespace) -> int:
    ensure_app_dirs()
    config_path = get_config_path()
    if config_path.exists():
        print(f"Config already exists: {config_path}")
    else:
        save_default_config(config_path)
        print(f"Wrote config: {config_path}")
    db = Database(get_db_path())
    db.init_schema()
    print(f"App dir: {get_app_dir()}")
    return 0


def _cmd_config_show(args: argparse.Namespace) -> int:
    cfg = load_config()
    data = cfg.to_dict()
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        import yaml

        print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return 0


def _cmd_config_set(args: argparse.Namespace) -> int:
    cfg = load_config()
    cfg.set_by_dotted_key(args.key, args.value)
    cfg.save()
    print(f"Updated config: {get_config_path()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lsi", description="Local semantic indexer (offline-first).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_init = sub.add_parser("init", help="Initialize app directories and default config.")
    sp_init.set_defaults(func=_cmd_init)

    sp_config = sub.add_parser("config", help="Show or set config.")
    sub_config = sp_config.add_subparsers(dest="subcmd", required=True)

    sp_show = sub_config.add_parser("show", help="Show config.")
    sp_show.add_argument("--json", action="store_true", help="Output JSON.")
    sp_show.set_defaults(func=_cmd_config_show)

    sp_set = sub_config.add_parser("set", help="Set config key (dot notation).")
    sp_set.add_argument("key")
    sp_set.add_argument("value")
    sp_set.set_defaults(func=_cmd_config_set)

    sp_index = sub.add_parser("index", help="Index roots (supports --daemon/--once).")
    sp_index.add_argument("--roots", nargs="*", help="Roots to scan (default: config roots).")
    sp_index.add_argument("--daemon", action="store_true", help="Run as a background daemon.")
    sp_index.add_argument("--once", action="store_true", help="Scan once then exit (default).")
    sp_index.add_argument("--rescan", action="store_true", help="Rebuild index from scratch.")
    sp_index.add_argument("--dry-run", action="store_true", help="Only print files that would be indexed.")
    sp_index.add_argument("--_daemon-child", action="store_true", help=argparse.SUPPRESS)
    sp_index.set_defaults(func=_cmd_index)

    sp_search = sub.add_parser("search", help="Search the index.")
    sp_search.add_argument("query")
    sp_search.add_argument("--top", type=int, default=20)
    sp_search.add_argument("--type", dest="type_filter", choices=["text", "pdf", "image", "audio", "video", "archive"])
    sp_search.add_argument("--path", dest="path_glob", help="Glob to filter paths.")
    sp_search.add_argument("--since", help="Only include files modified since YYYY-MM-DD.")
    sp_search.add_argument("--json", action="store_true")
    sp_search.add_argument("--no-llm", action="store_true", help="Disable local LLM rerank/explain.")
    sp_search.set_defaults(func=_cmd_search)

    sp_status = sub.add_parser("status", help="Show indexer status.")
    sp_status.add_argument("--json", action="store_true")
    sp_status.set_defaults(func=_cmd_status)

    sp_cleanup = sub.add_parser("cleanup", help="Cleanup caches.")
    sp_cleanup.add_argument("--all", action="store_true", help="Also remove vectors (keeps DB).")
    sp_cleanup.set_defaults(func=_cmd_cleanup)

    return p


def _wipe_index() -> None:
    db_path = get_db_path()
    if db_path.exists():
        db_path.unlink()
    vec_dir = get_vectors_dir()
    if vec_dir.exists():
        for p in vec_dir.glob("*"):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass


def _cmd_index(args: argparse.Namespace) -> int:
    cfg = load_config()
    roots = args.roots if args.roots else cfg.roots
    if not roots:
        print("No roots configured. Set config.roots or pass --roots.")
        return 2
    if cfg.allow_roots:
        disallowed = [r for r in roots if not is_within_any_root(os.path.expanduser(r), cfg.allow_roots)]
        if disallowed:
            print("Some --roots are not under config.allow_roots and will be skipped:")
            for r in disallowed:
                print(f"  - {r}")
            print("Update allow_roots via `lsi config set allow_roots '[\"/path\"]'` if needed.")

    ensure_app_dirs()
    if args.rescan and not args._daemon_child:
        _wipe_index()

    db = Database(get_db_path())
    db.init_schema()

    embedder = build_embedder(cfg)
    text_embedder = build_text_embedder(cfg)
    # Support separate text embedding dimension
    text_dim = int(text_embedder.dim) if text_embedder is not None else int(embedder.dim)
    vectors = VectorStore(get_vectors_dir(), dim=int(embedder.dim), text_dim=text_dim)
    transcriber = build_transcriber(cfg)

    svc = IndexingService(
        cfg=cfg,
        roots=roots,
        db=db,
        vectors=vectors,
        embedder=embedder,
        transcriber=transcriber,
        rescan=args.rescan,
        dry_run=args.dry_run,
        text_embedder=text_embedder,
    )

    if args.daemon and not args._daemon_child:
        # Spawn detached child process.
        log_path = get_logs_dir() / "daemon.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "lsi", "index", "--daemon", "--_daemon-child"]
        if args.rescan:
            cmd.append("--rescan")
        if args.dry_run:
            cmd.append("--dry-run")
        if roots:
            cmd += ["--roots", *roots]
        with open(log_path, "a", encoding="utf-8") as logf:
            p = subprocess.Popen(cmd, stdout=logf, stderr=logf, start_new_session=True)
        db.set_state("daemon_pid", p.pid)
        print(f"Started daemon PID {p.pid}. Logs: {log_path}")
        return 0

    # Default behavior: run once.
    if not args.daemon and not args.once:
        args.once = True
    with disallow_network(allow_localhost=True) if cfg.security.disallow_network else _nullcontext():
        if args.daemon and args._daemon_child:
            svc.run_daemon()
        else:
            svc.run_once()
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    cfg = load_config()
    db_path = get_db_path()
    if not db_path.exists():
        print("Index not initialized. Run `lsi init` first.")
        return 2
    db = Database(db_path)
    db.init_schema()
    embedder = build_embedder(cfg)
    text_embedder = build_text_embedder(cfg)
    # Support separate text embedding dimension (e.g., sentence-transformers uses 384, CLIP uses 512)
    text_dim = int(text_embedder.dim) if text_embedder is not None else int(embedder.dim)
    vectors = VectorStore(get_vectors_dir(), dim=int(embedder.dim), text_dim=text_dim)

    query = args.query
    type_filter = args.type_filter
    path_glob = args.path_glob
    since = args.since

    with disallow_network(allow_localhost=True) if cfg.security.disallow_network else _nullcontext():
        if cfg.models.ollama.enabled and not args.no_llm and cfg.models.ollama.query_parse:
            try:
                from .embedding.ollama_client import OllamaClient

                ollama = OllamaClient(base_url=cfg.models.ollama.base_url, model=cfg.models.ollama.model)
                parsed = llm_parse_query(ollama, query) or {}
                if isinstance(parsed.get("query"), str) and parsed["query"].strip():
                    query = parsed["query"].strip()
                if type_filter is None and isinstance(parsed.get("type"), str):
                    type_filter = parsed["type"]
                if path_glob is None and isinstance(parsed.get("path_glob"), str):
                    path_glob = parsed["path_glob"]
                if since is None and isinstance(parsed.get("since"), str):
                    since = parsed["since"]
            except Exception:
                pass

        results = hybrid_search(
            db=db,
            vectors=vectors,
            embedder=embedder,
            query=query,
            top=args.top,
            type_filter=type_filter,
            path_glob=path_glob,
            since=since,
            text_embedder=text_embedder,
        )

        if cfg.models.ollama.enabled and not args.no_llm and cfg.models.ollama.rerank and results:
            try:
                from .embedding.ollama_client import OllamaClient

                ollama = OllamaClient(base_url=cfg.models.ollama.base_url, model=cfg.models.ollama.model)
                results = llm_rerank(ollama, query, results, max_items=30).results
            except Exception:
                pass

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "path": r.path,
                        "type": r.type,
                        "score": r.score,
                        "snippet": r.snippet,
                        "timestamp_ms": r.timestamp_ms,
                        "sources": sorted(r.sources),
                        "explanation": r.explanation,
                    }
                    for r in results
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    for r in results:
        ts = f" @{r.timestamp_ms}ms" if r.timestamp_ms is not None else ""
        src = ",".join(sorted(r.sources))
        snip = (r.snippet or "").replace("\n", " ").strip()
        if len(snip) > 140:
            snip = snip[:140] + "…"
        print(f"{r.score:6.3f} [{r.type}] {r.path}{ts} ({src})")
        if r.explanation:
            print(f"  {r.explanation}")
        elif snip:
            print(f"  {snip}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    db_path = get_db_path()
    if not db_path.exists():
        print("Index not initialized. Run `lsi init`.")
        return 0
    db = Database(db_path)
    db.init_schema()
    status = db.get_state("status", {})
    if args.json:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    if not status:
        print(f"DB: {db_path}")
        return 0
    try:
        db_mb = db_path.stat().st_size / (1024 * 1024)
    except Exception:
        db_mb = 0.0
    vec_bytes = 0
    try:
        for p in get_vectors_dir().glob("*"):
            if p.is_file():
                vec_bytes += p.stat().st_size
    except Exception:
        pass
    vec_mb = vec_bytes / (1024 * 1024)
    print(
        f"Queue: {status.get('queue_length', 0)} | Processed: {status.get('processed', 0)} | "
        f"Skipped: {status.get('skipped', 0)} | Errors: {status.get('errors', 0)} | "
        f"DB: {db_mb:.1f}MB | Vectors: {vec_mb:.1f}MB"
    )
    if status.get("paused"):
        dur = status.get("paused_for_s")
        dur_s = f"{dur:.0f}s" if isinstance(dur, (int, float)) else "?"
        print(f"Paused: yes ({status.get('pause_reason')}, {dur_s})")
    else:
        print("Paused: no")
    pid = db.get_state("daemon_pid")
    if pid:
        print(f"Daemon PID: {pid}")
    return 0


def _cmd_cleanup(args: argparse.Namespace) -> int:
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        for sub in cache_dir.glob("*"):
            try:
                if sub.is_dir():
                    import shutil

                    shutil.rmtree(sub, ignore_errors=True)
                else:
                    sub.unlink()
            except Exception:
                pass
    if args.all:
        vec_dir = get_vectors_dir()
        if vec_dir.exists():
            for p in vec_dir.glob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                except Exception:
                    pass
    print("Cleanup done.")
    return 0


class _nullcontext:
    def __enter__(self):  # type: ignore[no-untyped-def]
        return None

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        return False


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return int(args.func(args))
