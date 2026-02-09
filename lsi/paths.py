from __future__ import annotations

import os
from pathlib import Path


def get_app_dir() -> Path:
    override = os.getenv("LSI_APP_DIR")
    if override:
        return Path(os.path.expanduser(override))
    return Path.home() / "Library" / "Application Support" / "local-semantic-indexer"


def get_config_path() -> Path:
    return get_app_dir() / "config.yaml"


def get_db_path() -> Path:
    return get_app_dir() / "index.sqlite3"


def get_vectors_dir() -> Path:
    return get_app_dir() / "vectors"


def get_cache_dir() -> Path:
    return get_app_dir() / "cache"


def get_logs_dir() -> Path:
    return get_app_dir() / "logs"


def ensure_app_dirs() -> None:
    app = get_app_dir()
    (app).mkdir(parents=True, exist_ok=True)
    get_vectors_dir().mkdir(parents=True, exist_ok=True)
    get_cache_dir().mkdir(parents=True, exist_ok=True)
    get_logs_dir().mkdir(parents=True, exist_ok=True)
