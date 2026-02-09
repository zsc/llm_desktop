from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .paths import ensure_app_dirs, get_config_path


@dataclass
class ArchiveConfig:
    large_threshold_bytes: int = 1_073_741_824
    enable_extract_small: bool = True
    extract_cache_max_gb: int = 10


@dataclass
class ClipConfig:
    name: str = "ViT-B-32"
    device_prefer: str = "mps"
    batch_size: int = 32
    checkpoint_path: str | None = None
    pretrained: str = "openai"  # pretrained tag for open_clip
    backend: str = "auto"  # "auto" | "hash" | "open_clip"


@dataclass
class WhisperConfig:
    model_size: str = "tiny"
    device_prefer: str = "mps"
    enabled: bool = True
    download_root: str | None = None


@dataclass
class OllamaConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:11434"
    model: str = "gemma3:1b"
    query_parse: bool = True
    rerank: bool = True
    explain: bool = True


@dataclass
class TextEmbedderConfig:
    backend: str = "clip"  # "clip" | "sentence_transformers"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device_prefer: str = "mps"
    batch_size: int = 32


@dataclass
class ModelsConfig:
    clip: ClipConfig = field(default_factory=ClipConfig)
    text_embedder: TextEmbedderConfig = field(default_factory=TextEmbedderConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)


@dataclass
class IndexingConfig:
    max_workers_io: int = 4
    max_workers_model: int = 1
    chunk_chars: int = 2000
    chunk_overlap_chars: int = 200
    rescan_interval_s: float = 30.0


@dataclass
class LoadSheddingConfig:
    cpu_pause_percent: float = 70.0
    mem_available_min_gb: float = 2.0
    pause_after_s: float = 10.0
    resume_after_s: float = 30.0


@dataclass
class VideoConfig:
    frame_every_seconds: float = 2.0
    max_frames_per_video: int = 2000


@dataclass
class SecurityConfig:
    local_only: bool = True
    disallow_network: bool = True


DEFAULT_IGNORE_PATTERNS = [
    "**/.ssh/**",
    "**/.gnupg/**",
    "**/Library/Keychains/**",
    "**/.env",
    "**/*.pem",
    "**/*.key",
    "**/*id_rsa*",
    "/System/**",
    "/Library/**",
]


@dataclass
class Config:
    roots: list[str] = field(default_factory=list)
    allow_roots: list[str] = field(default_factory=list)
    ignore_patterns: list[str] = field(default_factory=lambda: list(DEFAULT_IGNORE_PATTERNS))
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    load_shedding: LoadSheddingConfig = field(default_factory=LoadSheddingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    _path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        def encode(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: encode(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, Path):
                return str(obj)
            return obj

        data = encode(self)
        data.pop("_path", None)
        return data

    def save(self) -> None:
        path = self._path or get_config_path()
        ensure_app_dirs()
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
        self._path = path

    def set_by_dotted_key(self, key: str, value: str) -> None:
        parts = key.split(".")
        obj: Any = self
        for p in parts[:-1]:
            if not hasattr(obj, p):
                raise KeyError(f"Unknown config key: {key}")
            obj = getattr(obj, p)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise KeyError(f"Unknown config key: {key}")
        current = getattr(obj, leaf)
        casted: Any = value
        if isinstance(current, bool):
            casted = value.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            casted = int(value)
        elif isinstance(current, float):
            casted = float(value)
        elif isinstance(current, list):
            v = value.strip()
            if v.startswith("["):
                casted = json.loads(v)
            else:
                casted = [x.strip() for x in v.split(",") if x.strip()]
        setattr(obj, leaf, casted)


def _expand_path(p: str) -> str:
    return os.path.expanduser(p)


def default_config() -> Config:
    roots = [str(Path.home() / "Desktop"), str(Path.home() / "Downloads")]
    return Config(
        roots=[_expand_path(r) for r in roots],
        allow_roots=[_expand_path(r) for r in roots],
    )


def save_default_config(path: Path) -> None:
    cfg = default_config()
    cfg._path = path
    cfg.save()


def load_config(path: Path | None = None) -> Config:
    ensure_app_dirs()
    path = path or get_config_path()
    if not path.exists():
        cfg = default_config()
        cfg._path = path
        return cfg
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = default_config()
    _apply_dict(cfg, raw)
    cfg._path = path
    return cfg


def _apply_dict(cfg: Config, raw: dict[str, Any]) -> None:
    for key, value in raw.items():
        if key == "archive":
            cfg.archive = ArchiveConfig(**(value or {}))
        elif key == "models":
            mv = value or {}
            clip = ClipConfig(**(mv.get("clip") or {}))
            text_embedder = TextEmbedderConfig(**(mv.get("text_embedder") or {}))
            whisper = WhisperConfig(**(mv.get("whisper") or {}))
            ollama = OllamaConfig(**(mv.get("ollama") or {}))
            cfg.models = ModelsConfig(clip=clip, text_embedder=text_embedder, whisper=whisper, ollama=ollama)
        elif key == "indexing":
            cfg.indexing = IndexingConfig(**(value or {}))
        elif key == "load_shedding":
            cfg.load_shedding = LoadSheddingConfig(**(value or {}))
        elif key == "video":
            cfg.video = VideoConfig(**(value or {}))
        elif key == "security":
            cfg.security = SecurityConfig(**(value or {}))
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
