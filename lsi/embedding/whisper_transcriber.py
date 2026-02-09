from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import TranscriptSegment


def _default_download_root() -> Path:
    default = Path.home() / ".cache"
    root = Path(os.getenv("XDG_CACHE_HOME", str(default))) / "whisper"
    return root


@dataclass
class WhisperTranscriber:
    model_size: str = "tiny"
    device_prefer: str = "mps"
    download_root: str | None = None

    def __post_init__(self) -> None:
        import whisper  # type: ignore
        import torch

        self._whisper = whisper
        self._torch = torch
        self._model: Any | None = None

    def _select_device(self) -> str:
        prefer = (self.device_prefer or "").lower()
        if prefer == "mps" and self._torch.backends.mps.is_available():
            return "mps"
        if prefer == "cuda" and self._torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _ensure_model_present(self) -> None:
        if self.model_size not in self._whisper._MODELS:  # type: ignore[attr-defined]
            return  # custom path supported by whisper.load_model
        url = self._whisper._MODELS[self.model_size]  # type: ignore[attr-defined]
        filename = os.path.basename(url)
        root = Path(self.download_root).expanduser() if self.download_root else _default_download_root()
        model_path = root / filename
        if not model_path.exists():
            raise RuntimeError(
                f"Whisper model not found locally: {model_path} (downloads are disabled). "
                f"Place {filename} under that directory or set models.whisper.download_root."
            )

    def load(self) -> None:
        if self._model is not None:
            return
        self._ensure_model_present()
        device = self._select_device()
        root = str(Path(self.download_root).expanduser()) if self.download_root else None
        try:
            self._model = self._whisper.load_model(self.model_size, device=device, download_root=root)
        except Exception:
            if device != "cpu":
                self._model = self._whisper.load_model(self.model_size, device="cpu", download_root=root)
            else:
                raise

    def transcribe(self, audio_path: Path) -> list[TranscriptSegment]:
        self.load()
        assert self._model is not None
        device = self._select_device()
        # MPS does not support FP16; force FP32 on MPS to avoid warnings and potential errors
        fp16 = device != "mps"
        result = self._model.transcribe(str(audio_path), verbose=False, fp16=fp16)  # type: ignore[call-arg]
        segments = result.get("segments") or []
        out: list[TranscriptSegment] = []
        for seg in segments:
            try:
                start_ms = int(float(seg.get("start", 0.0)) * 1000)
                end_ms = int(float(seg.get("end", 0.0)) * 1000)
                text = str(seg.get("text") or "").strip()
            except Exception:
                continue
            if text:
                out.append(TranscriptSegment(start_ms=start_ms, end_ms=end_ms, text=text))
        return out

