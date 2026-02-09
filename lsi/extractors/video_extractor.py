from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExtractedFrame:
    path: Path
    timestamp_ms: int


@dataclass(frozen=True)
class VideoExtraction:
    audio_wav: Path | None
    frames: list[ExtractedFrame]


class VideoExtractor:
    def __init__(self, *, cache_dir: Path, frame_every_seconds: float, max_frames_per_video: int):
        self._cache_dir = cache_dir
        self._frame_every_seconds = float(frame_every_seconds)
        self._max_frames_per_video = int(max_frames_per_video)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, path: Path, *, key: str) -> VideoExtraction:
        work_dir = self._cache_dir / "video" / key
        work_dir.mkdir(parents=True, exist_ok=True)

        audio_wav = work_dir / "audio.wav"
        frames_dir = work_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        audio_ok = self._extract_audio(path, audio_wav)
        frames = self._extract_frames(path, frames_dir)
        return VideoExtraction(audio_wav=audio_wav if audio_ok else None, frames=frames)

    def _extract_audio(self, path: Path, out_wav: Path) -> bool:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(out_wav),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_wav.exists() and out_wav.stat().st_size > 0
        except Exception:
            return False

    def _extract_frames(self, path: Path, frames_dir: Path) -> list[ExtractedFrame]:
        every = max(0.1, self._frame_every_seconds)
        # Approximate timestamps as N * every; we don't parse real PTS yet.
        pattern = str(frames_dir / "frame_%06d.jpg")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-vf",
            f"fps=1/{every}",
            "-frames:v",
            str(self._max_frames_per_video),
            "-q:v",
            "2",
            pattern,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return []

        frames: list[ExtractedFrame] = []
        for i, frame_path in enumerate(sorted(frames_dir.glob("frame_*.jpg"))):
            ts_ms = int(math.floor(i * every * 1000.0))
            frames.append(ExtractedFrame(path=frame_path, timestamp_ms=ts_ms))
        return frames

