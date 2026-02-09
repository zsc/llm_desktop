from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass
class OllamaClient:
    base_url: str
    model: str
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        u = urlparse(self.base_url)
        host = u.hostname or ""
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(f"Ollama base_url must be localhost/127.0.0.1/::1, got {self.base_url!r}")

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            raise RuntimeError("Ollama returned invalid JSON.") from e

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        payload: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        data = self._post_json("/api/generate", payload)
        return str(data.get("response") or "")
