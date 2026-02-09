from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _select_torch_device(prefer: str) -> str:
    import torch

    prefer = (prefer or "").lower()
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _l2_normalize(x):
    import torch

    return x / (x.norm(dim=-1, keepdim=True) + 1e-12)


@dataclass
class OpenClipEmbedder:
    model_name: str
    checkpoint_path: str | None = None
    pretrained: str | None = "openai"
    device_prefer: str = "mps"
    batch_size: int = 32

    def __post_init__(self) -> None:
        try:
            import open_clip  # type: ignore
        except Exception as e:
            raise RuntimeError("CLIP support requires `open_clip_torch` (install extras: local-semantic-indexer[clip]).") from e

        import torch

        self.name = f"open_clip:{self.model_name}"
        self._open_clip = open_clip
        self._torch = torch

        device = _select_torch_device(self.device_prefer)
        self._device = self._torch.device(device)

        # Load model - either from checkpoint_path or using pretrained tag
        if self.checkpoint_path:
            ckpt = Path(self.checkpoint_path).expanduser()
            if not ckpt.exists():
                raise RuntimeError(f"CLIP checkpoint not found: {ckpt}")
            created = self._open_clip.create_model_and_transforms(self.model_name, pretrained=None)
            if len(created) == 3:
                model, _, preprocess = created
            else:  # pragma: no cover
                model, preprocess = created[0], created[-1]
            tokenizer = self._open_clip.get_tokenizer(self.model_name)

            state = self._torch.load(str(ckpt), map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                sd = state["state_dict"]
            elif isinstance(state, dict) and "model_state_dict" in state:
                sd = state["model_state_dict"]
            else:
                sd = state
            model.load_state_dict(sd, strict=False)
        else:
            # Load using pretrained tag (will use HuggingFace cache)
            pretrained_tag = self.pretrained or "openai"
            created = self._open_clip.create_model_and_transforms(
                self.model_name, 
                pretrained=pretrained_tag,
                device=self._device
            )
            if len(created) == 3:
                model, _, preprocess = created
            else:  # pragma: no cover
                model, preprocess = created[0], created[-1]
            tokenizer = self._open_clip.get_tokenizer(self.model_name)

        model.eval()
        model.to(self._device)

        # Infer embedding dimension.
        with self._torch.no_grad():
            dummy = tokenizer(["hello"]).to(self._device)
            feat = model.encode_text(dummy)
        self.dim = int(feat.shape[-1])

        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        feats: list[np.ndarray] = []
        bs = max(1, int(self.batch_size))
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            with self._torch.no_grad():
                tokens = self._tokenizer(batch).to(self._device)
                f = _l2_normalize(self._model.encode_text(tokens)).detach().cpu().numpy().astype(np.float32)
            feats.append(f)
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, self.dim), dtype=np.float32)

    def embed_images(self, images: list[Any]) -> np.ndarray:
        feats: list[np.ndarray] = []
        bs = max(1, int(self.batch_size))
        for i in range(0, len(images), bs):
            batch = images[i : i + bs]
            with self._torch.no_grad():
                tensors = self._torch.stack([self._preprocess(im) for im in batch]).to(self._device)
                f = _l2_normalize(self._model.encode_image(tensors)).detach().cpu().numpy().astype(np.float32)
            feats.append(f)
        return np.concatenate(feats, axis=0) if feats else np.zeros((0, self.dim), dtype=np.float32)

