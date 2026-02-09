from __future__ import annotations

from dataclasses import dataclass
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


def _mean_pooling(model_output, attention_mask):
    """Mean pooling to get sentence embeddings from token embeddings."""
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return np.sum(token_embeddings.detach().cpu().numpy() * input_mask_expanded.cpu().numpy(), axis=1) / np.clip(
        np.sum(input_mask_expanded.cpu().numpy(), axis=1), a_min=1e-9, a_max=None
    )


@dataclass
class SentenceTransformersEmbedder:
    """Sentence-transformers based text embedder for better semantic matching.
    
    Uses HuggingFace transformers directly to avoid sentence-transformers dependency issues.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device_prefer: str = "mps"
    batch_size: int = 32

    def __post_init__(self) -> None:
        import os
        os.environ['TRANSFORMERS_NO_TF'] = '1'  # Force PyTorch only
        
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Text embedder requires `transformers`. "
                "Install: pip install transformers"
            ) from e

        import torch

        self._torch = torch
        self.name = f"st:{self.model_name}"

        device = _select_torch_device(self.device_prefer)
        self._device = device
        
        # Load model and tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(device)
        
        # Get embedding dimension
        self.dim = self._model.config.hidden_size

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts to vectors using mean pooling."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Tokenize
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            
            # Get model output
            with self._torch.no_grad():
                model_output = self._model(**encoded)
            
            # Mean pooling
            attention_mask = encoded['attention_mask']
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = self._torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = self._torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
            
            # Normalize
            embeddings = self._torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings).astype(np.float32)

    def embed_images(self, images: list[Any]) -> np.ndarray:
        """Not supported for text-only embedder - returns zero vectors."""
        return np.zeros((len(images), self.dim), dtype=np.float32)
