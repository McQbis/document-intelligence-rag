from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper over SentenceTransformer providing normalized embeddings for retrieval."""
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        max_seq_length: int = 256,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length


    def embed_text(self, text: str) -> np.ndarray:
        """Encode single text into normalized embedding vector."""
        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=1,
            show_progress_bar=False,
        )


    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Encode batch of texts into normalized embedding matrix."""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )