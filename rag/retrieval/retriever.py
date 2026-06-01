from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import faiss
import bm25s
from sentence_transformers import CrossEncoder

from rag.ingestion.base import TextChunk
from rag.retrieval.embeddings import EmbeddingModel


class HybridRetriever:
    """Hybrid retrieval system: BM25 + FAISS + Cross-Encoder reranking."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        reranker_model: str = "BAAI/bge-reranker-base",
        rerank_top_n: int = 15,
    ):
        self.embedding_model = embedding_model
        self.rerank_top_n = rerank_top_n

        self.chunks: List[TextChunk] = []
        self._chunk_id_to_idx: dict[int, int] = {}

        self.index: faiss.IndexHNSWFlat | None = None
        self.bm25: bm25s.BM25 | None = None
        self.dimension: int | None = None

        self.reranker = CrossEncoder(reranker_model, device="cpu")

    def build_index(self, chunks: List[TextChunk]) -> None:
        """Build BM25 + FAISS indexes from chunks."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self._chunk_id_to_idx = {id(c): i for i, c in enumerate(chunks)}

        tokens = bm25s.tokenize(texts, stopwords="en")
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokens)

        embeddings = np.array(
            self.embedding_model.embed_batch(texts), dtype="float32"
        )

        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexHNSWFlat(
            self.dimension, 32, faiss.METRIC_INNER_PRODUCT
        )
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 64
        self.index.add(embeddings)

    def add_chunks(self, new_chunks: List[TextChunk]) -> None:
        """Incrementally update retrieval indexes with new chunks."""
        if self.index is None:
            self.build_index(new_chunks)
            return

        offset = len(self.chunks)

        for i, c in enumerate(new_chunks):
            self._chunk_id_to_idx[id(c)] = offset + i

        self.chunks.extend(new_chunks)

        tokens = bm25s.tokenize([c.text for c in self.chunks], stopwords="en")
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokens)

        embeddings = np.array(
            self.embedding_model.embed_batch([c.text for c in new_chunks]),
            dtype="float32",
        )
        self.index.add(embeddings)

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 30,
        rerank: bool = True,
    ) -> List[Tuple[TextChunk, float]]:
        """Hybrid search with optional reranking."""

        if not self.chunks:
            return []

        q_emb = np.array(
            [self.embedding_model.embed_text(query)],
            dtype="float32",
        )
        q_tokens = bm25s.tokenize([query], stopwords="en")

        _, faiss_idx = self.index.search(q_emb, candidate_k)
        faiss_idx = faiss_idx[0]

        bm25_res, _ = self.bm25.retrieve(
            q_tokens, k=min(candidate_k, len(self.chunks))
        )
        bm25_idx = bm25_res[0]

        rrf = defaultdict(float)
        k = 60

        for r, idx in enumerate(faiss_idx):
            if idx != -1:
                rrf[int(idx)] += 1.0 / (k + r + 1)

        for r, idx in enumerate(bm25_idx):
            rrf[int(idx)] += 1.0 / (k + r + 1)

        candidates = sorted(rrf.items(), key=lambda x: x[1], reverse=True)

        if not rerank:
            return [(self.chunks[i], float(s)) for i, s in candidates[:top_k]]

        rerank_pool = candidates[: self.rerank_top_n]
        pairs = [(query, self.chunks[i].text) for i, _ in rerank_pool]

        scores = self.reranker.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        reranked = sorted(
            zip(rerank_pool, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            (self.chunks[i], float(score))
            for (i, _), score in reranked[:top_k]
        ]

    @property
    def is_built(self) -> bool:
        return self.index is not None and self.bm25 is not None