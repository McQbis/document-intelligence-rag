from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np

from rag.ingestion.base import TextChunk
from rag.retrieval.retriever import HybridRetriever


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class CacheEntry:
    __slots__ = ("results", "ts")

    def __init__(self, results: List[Tuple[TextChunk, float]]):
        self.results = results
        self.ts = time.monotonic()


class QueryCache:
    def __init__(
        self,
        retriever: HybridRetriever,
        max_size: int = 512,
        semantic_threshold: float = 0.92,
        ttl_seconds: Optional[float] = None,
    ):
        self.retriever = retriever
        self.max_size = max_size
        self.semantic_threshold = semantic_threshold
        self.ttl = ttl_seconds

        self._exact: OrderedDict[str, CacheEntry] = OrderedDict()
        self._semantic_keys: List[str] = []
        self._semantic_embs: List[np.ndarray] = []


    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 30,
    ) -> Tuple[List[Tuple[TextChunk, float]], bool]:
        key = _hash(query)

        if key in self._exact:
            entry = self._exact[key]
            if self._is_alive(entry):
                self._exact.move_to_end(key)
                return entry.results, True
            else:
                del self._exact[key]

        q_emb = self.retriever.embedding_model.embed_text(query)
        semantic_hit = self._semantic_lookup(q_emb)
        if semantic_hit is not None:
            return semantic_hit, True

        results = self.retriever.search(query, top_k=top_k, candidate_k=candidate_k)
        self._store(key, query, q_emb, results)
        return results, False


    def invalidate(self) -> None:
        self._exact.clear()
        self._semantic_keys.clear()
        self._semantic_embs.clear()


    def _is_alive(self, entry: CacheEntry) -> bool:
        if self.ttl is None:
            return True
        return (time.monotonic() - entry.ts) < self.ttl

    def _semantic_lookup(
        self, q_emb: np.ndarray
    ) -> Optional[List[Tuple[TextChunk, float]]]:
        if not self._semantic_embs:
            return None

        matrix = np.stack(self._semantic_embs)
        sims = matrix @ q_emb
        best_idx = int(np.argmax(sims))

        if sims[best_idx] >= self.semantic_threshold:
            best_key = _hash(self._semantic_keys[best_idx])
            entry = self._exact.get(best_key)
            if entry and self._is_alive(entry):
                return entry.results

        return None

    def _store(
        self,
        key: str,
        query: str,
        emb: np.ndarray,
        results: List[Tuple[TextChunk, float]],
    ) -> None:
        while len(self._exact) >= self.max_size:
            oldest_key, _ = self._exact.popitem(last=False)
            try:
                i = self._semantic_keys.index(oldest_key)
                self._semantic_keys.pop(i)
                self._semantic_embs.pop(i)
            except ValueError:
                pass

        self._exact[key] = CacheEntry(results)
        self._semantic_keys.append(query)
        self._semantic_embs.append(emb)