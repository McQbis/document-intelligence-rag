from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from rag.ingestion.base import TextChunk
from rag.retrieval.retriever import HybridRetriever


class RouteMode(str, Enum):
    FAST = "fast"
    DEEP = "deep"
    AUTO = "auto"


# Heuristics for AUTO mode
_DEEP_KEYWORDS = {
    "explain", "summarize", "compare", "analyze", "describe",
    "elaborate", "detail", "comprehensive", "thorough", "research",
    "differences", "similarities",
}
_SHORT_QUERY_THRESHOLD = 6  # word count below → probably keyword search → fast


class QueryRouter:

    def __init__(
        self,
        retriever: HybridRetriever,
        top_k: int = 10,
        candidate_k: int = 30,
    ):
        # TODO: add cache for recent queries and their resolved modes (LRU, 100 entries?) to skip rerunning heuristics
        self.retriever = retriever
        self.top_k = top_k
        self.candidate_k = candidate_k


    def search(
        self,
        query: str,
        mode: RouteMode = RouteMode.AUTO,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
    ) -> List[Tuple[TextChunk, float]]:
        resolved_mode = self._resolve_mode(query, mode)
        _top_k = top_k or self.top_k
        _cand_k = candidate_k or self.candidate_k
        use_rerank = (resolved_mode == RouteMode.DEEP)

        return self.retriever.search(
            query, top_k=_top_k, candidate_k=_cand_k, rerank=use_rerank
        )


    def classify(self, query: str) -> RouteMode:
        return self._resolve_mode(query, RouteMode.AUTO)


    def _resolve_mode(self, query: str, mode: RouteMode) -> RouteMode:
        if mode != RouteMode.AUTO:
            return mode

        words = query.lower().split()
        if len(words) > _SHORT_QUERY_THRESHOLD:
            return RouteMode.DEEP
        if any(w in _DEEP_KEYWORDS for w in words):
            return RouteMode.DEEP
        return RouteMode.FAST