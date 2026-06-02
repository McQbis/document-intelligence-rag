from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from rag.retrieval.embeddings import EmbeddingModel
from rag.retrieval.retriever import HybridRetriever
from rag.cache.query_cache import QueryCache
from rag.routing.router import QueryRouter
from rag.ingestion.pipeline import IngestionPipeline


MAX_SESSIONS = 5
MAX_FILES_PER_SESSION = 5
MAX_TOTAL_BYTES = 10 * 1024 * 1024   # 10 MB per session
SESSION_TTL_SECONDS = 15 * 60        # 15 minutes


@dataclass
class UploadedFile:
    filename: str
    size_bytes: int
    chunks: int
    uploaded_at: float = field(default_factory=time.monotonic)


@dataclass
class Session:
    """Isolated retrieval workspace owned by a single client session."""
    session_id: str
    created_at: float
    last_active: float

    retriever: HybridRetriever
    cache: QueryCache
    router: QueryRouter
    pipeline: IngestionPipeline

    files: list[UploadedFile] = field(default_factory=list)
    total_bytes: int = 0
    query_count: int = 0

    # Per-session lock — prevents concurrent uploads/searches racing
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def touch(self) -> None:
        self.last_active = time.monotonic()

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > SESSION_TTL_SECONDS

    @property
    def seconds_remaining(self) -> int:
        elapsed = time.monotonic() - self.created_at
        return max(0, int(SESSION_TTL_SECONDS - elapsed))

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def bytes_remaining(self) -> int:
        return max(0, MAX_TOTAL_BYTES - self.total_bytes)


class SessionManager:
    """Creates, tracks and expires active retrieval sessions."""
    def __init__(self, embedding_model: EmbeddingModel, reranker_model: str):
        self._sessions: dict[str, Session] = {}
        self._registry_lock = asyncio.Lock()
        self._embedding_model = embedding_model
        self._reranker_model = reranker_model

    async def create_session(self) -> Session:
        async with self._registry_lock:
            self._evict_expired()

            if len(self._sessions) >= MAX_SESSIONS:
                raise RuntimeError("server_busy")

            # Each session owns an independent retrieval stack.
            retriever = HybridRetriever(
                self._embedding_model,
                reranker_model=self._reranker_model,
            )
            cache = QueryCache(retriever)
            router = QueryRouter(retriever, cache=cache)
            pipeline = IngestionPipeline()

            session = Session(
                session_id=str(uuid.uuid4()),
                created_at=time.monotonic(),
                last_active=time.monotonic(),
                retriever=retriever,
                cache=cache,
                router=router,
                pipeline=pipeline,
            )
            self._sessions[session.session_id] = session
            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        async with self._registry_lock:
            self._evict_expired()
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired:
                del self._sessions[session_id]
                return None
            session.touch()
            return session

    async def delete_session(self, session_id: str) -> None:
        async with self._registry_lock:
            self._sessions.pop(session_id, None)

    @property
    def active_count(self) -> int:
        return len(self._sessions)


    def _evict_expired(self) -> None:
        """Must be called while holding _registry_lock."""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired]
        for sid in expired:
            del self._sessions[sid]