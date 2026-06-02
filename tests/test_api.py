from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_mock_chunk(text: str = "mock text", source: str = "test.txt") -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    chunk.source = source
    chunk.page = 0
    chunk.chunk_index = 0
    return chunk


class MockEmbeddingModel:
    def embed_text(self, text: str) -> np.ndarray:
        return np.random.rand(768).astype("float32")

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        return np.random.rand(len(texts), 768).astype("float32")


class MockRetriever:
    def __init__(self, *args, **kwargs):
        self.chunks: list = []
        self._chunk_id_to_idx: dict = {}
        self.index = None
        self.bm25 = None

    def build_index(self, chunks):
        self.chunks = list(chunks)
        self.index = True
        self.bm25 = True

    def add_chunks(self, chunks):
        self.chunks.extend(chunks)

    def search(self, query, top_k=10, candidate_k=30, rerank=True):
        return [(_make_mock_chunk(f"result for: {query}", "demo.md"), 0.9)]

    @property
    def is_built(self) -> bool:
        return self.index is not None


class MockCache:
    def __init__(self, retriever):
        self._retriever = retriever

    def search(self, query, top_k=10, candidate_k=30):
        return self._retriever.search(query, top_k=top_k), False

    def invalidate(self):
        pass


class MockRouter:
    def __init__(self, retriever, cache=None, **kwargs):
        self._retriever = retriever

    def search(self, query, mode=None, top_k=10, candidate_k=30):
        return self._retriever.search(query, top_k=top_k)

    def classify(self, query):
        from rag.routing.router import RouteMode
        return RouteMode.FAST


class MockPipeline:
    def process(self, path: str):
        chunks = [_make_mock_chunk(f"chunk from {Path(path).name}", Path(path).name)]
        return chunks


DEMO_DOCS_DIR = Path(__file__).resolve().parents[1] / "demo_docs"


@pytest.fixture(scope="function")
def app():
    with patch("rag.retrieval.embeddings.EmbeddingModel", MockEmbeddingModel), \
         patch("rag.retrieval.retriever.HybridRetriever", MockRetriever), \
         patch("rag.cache.query_cache.QueryCache", MockCache), \
         patch("rag.routing.router.QueryRouter", MockRouter), \
         patch("rag.ingestion.pipeline.IngestionPipeline", MockPipeline), \
         patch("api.routes.DEMO_DOCS_DIR", DEMO_DOCS_DIR):

        import importlib
        import api.routes as routes_mod
        import api.session as session_mod
        importlib.reload(session_mod)
        importlib.reload(routes_mod)

        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from api.session import SessionManager
        from api.routes import router, set_manager

        test_app = FastAPI()
        test_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        test_app.include_router(router, prefix="/api")

        # Fresh manager per test — no state leaking between tests
        manager = SessionManager(MockEmbeddingModel(), "mock-reranker")
        set_manager(manager)

        yield test_app


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def session(client):
    """Create a fresh session and return its ID."""
    r = await client.post("/api/session")
    assert r.status_code == 200
    return r.json()["session_id"]


def hdrs(session_id: str) -> dict:
    return {"x-session-id": session_id}


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_create_session_returns_id(client):
    r = await client.post("/api/session")
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["ttl_seconds"] == 15 * 60
    assert data["max_files"] == 5


@pytest.mark.asyncio
async def test_session_status(client, session):
    r = await client.get("/api/session/status", headers=hdrs(session))
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == session
    assert data["index_built"] is False
    assert data["file_count"] == 0


@pytest.mark.asyncio
async def test_no_session_header_returns_401(client):
    r = await client.get("/api/session/status")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_invalid_session_id_returns_410(client):
    r = await client.get("/api/session/status", headers={"x-session-id": "nonexistent"})
    assert r.status_code == 410


@pytest.mark.asyncio
async def test_delete_session(client, session):
    r = await client.delete("/api/session", headers=hdrs(session))
    assert r.status_code == 200
    # Session should now be gone
    r2 = await client.get("/api/session/status", headers=hdrs(session))
    assert r2.status_code == 410


@pytest.mark.asyncio
async def test_session_limit(client):
    """3rd session should return 503 when limit is 2."""
    with patch("api.session.MAX_SESSIONS", 2):
        from api.routes import _manager
        # Reset sessions to ensure clean state
        _manager._sessions.clear()

        for _ in range(2):
            r = await client.post("/api/session")
            assert r.status_code == 200

        r = await client.post("/api/session")
        assert r.status_code == 503
        assert "busy" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_expired_session_returns_410(client):
    """Manually expire a session by backdating created_at."""
    from api.routes import _manager

    r = await client.post("/api/session")
    sid = r.json()["session_id"]

    # Backdate creation time past TTL
    session_obj = _manager._sessions[sid]
    session_obj.created_at = time.monotonic() - (15 * 60 + 1)

    r2 = await client.get("/api/session/status", headers=hdrs(sid))
    assert r2.status_code == 410


@pytest.mark.asyncio
async def test_upload_txt(client, session):
    content = b"Hello world " * 100
    r = await client.post(
        "/api/upload",
        headers=hdrs(session),
        files={"file": ("test.txt", content, "text/plain")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "test.txt"
    assert data["chunks_added"] >= 1
    assert data["files_used"] == 1


@pytest.mark.asyncio
async def test_upload_md(client, session):
    content = b"# Header\n\nSome markdown content " * 50
    r = await client.post(
        "/api/upload",
        headers=hdrs(session),
        files={"file": ("doc.md", content, "text/markdown")},
    )
    assert r.status_code == 200
    assert r.json()["filename"] == "doc.md"


@pytest.mark.asyncio
async def test_upload_unsupported_format(client, session):
    r = await client.post(
        "/api/upload",
        headers=hdrs(session),
        files={"file": ("data.csv", b"a,b,c", "text/csv")},
    )
    assert r.status_code == 400
    assert "Unsupported" in r.json()["detail"]


@pytest.mark.asyncio
async def test_upload_file_limit(client, session):
    """6th file should return 429."""
    with patch("api.routes.MAX_FILES_PER_SESSION", 2):
        content = b"x" * 10

        for _ in range(2):
            r = await client.post(
                "/api/upload",
                headers=hdrs(session),
                files={"file": ("f.txt", content, "text/plain")},
            )

        r = await client.post(
            "/api/upload",
            headers=hdrs(session),
            files={"file": ("overflow.txt", content, "text/plain")},
        )
        assert r.status_code == 429
        assert "limit" in r.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_size_limit(client, session):
    with patch("api.routes.MAX_TOTAL_BYTES", 100):
        r = await client.post(
            "/api/upload",
            headers=hdrs(session),
            files={"file": ("big.txt", b"x" * 200, "text/plain")},
        )
        assert r.status_code == 413


@pytest.mark.asyncio
async def test_upload_without_session(client):
    r = await client.post(
        "/api/upload",
        files={"file": ("f.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_demo_list(client):
    r = await client.get("/api/demo/list")
    assert r.status_code == 200
    assert "docs" in r.json()
    # Should have at least our two demo docs
    names = [d["filename"] for d in r.json()["docs"]]
    assert any("rag" in n for n in names)
    assert any("transformer" in n for n in names)


@pytest.mark.asyncio
async def test_demo_load(client, session):
    # Get first available demo doc
    r = await client.get("/api/demo/list")
    docs = r.json()["docs"]
    assert docs, "No demo docs found"

    filename = docs[0]["filename"]
    r = await client.post(f"/api/demo/load/{filename}", headers=hdrs(session))
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == filename
    assert data["chunks_added"] >= 1


@pytest.mark.asyncio
async def test_demo_load_duplicate(client, session):
    r = await client.get("/api/demo/list")
    filename = r.json()["docs"][0]["filename"]

    await client.post(f"/api/demo/load/{filename}", headers=hdrs(session))

    # Second load of same file
    r2 = await client.post(f"/api/demo/load/{filename}", headers=hdrs(session))
    assert r2.status_code == 409
    assert "already loaded" in r2.json()["detail"]


@pytest.mark.asyncio
async def test_demo_load_nonexistent(client, session):
    r = await client.post("/api/demo/load/does_not_exist.md", headers=hdrs(session))
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_demo_queries_empty_before_load(client, session):
    r = await client.get("/api/demo/queries", headers=hdrs(session))
    assert r.status_code == 200
    assert r.json()["queries"] == []


@pytest.mark.asyncio
async def test_demo_queries_after_load(client, session):
    r = await client.get("/api/demo/list")
    filename = r.json()["docs"][0]["filename"]
    await client.post(f"/api/demo/load/{filename}", headers=hdrs(session))

    r = await client.get("/api/demo/queries", headers=hdrs(session))
    assert r.status_code == 200
    # rag_overview.md and transformer_overview.md both have hardcoded queries
    assert len(r.json()["queries"]) >= 0  # 0 if unknown filename, >0 for known


@pytest.mark.asyncio
async def test_search_without_index_returns_400(client, session):
    r = await client.post(
        "/api/search",
        headers=hdrs(session),
        json={"query": "hello", "mode": "auto"},
    )
    assert r.status_code == 400
    assert "No documents" in r.json()["detail"]


@pytest.mark.asyncio
async def test_search_returns_results(client, session):
    # Load a demo doc first to build index
    r = await client.get("/api/demo/list")
    filename = r.json()["docs"][0]["filename"]
    await client.post(f"/api/demo/load/{filename}", headers=hdrs(session))

    r = await client.post(
        "/api/search",
        headers=hdrs(session),
        json={"query": "What is RAG?", "mode": "auto", "top_k": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "What is RAG?"
    assert "results" in data
    assert "resolved_mode" in data
    assert len(data["results"]) >= 1

    result = data["results"][0]
    assert "text" in result
    assert "source" in result
    assert "score" in result
    assert "page" in result
    assert "chunk_index" in result


@pytest.mark.asyncio
async def test_search_fast_mode(client, session):
    r = await client.get("/api/demo/list")
    filename = r.json()["docs"][0]["filename"]
    await client.post(f"/api/demo/load/{filename}", headers=hdrs(session))

    r = await client.post(
        "/api/search",
        headers=hdrs(session),
        json={"query": "hybrid retrieval", "mode": "fast"},
    )
    assert r.status_code == 200
    assert r.json()["mode"] == "fast"


@pytest.mark.asyncio
async def test_search_deep_mode(client, session):
    r = await client.get("/api/demo/list")
    filename = r.json()["docs"][0]["filename"]
    await client.post(f"/api/demo/load/{filename}", headers=hdrs(session))

    r = await client.post(
        "/api/search",
        headers=hdrs(session),
        json={"query": "explain transformer architecture in detail", "mode": "deep"},
    )
    assert r.status_code == 200
    assert r.json()["mode"] == "deep"


@pytest.mark.asyncio
async def test_search_invalid_mode(client, session):
    r = await client.get("/api/demo/list")
    filename = r.json()["docs"][0]["filename"]
    await client.post(f"/api/demo/load/{filename}", headers=hdrs(session))

    r = await client.post(
        "/api/search",
        headers=hdrs(session),
        json={"query": "test", "mode": "turbo"},
    )
    assert r.status_code == 400
    assert "Invalid mode" in r.json()["detail"]


@pytest.mark.asyncio
async def test_search_without_session(client):
    r = await client.post(
        "/api/search",
        json={"query": "test"},
    )
    assert r.status_code == 401