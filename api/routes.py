from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, UploadFile, File
from pydantic import BaseModel

from api.session import (
    SessionManager,
    MAX_FILES_PER_SESSION,
    MAX_TOTAL_BYTES,
    SESSION_TTL_SECONDS,
    UploadedFile,
)
from rag.routing.router import RouteMode

DEMO_DOCS_DIR = Path(__file__).parent.parent / "demo_docs"

router = APIRouter()
_manager: SessionManager | None = None


def set_manager(m: SessionManager) -> None:
    global _manager
    _manager = m



class SearchRequest(BaseModel):
    query: str
    mode: str = "auto"
    top_k: int = 10
    candidate_k: int = 30


class ChunkResult(BaseModel):
    text: str
    source: str
    page: int
    score: float
    chunk_index: int


class SearchResponse(BaseModel):
    query: str
    mode: str
    resolved_mode: str
    results: list[ChunkResult]
    cache_hit: bool


class SessionStatus(BaseModel):
    session_id: str
    chunks_indexed: int
    index_built: bool
    files: list[dict]
    file_count: int
    max_files: int
    bytes_used: int
    max_bytes: int
    seconds_remaining: int
    active_sessions: int
    max_sessions: int



X_SESSION_ID = "x-session-id"


async def _require_session(session_id: str | None):
    if not session_id:
        raise HTTPException(401, "No session. Call POST /session first.")
    session = await _manager.get_session(session_id)
    if session is None:
        raise HTTPException(410, "Session expired or not found.")
    return session



@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/session")
async def create_session():
    try:
        session = await _manager.create_session()
    except RuntimeError as e:
        if "server_busy" in str(e):
            raise HTTPException(503, "Server is busy — all session slots are taken. Try again shortly.")
        raise

    return {
        "session_id": session.session_id,
        "ttl_seconds": SESSION_TTL_SECONDS,
        "max_files": MAX_FILES_PER_SESSION,
        "max_bytes": MAX_TOTAL_BYTES,
    }


@router.delete("/session")
async def delete_session(
    session_id: Annotated[str | None, Header(alias=X_SESSION_ID)] = None,
):
    if session_id:
        await _manager.delete_session(session_id)
    return {"status": "session deleted"}


@router.get("/session/status", response_model=SessionStatus)
async def session_status(
    session_id: Annotated[str | None, Header(alias=X_SESSION_ID)] = None,
):
    session = await _require_session(session_id)
    return SessionStatus(
        session_id=session.session_id,
        chunks_indexed=len(session.retriever.chunks),
        index_built=session.retriever.is_built,
        files=[{"filename": f.filename, "chunks": f.chunks, "size": f.size_bytes} for f in session.files],
        file_count=session.file_count,
        max_files=MAX_FILES_PER_SESSION,
        bytes_used=session.total_bytes,
        max_bytes=MAX_TOTAL_BYTES,
        seconds_remaining=session.seconds_remaining,
        active_sessions=_manager.active_count,
        max_sessions=5,
    )


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    session_id: Annotated[str | None, Header(alias=X_SESSION_ID)] = None,
):
    session = await _require_session(session_id)

    async with session.lock:
        if session.file_count >= MAX_FILES_PER_SESSION:
            raise HTTPException(429, f"File limit reached ({MAX_FILES_PER_SESSION} files per session).")

        suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if suffix not in {".pdf", ".md", ".txt"}:
            raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: pdf, md, txt.")

        content = await file.read()
        size = len(content)

        if session.total_bytes + size > MAX_TOTAL_BYTES:
            raise HTTPException(413, f"Total upload limit exceeded. {session.bytes_remaining // 1024} KB remaining.")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            chunks = await asyncio.get_event_loop().run_in_executor(
                None, session.pipeline.process, tmp_path
            )
            for c in chunks:
                c.source = file.filename
        finally:
            os.unlink(tmp_path)

        if not chunks:
            raise HTTPException(422, "No text could be extracted from this file.")

        if session.retriever.is_built:
            await asyncio.get_event_loop().run_in_executor(
                None, session.retriever.add_chunks, chunks
            )
        else:
            await asyncio.get_event_loop().run_in_executor(
                None, session.retriever.build_index, chunks
            )

        session.cache.invalidate()
        session.files.append(UploadedFile(filename=file.filename, size_bytes=size, chunks=len(chunks)))
        session.total_bytes += size

    return {
        "filename": file.filename,
        "chunks_added": len(chunks),
        "total_chunks": len(session.retriever.chunks),
        "files_used": session.file_count,
        "bytes_used": session.total_bytes,
    }


@router.get("/demo/list")
def demo_list():
    docs = []
    for p in sorted(DEMO_DOCS_DIR.glob("*.md")) + sorted(DEMO_DOCS_DIR.glob("*.txt")):
        docs.append({"filename": p.name, "size": p.stat().st_size})
    return {"docs": docs}


@router.post("/demo/load/{filename}")
async def demo_load(
    filename: str,
    session_id: Annotated[str | None, Header(alias=X_SESSION_ID)] = None,
):
    session = await _require_session(session_id)

    path = DEMO_DOCS_DIR / filename
    if not path.exists() or path.parent != DEMO_DOCS_DIR:
        raise HTTPException(404, "Demo document not found.")

    async with session.lock:
        if session.file_count >= MAX_FILES_PER_SESSION:
            raise HTTPException(429, f"File limit reached ({MAX_FILES_PER_SESSION} files per session).")

        size = path.stat().st_size
        if session.total_bytes + size > MAX_TOTAL_BYTES:
            raise HTTPException(413, "Total size limit exceeded.")

        if any(f.filename == filename for f in session.files):
            raise HTTPException(409, f"'{filename}' is already loaded.")

        chunks = await asyncio.get_event_loop().run_in_executor(
            None, session.pipeline.process, str(path)
        )
        for c in chunks:
            c.source = filename

        if session.retriever.is_built:
            await asyncio.get_event_loop().run_in_executor(
                None, session.retriever.add_chunks, chunks
            )
        else:
            await asyncio.get_event_loop().run_in_executor(
                None, session.retriever.build_index, chunks
            )

        session.cache.invalidate()
        session.files.append(UploadedFile(filename=filename, size_bytes=size, chunks=len(chunks)))
        session.total_bytes += size

    return {
        "filename": filename,
        "chunks_added": len(chunks),
        "total_chunks": len(session.retriever.chunks),
    }


DEMO_QUERIES: dict[str, list[str]] = {
    "rag_overview.md": [
        "What is Retrieval-Augmented Generation?",
        "How does hybrid retrieval and RRF work?",
        "How are RAG systems evaluated?",
        "What is the difference between a bi-encoder and a cross-encoder?",
        "What advanced RAG techniques exist?",
    ],
    "transformer_overview.md": [
        "How does self-attention work?",
        "What is the difference between BERT and GPT?",
        "What are Transformer scaling laws?",
        "How are Transformers used in retrieval models?",
        "What is positional encoding?",
    ],
}


@router.get("/demo/queries")
async def demo_queries(
    session_id: Annotated[str | None, Header(alias=X_SESSION_ID)] = None,
):
    session = await _require_session(session_id)
    loaded = {f.filename for f in session.files}
    queries = []
    for fname in loaded:
        queries.extend(DEMO_QUERIES.get(fname, []))
    return {"queries": queries}


@router.post("/search", response_model=SearchResponse)
async def search(
    req: SearchRequest,
    session_id: Annotated[str | None, Header(alias=X_SESSION_ID)] = None,
):
    session = await _require_session(session_id)

    if not session.retriever.is_built:
        raise HTTPException(400, "No documents indexed yet. Upload a file or load a demo document first.")

    try:
        mode = RouteMode(req.mode)
    except ValueError:
        raise HTTPException(400, f"Invalid mode '{req.mode}'. Use: auto, fast, deep.")

    resolved = session.router.classify(req.query) if mode == RouteMode.AUTO else mode

    results = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: session.router.search(
            req.query,
            mode=mode,
            top_k=req.top_k,
            candidate_k=req.candidate_k,
        )
    )

    session.query_count += 1

    return SearchResponse(
        query=req.query,
        mode=req.mode,
        resolved_mode=resolved.value,
        cache_hit=False,
        results=[
            ChunkResult(
                text=chunk.text,
                source=chunk.source,
                page=chunk.page,
                score=round(score, 4),
                chunk_index=chunk.chunk_index,
            )
            for chunk, score in results
        ],
    )