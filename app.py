from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from rag.retrieval.embeddings import EmbeddingModel
from api.session import SessionManager
from api.routes import router, set_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared application resources during startup."""
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    reranker_name = os.getenv("RERANKER_MODEL",  "BAAI/bge-reranker-base")

    print(f"[startup] Loading embedding model : {model_name}")
    emb = EmbeddingModel(model_name=model_name)

    print(f"[startup] Reranker                : {reranker_name}")
    manager = SessionManager(emb, reranker_name)
    set_manager(manager)

    print("[startup] Ready.")
    yield


app = FastAPI(title="RAG System", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(router, prefix="/api")

app.mount("/static", StaticFiles(directory="static"), name="static")

# @app.get("/")
# def root():
#     return FileResponse("static/index.html")