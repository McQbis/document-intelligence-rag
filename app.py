from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Loads .env into os.environ if present (local dev only — on HF Spaces /
# Cloud Run, GROQ_API_KEY etc. come from real platform secrets, and
# load_dotenv() is a harmless no-op there since there's no .env file).
load_dotenv()

from rag.retrieval.embeddings import EmbeddingModel
from rag.generation import AnswerGenerator
from api.session import SessionManager
from api.routes import router, set_manager, set_generator


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

    # LLM generation (LangChain + Groq) is optional: the app still works in
    # retrieval-only mode if GROQ_API_KEY isn't set.
    generator = AnswerGenerator()
    print(f"[startup] LLM generation (Groq)   : {'enabled' if generator.is_configured else 'disabled (no GROQ_API_KEY)'}")
    set_generator(generator)

    print("[startup] Ready.")
    yield


app = FastAPI(title="Document Intelligence RAG", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(router, prefix="/api")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")