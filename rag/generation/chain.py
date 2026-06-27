from __future__ import annotations

import os
from typing import List, Tuple

import groq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_groq import ChatGroq

from rag.ingestion.base import TextChunk

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are a document Q&A assistant. Answer the user's question using ONLY "
    "the numbered context passages below. Each passage is tagged with a source "
    "id like [1], [2]. Cite the ids you relied on inline, e.g. 'according to [2]'. "
    "If the context does not contain the answer, say so explicitly instead of "
    "guessing.\n\nContext:\n{context}"
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def format_context(results: List[Tuple[TextChunk, float]]) -> str:
    """Render retrieved (and optionally reranked) chunks into a numbered,
    source-attributed block the LLM can cite back to the user."""
    lines = []
    for i, (chunk, _score) in enumerate(results, start=1):
        lines.append(f"[{i}] (source: {chunk.source}, page {chunk.page})\n{chunk.text}")
    return "\n\n".join(lines) if lines else "No documents retrieved."


def build_answer_chain(llm: Runnable) -> Runnable:
    """LCEL chain orchestrating: retrieved context -> prompt -> LLM -> text.

    This is the orchestration layer: the retriever/reranker stay as the
    existing custom hybrid pipeline (rag.retrieval.HybridRetriever); LangChain
    is only responsible for composing the retrieved context with the prompt
    and the Groq call, and for making that swap (model/provider) a one-line
    change instead of hand-rolled HTTP/string-formatting code.
    """
    return (
        {
            "context": RunnableLambda(lambda x: format_context(x["results"])),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | _PROMPT
        | llm
        | StrOutputParser()
    )


class GenerationError(Exception):
    """Raised when the LLM call itself fails (rate limit, quota, Groq outage,
    etc.) — distinct from RuntimeError, which means 'not configured at all'.
    Carries an HTTP-style status_code so the API layer can return a clean
    response instead of a bare 500.
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class AnswerGenerator:
    """Thin wrapper around the LangChain answer chain.

    Lazily initializes the Groq client so the rest of the app (retrieval-only
    mode) keeps working even if GROQ_API_KEY isn't configured.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.1):
        self._model_name = model or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self._temperature = temperature
        self._chain: Runnable | None = None

    @property
    def is_configured(self) -> bool:
        return bool(os.getenv("GROQ_API_KEY"))

    def _get_chain(self) -> Runnable:
        if self._chain is None:
            if not self.is_configured:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Generation is disabled; "
                    "/api/search (retrieval-only) still works."
                )
            llm = ChatGroq(model=self._model_name, temperature=self._temperature)
            self._chain = build_answer_chain(llm)
        return self._chain

    def generate(self, query: str, results: List[Tuple[TextChunk, float]]) -> str:
        chain = self._get_chain()
        try:
            return chain.invoke({"question": query, "results": results})
        except groq.RateLimitError as e:
            # Free-tier requests/tokens-per-minute or per-day limit exceeded.
            raise GenerationError(
                429,
                "Groq API rate limit reached (free-tier quota exceeded). "
                "Try again in a moment, or use retrieval-only results.",
            ) from e
        except groq.AuthenticationError as e:
            raise GenerationError(
                401,
                "Groq API key was rejected (invalid or revoked).",
            ) from e
        except groq.APIStatusError as e:
            # Any other non-2xx response from Groq (bad request, 5xx, etc.)
            raise GenerationError(
                e.status_code or 502,
                f"Groq API error: {e.message or str(e)}",
            ) from e