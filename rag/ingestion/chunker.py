from __future__ import annotations

import re
from typing import List

from rag.ingestion.base import TextChunk


_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    return _WS_RE.sub(" ", text).replace("\n", " ").replace("\t", " ").strip()


def chunk_text(
    text: str,
    source: str,
    page: int,
    file_type: str,
    chunk_size: int = 400,
    overlap: int = 80,
    clean: bool = True,
) -> List[TextChunk]:
    if not text:
        return []

    if clean:
        text = clean_text(text)

    step = max(1, chunk_size - overlap)

    return [
        TextChunk(
            text=text[i : i + chunk_size],
            source=source,
            page=page,
            chunk_index=idx,
            file_type=file_type,
        )
        for idx, i in enumerate(range(0, len(text), step))
    ]