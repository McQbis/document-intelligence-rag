from __future__ import annotations

import re
from typing import List

from rag.ingestion.base import TextChunk


def clean_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(
    text: str,
    source: str,
    page: int,
    file_type: str,
    chunk_size: int = 400,
    overlap: int = 150,
    clean: bool = True,
) -> List[TextChunk]:
    if clean:
        text = clean_text(text)

    if not text:
        return []

    chunks: List[TextChunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_str = text[start:end]

        chunks.append(
            TextChunk(
                text=chunk_str,
                source=source,
                page=page,
                chunk_index=chunk_index,
                file_type=file_type,
            )
        )

        chunk_index += 1
        start += chunk_size - overlap

    return chunks