from __future__ import annotations

from pathlib import Path
from typing import List

from rag.ingestion.base import TextChunk
from rag.ingestion.chunker import chunk_text


class TextLoader:
    """Loads plain text files and converts them into TextChunks."""

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load(self, file_path: str) -> List[TextChunk]:
        path = Path(file_path)

        text = path.read_text(encoding="utf-8", errors="ignore")

        return chunk_text(
            text=text,
            source=path.name,   # cleaner metadata than full path
            page=0,
            file_type="txt",
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )