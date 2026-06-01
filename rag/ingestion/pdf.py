from __future__ import annotations

from typing import List

from pypdf import PdfReader

from rag.ingestion.base import TextChunk
from rag.ingestion.chunker import chunk_text


class PDFLoader:
    """Extracts text from PDF pages and converts them into TextChunks."""

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load(self, file_path: str) -> List[TextChunk]:
        reader = PdfReader(file_path)

        chunks: List[TextChunk] = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()

            if not text:
                continue

            chunks.extend(
                chunk_text(
                    text=text,
                    source=file_path.split("/")[-1],  # cleaner metadata
                    page=page_num,
                    file_type="pdf",
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                )
            )

        return chunks