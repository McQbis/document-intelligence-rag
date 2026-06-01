from __future__ import annotations

from typing import List

from pypdf import PdfReader

from rag.ingestion.base import TextChunk
from rag.ingestion.chunker import chunk_text


class PDFLoader:
    def __init__(self, chunk_size: int = 800, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load(self, file_path: str) -> List[TextChunk]:
        reader = PdfReader(file_path)
        chunks: List[TextChunk] = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if not text:
                continue

            page_chunks = chunk_text(
                text=text,
                source=file_path,
                page=page_num,
                file_type="pdf",
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )
            chunks.extend(page_chunks)

        return chunks