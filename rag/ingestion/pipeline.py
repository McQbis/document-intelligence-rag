from __future__ import annotations

from pathlib import Path
from typing import List, Union

from rag.ingestion.base import TextChunk
from rag.ingestion.text import TextLoader
from rag.ingestion.markdown import MarkdownLoader
from rag.ingestion.pdf import PDFLoader


SUPPORTED = {
    ".pdf": PDFLoader,
    ".md": MarkdownLoader,
    ".txt": TextLoader,
}


class IngestionPipeline:
    """Unified document ingestion → TextChunk pipeline."""

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

        self._loaders = {
            ext: loader(chunk_size, overlap)
            for ext, loader in SUPPORTED.items()
        }

    def process(self, file_path: Union[str, Path]) -> List[TextChunk]:
        path = Path(file_path)
        loader = self._loaders.get(path.suffix.lower())

        if not loader:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        return loader.load(str(path))

    def process_many(self, file_paths: List[Union[str, Path]]) -> List[TextChunk]:
        all_chunks: List[TextChunk] = []

        for fp in file_paths:
            try:
                all_chunks.extend(self.process(fp))
            except ValueError:
                continue

        return all_chunks

    def process_raw(
        self,
        text: str,
        source: str = "raw",
        file_type: str = "txt",
    ) -> List[TextChunk]:
        from rag.ingestion.chunker import chunk_text

        return chunk_text(
            text=text,
            source=source,
            page=0,
            file_type=file_type,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )