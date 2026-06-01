from __future__ import annotations

from pathlib import Path
from typing import List, Union

from rag.ingestion.base import TextChunk
from rag.ingestion.text import TextLoader
from rag.ingestion.markdown import MarkdownLoader
from rag.ingestion.pdf import PDFLoader


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


class IngestionPipeline:
    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

        self._pdf = PDFLoader(chunk_size, overlap)
        self._md = MarkdownLoader(chunk_size, overlap)
        self._txt = TextLoader(chunk_size, overlap)


    def process(self, file_path: Union[str, Path]) -> List[TextChunk]:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._pdf.load(str(path))
        elif ext == ".md":
            return self._md.load(str(path))
        elif ext == ".txt":
            return self._txt.load(str(path))
        else:
            raise ValueError(
                f"Unsupported file type: {ext!r}. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

    def process_many(
        self, file_paths: List[Union[str, Path]]
    ) -> List[TextChunk]:
        import warnings

        all_chunks: List[TextChunk] = []
        for fp in file_paths:
            try:
                all_chunks.extend(self.process(fp))
            except ValueError as e:
                warnings.warn(str(e))
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