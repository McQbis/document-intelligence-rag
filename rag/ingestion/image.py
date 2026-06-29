from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytesseract
from PIL import Image

from rag.ingestion.base import TextChunk
from rag.ingestion.chunker import chunk_text


class ImageLoader:
    """Extracts text from images via OCR, optionally cleans it up with an
    LLM correction pass, and converts it into TextChunks — same output
    shape as PDFLoader/TextLoader/MarkdownLoader, so the rest of the
    pipeline (retriever, reranker, generation) doesn't need to know the
    text came from an image rather than a native text file.

    Pipeline: image -> pytesseract OCR -> raw text
                     -> OCRCorrector (LLM, optional) -> cleaned text
                     -> chunk_text (same chunker as every other loader)
    """

    def __init__(self, chunk_size: int = 400, overlap: int = 80, corrector: Optional[object] = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Lazily constructed so importing this module doesn't require
        # GROQ_API_KEY / langchain at all if correction is never used.
        self._corrector = corrector

    def _get_corrector(self):
        if self._corrector is None:
            from rag.generation import OCRCorrector

            self._corrector = OCRCorrector()
        return self._corrector

    def load(self, file_path: str) -> List[TextChunk]:
        path = Path(file_path)

        with Image.open(path) as img:
            raw_text = pytesseract.image_to_string(img)

        if not raw_text.strip():
            return []

        corrector = self._get_corrector()
        text = corrector.correct(raw_text)

        return chunk_text(
            text=text,
            source=path.name,
            page=0,
            file_type="image",
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )