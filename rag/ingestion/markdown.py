from __future__ import annotations

import re
from pathlib import Path
from typing import List

from rag.ingestion.base import TextChunk
from rag.ingestion.chunker import chunk_text


_MD_CLEANERS = [
    (r"^#{1,6}\s+", ""),                 # headers
    (r"\*{1,3}(.+?)\*{1,3}", r"\1"),     # bold/italic *
    (r"_{1,3}(.+?)_{1,3}", r"\1"),       # bold/italic _
    (r"`(.+?)`", r"\1"),                # inline code
    (r"!\[.*?\]\(.+?\)", ""),           # images
    (r"\[(.+?)\]\(.+?\)", r"\1"),       # links
    (r"^[-*_]{3,}\s*$", ""),            # rules
]


def clean_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]+?```", "", text)  # remove code blocks first

    for pattern, repl in _MD_CLEANERS:
        text = re.sub(pattern, repl, text, flags=re.MULTILINE)

    return text


class MarkdownLoader:
    """Parses Markdown files into clean TextChunks for retrieval."""

    def __init__(self, chunk_size: int = 400, overlap: int = 80, strip_syntax: bool = True):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.strip_syntax = strip_syntax

    def load(self, file_path: str) -> List[TextChunk]:
        path = Path(file_path)
        raw = path.read_text(encoding="utf-8", errors="ignore")

        text = clean_markdown(raw) if self.strip_syntax else raw

        return chunk_text(
            text=text,
            source=path.name,
            page=0,
            file_type="md",
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )