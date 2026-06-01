from __future__ import annotations

from dataclasses import dataclass, field
import uuid


@dataclass
class TextChunk:
    text: str
    source: str = ""          # file path or document title
    page: int = 0
    chunk_index: int = 0
    file_type: str = ""       # "pdf" | "md" | "txt"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"TextChunk(source={self.source!r}, page={self.page}, "
            f"chunk_index={self.chunk_index}, text={preview!r}...)"
        )