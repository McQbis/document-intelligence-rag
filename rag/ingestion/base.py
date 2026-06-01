from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import uuid


@dataclass(frozen=True)
class TextChunk:
    text: str
    source: str = ""
    page: int = 0
    chunk_index: int = 0
    file_type: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"TextChunk(source={self.source}, page={self.page}, "
            f"chunk_index={self.chunk_index}, text='{preview}...')"
        )