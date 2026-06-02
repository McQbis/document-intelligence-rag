"""
Unit tests for ingestion + routing + retrieval logic.

Covers:
- text cleaning
- chunking logic
- ingestion pipeline
- query routing (fast/deep)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from rag.ingestion.base import TextChunk
from rag.ingestion.chunker import chunk_text, clean_text
from rag.ingestion.pipeline import IngestionPipeline
from rag.routing.router import QueryRouter, RouteMode


def test_clean_text_collapses_whitespace():
    assert clean_text("hello   \t  world\n") == "hello world"


def test_chunk_text_produces_correct_count():
    text = "a" * 1000
    chunks = chunk_text(text, source="test", page=0, file_type="txt",
                        chunk_size=300, overlap=50)
    assert len(chunks) >= 3
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_chunk_text_overlap():
    text = "abcdefghij" * 100   # 1000 chars
    chunks = chunk_text(text, source="t", page=0, file_type="txt",
                        chunk_size=100, overlap=20)
    assert chunks[1].chunk_index == 1
    assert chunks[0].text[80:] == chunks[1].text[:20]



def test_pipeline_raises_on_unsupported_type(tmp_path):
    f = tmp_path / "file.xyz"
    f.write_text("data")
    pipeline = IngestionPipeline()
    with pytest.raises(ValueError, match="Unsupported file type"):
        pipeline.process(str(f))


def test_pipeline_process_many_skips_bad(tmp_path):
    good = tmp_path / "doc.txt"
    good.write_text("Hello world " * 50)

    bad = tmp_path / "doc.xyz"
    bad.write_text("ignored")

    pipeline = IngestionPipeline()
    chunks = pipeline.process_many([str(good), str(bad)])

    assert len(chunks) > 0


def test_pipeline_txt(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello RAG world! " * 100)
    pipeline = IngestionPipeline()
    chunks = pipeline.process(str(f))
    assert len(chunks) > 0
    assert chunks[0].file_type == "txt"


def test_pipeline_md(tmp_path):
    f = tmp_path / "test.md"
    f.write_text("# Header\n\nSome **bold** text and a [link](https://x.com).\n" * 30)
    pipeline = IngestionPipeline()
    chunks = pipeline.process(str(f))
    assert len(chunks) > 0
    assert chunks[0].file_type == "md"
    assert "#" not in chunks[0].text


def test_pipeline_raw():
    pipeline = IngestionPipeline(chunk_size=200, overlap=20)
    chunks = pipeline.process_raw("word " * 200, source="api_upload")
    assert len(chunks) > 1
    assert chunks[0].source == "api_upload"


class _MockRetriever:
    """No-op retriever for router tests."""
    def search(self, query, top_k=5, candidate_k=20, rerank=True):
        return []
    def build_index(self, chunks):
        pass


def test_router_fast_for_short_query():
    r = _MockRetriever()
    router = QueryRouter(r)
    assert router.classify("what is RAG") == RouteMode.FAST


def test_router_deep_for_long_query():
    r = _MockRetriever()
    router = QueryRouter(r)
    assert router.classify("can you explain in detail how hybrid retrieval works") == RouteMode.DEEP


def test_router_deep_for_keyword():
    r = _MockRetriever()
    router = QueryRouter(r)
    assert router.classify("summarize this") == RouteMode.DEEP


def test_router_force_mode():
    r = _MockRetriever()
    router = QueryRouter(r)
    results = router.search("short", mode=RouteMode.DEEP)
    assert results == []   # mock returns empty


def test_text_chunk_has_unique_ids():
    c1 = TextChunk(text="hello")
    c2 = TextChunk(text="hello")
    assert c1.id != c2.id