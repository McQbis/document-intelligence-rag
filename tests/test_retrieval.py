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


# ======================================================================
# Image ingestion (OCR -> optional LLM correction -> chunking)
# ======================================================================

class _StubCorrector:
    """Stands in for OCRCorrector — lets us test ImageLoader without a real
    Tesseract binary or GROQ_API_KEY in CI."""

    def __init__(self, fixed_output=None):
        self.fixed_output = fixed_output
        self.calls = []

    def correct(self, raw_text):
        self.calls.append(raw_text)
        return self.fixed_output if self.fixed_output is not None else raw_text


def _make_test_png(path, text="Retrieval Augmented Generation"):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (500, 80), color="white")
    ImageDraw.Draw(img).text((10, 10), text, fill="black")
    img.save(path)


def test_image_loader_registered_in_pipeline():
    from rag.ingestion.image import ImageLoader

    pipeline = IngestionPipeline()
    assert isinstance(pipeline._loaders.get(".png"), ImageLoader)
    assert isinstance(pipeline._loaders.get(".jpg"), ImageLoader)
    assert isinstance(pipeline._loaders.get(".jpeg"), ImageLoader)


def test_image_loader_uses_correction_output(tmp_path, monkeypatch):
    """ImageLoader must run OCR output through the corrector and chunk
    *its* output, not the raw OCR text — without needing a real LLM call."""
    from rag.ingestion.image import ImageLoader

    monkeypatch.setattr(
        "pytesseract.image_to_string", lambda img: "Retreival Augmentd Generaton"
    )

    stub = _StubCorrector(fixed_output="Retrieval Augmented Generation")
    loader = ImageLoader(corrector=stub)

    f = tmp_path / "doc.png"
    _make_test_png(f)

    chunks = loader.load(str(f))

    assert len(chunks) == 1
    assert chunks[0].text == "Retrieval Augmented Generation"
    assert chunks[0].file_type == "image"
    assert chunks[0].source == "doc.png"
    assert stub.calls == ["Retreival Augmentd Generaton"]


def test_image_loader_skips_correction_for_blank_ocr(tmp_path, monkeypatch):
    """If OCR finds no text at all, ImageLoader shouldn't bother calling
    the corrector or produce any chunks."""
    from rag.ingestion.image import ImageLoader

    monkeypatch.setattr("pytesseract.image_to_string", lambda img: "   \n  ")

    stub = _StubCorrector()
    loader = ImageLoader(corrector=stub)

    f = tmp_path / "blank.png"
    _make_test_png(f, text="")

    chunks = loader.load(str(f))

    assert chunks == []
    assert stub.calls == []


def test_ocr_corrector_falls_back_to_raw_text_when_not_configured(monkeypatch):
    """Without GROQ_API_KEY, correction is a no-op — OCR ingestion must
    still work end-to-end, just without the cleanup pass."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from rag.generation import OCRCorrector

    corrector = OCRCorrector()
    assert corrector.is_configured is False
    assert corrector.correct("Retreival Augmentd Generaton") == "Retreival Augmentd Generaton"


def test_ocr_corrector_falls_back_on_groq_failure(monkeypatch):
    """If the LLM call itself fails (rate limit, outage), correction must
    degrade to the raw OCR text rather than raising — OCR ingestion should
    never hard-fail just because the optional cleanup pass couldn't run."""
    import groq
    import httpx
    from rag.generation import OCRCorrector

    monkeypatch.setenv("GROQ_API_KEY", "fake-key-for-test")
    corrector = OCRCorrector()

    class _ExplodingChain:
        def invoke(self, _input):
            raise groq.APIConnectionError(
                request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            )

    corrector._chain = _ExplodingChain()

    assert corrector.correct("Retreival Augmentd Generaton") == "Retreival Augmentd Generaton"