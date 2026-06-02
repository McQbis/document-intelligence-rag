---
title: Document Intelligence RAG
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Document Intelligence RAG

Hybrid document retrieval system combining **BM25s**, **FAISS HNSW** and **CrossEncoder reranking**, evaluated on BEIR benchmarks.

![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)
![License](https://img.shields.io/github/license/McQbis/document-intelligence-rag?style=flat-square)
![Tests](https://img.shields.io/badge/tests-pytest-green?style=flat-square)

---

## Architecture

```
Query
  │
  ▼
QueryCache ── hit ──────────────────────────────────────── Results
  │ miss                                                      ▲
  ▼                                                           │
QueryRouter                                                   │
  │                                                           │
  ├─ fast (short / keyword query) ────────────────────────────│
  │     ├─ BM25s (lexical)   ─┐                               │
  │     └─ FAISS HNSW (dense)─┴─ RRF fusion                   │
  │                                                           │
  └─ deep (analytical query)  ────────────────────────────────┘
        ├─ BM25s (lexical)   ─┐
        └─ FAISS HNSW (dense)─┴─ RRF fusion ─ CrossEncoder rerank
```

**QueryCache** — two-layer cache: exact LRU match (hash) and semantic near-duplicate detection (cosine similarity). Bypassed in deep mode to always return fresh results.

**QueryRouter** — classifies queries automatically based on length and analytical keywords. Can be overridden explicitly with `fast`, `auto`, or `deep` mode.

---

## Demo

![Demo screenshot](assets/screenshot.png)

A live demo is available on Hugging Face Spaces. Upload your own PDF, MD, or TXT files and query them with fast or deep retrieval mode.

👉 https://huggingface.co/spaces/McQbis/document-intelligence-rag

---

## Quick Start

```python
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.embeddings import EmbeddingModel
from rag.retrieval.retriever import HybridRetriever
from rag.routing.router import QueryRouter, RouteMode

# Ingest
pipeline = IngestionPipeline()
chunks = pipeline.process("document.pdf")   # also .md and .txt

# Build index
emb = EmbeddingModel()
retriever = HybridRetriever(emb)
retriever.build_index(chunks)

# Search
router = QueryRouter(retriever)
results = router.search("What is retrieval-augmented generation?", mode=RouteMode.AUTO)

for chunk, score in results:
    print(f"[{score:.3f}] {chunk.source} — {chunk.text[:120]}")
```

---

## Installation

```bash
git clone https://github.com/McQbis/document-intelligence-rag.git
cd document-intelligence-rag
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
# .venv\Scripts\activate    # Windows
pip install -e .
```

### Optional — testing, evaluation and finetuning

```bash
pip install -e ".[dev]"
```

---

## Testing

The test suite covers the full stack without loading any real models — all embedding and retrieval components are mocked, so tests run fast and work offline in CI.

- Unit tests — chunking, routing, ingestion pipeline
- API integration tests — session lifecycle, upload limits, search flows

```bash
pytest
```

---

## Evaluation (BEIR)

Offline evaluation on BEIR benchmark datasets. Supports multiple datasets (`fiqa`, `scifact`, `nfcorpus`, `nq`, …) and all retrieval modes.

```bash
python scripts/evaluate.py --dataset scifact --router deep
python scripts/evaluate.py --dataset fiqa --router fast   # no reranker
```

## Finetuning (offline)

Fine-tunes the bi-encoder on BEIR training data using MultipleNegativesRankingLoss. The finetuned model can be passed directly to the evaluator.

```bash
python scripts/finetune.py --dataset fiqa --output-dir models/finetuned
python scripts/evaluate.py --dataset fiqa --model models/finetuned --router deep
```

---

## CI/CD
 
Every push to `main` triggers a GitHub Actions pipeline:
 
1. **Test** — runs the full test suite (`pytest tests/ -v`) with all dependencies mocked, no real models loaded
2. **Deploy** — if tests pass, automatically pushes to HF Spaces
Pull requests only run tests — deploy fires on merge to `main`.
 
---

## Results

| Dataset | Mode | Recall@10 | nDCG@10 |
|---------|------|-----------|---------|
| FiQA | fast (no reranker) | 0.6636 | 0.3533 |
| FiQA | deep (reranker) | 0.6759 | 0.3750 |
| FiQA | finetuned + deep | 0.6775 | **0.3815** |
| SciFact | deep (reranker) | 0.8767 | 0.7293 |
| SciFact | finetuned + deep | 0.9033 | **0.7390** |

👉 **Full report:** [RESULTS.md](RESULTS.md) — includes fast vs deep comparison, finetuning analysis and future work.
