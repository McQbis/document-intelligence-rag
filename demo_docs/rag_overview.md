# Retrieval-Augmented Generation (RAG) — Overview

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system with a language model to produce more accurate and grounded answers. Instead of relying solely on the parametric knowledge baked into the model during training, RAG retrieves relevant documents from an external corpus at query time and uses them as context for generation.

The core idea was introduced in the 2020 paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Lewis et al. (Facebook AI Research). The paper demonstrated that augmenting a seq2seq model with a dense retrieval component significantly improved performance on open-domain QA, fact verification, and knowledge-grounded generation tasks.

## Components of a RAG Pipeline

### 1. Ingestion
Documents are split into chunks and encoded into dense vector representations (embeddings) using a bi-encoder model. These embeddings are stored in a vector database alongside the raw text.

Common chunking strategies include fixed-size character splits with overlap (simple and fast), sentence-level splits (more semantically coherent), and recursive splits (respects document structure).

### 2. Retrieval
At query time, the question is encoded using the same bi-encoder. Nearest neighbours are retrieved from the vector store using approximate nearest-neighbour search (FAISS, HNSW, ScaNN). Hybrid retrieval combines dense vectors with sparse BM25 scores using Reciprocal Rank Fusion (RRF).

### 3. Reranking
A cross-encoder model (e.g. BGE-reranker, Cohere Rerank) rescores the top-K candidates from retrieval. Cross-encoders attend jointly to the query and document and are significantly more accurate than bi-encoders, at the cost of higher latency.

### 4. Generation
The retrieved and reranked passages are injected into the language model's context window as grounding evidence. The model is prompted to answer using only the provided context, reducing hallucination.

## Evaluation Metrics

RAG systems are typically evaluated on retrieval quality and end-to-end generation quality separately.

For retrieval:
- **Recall@K**: fraction of queries for which the relevant document appears in the top-K results
- **nDCG@K**: normalised Discounted Cumulative Gain, measures ranking quality (relevant documents ranked higher score better)
- **MRR**: Mean Reciprocal Rank

For generation:
- **Faithfulness**: whether the answer is supported by the retrieved context
- **Answer Relevance**: whether the answer addresses the question
- **Context Precision / Recall**: RAGAS framework metrics

## Hybrid Retrieval and RRF

Hybrid retrieval combines two fundamentally different signals: lexical matching (BM25) and semantic similarity (dense embeddings). BM25 excels at exact keyword matches and rare terms; dense retrieval captures paraphrase and semantic overlap.

Reciprocal Rank Fusion merges ranked lists from multiple retrievers without requiring score calibration. For each document, RRF score = Σ 1/(k + rank_i), where k is a constant (typically 60) and rank_i is the document's rank in each list. The final ranking is by descending RRF score.

## BEIR Benchmark

BEIR (Benchmarking IR) is a heterogeneous benchmark for zero-shot evaluation of information retrieval models. It covers 18 datasets across diverse domains including biomedical (TREC-COVID, NFCorpus), financial QA (FiQA), scientific fact checking (SciFact), and web search (MS MARCO).

Strong baselines on BEIR:
- BM25: nDCG@10 ≈ 0.23 on FiQA
- dense bi-encoder (BGE-base): nDCG@10 ≈ 0.33 on FiQA
- hybrid + reranker: nDCG@10 ≈ 0.38-0.42 on FiQA

## Advanced Techniques

**Late chunking**: instead of chunking before embedding, encode the full document and then pool token embeddings per chunk. Preserves cross-chunk context.

**HyDE (Hypothetical Document Embeddings)**: generate a hypothetical answer using a language model, embed it, and retrieve documents similar to that hypothetical. Bridges the query-document gap.

**Multi-vector retrieval (ColBERT)**: represent queries and documents as sets of token-level vectors and compute MaxSim scores. More expressive than single-vector bi-encoders.

**RAPTOR**: recursively summarise document clusters into higher-level nodes, enabling retrieval at multiple granularities.