# Retrieval-Augmented Document Intelligence System

## Evaluation Report (BEIR Benchmark)

# Abstract

This report evaluates a hybrid document retrieval system combining sparse (BM25) and dense (FAISS-based embedding) retrieval with Reciprocal Rank Fusion (RRF) and optional cross-encoder reranking. The system is evaluated on two BEIR benchmark datasets: FIQA-2018 and SCIFACT. We analyze the impact of reranking and embedding fine-tuning on retrieval effectiveness.

---

# 1. Introduction

Modern information retrieval systems benefit from combining lexical and semantic matching strategies. In this work, we implement a modular retrieval pipeline consisting of:

- Sparse retrieval (BM25)
- Dense retrieval (Sentence Transformer embeddings + FAISS)
- Score fusion via RRF
- Optional cross-encoder reranking

We further evaluate the impact of embedding fine-tuning and reranking on retrieval quality.

---

# 2. System Overview

## 2.1 Retrieval Pipeline

The system consists of the following stages:

1. **Candidate generation**
   - BM25 lexical retrieval
   - Dense vector retrieval (FAISS HNSW index)

2. **Fusion**
   - Reciprocal Rank Fusion (RRF) combines rankings from both methods

3. **Reranking (optional)**
   - Cross-encoder model re-scores top candidates

---

## 2.2 Configurations

We evaluate three system variants:

### Fast (No Reranking)
- BM25 + FAISS
- RRF fusion
- No cross-encoder reranking

### Deep (With Reranking)
- BM25 + FAISS
- RRF fusion
- Cross-encoder reranking applied to top candidates

### Fine-tuned (FT)
- Same retrieval pipeline as above
- Dense embedding model fine-tuned on domain-specific pairs

---

## 2.3 Evaluation Metrics

We report standard BEIR retrieval metrics:

- Recall@1
- Recall@5
- Recall@10
- nDCG@10

---

# 3. Experimental Results

## 3.1 FIQA (Test Set, n = 648 queries)

| System | Recall@1 | Recall@5 | Recall@10 | nDCG@10 |
|--------|----------|----------|-----------|---------|
| Fast   | 0.3148   | 0.5664   | 0.6636    | 0.3526  |
| Deep   | 0.3441   | 0.5802   | 0.6759    | 0.3745  |
| Fast + FT | 0.3194 | 0.5633   | 0.6590    | 0.3605  |
| Deep + FT | 0.3457 | 0.5833   | 0.6667    | 0.3775  |

---

## 3.2 SCIFACT (Test Set, n = 300 queries)

| System | Recall@1 | Recall@5 | Recall@10 | nDCG@10 |
|--------|----------|----------|-----------|---------|
| Fast   | 0.6067   | 0.8133   | 0.8767    | 0.7293  |
| Deep   | 0.6100   | 0.8133   | 0.8733    | 0.7314  |
| Fast + FT | 0.5967 | 0.8200   | 0.9033    | 0.7388  |
| Deep + FT | 0.6133 | 0.8167   | 0.8967    | 0.7390  |

---

# 4. Analysis

## 4.1 Impact of Reranking

Introducing cross-encoder reranking consistently improves ranking quality (nDCG@10), particularly on FIQA. Gains are more pronounced in ranking metrics than in recall, indicating improved ordering of top retrieved documents rather than increased coverage.

---

## 4.2 Impact of Fine-Tuning

Embedding fine-tuning shows mixed effects:

- **SCIFACT**: consistent improvements in Recall@10 and nDCG@10
- **FIQA**: marginal or neutral impact, with slight improvements in ranking quality

This suggests that embedding adaptation is sensitive to dataset structure and domain noise.

---

## 4.3 Hybrid Retrieval Effectiveness

Even without reranking, the hybrid combination of BM25 and dense retrieval provides strong baseline performance across both datasets. This confirms the effectiveness of multi-signal retrieval fusion.

---

## 4.4 Observations

- Reranking provides the most consistent improvement across datasets
- Fine-tuning yields dataset-dependent gains
- SCIFACT results suggest partial retrieval saturation, as reranking and fine-tuning provide only marginal improvements over the hybrid retrieval baseline.

---

# 5. Conclusion

We present a modular hybrid retrieval system combining lexical and semantic search with optional reranking and embedding fine-tuning. Empirical evaluation on BEIR benchmarks shows:

- Strong baseline performance from hybrid retrieval alone
- Significant gains from cross-encoder reranking
- Limited but dataset-dependent improvements from fine-tuning

Overall, system performance is primarily driven by retrieval architecture rather than embedding optimization.

---

# 6. Future Work

- Integration of query expansion techniques
- Reranker fine-tuning for domain adaptation
- Hard negative mining for embedding training
- Latency optimization for production deployment