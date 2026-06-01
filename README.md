# document-intelligence-rag
Hybrid document retrieval system combining BM25, FAISS and reranking, evaluated on BEIR benchmarks.

## Instalation

```
git clone https://github.com/McQbis/document-intelligence-rag.git
cd document-intelligence-rag
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
# .venv\Scripts\activate    # Windows
pip install -e .
```

## Testing

```
pip install -e ".[dev]"
pytest
```

## Evaluation (BEIR)

```
python scripts/evaluate.py --dataset scifact --router deep
```

## Results

The retrieval pipeline was evaluated on BEIR benchmark datasets.

| Dataset | Recall@10 | nDCG@10 |
|----------|----------|----------|
| FIQA | 0.6759 | 0.3775 |
| SCIFACT | 0.9033 | 0.7390 |

👉 **Full evaluation report:** [RESULTS.md](RESULTS.md)

Includes:
- Fast vs Deep retrieval comparison
- Fine-tuning experiments
- FIQA and SCIFACT benchmarks
- Retrieval analysis and future work

## Finetuning (offline)

```bash
python scripts/finetune.py --dataset fiqa --output-dir models/finetuned
```