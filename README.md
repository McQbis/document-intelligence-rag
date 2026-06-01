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