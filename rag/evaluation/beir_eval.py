from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tqdm import tqdm


@dataclass
class EvalResults:
    """BEIR evaluation summary for a single dataset/split."""

    dataset: str
    split: str
    top_k: int
    queries_evaluated: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float = 0.0
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"\n{'='*40}",
            f"BEIR Evaluation — {self.dataset} ({self.split})",
            f"{'='*40}",
            f"Queries evaluated : {self.queries_evaluated}",
            f"Recall@1          : {self.recall_at_1:.4f}",
            f"Recall@5          : {self.recall_at_5:.4f}",
            f"Recall@10         : {self.recall_at_10:.4f}",
            f"nDCG@10           : {self.ndcg_at_10:.4f}",
            f"{'='*40}",
        ]
        return "\n".join(lines)


class BEIREvaluator:
    BEIR_BASE_URL = (
        "https://public.ukp.informatik.tu-darmstadt.de/"
        "thakur/BEIR/datasets/{dataset}.zip"
    )

    def __init__(
        self,
        retriever,  # HybridRetriever or QueryRouter
        data_dir: str = "beir-data",
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
    ):
        self.retriever = retriever
        self.data_dir = data_dir
        self.query_prefix = query_prefix


    def run(
        self,
        dataset: str,
        split: str = "test",
        top_k: int = 10,
        max_queries: Optional[int] = None,
    ) -> EvalResults:
        """Download dataset (if needed), build index, evaluate."""
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        print(f"[beir] Downloading {dataset}…")
        url = self.BEIR_BASE_URL.format(dataset=dataset)
        data_path = util.download_and_unzip(url, self.data_dir)

        print(f"[beir] Loading {dataset}/{split}")
        corpus, queries, qrels = GenericDataLoader(
            data_folder=os.path.join(data_path)
        ).load(split=split)
        print(f"[beir] corpus={len(corpus):,}  queries={len(queries):,}")

        chunks, doc_id_mapping = self._corpus_to_chunks(corpus)
        chunk_to_doc: Dict[int, str] = {
            id(c): doc_id_mapping[i] for i, c in enumerate(chunks)
        }

        print("[beir] Building index…")
        self.retriever.build_index(chunks)
        print("[beir] Index ready.")

        hits_1 = hits_5 = hits_10 = 0
        ndcg_sum = 0.0
        evaluated = 0

        query_items = list(queries.items())
        if max_queries:
            query_items = query_items[:max_queries]

        for query_id, query_text in tqdm(query_items, desc="eval"):
            relevant = set(qrels[query_id].keys())
            if not relevant:
                continue
            evaluated += 1

            # instruction-style prefix improves embedding models (e.g., BGE)
            prefixed = self.query_prefix + query_text
            results = self._search(prefixed, top_k=top_k)
            retrieved = [chunk_to_doc[id(chunk)] for chunk, _ in results]

            if any(d in relevant for d in retrieved[:1]):
                hits_1 += 1
            if any(d in relevant for d in retrieved[:5]):
                hits_5 += 1
            if any(d in relevant for d in retrieved[:10]):
                hits_10 += 1

            ndcg_sum += self._ndcg_at_k(retrieved, relevant, k=10)

        return EvalResults(
            dataset=dataset,
            split=split,
            top_k=top_k,
            queries_evaluated=evaluated,
            recall_at_1=hits_1 / evaluated,
            recall_at_5=hits_5 / evaluated,
            recall_at_10=hits_10 / evaluated,
            ndcg_at_10=ndcg_sum / evaluated,
        )


    def _search(self, query: str, top_k: int):
        if hasattr(self.retriever, "search"):
            return self.retriever.search(query, top_k=top_k)
        raise TypeError(f"Unsupported retriever type: {type(self.retriever)}")

    @staticmethod
    def _corpus_to_chunks(corpus: dict):
        """Convert BEIR corpus into unified chunk format."""

        class _Chunk:
            def __init__(self, text: str):
                self.text = text
                self.page = 0
                self.source = ""
                self.file_type = "beir"
                self.chunk_index = 0
                self.metadata: dict = {}

        chunks = []
        doc_ids = []
        for doc_id, doc in corpus.items():
            parts = []
            if doc.get("title"):
                parts.append(doc["title"])
            if doc.get("text"):
                parts.append(doc["text"])
            chunks.append(_Chunk(text="\n".join(parts)))
            doc_ids.append(doc_id)

        return chunks, doc_ids

    @staticmethod
    def _ndcg_at_k(retrieved: List[str], relevant: set, k: int) -> float:
        import math

        dcg = sum(
            1.0 / math.log2(i + 2)
            for i, doc_id in enumerate(retrieved[:k])
            if doc_id in relevant
        )
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0