import argparse
import sys
from pathlib import Path

# allow running script without installing package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.retrieval.embeddings import EmbeddingModel
from rag.retrieval.retriever import HybridRetriever
from rag.routing.router import QueryRouter, RouteMode
from rag.evaluation.beir_eval import BEIREvaluator


def parse_args():
    p = argparse.ArgumentParser(description="BEIR offline evaluation")
    p.add_argument("--dataset", default="fiqa", help="BEIR dataset name")
    p.add_argument("--split", default="test")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--candidate-k", type=int, default=30)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--model", default="BAAI/bge-base-en-v1.5", help="Embedding model or local path")
    p.add_argument("--reranker", default="BAAI/bge-reranker-base")
    p.add_argument("--data-dir", default="beir-data")
    p.add_argument(
        "--router",
        default=None,
        choices=["auto", "fast", "deep"],
        help="Użyj QueryRouter zamiast HybridRetriever. "
             "auto=heurystyka, fast=bez rerankera, deep=zawsze reranker",
    )
    return p.parse_args()


def build_retriever(args):
    emb = EmbeddingModel(model_name=args.model)
    retriever = HybridRetriever(emb, reranker_model=args.reranker)

    if args.router is None:
        print(f"[eval] Retriever   : HybridRetriever (rerank=True)")
        return retriever

    mode_map = {"auto": RouteMode.AUTO, "fast": RouteMode.FAST, "deep": RouteMode.DEEP}
    forced_mode = mode_map[args.router]
    router = QueryRouter(
        retriever,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )
    print(f"[eval] Retriever   : QueryRouter (mode={args.router})")

    class _ForcedRouter:
        # Adapter forcing a fixed routing strategy for evaluation
        def __init__(self, router, mode):
            self._router = router
            self._mode = mode

        def build_index(self, chunks):
            self._router.retriever.build_index(chunks)

        def search(self, query, top_k=10, candidate_k=30):
            return self._router.search(query, mode=self._mode, top_k=top_k, candidate_k=candidate_k)

    return _ForcedRouter(router, forced_mode)


def main():
    args = parse_args()

    print(f"[eval] Model       : {args.model}")
    print(f"[eval] Reranker    : {args.reranker}")
    print(f"[eval] Dataset     : {args.dataset}/{args.split}")
    print(f"[eval] Top-K       : {args.top_k}")
    print(f"[eval] Candidate-K : {args.candidate_k}")

    retriever = build_retriever(args)

    evaluator = BEIREvaluator(retriever, data_dir=args.data_dir)
    results = evaluator.run(
        dataset=args.dataset,
        split=args.split,
        top_k=args.top_k,
        max_queries=args.max_queries,
    )

    print(results)


if __name__ == "__main__":
    main()