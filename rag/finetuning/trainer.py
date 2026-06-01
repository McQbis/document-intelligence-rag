from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass
class FinetuneConfig:
    """Hyperparameters for bi-encoder fine-tuning."""

    base_model: str = "BAAI/bge-base-en-v1.5"
    output_dir: str = "models/finetuned"
    dataset: str = "fiqa"
    data_dir: str = "beir-data"
    split: str = "train"          # use train split for finetuning
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 256
    negatives_per_positive: int = 5
    seed: int = 42


class BiEncoderTrainer:
    def __init__(self, config: FinetuneConfig):
        self.config = config


    def run(self) -> str:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        cfg = self.config
        random.seed(cfg.seed)
        os.makedirs(cfg.output_dir, exist_ok=True)

        corpus, queries, qrels = self._load_beir(cfg.dataset, cfg.split)

        # dataset statistics (useful for debugging training scale)
        print(f"[finetune] corpus={len(corpus):,}  queries={len(queries):,}")

        examples = self._build_examples(corpus, queries, qrels, cfg.negatives_per_positive)
        print(f"[finetune] Training examples: {len(examples):,}")

        model = SentenceTransformer(cfg.base_model)
        model.max_seq_length = cfg.max_seq_length

        loader = DataLoader(examples, batch_size=cfg.batch_size, shuffle=True)
        loss = losses.MultipleNegativesRankingLoss(model)

        warmup_steps = int(len(loader) * cfg.epochs * cfg.warmup_ratio)

        model.fit(
            train_objectives=[(loader, loss)],
            epochs=cfg.epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": cfg.learning_rate},
            output_path=cfg.output_dir,
            show_progress_bar=True,
        )

        return cfg.output_dir


    def _load_beir(self, dataset: str, split: str):
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        url = (
            "https://public.ukp.informatik.tu-darmstadt.de/"
            f"thakur/BEIR/datasets/{dataset}.zip"
        )
        data_path = util.download_and_unzip(url, self.config.data_dir)
        return GenericDataLoader(data_folder=data_path).load(split=split)

    @staticmethod
    def _doc_text(doc: dict) -> str:
        parts = []
        if doc.get("title"):
            parts.append(doc["title"])
        if doc.get("text"):
            parts.append(doc["text"])
        return "\n".join(parts)

    def _build_examples(
        self,
        corpus: dict,
        queries: dict,
        qrels: dict,
        negatives_per_pos: int,
    ):
        from sentence_transformers import InputExample

        doc_ids = list(corpus.keys())
        examples = []

        for query_id, query_text in queries.items():
            rel_ids = set(qrels.get(query_id, {}).keys())
            if not rel_ids:
                continue

            for pos_id in rel_ids:
                if pos_id not in corpus:
                    continue
                pos_text = self._doc_text(corpus[pos_id])

                # naive negative sampling (replace with hard negatives for better performance)
                neg_pool = [d for d in doc_ids if d not in rel_ids]
                negatives = random.sample(neg_pool, min(negatives_per_pos, len(neg_pool)))
                neg_texts = [self._doc_text(corpus[n]) for n in negatives]

                # MultipleNegativesRankingLoss format: [query, positive, negatives...]
                examples.append(
                    InputExample(texts=[query_text, pos_text] + neg_texts)
                )

        return examples