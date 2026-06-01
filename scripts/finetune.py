import argparse
import sys
from pathlib import Path

# allow running without package install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.finetuning.trainer import BiEncoderTrainer, FinetuneConfig


def parse_args():
    p = argparse.ArgumentParser(description="Bi-encoder finetuning on BEIR")
    p.add_argument("--dataset", default="fiqa")
    p.add_argument("--split", default="train")
    p.add_argument("--base-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--output-dir", default="models/finetuned")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--negatives", type=int, default=5, help="Hard negatives per positive")
    p.add_argument("--data-dir", default="beir-data")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # CLI -> training config mapping
    cfg = FinetuneConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        dataset=args.dataset,
        data_dir=args.data_dir,
        split=args.split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        negatives_per_positive=args.negatives,
        seed=args.seed,
    )

    print(f"[finetune] Config: {cfg}")
    trainer = BiEncoderTrainer(cfg)
    output_path = trainer.run()
    print(f"\n[finetune] Done! Model saved to: {output_path}")
    print(f"[finetune] Evaluate with:")
    print(f"  python scripts/evaluate.py --dataset {args.dataset} --model {output_path}")


if __name__ == "__main__":
    main()