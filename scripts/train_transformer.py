from __future__ import annotations



import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from fake_news.training.transformer_train import train_transformer
from fake_news.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)

    metrics = train_transformer(
        train_df=train_df,
        val_df=val_df,
        output_dir=args.out_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_len=args.max_len,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        use_class_weights=not args.no_class_weights,
        patience=args.patience,
    )
    save_json(metrics, args.out_dir / "metrics_transformer.json")
    print(metrics)


if __name__ == "__main__":
    main()

