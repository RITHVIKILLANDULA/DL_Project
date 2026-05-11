from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd

from fake_news.config import TrainingConfig
from fake_news.training.classic import train_classic_model
from fake_news.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument("--model-type", type=str, default="lstm", choices=["lstm", "rnn"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-suffix", type=str, default="", help="Suffix appended to out-dir, e.g. '_seed1'")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)

    if args.max_train_samples > 0:
        train_df = train_df.sample(n=min(len(train_df), args.max_train_samples), random_state=42)
    if args.max_val_samples > 0:
        val_df = val_df.sample(n=min(len(val_df), args.max_val_samples), random_state=42)

    cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_len=args.max_len,
        seed=args.seed,
    )

    out_dir = args.out_dir / f"classic_{args.model_type}{args.out_suffix}"
    metrics = train_classic_model(
        model_type=args.model_type,
        train_df=train_df,
        val_df=val_df,
        cfg=cfg,
        output_dir=out_dir,
    )
    save_json(metrics, out_dir / f"metrics_{args.model_type}.json")
    print(metrics)


if __name__ == "__main__":
    main()
