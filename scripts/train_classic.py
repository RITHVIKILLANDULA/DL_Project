from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fake_news.config import TrainingConfig
from fake_news.training.classic import train_classic_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rnn", "lstm"], required=True)
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    cfg = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len)

    result = train_classic_model(args.model, train_df, val_df, cfg, args.out_dir)
    print(result)


if __name__ == "__main__":
    main()
