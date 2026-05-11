"""Generate test predictions for the trained DistilBERT model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fake_news.training.transformer_train import TransformerTextDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_transformer"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/predictions_transformer.csv"))
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    test_df = pd.read_csv(args.test)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    ds = TransformerTextDataset(
        test_df["text"].astype(str).tolist(),
        test_df["label"].astype(int).tolist(),
        tokenizer,
        max_len=args.max_len,
    )
    loader = DataLoader(ds, batch_size=args.batch_size)

    preds, probs = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            p = torch.softmax(logits, dim=1)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            probs.extend(p[:, 1].cpu().tolist())

    out = pd.DataFrame(
        {
            "text": test_df["text"],
            "true_label": test_df["label"],
            "predicted_label": preds,
            "prob_fake": probs,
            "source_dataset": test_df["source_dataset"],
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Saved {args.out}: {len(out)} rows")


if __name__ == "__main__":
    main()
