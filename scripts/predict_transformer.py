from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fake_news.utils import save_json


class TransformerEvalDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int) -> None:
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_transformer"))
    parser.add_argument("--input-csv", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--out-predictions", type=Path, default=Path("reports/predictions_transformer.csv"))
    parser.add_argument("--out-metrics", type=Path, default=Path("reports/metrics_test_transformer.json"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=64)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.max_samples and args.max_samples > 0:
        df = df.head(args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = TransformerEvalDataset(df["text"].tolist(), df["label"].astype(int).tolist(), tokenizer, args.max_len)
    loader = DataLoader(dataset, batch_size=8)

    predictions: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_predictions = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            predictions.extend(batch_predictions)
            labels.extend(batch["labels"].tolist())

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "classification_report": classification_report(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "count": len(labels),
    }

    args.out_predictions.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pred": predictions}).to_csv(args.out_predictions, index=False)
    save_json(metrics, args.out_metrics)
    print(metrics)


if __name__ == "__main__":
    main()
