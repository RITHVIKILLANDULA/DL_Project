from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fake_news.utils import save_json, set_seed


@dataclass
class TransformerBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class TransformerTextDataset(Dataset):
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


def _compute_metrics(labels: list[int], preds: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def _train_epoch(model, loader, optimizer, device) -> dict[str, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss if outputs.loss is not None else loss_fn(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(outputs.logits, dim=1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    metrics = _compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def _evaluate_epoch(model, loader, device) -> dict[str, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels: list[int] = []
    all_preds: list[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss if outputs.loss is not None else loss_fn(outputs.logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(outputs.logits, dim=1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    metrics = _compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def train_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_len: int = 128,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
) -> dict:
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_subset = train_df.reset_index(drop=True)
    val_subset = val_df.reset_index(drop=True)
    if max_train_samples and max_train_samples > 0:
        train_subset = train_subset.head(max_train_samples).reset_index(drop=True)
    if max_val_samples and max_val_samples > 0:
        val_subset = val_subset.head(max_val_samples).reset_index(drop=True)

    train_ds = TransformerTextDataset(train_subset["text"].tolist(), train_subset["label"].astype(int).tolist(), tokenizer, max_len)
    val_ds = TransformerTextDataset(val_subset["text"].tolist(), val_subset["label"].astype(int).tolist(), tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_val_f1 = -1.0
    history: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = _train_epoch(model, train_loader, optimizer, device)
        val_metrics = _evaluate_epoch(model, val_loader, device)

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_path = output_dir / "best_transformer"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    metrics = {"history": history, "best_val_f1": best_val_f1, "train_subset_size": len(train_subset), "val_subset_size": len(val_subset)}
    save_json(metrics, output_dir / "metrics_transformer.json")
    return metrics
