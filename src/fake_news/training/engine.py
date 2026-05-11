from __future__ import annotations



from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def _compute_metrics(labels: list[int], preds: list[int], avg: str = "binary") -> tuple[float, float, float, float]:
    return (
        accuracy_score(labels, preds),
        precision_score(labels, preds, average=avg, zero_division=0),
        recall_score(labels, preds, average=avg, zero_division=0),
        f1_score(labels, preds, average=avg, zero_division=0),
    )


def train_epoch(model, loader, optimizer, criterion, device, grad_clip: float = 1.0) -> EpochMetrics:
    model.train()
    all_preds: list[int] = []
    all_labels: list[int] = []
    running_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

    acc, prec, rec, f1 = _compute_metrics(all_labels, all_preds)
    return EpochMetrics(loss=running_loss / len(loader.dataset), accuracy=acc, precision=prec, recall=rec, f1=f1)


def evaluate_epoch(model, loader, criterion, device) -> EpochMetrics:
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    acc, prec, rec, f1 = _compute_metrics(all_labels, all_preds)
    return EpochMetrics(loss=running_loss / len(loader.dataset), accuracy=acc, precision=prec, recall=rec, f1=f1)

