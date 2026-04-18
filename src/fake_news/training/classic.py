from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from fake_news.config import TrainingConfig
from fake_news.data.dataset import NewsDataset, build_vocab
from fake_news.models.lstm import LSTMClassifier
from fake_news.models.rnn import RNNClassifier
from fake_news.training.engine import evaluate_epoch, train_epoch
from fake_news.utils import save_json, set_seed


def _build_model(model_type: str, cfg: TrainingConfig, vocab_size: int, pad_id: int):
    if model_type == "rnn":
        return RNNClassifier(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_id=pad_id,
        )
    if model_type == "lstm":
        return LSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_id=pad_id,
        )
    raise ValueError("model_type must be one of: rnn, lstm")


def train_classic_model(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: TrainingConfig,
    output_dir: Path,
) -> dict:
    set_seed(cfg.seed)

    vocab = build_vocab(train_df["text"].tolist())
    train_ds = NewsDataset(train_df["text"].tolist(), train_df["label"].tolist(), vocab, cfg.max_len)
    val_ds = NewsDataset(val_df["text"].tolist(), val_df["label"].tolist(), vocab, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_type, cfg, len(vocab.itos), vocab.pad_id).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    history: list[dict] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train": train_metrics.__dict__,
            "val": val_metrics.__dict__,
        }
        history.append(record)

        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab.__dict__,
                    "config": cfg.__dict__,
                    "model_type": model_type,
                },
                output_dir / f"best_{model_type}.pt",
            )

    save_json({"history": history, "best_val_f1": best_val_f1}, output_dir / f"metrics_{model_type}.json")
    return {"best_val_f1": best_val_f1, "history": history}
