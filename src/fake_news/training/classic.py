from __future__ import annotations



from pathlib import Path

import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from fake_news.config import TrainingConfig
from fake_news.data.dataset import NewsDataset, build_vocab
from fake_news.models.lstm import LSTMClassifier
from fake_news.models.rnn import RNNClassifier
from fake_news.training.engine import evaluate_epoch, train_epoch
from fake_news.utils import save_json, set_seed
from fake_news.embeddings import load_glove_embeddings


def _build_model(model_type: str, cfg: TrainingConfig, vocab_size: int, pad_id: int, embed_weights=None):
    """Build RNN or LSTM model.
    
    Args:
        model_type: 'rnn' or 'lstm'
        cfg: TrainingConfig
        vocab_size: Size of vocabulary
        pad_id: Index of padding token
        embed_weights: Pre-trained embedding weights (optional)
    """
    if model_type == "rnn":
        model = RNNClassifier(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_id=pad_id,
        )
    elif model_type == "lstm":
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_id=pad_id,
        )
    else:
        raise ValueError("model_type must be one of: rnn, lstm")

    # Load pre-trained embeddings if provided
    if embed_weights is not None:
        model.embedding.weight.data.copy_(torch.from_numpy(embed_weights))
        model.embedding.weight.data[pad_id] = 0  # Ensure padding is zero
        print("✅ Loaded pre-trained embeddings")

    return model


def train_classic_model(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: TrainingConfig,
    output_dir: Path,
    use_glove: bool = True,
) -> dict:
    """Train RNN or LSTM model with optional GloVe embeddings.
    
    Args:
        model_type: 'rnn' or 'lstm'
        train_df: Training dataframe
        val_df: Validation dataframe
        cfg: TrainingConfig
        output_dir: Output directory for checkpoints
        use_glove: Whether to use GloVe pre-trained embeddings
    """
    set_seed(cfg.seed)

    vocab = build_vocab(train_df["text"].tolist())
    train_ds = NewsDataset(train_df["text"].tolist(), train_df["label"].tolist(), vocab, cfg.max_len)
    val_ds = NewsDataset(val_df["text"].tolist(), val_df["label"].tolist(), vocab, cfg.max_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to load GloVe embeddings
    embed_weights = None
    if use_glove:
        try:
            glove_path = Path("data/raw/glove_embeddings/glove.6B.100d.txt")
            if not glove_path.exists():
                print("📥 GloVe not found, downloading...")
                from fake_news.embeddings import download_glove
                glove_path = download_glove()
            
            print(f"📖 Loading GloVe embeddings...")
            embed_weights = load_glove_embeddings(glove_path, vocab, embed_dim=cfg.embed_dim)
        except Exception as e:
            print(f"⚠️  Could not load GloVe embeddings: {e}")
            print("   Falling back to random embeddings")
            use_glove = False

    # Build model with optional embeddings
    model = _build_model(model_type, cfg, len(vocab.itos), vocab.pad_id, embed_weights).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    # Compute class weights to handle imbalance
    counts = train_df["label"].value_counts().sort_index()
    total = float(counts.sum())
    weights = [total / (2.0 * float(counts.get(i, 1))) for i in range(2)]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"\n{'='*80}")
    print(f"Training {model_type.upper()} model")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Embedding dim: {cfg.embed_dim} (using {'GloVe' if use_glove else 'random'})")
    print(f"Class weights: {weights}")
    print(f"{'='*80}\n")

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

    save_json(
        {"history": history, "best_val_f1": best_val_f1, "used_glove": use_glove},
        output_dir / f"metrics_{model_type}.json"
    )
    return {"best_val_f1": best_val_f1, "history": history, "used_glove": use_glove}

