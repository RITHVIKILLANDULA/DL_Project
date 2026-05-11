from __future__ import annotations



import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fake_news.config import TrainingConfig
from fake_news.data.dataset import NewsDataset, Vocab
from fake_news.models.lstm import LSTMClassifier
from fake_news.models.rnn import RNNClassifier


def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = Vocab(stoi=ckpt["vocab"]["stoi"], itos=ckpt["vocab"]["itos"])
    cfg = TrainingConfig(**ckpt["config"])
    model_type = ckpt["model_type"]

    if model_type == "rnn":
        model = RNNClassifier(
            vocab_size=len(vocab.itos),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_id=vocab.pad_id,
        )
    else:
        model = LSTMClassifier(
            vocab_size=len(vocab.itos),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_id=vocab.pad_id,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/predictions.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, vocab, cfg = _load_model(args.checkpoint, device)
    ds = NewsDataset(df["text"].tolist(), df["label"].tolist(), vocab, cfg.max_len)
    loader = DataLoader(ds, batch_size=64)

    preds: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())

    out_df = pd.DataFrame({"pred": preds})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out.resolve()}")


if __name__ == "__main__":
    main()

