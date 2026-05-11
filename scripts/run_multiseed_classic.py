"""3-seed RNN + LSTM runner. Trains each model with seeds {42, 1, 7},
predicts on test, and writes a separate aggregate JSON.

DOES NOT modify final_report.md, full_evaluation.json, or summary_table.csv.
Outputs all go to:
  models/classic_{rnn,lstm}_seed{42,1,7}/
  reports/predictions_{rnn,lstm}_seed{42,1,7}.csv
  reports/multiseed_aggregate.json
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fake_news.config import TrainingConfig
from fake_news.data.dataset import NewsDataset, Vocab
from fake_news.models.lstm import LSTMClassifier
from fake_news.models.rnn import RNNClassifier


SEEDS = [42, 1, 7]
MODELS = ["rnn", "lstm"]
EPOCHS = {"rnn": 10, "lstm": 12}


def train(model_type: str, seed: int) -> Path:
    suffix = f"_seed{seed}"
    out_dir = Path("models") / f"classic_{model_type}{suffix}"
    if (out_dir / f"best_{model_type}.pt").exists():
        print(f"[skip-train] {out_dir}/best_{model_type}.pt already exists")
        return out_dir / f"best_{model_type}.pt"
    cmd = [
        sys.executable,
        "scripts/train_classic.py",
        "--model-type", model_type,
        "--epochs", str(EPOCHS[model_type]),
        "--max-len", "256",
        "--learning-rate", "1e-3",
        "--seed", str(seed),
        "--out-suffix", suffix,
    ]
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, env={"PYTHONIOENCODING": "utf-8", **dict(__import__("os").environ)})
    if proc.returncode != 0:
        raise RuntimeError(f"Train failed for {model_type} seed {seed}")
    return out_dir / f"best_{model_type}.pt"


def predict(ckpt_path: Path, test_df: pd.DataFrame, model_type: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab = Vocab(stoi=ckpt["vocab"]["stoi"], itos=ckpt["vocab"]["itos"])
    cfg = TrainingConfig(**ckpt["config"])

    if model_type == "rnn":
        m = RNNClassifier(
            vocab_size=len(vocab.itos),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_id=vocab.pad_id,
        )
    else:
        m = LSTMClassifier(
            vocab_size=len(vocab.itos),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_id=vocab.pad_id,
        )
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()

    ds = NewsDataset(test_df["text"].tolist(), test_df["label"].tolist(), vocab, cfg.max_len)
    loader = DataLoader(ds, batch_size=64)
    preds = []
    with torch.no_grad():
        for batch in loader:
            logits = m(batch["input_ids"])
            preds.extend(torch.argmax(logits, dim=1).tolist())

    y_true = test_df["label"].astype(int).to_numpy()
    y_pred = np.array(preds)
    src = test_df["source_dataset"].to_numpy()

    def _m(yt, yp):
        return {
            "accuracy": float(accuracy_score(yt, yp)),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "confusion_matrix": confusion_matrix(yt, yp, labels=[0, 1]).tolist(),
            "n": int(len(yt)),
        }

    overall = _m(y_true, y_pred)
    per_source = {}
    for s in ("liar", "fakenewsnet"):
        mask = src == s
        if mask.sum() > 0:
            per_source[s] = _m(y_true[mask], y_pred[mask])

    return {"overall": overall, "per_source": per_source, "predictions": y_pred.tolist()}


def aggregate(results: dict) -> dict:
    """Compute mean ± std across seeds for each model and per-source slice."""
    agg = {}
    for model_type, by_seed in results.items():
        slices = ["overall"] + list(next(iter(by_seed.values()))["per_source"].keys())
        slice_agg = {}
        for slice_name in slices:
            metrics_per_seed = []
            for seed, r in by_seed.items():
                m = r[slice_name] if slice_name == "overall" else r["per_source"][slice_name]
                metrics_per_seed.append({k: v for k, v in m.items() if isinstance(v, (int, float))})
            keys = metrics_per_seed[0].keys()
            slice_agg[slice_name] = {
                k: {
                    "mean": float(np.mean([m[k] for m in metrics_per_seed])),
                    "std": float(np.std([m[k] for m in metrics_per_seed], ddof=1)) if len(metrics_per_seed) > 1 else 0.0,
                    "values": [m[k] for m in metrics_per_seed],
                }
                for k in keys
            }
        agg[model_type] = slice_agg
    return agg


def main() -> None:
    test_df = pd.read_csv("data/processed/test.csv")
    results: dict[str, dict[int, dict]] = {m: {} for m in MODELS}

    for model_type in MODELS:
        for seed in SEEDS:
            print(f"\n=== {model_type.upper()} seed={seed} ===")
            ckpt = train(model_type, seed)
            r = predict(ckpt, test_df, model_type)
            results[model_type][seed] = r

            pred_csv = Path(f"reports/predictions_{model_type}_seed{seed}.csv")
            out = pd.DataFrame({
                "text": test_df["text"],
                "true_label": test_df["label"],
                "predicted_label": r["predictions"],
                "source_dataset": test_df["source_dataset"],
            })
            out.to_csv(pred_csv, index=False)
            print(f"  acc={r['overall']['accuracy']:.4f} f1={r['overall']['f1']:.4f}  pred={pred_csv}")

    agg = aggregate(results)
    out_path = Path("reports/multiseed_aggregate.json")
    out_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\nSaved aggregate: {out_path}")

    print("\n=== Summary (mean ± std across 3 seeds) ===")
    for model_type, slices in agg.items():
        print(f"\n{model_type.upper()}:")
        for slice_name, metrics in slices.items():
            acc = metrics["accuracy"]
            f1 = metrics["f1"]
            print(f"  {slice_name:<15} acc={acc['mean']:.4f} ± {acc['std']:.4f}  f1={f1['mean']:.4f} ± {f1['std']:.4f}  (n={int(metrics['n']['values'][0])})")


if __name__ == "__main__":
    main()
