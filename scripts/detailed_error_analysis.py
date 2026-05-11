"""Quantitative error analysis: text length, source dataset, and error patterns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report


LENGTH_BINS = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 5000)]


def length_bin(text: str) -> str:
    n = len(text)
    for lo, hi in LENGTH_BINS:
        if lo <= n < hi:
            return f"{lo}-{hi}"
    return f">={LENGTH_BINS[-1][1]}"


def per_bin_metrics(df: pd.DataFrame, true_col: str, pred_col: str, group_col: str) -> dict:
    out = {}
    for g, sub in df.groupby(group_col):
        if len(sub) == 0:
            continue
        y_true = sub[true_col].to_numpy()
        y_pred = sub[pred_col].to_numpy()
        correct = (y_true == y_pred).sum()
        out[str(g)] = {
            "n": int(len(sub)),
            "accuracy": float(correct / len(sub)),
            "n_real": int((y_true == 0).sum()),
            "n_fake": int((y_true == 1).sum()),
            "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
            "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        }
    return out


def confusion_breakdown(df: pd.DataFrame) -> dict:
    y_true = df["true_label"].to_numpy()
    y_pred = df["predicted_label"].to_numpy()
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    return {
        "total": int(len(df)),
        "correct": int((y_true == y_pred).sum()),
        "false_positives": {
            "count": int(fp_mask.sum()),
            "mean_length": float(df.loc[fp_mask, "text"].str.len().mean()) if fp_mask.any() else 0.0,
            "by_source": df.loc[fp_mask, "source_dataset"].value_counts().to_dict(),
        },
        "false_negatives": {
            "count": int(fn_mask.sum()),
            "mean_length": float(df.loc[fn_mask, "text"].str.len().mean()) if fn_mask.any() else 0.0,
            "by_source": df.loc[fn_mask, "source_dataset"].value_counts().to_dict(),
        },
    }


def analyze_predictions(pred_csv: Path, test_csv: Path, out_dir: Path, model_name: str) -> dict:
    pred_df = pd.read_csv(pred_csv)
    test_df = pd.read_csv(test_csv)

    pred_col = "predicted_label" if "predicted_label" in pred_df.columns else "pred"
    if "true_label" not in pred_df.columns:
        pred_df["true_label"] = test_df["label"].to_numpy()
    pred_df["predicted_label"] = pred_df[pred_col]
    if "text" not in pred_df.columns:
        pred_df["text"] = test_df["text"].to_numpy()
    if "source_dataset" not in pred_df.columns:
        pred_df["source_dataset"] = test_df["source_dataset"].to_numpy()
    pred_df["length_bin"] = pred_df["text"].astype(str).map(length_bin)

    out_dir.mkdir(parents=True, exist_ok=True)

    fp = pred_df[(pred_df["true_label"] == 0) & (pred_df["predicted_label"] == 1)]
    fn = pred_df[(pred_df["true_label"] == 1) & (pred_df["predicted_label"] == 0)]
    fp.to_csv(out_dir / f"false_positives_{model_name}.csv", index=False)
    fn.to_csv(out_dir / f"false_negatives_{model_name}.csv", index=False)

    return {
        "model": model_name,
        "overall": confusion_breakdown(pred_df),
        "by_length_bin": per_bin_metrics(pred_df, "true_label", "predicted_label", "length_bin"),
        "by_source": per_bin_metrics(pred_df, "true_label", "predicted_label", "source_dataset"),
        "classification_report": classification_report(
            pred_df["true_label"], pred_df["predicted_label"], target_names=["real", "fake"], output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/error_analysis_detailed.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    models = {
        "baseline_tfidf": Path("reports/predictions_baseline.csv"),
        "rnn": Path("reports/predictions_rnn.csv"),
        "lstm": Path("reports/predictions_lstm.csv"),
        "transformer": Path("reports/predictions_transformer.csv"),
    }

    results = {}
    for name, path in models.items():
        if not path.exists():
            print(f"[skip] {name}: no predictions at {path}")
            continue
        print(f"[analyze] {name}")
        results[name] = analyze_predictions(path, args.test, args.out_dir, name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {args.out}")

    print("\n=== Accuracy by text-length bin ===")
    print(f"{'Bin':<12}", *[f"{m:>12}" for m in results.keys()])
    def _bin_key(s: str) -> int:
        s = s.replace(">=", "")
        return int(s.split("-")[0])
    bins = sorted({b for r in results.values() for b in r["by_length_bin"]}, key=_bin_key)
    for b in bins:
        cells = []
        for m, r in results.items():
            row = r["by_length_bin"].get(b)
            cells.append(f"{row['accuracy']:.3f} (n={row['n']})" if row else "n/a")
        print(f"{b:<12}", *[f"{c:>12}" for c in cells])

    print("\n=== Accuracy by source dataset ===")
    sources = sorted({s for r in results.values() for s in r["by_source"]})
    print(f"{'Source':<12}", *[f"{m:>12}" for m in results.keys()])
    for s in sources:
        cells = []
        for m, r in results.items():
            row = r["by_source"].get(s)
            cells.append(f"{row['accuracy']:.3f} (n={row['n']})" if row else "n/a")
        print(f"{s:<12}", *[f"{c:>12}" for c in cells])


if __name__ == "__main__":
    main()
