from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _metric_tuple(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n": int(len(y_true)),
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, precs, recs, f1s = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        accs.append(accuracy_score(yt, yp))
        precs.append(precision_score(yt, yp, zero_division=0))
        recs.append(recall_score(yt, yp, zero_division=0))
        f1s.append(f1_score(yt, yp, zero_division=0))

    def _ci(xs):
        a = np.asarray(xs)
        return {
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)),
            "ci_low": float(np.percentile(a, 2.5)),
            "ci_high": float(np.percentile(a, 97.5)),
        }

    return {
        "n_bootstrap": n_boot,
        "accuracy": _ci(accs),
        "precision": _ci(precs),
        "recall": _ci(recs),
        "f1": _ci(f1s),
    }


def evaluate_predictions(pred_csv: Path, test_csv: Path) -> dict:
    pred_df = pd.read_csv(pred_csv)
    test_df = pd.read_csv(test_csv)
    if len(pred_df) != len(test_df):
        raise ValueError(f"Row mismatch: pred={len(pred_df)} test={len(test_df)} for {pred_csv}")

    pred_col = "predicted_label" if "predicted_label" in pred_df.columns else "pred"
    true_col = "true_label" if "true_label" in pred_df.columns else None

    y_pred = pred_df[pred_col].to_numpy()
    y_true = pred_df[true_col].to_numpy() if true_col else test_df["label"].to_numpy()
    source = test_df["source_dataset"].to_numpy()

    overall = _metric_tuple(y_true, y_pred)
    overall["bootstrap"] = bootstrap_ci(y_true, y_pred)

    per_source = {}
    for src in np.unique(source):
        mask = source == src
        yt, yp = y_true[mask], y_pred[mask]
        if len(yt) == 0:
            continue
        per_source[str(src)] = _metric_tuple(yt, yp)
        if len(yt) >= 25:
            per_source[str(src)]["bootstrap"] = bootstrap_ci(yt, yp)

    return {"overall": overall, "per_source": per_source}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, default=Path("data/processed/test.csv"))
    p.add_argument("--out", type=Path, default=Path("reports/full_evaluation.json"))
    args = p.parse_args()

    models = {
        "baseline_tfidf": Path("reports/predictions_baseline.csv"),
        "rnn": Path("reports/predictions_rnn.csv"),
        "lstm": Path("reports/predictions_lstm.csv"),
        "transformer": Path("reports/predictions_transformer.csv"),
    }

    results = {}
    test_df = pd.read_csv(args.test)
    majority = max(test_df["label"].value_counts().to_dict().values()) / len(test_df)

    for name, path in models.items():
        if not path.exists():
            print(f"[skip] missing predictions: {path}")
            continue
        print(f"[eval] {name} from {path}")
        results[name] = evaluate_predictions(path, args.test)

    dataset_summary = {
        "test_n": int(len(test_df)),
        "test_label_counts": test_df["label"].value_counts().to_dict(),
        "test_source_counts": test_df["source_dataset"].value_counts().to_dict(),
        "majority_class_accuracy": float(majority),
    }

    output = {"dataset": dataset_summary, "models": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {args.out}")

    print("\n=== Overall test metrics ===")
    print(f"{'Model':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'F1 95%CI':>22}")
    print("-" * 70)
    for name, r in results.items():
        o = r["overall"]
        ci = o.get("bootstrap", {}).get("f1", {})
        ci_str = f"[{ci.get('ci_low', 0):.3f},{ci.get('ci_high', 0):.3f}]" if ci else "n/a"
        print(f"{name:<20} {o['accuracy']:>6.3f} {o['precision']:>6.3f} {o['recall']:>6.3f} {o['f1']:>6.3f} {ci_str:>22}")
    print(f"\nMajority-class baseline accuracy: {majority:.3f}")

    print("\n=== Per-dataset breakdown ===")
    for name, r in results.items():
        print(f"\n{name}:")
        for src, m in r["per_source"].items():
            print(f"  {src:<15} n={m['n']:>4}  acc={m['accuracy']:.3f}  f1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
