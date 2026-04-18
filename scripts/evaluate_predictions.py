from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fake_news.analysis.error_analysis import collect_error_slices
from fake_news.evaluation import evaluate_predictions
from fake_news.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(args.test)
    pred_df = pd.read_csv(args.predictions)

    if "pred" not in pred_df.columns:
        raise ValueError("Predictions CSV must contain a 'pred' column.")

    y_true = test_df["label"].astype(int).tolist()
    y_pred = pred_df["pred"].astype(int).tolist()

    metrics = evaluate_predictions(test_df, y_true, y_pred)
    save_json(metrics, args.out_dir / "metrics_test.json")

    errors = collect_error_slices(test_df, y_true, y_pred)
    errors.false_positives.to_csv(args.out_dir / "false_positives.csv", index=False)
    errors.false_negatives.to_csv(args.out_dir / "false_negatives.csv", index=False)

    print("Saved metrics and error slices in reports/")


if __name__ == "__main__":
    main()
