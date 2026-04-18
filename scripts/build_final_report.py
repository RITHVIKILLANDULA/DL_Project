from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news.analysis.error_analysis import collect_error_slices
from fake_news.utils import save_json


def _truncate_text(value: str, limit: int = 180) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _format_confusion_matrix(matrix: list[list[int]] | None) -> str:
    if not matrix or len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        return "n/a"
    return f"[[{matrix[0][0]}, {matrix[0][1]}], [{matrix[1][0]}, {matrix[1][1]}]]"


def _load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_predictions(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "pred" not in df.columns:
        raise ValueError(f"Predictions file missing 'pred' column: {path}")
    return df["pred"].astype(int)


def _model_rows(metrics_by_model: dict[str, dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, metrics in metrics_by_model.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "count": metrics.get("count"),
                "confusion_matrix": _format_confusion_matrix(metrics.get("confusion_matrix")),
            }
        )
    return rows


def _example_lines(df: pd.DataFrame, limit: int) -> list[str]:
    lines: list[str] = []
    for _, row in df.head(limit).iterrows():
        lines.append(
            f"- true={int(row['y_true'])} pred={int(row['y_pred'])}: {_truncate_text(row['text'])}"
        )
    if not lines:
        lines.append("- none")
    return lines


def _render_markdown_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No model metrics found."

    headers = ["model", "accuracy", "precision", "recall", "f1", "count", "confusion_matrix"]
    table_rows = [[str(row.get(header, "")) for header in headers] for row in rows]
    widths = [len(header) for header in headers]
    for row in table_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in table_rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/final_report.md"))
    parser.add_argument("--out-json", type=Path, default=Path("reports/final_report.json"))
    parser.add_argument("--error-limit", type=int, default=5)
    args = parser.parse_args()

    reports_dir = args.out_md.parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(args.test_csv)
    train_df = pd.read_csv(args.train_csv) if args.train_csv.exists() else None

    metrics_paths = {
        "rnn": reports_dir / "metrics_test_rnn.json",
        "lstm": reports_dir / "metrics_test_lstm.json",
        "transformer": reports_dir / "metrics_test_transformer.json",
    }
    prediction_paths = {
        "rnn": reports_dir / "predictions_rnn.csv",
        "lstm": reports_dir / "predictions_lstm.csv",
        "transformer": reports_dir / "predictions_transformer.csv",
    }

    metrics_by_model: dict[str, dict] = {}
    error_slices: dict[str, dict[str, pd.DataFrame]] = {}

    for model_name, metrics_path in metrics_paths.items():
        metrics = _load_metrics(metrics_path)
        if metrics is None:
            continue

        metrics_by_model[model_name] = metrics

        pred_series = _load_predictions(prediction_paths[model_name])
        if pred_series is None:
            continue

        y_true = test_df["label"].astype(int).tolist()
        y_pred = pred_series.tolist()
        errors = collect_error_slices(test_df, y_true, y_pred, top_k=args.error_limit)
        error_slices[model_name] = {
            "false_positives": errors.false_positives,
            "false_negatives": errors.false_negatives,
        }
        errors.false_positives.to_csv(reports_dir / f"false_positives_{model_name}.csv", index=False)
        errors.false_negatives.to_csv(reports_dir / f"false_negatives_{model_name}.csv", index=False)

    rows = _model_rows(metrics_by_model)
    comparison_table = pd.DataFrame(rows).sort_values(by="f1", ascending=False) if rows else pd.DataFrame()

    train_label_counts = None
    test_label_counts = test_df["label"].value_counts().sort_index().to_dict()
    source_counts = None
    if train_df is not None and "source_dataset" in train_df.columns:
        source_counts = train_df["source_dataset"].value_counts().to_dict()
    if train_df is not None:
        train_label_counts = train_df["label"].value_counts().sort_index().to_dict()

    summary = {
        "dataset": {
            "train_rows": int(len(train_df)) if train_df is not None else None,
            "test_rows": int(len(test_df)),
            "train_label_counts": train_label_counts,
            "test_label_counts": test_label_counts,
            "source_counts": source_counts,
            "label_mapping": {"0": "real", "1": "fake"},
        },
        "models": rows,
        "generated_error_counts": {
            model_name: {
                "false_positives": int(len(slices["false_positives"])),
                "false_negatives": int(len(slices["false_negatives"])),
            }
            for model_name, slices in error_slices.items()
        },
    }
    save_json(summary, args.out_json)

    markdown_lines = [
        "# Final Model Comparison",
        "",
        "## Dataset Summary",
        f"- Train rows: {summary['dataset']['train_rows']}",
        f"- Test rows: {summary['dataset']['test_rows']}",
        f"- Train label counts: {summary['dataset']['train_label_counts']}",
        f"- Test label counts: {summary['dataset']['test_label_counts']}",
        f"- Source counts: {summary['dataset']['source_counts']}",
        f"- Label mapping: {summary['dataset']['label_mapping']}",
        "",
        "## Model Comparison",
        _render_markdown_table(rows),
        "",
    ]

    for model_name in comparison_table["model"].tolist() if not comparison_table.empty else []:
        metrics = metrics_by_model[model_name]
        slices = error_slices.get(model_name, {})
        fp_df = slices.get("false_positives", pd.DataFrame())
        fn_df = slices.get("false_negatives", pd.DataFrame())
        markdown_lines.extend(
            [
                f"## {model_name.upper()}",
                f"- Accuracy: {metrics.get('accuracy')}",
                f"- Precision: {metrics.get('precision')}",
                f"- Recall: {metrics.get('recall')}",
                f"- F1: {metrics.get('f1')}",
                f"- Confusion matrix: {_format_confusion_matrix(metrics.get('confusion_matrix'))}",
                f"- False positives saved: {len(fp_df)}",
                f"- False negatives saved: {len(fn_df)}",
                "",
                f"### {model_name.upper()} False Positives",
                *(_example_lines(fp_df, args.error_limit)),
                "",
                f"### {model_name.upper()} False Negatives",
                *(_example_lines(fn_df, args.error_limit)),
                "",
            ]
        )

    markdown_lines.extend(
        [
            "## Notes",
            "- The transformer baseline is trained on a small subset for local feasibility.",
            "- The project uses a binary mapping: 0 = real, 1 = fake.",
            "- FakeNewsNet social-context collection is intentionally out of scope here; the pipeline uses article text only.",
            "",
        ]
    )

    args.out_md.write_text("\n".join(markdown_lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()