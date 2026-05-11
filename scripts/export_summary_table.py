"""Export a flat CSV summary of the final test metrics for the report."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    eval_path = Path("reports/full_evaluation.json")
    out_path = Path("reports/summary_table.csv")
    data = json.loads(eval_path.read_text(encoding="utf-8"))

    rows = []
    for name, r in data["models"].items():
        o = r["overall"]
        b = o.get("bootstrap", {})
        row = {
            "model": name,
            "test_n": o["n"],
            "accuracy": round(o["accuracy"], 4),
            "precision": round(o["precision"], 4),
            "recall": round(o["recall"], 4),
            "f1": round(o["f1"], 4),
            "f1_ci_low": round(b.get("f1", {}).get("ci_low", 0), 4),
            "f1_ci_high": round(b.get("f1", {}).get("ci_high", 0), 4),
            "tp": o["confusion_matrix"][1][1],
            "fn": o["confusion_matrix"][1][0],
            "fp": o["confusion_matrix"][0][1],
            "tn": o["confusion_matrix"][0][0],
        }
        for src in ("liar", "fakenewsnet"):
            m = r["per_source"].get(src, {})
            row[f"{src}_acc"] = round(m.get("accuracy", 0), 4)
            row[f"{src}_f1"] = round(m.get("f1", 0), 4)
            row[f"{src}_n"] = m.get("n", 0)
        rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")
    print(f"Majority class accuracy: {data['dataset']['majority_class_accuracy']:.4f}")


if __name__ == "__main__":
    main()
