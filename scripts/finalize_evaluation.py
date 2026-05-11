"""End-to-end finalize step:
1. Predict on test with DistilBERT.
2. Run full evaluation (incl. bootstrap CIs) over all four models.
3. Run detailed error analysis.
4. Regenerate figures.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, env={"PYTHONIOENCODING": "utf-8", **dict(__import__("os").environ)})
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    py = sys.executable

    if Path("models/best_transformer/model.safetensors").exists():
        run([py, "scripts/predict_transformer_test.py"])
    else:
        print("[WARN] models/best_transformer/model.safetensors missing — skipping transformer predictions")

    run([py, "scripts/full_evaluation.py"])
    run([py, "scripts/detailed_error_analysis.py"])
    run([py, "scripts/plot_results.py"])

    print("\nFinalize complete. See reports/full_evaluation.json, error_analysis_detailed.json, fig_*.png")


if __name__ == "__main__":
    main()
