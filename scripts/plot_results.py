"""Generate result figures for the report.

Outputs (PNG, no display backend):
  reports/fig_metrics_with_ci.png   - bar chart of F1 with 95% CIs
  reports/fig_per_dataset.png       - per-dataset accuracy breakdown
  reports/fig_length_bins.png       - accuracy by text-length bin
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fig_metrics_with_ci(full_eval: dict, out: Path) -> None:
    models = list(full_eval["models"].keys())
    f1_means = []
    f1_low = []
    f1_high = []
    for m in models:
        o = full_eval["models"][m]["overall"]
        f1_means.append(o["f1"])
        b = o.get("bootstrap", {}).get("f1", {})
        f1_low.append(b.get("ci_low", o["f1"]))
        f1_high.append(b.get("ci_high", o["f1"]))

    f1_means = np.array(f1_means)
    err_low = f1_means - np.array(f1_low)
    err_high = np.array(f1_high) - f1_means

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    bars = ax.bar(x, f1_means, color=["#94a3b8", "#fb923c", "#3b82f6", "#10b981"][: len(models)])
    ax.errorbar(x, f1_means, yerr=[err_low, err_high], fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("F1 (fake class)")
    ax.set_title("Test F1 with bootstrap 95% CIs (n=1000)")
    ax.axhline(full_eval["dataset"]["majority_class_accuracy"], color="red", linestyle="--", linewidth=0.8, label=f"majority-class acc = {full_eval['dataset']['majority_class_accuracy']:.3f}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0, max(0.85, max(f1_high) + 0.05))
    for bar, m in zip(bars, f1_means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005, f"{m:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_per_dataset(full_eval: dict, out: Path) -> None:
    models = list(full_eval["models"].keys())
    sources = sorted({s for m in models for s in full_eval["models"][m]["per_source"].keys()})

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(1, len(sources))
    x = np.arange(len(models))
    for i, src in enumerate(sources):
        vals = []
        ns = []
        for m in models:
            r = full_eval["models"][m]["per_source"].get(src)
            vals.append(r["accuracy"] if r else 0.0)
            ns.append(r["n"] if r else 0)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=f"{src} (n≈{ns[0]})")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-dataset test accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_length_bins(detailed: dict, out: Path) -> None:
    models = list(detailed.keys())
    bins = sorted({b for m in models for b in detailed[m]["by_length_bin"].keys()}, key=lambda s: int(s.replace(">=", "").split("-")[0]))

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / max(1, len(models))
    x = np.arange(len(bins))
    for i, m in enumerate(models):
        vals = [detailed[m]["by_length_bin"].get(b, {}).get("accuracy", 0.0) for b in bins]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=20)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by text-length bin")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    full_eval = _load(Path("reports/full_evaluation.json"))
    detailed = _load(Path("reports/error_analysis_detailed.json"))
    Path("reports").mkdir(exist_ok=True)
    fig_metrics_with_ci(full_eval, Path("reports/fig_metrics_with_ci.png"))
    fig_per_dataset(full_eval, Path("reports/fig_per_dataset.png"))
    fig_length_bins(detailed, Path("reports/fig_length_bins.png"))
    print("Saved figures: fig_metrics_with_ci.png, fig_per_dataset.png, fig_length_bins.png")


if __name__ == "__main__":
    main()
