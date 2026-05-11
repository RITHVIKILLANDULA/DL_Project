#!/usr/bin/env python
"""Train TF-IDF + Logistic Regression baseline model for fake news detection."""

import argparse
from pathlib import Path

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json

def compute_metrics(y_true, y_pred):
    """Compute all metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression baseline")
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.csv"))
    parser.add_argument("--val", type=Path, default=Path("data/processed/val.csv"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.csv"))
    parser.add_argument("--max-features", type=int, default=5000, help="Max features for TF-IDF")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--out-pred", type=Path, default=Path("reports/predictions_baseline.csv"))
    args = parser.parse_args()

    print("\n" + "="*80)
    print("TF-IDF + LOGISTIC REGRESSION BASELINE")
    print("="*80)

    # Load data
    print(f"\n📂 Loading data...")
    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.val)
    test_df = pd.read_csv(args.test)

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    # Combine train + val for vectorizer training (common practice)
    combined_texts = list(train_df["text"]) + list(val_df["text"])
    combined_labels = list(train_df["label"]) + list(val_df["label"])

    # Create vectorizer
    print(f"\n🔤 Creating TF-IDF vectorizer (max_features={args.max_features})...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        min_df=2,
        max_df=0.95,
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
    )

    # Transform data
    print(f"📊 Transforming texts to TF-IDF vectors...")
    X_combined = vectorizer.fit_transform(combined_texts)
    X_test = vectorizer.transform(test_df["text"])

    print(f"  Feature dimension: {X_combined.shape[1]}")
    print(f"  Sparsity: {(X_combined == 0).sum() / (X_combined.shape[0] * X_combined.shape[1]) * 100:.1f}%")

    # Train Logistic Regression
    print(f"\n🧠 Training Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # Handle class imbalance
        random_state=42,
        verbose=1,
    )
    model.fit(X_combined, combined_labels)

    # Evaluate on test set
    print(f"\n📈 Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    metrics = compute_metrics(test_df["label"], y_pred)

    print(f"\n  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

    # Save model and vectorizer
    print(f"\n💾 Saving artifacts...")
    model_dir = args.out_dir / "baseline_tfidf"
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    args.out_pred.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame({
        "text": test_df["text"],
        "true_label": test_df["label"],
        "predicted_label": y_pred,
        "prob_real": y_pred_proba[:, 0],
        "prob_fake": y_pred_proba[:, 1],
    })
    results_df.to_csv(args.out_pred, index=False)

    print(f"  ✅ Model saved to: {model_dir}")
    print(f"  ✅ Predictions saved to: {args.out_pred}")

    print("\n" + "="*80)
    print("BASELINE COMPLETE - This is your comparison point!")
    print("="*80 + "\n")

    return metrics


if __name__ == "__main__":
    main()
