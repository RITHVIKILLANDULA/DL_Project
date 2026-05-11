#!/usr/bin/env python
"""Ablation study: Compare RNN/LSTM with different embedding dimensions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json

from fake_news.config import TrainingConfig
from fake_news.data.dataset import NewsDataset, build_vocab
from fake_news.models.lstm import LSTMClassifier
from fake_news.training.engine import evaluate_epoch, train_epoch
from fake_news.utils import set_seed, save_json


def run_ablation(embedding_dims=[64, 100, 128, 200, 256]):
    """Run ablation study with different embedding dimensions."""
    print("\n" + "="*80)
    print("ABLATION STUDY: Impact of Embedding Dimension")
    print("="*80)
    
    # Load data
    print(f"\n📂 Loading data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    vocab = build_vocab(train_df["text"].tolist())
    
    results = {}
    
    for embed_dim in embedding_dims:
        print(f"\n{'='*80}")
        print(f"Testing embedding_dim = {embed_dim}")
        print(f"{'='*80}")
        
        set_seed(42)
        
        # Create datasets
        train_ds = NewsDataset(train_df["text"].tolist(), train_df["label"].tolist(), vocab, max_len=128)
        val_ds = NewsDataset(val_df["text"].tolist(), val_df["label"].tolist(), vocab, max_len=128)
        test_ds = NewsDataset(test_df["text"].tolist(), test_df["label"].tolist(), vocab, max_len=128)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32)
        test_loader = DataLoader(test_ds, batch_size=32)
        
        # Build model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMClassifier(
            vocab_size=len(vocab.itos),
            embed_dim=embed_dim,
            hidden_dim=128,
            num_layers=1,
            dropout=0.2,
            pad_id=vocab.pad_id,
        ).to(device)
        
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # Compute class weights
        counts = train_df["label"].value_counts().sort_index()
        total = float(counts.sum())
        weights = [total / (2.0 * float(counts.get(i, 1))) for i in range(2)]
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Train
        best_val_f1 = -1.0
        best_epoch = 0
        
        for epoch in range(1, 11):  # 10 epochs for ablation
            train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate_epoch(model, val_loader, criterion, device)
            
            if val_metrics.f1 > best_val_f1:
                best_val_f1 = val_metrics.f1
                best_epoch = epoch
        
        # Evaluate on test set
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        results[embed_dim] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "best_val_f1": float(best_val_f1),
            "best_epoch": best_epoch,
        }
        
        print(f"\nResults for embed_dim={embed_dim}:")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test Precision: {precision:.4f}")
        print(f"  Test Recall: {recall:.4f}")
        print(f"  Test F1: {f1:.4f}")
        print(f"  Best Val F1: {best_val_f1:.4f} (epoch {best_epoch})")
    
    # Save results
    output_path = Path("reports/ablation_embedding_dim.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Embed Dim':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*60)
    
    for embed_dim in embedding_dims:
        r = results[embed_dim]
        print(f"{embed_dim:<12} {r['accuracy']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f}")
    
    print(f"\n✅ Ablation study complete: {output_path}\n")
    
    return results


if __name__ == "__main__":
    run_ablation()
