#!/usr/bin/env python
"""Flask web app for fake news detection.

Loads every trained checkpoint at startup and exposes a /api/predict endpoint
with a `model` parameter so the UI can switch between DistilBERT, BiLSTM, and BiRNN.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fake_news.config import TrainingConfig
from fake_news.data.dataset import Vocab, encode_text
from fake_news.data.preprocess import clean_text
from fake_news.models.lstm import LSTMClassifier
from fake_news.models.rnn import RNNClassifier


app = Flask(__name__)
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODELS: dict[str, dict[str, Any]] = {}
DEFAULT_MODEL = "distilbert"


def _load_classic(path: Path, model_type: str, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = Vocab(stoi=ckpt["vocab"]["stoi"], itos=ckpt["vocab"]["itos"])
    cfg = TrainingConfig(**ckpt["config"])
    cls = RNNClassifier if model_type == "rnn" else LSTMClassifier
    model = cls(
        vocab_size=len(vocab.itos),
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pad_id=vocab.pad_id,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return {"type": "classic", "model": model, "vocab": vocab, "cfg": cfg, "device": device}


def _load_transformer(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(str(path))
    model.to(device).eval()
    return {"type": "transformer", "model": model, "tokenizer": tokenizer, "device": device, "max_len": 128}


def load_models_global() -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading models from {MODELS_DIR}/ (device: {device})")

    transformer_path = MODELS_DIR / "best_transformer"
    if (transformer_path / "model.safetensors").exists() or (transformer_path / "pytorch_model.bin").exists():
        try:
            MODELS["distilbert"] = _load_transformer(transformer_path, device)
            print(f"  loaded distilbert from {transformer_path}")
        except Exception as e:
            print(f"  failed distilbert: {e}")

    lstm_ckpt = MODELS_DIR / "classic_lstm" / "best_lstm.pt"
    if lstm_ckpt.exists():
        try:
            MODELS["lstm"] = _load_classic(lstm_ckpt, "lstm", device)
            print(f"  loaded lstm from {lstm_ckpt}")
        except Exception as e:
            print(f"  failed lstm: {e}")

    rnn_ckpt = MODELS_DIR / "classic_rnn" / "best_rnn.pt"
    if rnn_ckpt.exists():
        try:
            MODELS["rnn"] = _load_classic(rnn_ckpt, "rnn", device)
            print(f"  loaded rnn from {rnn_ckpt}")
        except Exception as e:
            print(f"  failed rnn: {e}")

    return len(MODELS) > 0


def _predict_probs(entry: dict[str, Any], clean: str) -> tuple[float, float]:
    if entry["type"] == "transformer":
        tokenizer = entry["tokenizer"]
        model = entry["model"]
        device = entry["device"]
        enc = tokenizer(clean, truncation=True, padding=True, max_length=entry["max_len"], return_tensors="pt")
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
            probs = torch.softmax(logits, dim=1)
            return float(probs[0][0].item()), float(probs[0][1].item())

    vocab = entry["vocab"]
    cfg = entry["cfg"]
    model = entry["model"]
    device = entry["device"]
    ids = encode_text(clean, vocab, cfg.max_len)
    tensor = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        return float(probs[0][0].item()), float(probs[0][1].item())


def _explanation(clean: str, p_fake: float, threshold: float) -> dict[str, Any]:
    text_lower = clean.lower()
    factors: list[str] = []
    if "!" in clean:
        factors.append("Contains exclamation/strong punctuation (sensational)")
    if any(w in text_lower for w in ["rumour", "rumor", "alleged", "allegedly", "claims", "claimed", "reportedly"]):
        factors.append("Contains hedging/rumour words (unverified claim)")
    if any(w in text_lower for w in ["according to", "announces", "confirms", "reveals", "according"]):
        factors.append('Has source attribution ("according to", "announces")')
    if any(w in text_lower for w in ["funeral", "family", "official", "statement"]):
        factors.append("Formal/official context words found")
    if any(w in text_lower for w in ["dating", "engaged", "speculation", "gossip"]):
        factors.append("Gossip/celebrity keywords detected")
    if any(w in text_lower for w in ["senator", "congress", "president", "politician", "budget"]):
        factors.append("Political topic words detected")
    if any(c.isdigit() for c in clean):
        factors.append("Contains numbers/specific details")
    if "'" in clean or '"' in clean:
        factors.append("Quotation marks detected (possible direct quote)")
    if not factors:
        factors.append("Neutral / no strong linguistic indicators found")

    short = (
        "Model flagged this as FAKE because: " + "; ".join(factors[:3])
        if p_fake >= threshold
        else "Model considered this REAL because: " + "; ".join(factors[:3])
    )

    return {
        "short": short,
        "factors": factors,
        "linguistic": {
            "word_count": len(clean.split()),
            "has_quotes": ("'" in clean) or ('"' in clean),
            "has_attribution": any(w in text_lower for w in ["according", "says", "announces", "reveals", "confirms"]),
            "sensationalism": (
                "High"
                if any(w in text_lower for w in ["horror", "hell", "shock", "scandal", "!"])
                else "Medium"
                if any(w in text_lower for w in ["dating", "rumor", "speculation"])
                else "Low"
            ),
        },
    }


def predict(text: str, threshold: float = 0.50, model_name: str = DEFAULT_MODEL) -> dict[str, Any] | None:
    if model_name not in MODELS:
        return None
    clean = clean_text(text)
    p_real, p_fake = _predict_probs(MODELS[model_name], clean)
    prediction = "FAKE" if p_fake >= threshold else "REAL"
    confidence = max(p_real, p_fake)
    return {
        "prediction": prediction,
        "p_real": p_real,
        "p_fake": p_fake,
        "confidence": confidence,
        "clean_text": clean,
        "model": model_name,
        "explanation": _explanation(clean, p_fake, threshold),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.json or {}
        text = (data.get("text") or "").strip()
        threshold = float(data.get("threshold", 0.50))
        model_name = data.get("model", DEFAULT_MODEL)

        if not text:
            return jsonify({"error": "Please enter some text"}), 400
        if len(text) < 5:
            return jsonify({"error": "Text too short (minimum 5 characters)"}), 400

        result = predict(text, threshold, model_name=model_name)
        if result is None:
            return jsonify({"error": f"Model '{model_name}' not loaded (available: {list(MODELS.keys())})"}), 500

        return jsonify(
            {
                "success": True,
                "model": result["model"],
                "prediction": result["prediction"],
                "p_fake": round(result["p_fake"], 4),
                "p_real": round(result["p_real"], 4),
                "confidence": round(result["confidence"], 4),
                "confidence_percent": round(result["confidence"] * 100, 1),
                "clean_text": result["clean_text"],
                "explanation": result["explanation"],
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Error: {exc}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "models_loaded": list(MODELS.keys()),
            "default": DEFAULT_MODEL,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify(
        {
            "available_models": {k: v["type"] for k, v in MODELS.items()},
            "default": DEFAULT_MODEL,
            "timestamp": datetime.now().isoformat(),
        }
    )


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FAKE NEWS DETECTION - FLASK WEB APP")
    print("=" * 70)
    if not load_models_global():
        print("\nNo models loaded. Train at least one checkpoint first. Exiting.")
        sys.exit(1)
    print(f"\nLoaded models: {list(MODELS.keys())} (default: {DEFAULT_MODEL})")
    print("\nServer: http://127.0.0.1:5000")
    print("Endpoints: GET / | POST /api/predict | GET /api/health | GET /api/models\n")
    app.run(debug=False, host="127.0.0.1", port=5000, threaded=True)
