from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add src directory to path so fake_news module can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fake_news.config import TrainingConfig
from fake_news.data.dataset import Vocab, encode_text
from fake_news.data.preprocess import clean_text
from fake_news.models.lstm import LSTMClassifier
from fake_news.models.rnn import RNNClassifier


PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


@st.cache_resource
def _load_classic_checkpoint(path_str: str):
    path = Path(path_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    vocab = Vocab(stoi=ckpt["vocab"]["stoi"], itos=ckpt["vocab"]["itos"])
    cfg = TrainingConfig(**ckpt["config"])

    if ckpt["model_type"] == "rnn":
        model = RNNClassifier(
            vocab_size=len(vocab.itos),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_id=vocab.pad_id,
        )
    else:
        model = LSTMClassifier(
            vocab_size=len(vocab.itos),
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            pad_id=vocab.pad_id,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return {
        "kind": "classic",
        "model": model,
        "vocab": vocab,
        "cfg": cfg,
        "device": device,
        "model_name": ckpt["model_type"].upper(),
    }


@st.cache_resource
def _load_transformer_checkpoint(path_str: str):
    path = Path(path_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(device)
    model.eval()
    return {
        "kind": "transformer",
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "max_len": 128,
        "model_name": "DistilBERT",
    }


def discover_models() -> list[tuple[str, str, str]]:
    """Return [(display_name, path, model_key)] for every trained model in models/.

    model_key is one of 'transformer', 'lstm', 'rnn' — used to look up reported metrics.
    Listed in recommended order: DistilBERT (best overall acc), BiLSTM, BiRNN.
    """
    found: list[tuple[str, str, str]] = []

    transformer_dir = MODELS_DIR / "best_transformer"
    if (transformer_dir / "model.safetensors").exists() or (transformer_dir / "pytorch_model.bin").exists():
        found.append(("DistilBERT (fine-tuned)", str(transformer_dir), "transformer"))

    lstm_ckpt = MODELS_DIR / "classic_lstm" / "best_lstm.pt"
    if lstm_ckpt.exists():
        found.append(("BiLSTM + GloVe", str(lstm_ckpt), "lstm"))

    rnn_ckpt = MODELS_DIR / "classic_rnn" / "best_rnn.pt"
    if rnn_ckpt.exists():
        found.append(("BiRNN + GloVe", str(rnn_ckpt), "rnn"))

    return found


def load_checkpoint(path: Path) -> dict[str, Any]:
    if path.is_dir():
        return _load_transformer_checkpoint(str(path))
    return _load_classic_checkpoint(str(path))


def _load_reported_metrics() -> dict[str, dict[str, float]]:
    eval_path = PROJECT_ROOT / "reports" / "full_evaluation.json"
    if not eval_path.exists():
        return {}
    try:
        data = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    mapped: dict[str, dict[str, float]] = {}
    for name, payload in data.get("models", {}).items():
        overall = payload.get("overall", {})
        per_source = payload.get("per_source", {})
        key = str(name).lower()
        if "transformer" in key or "distilbert" in key:
            mapped["transformer"] = {**overall, "per_source": per_source}
        elif "lstm" in key:
            mapped["lstm"] = {**overall, "per_source": per_source}
        elif "rnn" in key:
            mapped["rnn"] = {**overall, "per_source": per_source}
        elif "baseline" in key or "tfidf" in key:
            mapped["baseline"] = {**overall, "per_source": per_source}
    return mapped


def predict_text(loaded: dict[str, Any], text: str, fake_threshold: float) -> dict[str, float | int]:
    clean = clean_text(text)

    if loaded["kind"] == "classic":
        vocab = loaded["vocab"]
        cfg = loaded["cfg"]
        model = loaded["model"]
        device = loaded["device"]
        ids = encode_text(clean, vocab, cfg.max_len)
        tensor = torch.tensor([ids], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            fake_prob = float(probs[0][1].item())
    else:
        tokenizer = loaded["tokenizer"]
        model = loaded["model"]
        device = loaded["device"]
        max_len = loaded["max_len"]
        enc = tokenizer(
            clean,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
            probs = torch.softmax(logits, dim=1)
            fake_prob = float(probs[0][1].item())

    real_prob = 1.0 - fake_prob
    pred = 1 if fake_prob >= fake_threshold else 0
    confidence = fake_prob if pred == 1 else real_prob
    return {
        "pred": pred,
        "confidence": confidence,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
    }


def main() -> None:
    st.set_page_config(page_title="Fake News Detector", page_icon=None, layout="centered")
    st.title("Fake News Detector")
    st.write("Pick a trained model, paste text, get a real/fake prediction.")

    available_models = discover_models()
    if not available_models:
        st.error(
            "No trained checkpoints found in `models/`. "
            "Train at least one model with `scripts/train_classic.py` or `scripts/train_transformer.py`."
        )
        return

    display_names = [name for name, _, _ in available_models]
    path_by_name = {name: path for name, path, _ in available_models}
    key_by_name = {name: key for name, _, key in available_models}

    selected_model = st.selectbox(
        "Model",
        display_names,
        index=0,
        help="DistilBERT is best on accuracy and on full-length articles; BiRNN has highest F1 on the test set; BiLSTM is balanced.",
    )
    model_key = key_by_name[selected_model]

    reported_metrics = _load_reported_metrics()
    if model_key in reported_metrics:
        m = reported_metrics[model_key]
        ps = m.get("per_source", {})
        liar = ps.get("liar", {})
        fnn = ps.get("fakenewsnet", {})
        cols = st.columns(4)
        cols[0].metric("Test Accuracy", f"{m.get('accuracy', 0.0):.3f}")
        cols[1].metric("Test F1", f"{m.get('f1', 0.0):.3f}")
        cols[2].metric("LIAR Acc", f"{liar.get('accuracy', 0.0):.3f}" if liar else "—")
        cols[3].metric("FNN Acc", f"{fnn.get('accuracy', 0.0):.3f}" if fnn else "—")
        st.caption("Offline metrics on the held-out test set (n=1,295), loaded from `reports/full_evaluation.json`.")

    fake_threshold = st.slider(
        "Fake threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.01,
        help="Lower = more aggressive fake detection (higher recall, lower precision). 0.50 matches the CLI evaluation.",
    )

    text = st.text_area(
        "News text",
        "Type or paste a statement or article here.",
        height=150,
    )

    if st.button("Predict", type="primary"):
        if not text.strip() or text.strip().startswith("Type or paste"):
            st.error("Please enter some text.")
            return

        with st.spinner(f"Running {selected_model}..."):
            path = Path(path_by_name[selected_model])
            loaded = load_checkpoint(path)
            out = predict_text(loaded, text, fake_threshold=fake_threshold)

        label = "FAKE" if out["pred"] == 1 else "REAL"
        if out["pred"] == 1:
            st.error(f"Prediction: **{label}** (confidence {out['confidence']:.3f})")
        else:
            st.success(f"Prediction: **{label}** (confidence {out['confidence']:.3f})")

        cols = st.columns(2)
        cols[0].metric("P(real)", f"{out['real_prob']:.3f}")
        cols[1].metric("P(fake)", f"{out['fake_prob']:.3f}")
        st.caption(
            f"Model: {loaded['model_name']} | threshold: {fake_threshold:.2f} | "
            "this model scores linguistic patterns; it does not verify external facts."
        )


if __name__ == "__main__":
    main()
