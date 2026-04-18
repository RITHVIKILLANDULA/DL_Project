from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch

from fake_news.config import TrainingConfig
from fake_news.data.dataset import Vocab, encode_text
from fake_news.data.preprocess import clean_text
from fake_news.models.lstm import LSTMClassifier
from fake_news.models.rnn import RNNClassifier


@st.cache_resource
def load_checkpoint(path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
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
    return model, vocab, cfg, device


def predict_text(model, vocab, cfg, device, text: str) -> tuple[int, float]:
    clean = clean_text(text)
    ids = encode_text(clean, vocab, cfg.max_len)
    tensor = torch.tensor([ids], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        conf = float(torch.max(probs, dim=1).values.item())
    return pred, conf


def main() -> None:
    st.title("Fake News Detector")
    st.write("Load a trained RNN/LSTM checkpoint and test new text.")

    model_path = st.text_input("Checkpoint path", "models/best_lstm.pt")
    text = st.text_area("News text", "Type or paste a statement/article here.")

    if st.button("Predict"):
        path = Path(model_path)
        if not path.exists():
            st.error("Checkpoint not found.")
            return

        model, vocab, cfg, device = load_checkpoint(path)
        pred, conf = predict_text(model, vocab, cfg, device, text)
        label = "FAKE" if pred == 1 else "REAL"
        st.success(f"Prediction: {label} (confidence: {conf:.3f})")


if __name__ == "__main__":
    main()
