#!/usr/bin/env python
"""Download and prepare GloVe embeddings for RNN/LSTM models."""

import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from collections import defaultdict

def download_glove():
    """Download GloVe embeddings (6B tokens version, 100d)."""
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = Path("data/raw/glove.6B.zip")
    extract_path = Path("data/raw/glove_embeddings")

    # Skip if already downloaded
    if (extract_path / "glove.6B.100d.txt").exists():
        print("✅ GloVe embeddings already downloaded")
        return extract_path / "glove.6B.100d.txt"

    print(f"📥 Downloading GloVe embeddings from {url}")
    print("   (This may take 1-2 minutes and ~822 MB)")

    try:
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Download complete")

        print(f"📦 Extracting to {extract_path}...")
        extract_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        print("✅ Extraction complete")
        return extract_path / "glove.6B.100d.txt"

    except Exception as e:
        print(f"❌ Error downloading GloVe: {e}")
        print("\nAlternative: Use FastText embeddings (smaller)")
        print("Or train Word2Vec on your corpus")
        raise


def load_glove_embeddings(glove_path, vocab, embed_dim=100):
    """Load GloVe embeddings and create embedding matrix for vocab.

    Args:
        glove_path: Path to glove.6B.XXXd.txt file
        vocab: Vocab object with stoi (string to index) mapping
        embed_dim: Embedding dimension (100 for glove.6B.100d)

    Returns:
        np.array of shape (vocab_size, embed_dim) with loaded embeddings
    """
    print(f"\n📖 Loading GloVe embeddings from {glove_path}")
    print(f"   This may take 30-60 seconds...")

    # Initialize with random embeddings
    embedding_matrix = np.random.randn(len(vocab.itos), embed_dim).astype(np.float32) * 0.01

    # Counter for coverage
    matched = 0
    total_vocab = len(vocab.itos)

    # Load GloVe vectors
    with open(glove_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if line_num % 100000 == 0:
                print(f"   Processing line {line_num}...")

            parts = line.rstrip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)

            # Check if word is in our vocabulary
            if word in vocab.stoi:
                word_idx = vocab.stoi[word]
                embedding_matrix[word_idx] = vector
                matched += 1

    coverage = matched / total_vocab * 100
    print(f"\n✅ GloVe loading complete!")
    print(f"   Matched: {matched}/{total_vocab} vocab words ({coverage:.1f}%)")
    print(f"   Dimension: {embed_dim}")

    return embedding_matrix


if __name__ == "__main__":
    # Test download
    glove_file = download_glove()
    print(f"\n✅ Ready to use GloVe embeddings at: {glove_file}")
