"""Download the DistilBERT checkpoint (model.safetensors, ~256 MB) from the
GitHub Release attached to this repository.

The other three files (config.json, tokenizer.json, tokenizer_config.json) are
already committed under models/best_transformer/. Only the large weights file
exceeds GitHub's 100 MB per-file limit and is hosted on the Release page instead.

Usage:
    python scripts/download_trained_models.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


RELEASE_URL = "https://github.com/RITHVIKILLANDULA/DL_Project/releases/download/v1.0-models/model.safetensors"
TARGET = Path(__file__).parent.parent / "models" / "best_transformer" / "model.safetensors"


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    pct = min(100.0, 100.0 * downloaded / total_size) if total_size > 0 else 0
    bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
    mb_down = downloaded / 1e6
    mb_total = total_size / 1e6
    sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%  {mb_down:7.1f} / {mb_total:7.1f} MB")
    sys.stdout.flush()


def main() -> int:
    if TARGET.exists():
        size_mb = TARGET.stat().st_size / 1e6
        print(f"Already present: {TARGET} ({size_mb:.1f} MB) — skipping download.")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DistilBERT weights from:\n  {RELEASE_URL}")
    print(f"Target:\n  {TARGET}")
    try:
        urllib.request.urlretrieve(RELEASE_URL, TARGET, reporthook=_progress)
        print()  # newline after the progress bar
        print(f"Saved: {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")
        return 0
    except Exception as exc:
        print()
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "Fallback: download the file manually from "
            "https://github.com/RITHVIKILLANDULA/DL_Project/releases/tag/v1.0-models",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
