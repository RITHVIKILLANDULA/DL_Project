from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from newspaper import Article

from fake_news.constants import FAKE_LABEL, REAL_LABEL
from fake_news.data.preprocess import clean_text


LIAR_URL = "https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip"
FAKENEWSNET_FILES = {
    "politifact_fake": [
        "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/main/dataset/politifact_fake.csv",
        "https://raw.githubusercontent.com/kaidmml/FakeNewsNet/main/dataset/politifact_fake.csv",
    ],
    "politifact_real": [
        "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/main/dataset/politifact_real.csv",
        "https://raw.githubusercontent.com/kaidmml/FakeNewsNet/main/dataset/politifact_real.csv",
    ],
    "gossipcop_fake": [
        "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/main/dataset/gossipcop_fake.csv",
        "https://raw.githubusercontent.com/kaidmml/FakeNewsNet/main/dataset/gossipcop_fake.csv",
    ],
    "gossipcop_real": [
        "https://raw.githubusercontent.com/KaiDMML/FakeNewsNet/main/dataset/gossipcop_real.csv",
        "https://raw.githubusercontent.com/kaidmml/FakeNewsNet/main/dataset/gossipcop_real.csv",
    ],
}


def _download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    out_path.write_bytes(response.content)


def download_liar(out_dir: Path) -> Path:
    zip_path = out_dir / "liar_dataset.zip"
    _download_file(LIAR_URL, zip_path)
    return zip_path


def download_fakenewsnet_csvs(out_dir: Path) -> list[Path]:
    csv_dir = out_dir / "fakenewsnet_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, urls in FAKENEWSNET_FILES.items():
        path = csv_dir / f"{name}.csv"
        last_error = None
        for url in urls:
            try:
                _download_file(url, path)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        paths.append(path)
    return paths


def _archive_url(url: str) -> str | None:
    try:
        archive_api = f"http://web.archive.org/cdx/search/cdx?url={url}&output=json"
        response = requests.get(archive_api, timeout=30)
        response.raise_for_status()
        rows = response.json()
        if len(rows) < 2:
            return None
        timestamp, original_url = rows[1][1], rows[1][2]
        return f"https://web.archive.org/web/{timestamp}/{original_url}"
    except Exception:
        return None


def _crawl_article(url: str) -> dict | None:
    candidates = [url]
    if url and not url.startswith("http"):
        candidates = [f"https://{url.lstrip('/')}", f"http://{url.lstrip('/')}"]

    for candidate in candidates:
        for final_url in (candidate, _archive_url(candidate) if candidate else None):
            if not final_url:
                continue
            try:
                article = Article(final_url)
                article.download()
                time.sleep(1)
                article.parse()
                if not article.is_parsed:
                    continue
                text = clean_text(article.text or "")
                if text:
                    return {
                        "url": url,
                        "source_url": final_url,
                        "text": text,
                        "title": article.title,
                        "publish_date": str(article.publish_date) if article.publish_date else None,
                    }
            except Exception:
                continue
    return None


def build_fakenewsnet_articles(csv_paths: list[Path], out_dir: Path) -> pd.DataFrame:
    records = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        label = FAKE_LABEL if "fake" in csv_path.stem else REAL_LABEL
        source = "politifact" if "politifact" in csv_path.stem else "gossipcop"
        for _, row in df.iterrows():
            url = str(row.get("url", "")).strip()
            title = str(row.get("title", "")).strip()
            article = _crawl_article(url)
            if article is None:
                text = clean_text(title)
                if not text:
                    continue
                article = {
                    "url": url,
                    "source_url": None,
                    "text": text,
                    "title": title,
                    "publish_date": None,
                }
            records.append(
                {
                    "text": article["text"],
                    "label": label,
                    "source_dataset": f"fakenewsnet_{source}",
                    "url": article["url"],
                    "title": article["title"],
                    "publish_date": article["publish_date"],
                }
            )
    df = pd.DataFrame(records).drop_duplicates(subset=["text"])
    out_path = out_dir / "fakenewsnet_articles.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def _extract_liar_zip(zip_path: Path, out_dir: Path) -> pd.DataFrame:
    import zipfile

    liar_dir = out_dir / "liar"
    liar_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(liar_dir)

    tsv_candidates = list(liar_dir.rglob("*.tsv"))
    if not tsv_candidates:
        raise FileNotFoundError("No TSV files found in LIAR zip")

    split_map = {"train": None, "valid": None, "test": None}
    for path in tsv_candidates:
        stem = path.stem.lower()
        if stem in split_map:
            split_map[stem] = path

    rows = []
    for split_name, path in split_map.items():
        if path is None:
            continue
        df = pd.read_csv(path, sep="\t", header=None)
        if df.shape[1] < 3:
            continue
        label_col = df.columns[1]
        text_col = df.columns[2]
        for _, row in df.iterrows():
            label_raw = str(row[label_col]).strip().lower()
            label = REAL_LABEL if label_raw in {"true", "mostly-true", "half-true"} else FAKE_LABEL
            text = clean_text(str(row[text_col]))
            if not text:
                continue
            rows.append({"text": text, "label": label, "source_dataset": f"liar_{split_name}"})

    out = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    out.to_csv(out_dir / "liar_cleaned.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/downloaded"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    liar_zip = download_liar(args.out_dir)
    fakenews_csvs = download_fakenewsnet_csvs(args.out_dir)

    liar_df = _extract_liar_zip(liar_zip, args.out_dir)
    fake_df = build_fakenewsnet_articles(fakenews_csvs, args.out_dir)

    summary = {
        "liar_rows": int(len(liar_df)),
        "fakenewsnet_rows": int(len(fake_df)),
        "liar_path": str((args.out_dir / "liar_cleaned.csv").resolve()),
        "fakenewsnet_path": str((args.out_dir / "fakenewsnet_articles.csv").resolve()),
    }
    (args.out_dir / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
