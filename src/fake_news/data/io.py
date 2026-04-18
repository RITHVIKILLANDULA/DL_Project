from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from fake_news.constants import LABEL_MAP
from fake_news.data.preprocess import clean_text


def _normalize_label(value: str | int | float) -> int | None:
    key = str(value).strip().lower()
    return LABEL_MAP.get(key)


def _safe_text_column(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    normalized = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in normalized:
            return df[normalized[candidate]].fillna("").astype(str)
    return pd.Series([""] * len(df))


def _postprocess_dataset(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["label"].map(_normalize_label)
    out = out.dropna(subset=["label"]).copy()
    out["text"] = out["text"].map(clean_text)
    out["source_dataset"] = source_name
    out = out[out["text"].str.len() > 0]
    return out[["text", "label", "source_dataset"]]


def load_liar(liar_path: Path) -> pd.DataFrame:
    def _load_liar_file(tsv_path: Path) -> pd.DataFrame:
        liar = pd.read_csv(tsv_path, sep="\t", header=None)
        liar.columns = [
            "id",
            "label",
            "statement",
            "subject",
            "speaker",
            "speaker_job",
            "state",
            "party",
            "barely_true",
            "false_count",
            "half_true",
            "mostly_true",
            "pants_fire",
            "context",
        ]
        return liar[["statement", "label"]].rename(columns={"statement": "text"})

    if liar_path.is_dir():
        tsv_files = [p for p in liar_path.glob("*.tsv") if p.name.lower() in {"train.tsv", "valid.tsv", "test.tsv"}]
        if not tsv_files:
            raise FileNotFoundError(f"No LIAR TSV files found in {liar_path}")
        liar = pd.concat([_load_liar_file(p) for p in sorted(tsv_files)], ignore_index=True)
    else:
        liar = _load_liar_file(liar_path)

    return _postprocess_dataset(liar, "liar")


def _load_fakenewsnet_labeled_csv(fake_path: Path) -> pd.DataFrame:
    fake = pd.read_csv(fake_path)
    label_col = None
    for candidate in ["label", "target", "class", "is_fake"]:
        if candidate in {c.lower() for c in fake.columns}:
            inv = {c.lower(): c for c in fake.columns}
            label_col = inv[candidate]
            break
    if label_col is None:
        raise ValueError(
            "Could not find a label column in FakeNewsNet file. Expected one of label,target,class,is_fake"
        )

    text_series = _safe_text_column(fake, ["text", "content", "article", "title"])
    if text_series.str.len().sum() == 0:
        title = _safe_text_column(fake, ["title"])
        body = _safe_text_column(fake, ["body", "content"])
        text_series = title + " " + body

    merged = pd.DataFrame({"text": text_series, "label": fake[label_col]})
    return _postprocess_dataset(merged, "fakenewsnet")


def _load_fakenewsnet_split_csvs(fake_dir: Path) -> pd.DataFrame:
    csv_files = list(fake_dir.glob("*.csv"))
    split_csvs = [p for p in csv_files if any(k in p.stem.lower() for k in ["politifact", "gossipcop"]) and any(k in p.stem.lower() for k in ["fake", "real"])]
    if not split_csvs:
        raise FileNotFoundError("No FakeNewsNet split CSV files found.")

    rows: list[dict] = []
    for path in split_csvs:
        stem = path.stem.lower()
        label = "fake" if "fake" in stem else "real"
        df = pd.read_csv(path)
        text_series = _safe_text_column(df, ["text", "content", "article", "title"])
        if text_series.str.len().sum() == 0:
            title = _safe_text_column(df, ["title"])
            body = _safe_text_column(df, ["body", "content"])
            text_series = title + " " + body
        for text in text_series.tolist():
            rows.append({"text": text, "label": label})

    merged = pd.DataFrame(rows)
    return _postprocess_dataset(merged, "fakenewsnet")


def _load_fakenewsnet_news_content(fake_dir: Path) -> pd.DataFrame:
    json_files = (
        list(fake_dir.rglob("news content.json"))
        + list(fake_dir.rglob("news_content.json"))
        + list(fake_dir.rglob("*-Webpage.json"))
        + list(fake_dir.rglob("*.json"))
    )
    seen: set[Path] = set()
    unique_files: list[Path] = []
    for p in json_files:
        if p not in seen:
            unique_files.append(p)
            seen.add(p)

    json_files = unique_files
    if not json_files:
        raise FileNotFoundError("No FakeNewsNet news content JSON files found.")

    rows: list[dict] = []
    for path in json_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        text = payload.get("text") if isinstance(payload, dict) else None
        if not text:
            continue

        label = None
        parts = [p.lower() for p in path.parts]
        stem = path.stem.lower()
        if "fake" in parts:
            label = "fake"
        elif "real" in parts:
            label = "real"
        elif "fakenewscontent" in parts or "_fake_" in stem:
            label = "fake"
        elif "realnewscontent" in parts or "_real_" in stem:
            label = "real"
        if label is None:
            continue

        rows.append({"text": text, "label": label})

    if not rows:
        raise ValueError("Found news content JSON files but could not parse labeled article text.")
    merged = pd.DataFrame(rows)
    return _postprocess_dataset(merged, "fakenewsnet")


def load_fakenewsnet(fake_path: Path) -> pd.DataFrame:
    if fake_path.is_file():
        return _load_fakenewsnet_labeled_csv(fake_path)

    if not fake_path.exists():
        raise FileNotFoundError(f"FakeNewsNet path does not exist: {fake_path}")

    try:
        return _load_fakenewsnet_news_content(fake_path)
    except Exception:
        return _load_fakenewsnet_split_csvs(fake_path)


def build_unified_dataset(liar_path: Path, fake_path: Path) -> pd.DataFrame:
    liar_df = load_liar(liar_path)
    fake_df = load_fakenewsnet(fake_path)
    unified = pd.concat([liar_df, fake_df], axis=0, ignore_index=True)
    unified = unified.drop_duplicates(subset=["text"])
    return unified
