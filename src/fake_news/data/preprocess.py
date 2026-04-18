from __future__ import annotations

import re

_WS_PATTERN = re.compile(r"\s+")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


def clean_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = _URL_PATTERN.sub(" ", text)
    text = text.replace("\n", " ")
    text = _WS_PATTERN.sub(" ", text)
    return text.strip()
