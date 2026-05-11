from __future__ import annotations


REAL_LABEL = 0
FAKE_LABEL = 1

LABEL_MAP = {
    # LIAR-like labels
    "pants-fire": FAKE_LABEL,
    "false": FAKE_LABEL,
    "barely-true": FAKE_LABEL,
    "half-true": REAL_LABEL,
    "mostly-true": REAL_LABEL,
    "true": REAL_LABEL,
    # Direct binary labels
    "fake": FAKE_LABEL,
    "real": REAL_LABEL,
    "0": REAL_LABEL,
    "1": FAKE_LABEL,
}

