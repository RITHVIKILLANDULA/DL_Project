from __future__ import annotations



from dataclasses import dataclass

import pandas as pd


@dataclass
class ErrorAnalysisResult:
    false_positives: pd.DataFrame
    false_negatives: pd.DataFrame


def collect_error_slices(df: pd.DataFrame, y_true: list[int], y_pred: list[int], top_k: int = 25) -> ErrorAnalysisResult:
    tmp = df.copy()
    tmp["y_true"] = y_true
    tmp["y_pred"] = y_pred

    fp = tmp[(tmp["y_true"] == 0) & (tmp["y_pred"] == 1)].head(top_k)
    fn = tmp[(tmp["y_true"] == 1) & (tmp["y_pred"] == 0)].head(top_k)
    return ErrorAnalysisResult(false_positives=fp, false_negatives=fn)

