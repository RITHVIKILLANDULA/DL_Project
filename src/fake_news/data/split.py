from __future__ import annotations



from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_dataframe(df: pd.DataFrame, seed: int = 42) -> SplitResult:
    train_df, tmp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=0.5,
        random_state=seed,
        stratify=tmp_df["label"],
    )
    return SplitResult(train=train_df.reset_index(drop=True), val=val_df.reset_index(drop=True), test=test_df.reset_index(drop=True))

