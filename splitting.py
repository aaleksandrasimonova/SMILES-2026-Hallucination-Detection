from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    val_size: float = 0.2,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:

    idx = np.arange(len(y))

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits = []

    for idx_train_val, idx_test in skf.split(idx, y):
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=val_size,
            random_state=random_state,
            stratify=y[idx_train_val],
        )

        splits.append((idx_train, idx_val, idx_test))

    return splits