"""Model training functions. Estimators receive transformed training features."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .models import build_models


def train_models(
    X_train_transformed: np.ndarray,
    y_train: pd.Series | np.ndarray,
    estimators: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Fit the four estimators using the already-fitted shared preprocessor output."""
    X = np.asarray(X_train_transformed, dtype=float)
    y = np.asarray(y_train, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X_train_transformed phải là ma trận 2 chiều.")
    if len(X) != len(y):
        raise ValueError("Số dòng X_train và y_train không khớp.")
    model_dict = dict(estimators) if estimators is not None else build_models()
    for name, model in model_dict.items():
        model.fit(X, y)
    return model_dict
