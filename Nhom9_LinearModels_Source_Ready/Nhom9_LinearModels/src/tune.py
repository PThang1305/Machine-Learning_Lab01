"""Cross-validation helpers. Preprocessing is fitted separately inside each fold."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS, RANDOM_STATE
from .models import build_models
from .preprocess import build_preprocessor

CV_SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def make_cv_pipeline(model) -> Pipeline:
    """Construct a fresh preprocessing+model pipeline for a CV fold."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", clone(model)),
    ])


def cv_scores(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, np.ndarray]:
    """Run shuffled K-fold CV; all encoding/scaling is fitted only on each fold's train rows."""
    splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return cross_validate(
        make_cv_pipeline(model),
        X_train,
        y_train,
        cv=splitter,
        scoring=CV_SCORING,
        return_train_score=False,
        error_score="raise",
    )


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: dict | None = None,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the four fixed models by CV and return summary + fold-level R²."""
    estimators = models if models is not None else build_models()
    rows = []
    fold_r2 = {}
    for name, model in estimators.items():
        scores = cv_scores(model, X_train, y_train, cv_folds, random_state)
        fold_r2[name] = scores["test_r2"]
        rows.append({
            "Model": name,
            "CV RMSE": -float(scores["test_rmse"].mean()),
            "SD RMSE": float(scores["test_rmse"].std(ddof=1)),
            "CV MAE": -float(scores["test_mae"].mean()),
            "CV R2": float(scores["test_r2"].mean()),
            "SD R2": float(scores["test_r2"].std(ddof=1)),
        })
    return pd.DataFrame(rows).set_index("Model"), pd.DataFrame(fold_r2, index=[f"fold {i}" for i in range(1, cv_folds + 1)])


def evaluate_parameter_grid(
    base_model,
    parameter_name: str,
    values: Iterable[float],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
) -> pd.DataFrame:
    """Evaluate one hyperparameter grid using CV on Train only."""
    rows = []
    for value in values:
        candidate = clone(base_model).set_params(**{parameter_name: value})
        scores = cv_scores(candidate, X_train, y_train, cv_folds=cv_folds)
        rows.append({
            parameter_name: float(value),
            "cv_rmse": -float(scores["test_rmse"].mean()),
            "cv_rmse_sd": float(scores["test_rmse"].std(ddof=1)),
            "cv_mae": -float(scores["test_mae"].mean()),
            "cv_r2": float(scores["test_r2"].mean()),
        })
    return pd.DataFrame(rows)
