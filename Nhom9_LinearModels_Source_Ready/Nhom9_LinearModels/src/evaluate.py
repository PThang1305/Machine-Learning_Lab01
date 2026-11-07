"""Evaluation metrics and coefficient tables for trained estimators."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def zero_weight_features(model: object, feature_names: list[str], zero_tol: float = 1e-4) -> list[str]:
    coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
    return [name for name, value in zip(feature_names, coefficients) if abs(value) < zero_tol]


def evaluate_models(
    models: Mapping[str, object],
    X_test_transformed: np.ndarray,
    y_test: pd.Series | np.ndarray,
    feature_names: list[str],
    zero_tol: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Return metric results, feature weights and predictions for all models."""
    X = np.asarray(X_test_transformed, dtype=float)
    y = np.asarray(y_test, dtype=float).reshape(-1)
    predictions: dict[str, np.ndarray] = {}
    rows = []
    weight_columns = {}

    for name, model in models.items():
        y_hat = np.asarray(model.predict(X), dtype=float).reshape(-1)
        predictions[name] = y_hat
        mse = float(mean_squared_error(y, y_hat))
        zeros = zero_weight_features(model, feature_names, zero_tol=zero_tol)
        rows.append({
            "Model": name,
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "MAE": float(mean_absolute_error(y, y_hat)),
            "R2": float(r2_score(y, y_hat)),
            "Zero_Weights": int(len(zeros)),
            "Zero_Features": ", ".join(zeros) or "-",
        })
        coef = np.asarray(model.coef_, dtype=float).reshape(-1)
        coef[np.abs(coef) < zero_tol] = 0.0
        weight_columns[name] = coef

    results = pd.DataFrame(rows).set_index("Model")
    weights = pd.DataFrame(weight_columns, index=feature_names)
    weights.index.name = "Feature"
    return results, weights, predictions


def save_evaluation_tables(
    results: pd.DataFrame,
    weights: pd.DataFrame,
    output_dir,
) -> tuple:
    from pathlib import Path

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results_path = directory / "model_results.csv"
    weights_path = directory / "model_weights.csv"
    results.to_csv(results_path)
    weights.to_csv(weights_path)
    return results_path, weights_path
