"""Safe, human-readable JSON persistence for linear-model inference parameters."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import MODEL_FILENAMES, MODEL_LABELS, MODELS_DIR, TARGET_COLUMN, ZERO_TOL, ensure_project_directories
from .preprocess import preprocessor_to_metadata


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Không thể chuyển kiểu {type(value).__name__} sang JSON.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    os.replace(temporary, path)


def save_model_artifacts(
    models: dict[str, object],
    preprocessor,
    feature_names: list[str],
    output_dir: str | Path | None = None,
    zero_tol: float = ZERO_TOL,
) -> dict[str, Path]:
    """Save one shared preprocessor file and one JSON file per fitted estimator."""
    ensure_project_directories()
    directory = Path(output_dir) if output_dir else MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    preprocessor_path = directory / "preprocessor.json"
    _write_json(preprocessor_path, preprocessor_to_metadata(preprocessor, feature_names))

    paths: dict[str, Path] = {"preprocessor": preprocessor_path}
    for name, model in models.items():
        coefficients = np.asarray(model.coef_, dtype=float).reshape(-1)
        if len(coefficients) != len(feature_names):
            raise ValueError(f"Số trọng số của {name} không khớp số đặc trưng.")
        parameters = model.get_params(deep=False)
        compact_params = {
            key: value for key, value in parameters.items()
            if key in {"alpha", "l1_ratio", "max_iter", "fit_intercept", "tol", "positive", "selection"}
        }
        payload = {
            "schema_version": 1,
            "model_key": name,
            "model_name": MODEL_LABELS.get(name, name),
            "estimator_class": model.__class__.__name__,
            "target": TARGET_COLUMN,
            "parameters": compact_params,
            "feature_names": list(feature_names),
            "coefficients": {feature: float(coef) for feature, coef in zip(feature_names, coefficients)},
            "intercept": float(np.asarray(model.intercept_).reshape(-1)[0] if np.asarray(model.intercept_).ndim else model.intercept_),
            "zero_weight_tolerance": float(zero_tol),
            "preprocessor_file": "preprocessor.json",
            "prediction_formula": "y_hat = intercept + sum((x_scaled_j * coefficient_j))",
        }
        path = directory / MODEL_FILENAMES[name]
        _write_json(path, payload)
        paths[name] = path
    return paths


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON metadata and validate its top-level structure."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {file_path}")
    with file_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"File JSON không đúng schema version 1: {file_path}")
    return payload


def load_model_bundle(models_dir: str | Path | None = None) -> tuple[dict, dict[str, dict]]:
    """Load the shared preprocessor metadata and all four model JSON files."""
    directory = Path(models_dir) if models_dir else MODELS_DIR
    preprocessor = load_json(directory / "preprocessor.json")
    models: dict[str, dict] = {}
    for name, filename in MODEL_FILENAMES.items():
        payload = load_json(directory / filename)
        if payload.get("model_key") != name:
            raise ValueError(f"model_key trong {filename} không khớp tên mô hình {name}.")
        if payload.get("feature_names") != preprocessor.get("feature_names"):
            raise ValueError(f"Thứ tự đặc trưng trong {filename} không khớp preprocessor.json.")
        models[name] = payload
    return preprocessor, models
