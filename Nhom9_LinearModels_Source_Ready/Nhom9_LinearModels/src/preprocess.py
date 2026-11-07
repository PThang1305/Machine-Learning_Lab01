"""Data loading, splitting, feature preprocessing and JSON preprocessor helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED_DIR,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    ensure_project_directories,
)


@dataclass
class DataSplits:
    """Raw-feature train/test partitions after duplicate removal."""

    raw_frame: pd.DataFrame
    clean_frame: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def validate_dataset(frame: pd.DataFrame) -> None:
    """Validate the dataset schema and essential values."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Dataset thiếu các cột bắt buộc: {missing_columns}")

    missing_values = frame[REQUIRED_COLUMNS].isna().sum()
    if int(missing_values.sum()) > 0:
        details = missing_values[missing_values > 0].to_dict()
        raise ValueError(f"Dataset có giá trị thiếu; cần xử lý trước khi huấn luyện: {details}")

    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.isna().any():
            raise ValueError(f"Cột '{column}' phải chứa dữ liệu số hợp lệ.")

    for column in CATEGORICAL_FEATURES:
        if frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Cột '{column}' có giá trị phân loại rỗng.")


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the local raw CSV. No silent network download is performed."""
    data_path = Path(path) if path is not None else RAW_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu: {data_path}. "
            "Hãy đặt insurance.csv vào data/raw/ hoặc truyền đường dẫn bằng tham số path."
        )
    frame = pd.read_csv(data_path)
    validate_dataset(frame)
    return frame


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove fully duplicated rows and normalize the required data types."""
    validate_dataset(frame)
    clean = frame[REQUIRED_COLUMNS].copy()
    for column in NUMERIC_FEATURES + [TARGET_COLUMN]:
        clean[column] = pd.to_numeric(clean[column], errors="raise")
    for column in CATEGORICAL_FEATURES:
        clean[column] = clean[column].astype(str).str.strip()
    return clean.drop_duplicates().reset_index(drop=True)


def split_data(
    clean_frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DataSplits:
    """Split cleaned raw features into train/test partitions."""
    if not 0 < test_size < 1:
        raise ValueError("test_size phải nằm trong khoảng (0, 1).")
    X = clean_frame[FEATURE_COLUMNS].copy()
    y = clean_frame[TARGET_COLUMN].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return DataSplits(
        raw_frame=clean_frame.copy(),
        clean_frame=clean_frame.copy(),
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )


def prepare_data(
    path: str | Path | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_processed: bool = True,
) -> DataSplits:
    """Load, clean, split and optionally save raw-feature train/test CSVs."""
    raw = load_raw_data(path)
    clean = clean_data(raw)
    splits = split_data(clean, test_size=test_size, random_state=random_state)
    splits.raw_frame = raw.copy()
    if save_processed:
        save_processed_splits(splits)
    return splits


def save_processed_splits(splits: DataSplits, output_dir: str | Path | None = None) -> dict[str, Path]:
    """Save split data before one-hot encoding/scaling, to keep the files human-readable."""
    ensure_project_directories()
    target_dir = Path(output_dir) if output_dir else DATA_PROCESSED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "X_train": target_dir / "X_train.csv",
        "X_test": target_dir / "X_test.csv",
        "y_train": target_dir / "y_train.csv",
        "y_test": target_dir / "y_test.csv",
    }
    splits.X_train.to_csv(files["X_train"], index=False)
    splits.X_test.to_csv(files["X_test"], index=False)
    splits.y_train.rename(TARGET_COLUMN).to_frame().to_csv(files["y_train"], index=False)
    splits.y_test.rename(TARGET_COLUMN).to_frame().to_csv(files["y_test"], index=False)
    return files


def load_processed_splits(input_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the four processed split files (still in original feature columns)."""
    directory = Path(input_dir) if input_dir else DATA_PROCESSED_DIR
    paths = {name: directory / f"{name}.csv" for name in ("X_train", "X_test", "y_train", "y_test")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu các file chia dữ liệu: {missing}. Hãy chạy prepare_data() trước.")
    X_train = pd.read_csv(paths["X_train"])
    X_test = pd.read_csv(paths["X_test"])
    y_train = pd.read_csv(paths["y_train"])[TARGET_COLUMN]
    y_test = pd.read_csv(paths["y_test"])[TARGET_COLUMN]
    validate_dataset(pd.concat([X_train.assign(charges=0), X_test.assign(charges=0)], ignore_index=True))
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Số dòng X và y không khớp trong các file processed.")
    return X_train, X_test, y_train, y_test


def build_preprocessor() -> Pipeline:
    """Build the shared OneHotEncoder + StandardScaler preprocessing pipeline."""
    encoder_and_columns = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("encode", encoder_and_columns),
            ("scale", StandardScaler()),
        ]
    )


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Return feature names in the exact order emitted by the fitted preprocessor."""
    if not hasattr(preprocessor, "named_steps") or "encode" not in preprocessor.named_steps:
        raise ValueError("Preprocessor chưa được fit hoặc không đúng cấu trúc.")
    names = preprocessor.named_steps["encode"].get_feature_names_out()
    return [str(name).replace("num__", "").replace("cat__", "") for name in names]


def fit_transform_features(
    preprocessor: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit preprocessing on Train only; transform both Train and Test."""
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor)
    if X_train_transformed.shape[1] != len(feature_names):
        raise RuntimeError("Số cột sau tiền xử lý không khớp với tên đặc trưng.")
    return np.asarray(X_train_transformed, dtype=float), np.asarray(X_test_transformed, dtype=float), feature_names


def preprocessor_to_metadata(preprocessor: Pipeline, feature_names: list[str] | None = None) -> dict[str, Any]:
    """Convert a fitted preprocessor to JSON-safe metadata for inference."""
    names = feature_names or get_feature_names(preprocessor)
    encoder = preprocessor.named_steps["encode"].named_transformers_["cat"]
    scaler: StandardScaler = preprocessor.named_steps["scale"]
    categories = {
        column: [str(value) for value in values.tolist()]
        for column, values in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }
    scaler_mean = np.asarray(scaler.mean_, dtype=float).tolist()
    scaler_scale = np.asarray(scaler.scale_, dtype=float).tolist()
    return {
        "schema_version": 1,
        "description": "JSON representation of the fitted OneHotEncoder + StandardScaler pipeline.",
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "categories": categories,
        "one_hot_drop": "first",
        "unknown_category_behavior": "all_zero_for_that_categorical_group",
        "feature_names": list(names),
        "scaler": {
            "with_mean": bool(scaler.with_mean),
            "with_std": bool(scaler.with_std),
            "mean": scaler_mean,
            "scale": scaler_scale,
        },
    }


def transform_with_metadata(frame: pd.DataFrame, metadata: dict[str, Any]) -> np.ndarray:
    """Apply the fitted preprocessing rules stored in preprocessor.json."""
    numeric = metadata["numeric_features"]
    categorical = metadata["categorical_features"]
    required = numeric + categorical
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dữ liệu dự đoán thiếu cột: {missing}")
    missing_values = [column for column in required if frame[column].isna().any()]
    if missing_values:
        raise ValueError(f"Dữ liệu dự đoán có giá trị thiếu ở các cột: {missing_values}")
    blank_categories = [
        column for column in categorical
        if frame[column].astype(str).str.strip().eq("").any()
    ]
    if blank_categories:
        raise ValueError(f"Dữ liệu dự đoán có giá trị phân loại rỗng: {blank_categories}")

    parts: list[np.ndarray] = []
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Cột số '{column}' có giá trị không hợp lệ hoặc bị thiếu.")
        parts.append(values.to_numpy(dtype=float).reshape(-1, 1))

    categories = metadata["categories"]
    for column in categorical:
        values = frame[column].astype(str).str.strip()
        known_categories = categories[column]
        # OneHotEncoder(drop='first'): category đầu tiên là nhóm tham chiếu.
        for category in known_categories[1:]:
            parts.append((values == category).to_numpy(dtype=float).reshape(-1, 1))
        # Category lạ được mã hóa thành toàn số 0, cùng hành vi handle_unknown='ignore'.

    encoded = np.hstack(parts) if parts else np.empty((len(frame), 0), dtype=float)
    scaler = metadata["scaler"]
    mean = np.asarray(scaler["mean"], dtype=float)
    scale = np.asarray(scaler["scale"], dtype=float)
    if encoded.shape[1] != len(mean) or len(mean) != len(metadata["feature_names"]):
        raise ValueError("Metadata tiền xử lý không nhất quán về số đặc trưng.")
    if bool(scaler.get("with_mean", True)):
        encoded = encoded - mean
    if bool(scaler.get("with_std", True)):
        encoded = encoded / scale
    return encoded
