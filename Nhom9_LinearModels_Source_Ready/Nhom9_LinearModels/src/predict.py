"""CSV inference using JSON model/preprocessor metadata (no pickle/joblib)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODEL_ORDER, MODELS_DIR
from .persistence import load_model_bundle
from .preprocess import transform_with_metadata


def predict_dataframe(
    input_frame: pd.DataFrame,
    models_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Predict charges with all four JSON models from raw customer feature columns."""
    preprocessor_meta, models = load_model_bundle(models_dir or MODELS_DIR)
    transformed = transform_with_metadata(input_frame, preprocessor_meta)
    output = input_frame.copy()
    for name in MODEL_ORDER:
        model = models[name]
        coefficients = np.array([model["coefficients"][feature] for feature in model["feature_names"]], dtype=float)
        intercept = float(model["intercept"])
        output[f"predicted_charges_{name.lower()}"] = transformed @ coefficients + intercept
    return output


def predict_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    models_dir: str | Path | None = None,
) -> Path:
    """Read a customer CSV, run all four models, and write predictions to CSV."""
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy CSV đầu vào: {input_path}")
    frame = pd.read_csv(input_path)
    predictions = predict_dataframe(frame, models_dir=models_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dự đoán charges bằng 4 mô hình JSON của Nhóm 9.")
    parser.add_argument("--input", required=True, help="Đường dẫn CSV khách hàng đầu vào.")
    parser.add_argument("--output", required=True, help="Đường dẫn CSV lưu kết quả dự đoán.")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="Thư mục chứa các file JSON mô hình.")
    args = parser.parse_args()
    result_path = predict_csv(args.input, args.output, args.models_dir)
    print(f"Đã lưu dự đoán tại: {result_path}")


if __name__ == "__main__":
    main()
