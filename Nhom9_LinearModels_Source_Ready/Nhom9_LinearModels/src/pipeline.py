"""End-to-end training/evaluation/export workflow for the group project."""
from __future__ import annotations

import pandas as pd

from .config import CV_FOLDS, FIGURES_DIR, MODELS_DIR, TABLES_DIR, ZERO_TOL, ensure_project_directories
from .evaluate import evaluate_models, save_evaluation_tables
from .persistence import save_model_artifacts
from .preprocess import build_preprocessor, fit_transform_features, prepare_data
from .train import train_models
from .tune import cross_validate_models


def run_pipeline() -> dict:
    """Prepare data, fit four models, evaluate, cross-validate and export JSON/CSV."""
    ensure_project_directories()
    splits = prepare_data(save_processed=True)

    preprocessor = build_preprocessor()
    X_train_transformed, X_test_transformed, feature_names = fit_transform_features(
        preprocessor, splits.X_train, splits.X_test
    )
    models = train_models(X_train_transformed, splits.y_train)

    results, weights, predictions = evaluate_models(
        models,
        X_test_transformed,
        splits.y_test,
        feature_names,
        zero_tol=ZERO_TOL,
    )
    result_files = save_evaluation_tables(results, weights, TABLES_DIR)

    cv_table, fold_r2 = cross_validate_models(
        splits.X_train,
        splits.y_train,
        models=models,
        cv_folds=CV_FOLDS,
    )
    cv_path = TABLES_DIR / "cross_validation_results.csv"
    cv_table.to_csv(cv_path)
    fold_path = TABLES_DIR / "cross_validation_folds.csv"
    fold_r2.to_csv(fold_path)

    model_files = save_model_artifacts(models, preprocessor, feature_names, MODELS_DIR, ZERO_TOL)

    return {
        "splits": splits,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "models": models,
        "results": results,
        "weights": weights,
        "predictions": predictions,
        "cv_table": cv_table,
        "cv_fold_r2": fold_r2,
        "csv_files": [*result_files, cv_path, fold_path],
        "model_files": model_files,
    }


def main() -> None:
    artifacts = run_pipeline()
    print("\n=== KẾT QUẢ TEST ===")
    print(artifacts["results"].round(4).to_string())
    print("\n=== KẾT QUẢ 5-FOLD CV TRÊN TRAIN ===")
    print(artifacts["cv_table"].round(4).to_string())
    print("\nĐã tạo 4 CSV Train/Test trong data/processed.")
    print("Đã tạo các JSON mô hình trong models/.")
    print("Đã tạo bảng CSV trong reports/tables/.")


if __name__ == "__main__":
    main()
