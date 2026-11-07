"""Tests that JSON-based inference reproduces scikit-learn predictions."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import MODEL_ORDER
from src.evaluate import evaluate_models
from src.persistence import save_model_artifacts
from src.predict import predict_dataframe, predict_csv
from src.preprocess import build_preprocessor, fit_transform_features, prepare_data
from src.train import train_models


class JsonPersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.models_dir = Path(cls.temp_dir.name) / "models"
        cls.splits = prepare_data(save_processed=False)
        cls.preprocessor = build_preprocessor()
        cls.X_train_t, cls.X_test_t, cls.feature_names = fit_transform_features(
            cls.preprocessor, cls.splits.X_train, cls.splits.X_test
        )
        cls.models = train_models(cls.X_train_t, cls.splits.y_train)
        save_model_artifacts(
            cls.models,
            cls.preprocessor,
            cls.feature_names,
            output_dir=cls.models_dir,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_json_predictions_match_sklearn(self):
        sample = self.splits.X_test.iloc[:25].reset_index(drop=True)
        actual = predict_dataframe(sample, models_dir=self.models_dir)
        for name in MODEL_ORDER:
            expected = self.models[name].predict(self.X_test_t[:25])
            column = f"predicted_charges_{name.lower()}"
            np.testing.assert_allclose(actual[column].to_numpy(), expected, rtol=0, atol=1e-6)

    def test_unknown_category_matches_encoder_ignore_behavior(self):
        sample = self.splits.X_test.iloc[:3].copy().reset_index(drop=True)
        sample.loc[0, "region"] = "unknown_region"
        actual = predict_dataframe(sample, models_dir=self.models_dir)
        transformed = self.preprocessor.transform(sample)
        for name in MODEL_ORDER:
            expected = self.models[name].predict(transformed)
            column = f"predicted_charges_{name.lower()}"
            np.testing.assert_allclose(actual[column].to_numpy(), expected, rtol=0, atol=1e-6)

    def test_predict_csv_writes_four_prediction_columns(self):
        input_path = Path(self.temp_dir.name) / "input.csv"
        output_path = Path(self.temp_dir.name) / "predictions.csv"
        self.splits.X_test.iloc[:4].to_csv(input_path, index=False)
        predict_csv(input_path, output_path, models_dir=self.models_dir)
        self.assertTrue(output_path.exists())
        import pandas as pd
        result = pd.read_csv(output_path)
        for name in MODEL_ORDER:
            self.assertIn(f"predicted_charges_{name.lower()}", result.columns)
        self.assertEqual(len(result), 4)


if __name__ == "__main__":
    unittest.main()
