"""Central project paths and experiment configuration."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RAW_DATA_PATH = DATA_RAW_DIR / "insurance.csv"
TARGET_COLUMN = "charges"
NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5
ZERO_TOL = 1e-4

ALPHA_RIDGE = 10.0
ALPHA_LASSO = 100.0
ALPHA_ELASTICNET = 100.0
ELASTICNET_L1_RATIO = 0.9998
MAX_ITER = 20_000

MODEL_ORDER = ["OLS", "Ridge", "Lasso", "ElasticNet"]
MODEL_LABELS = {
    "OLS": "Linear Regression (OLS)",
    "Ridge": "Ridge (L2, alpha=10)",
    "Lasso": "Lasso (L1, alpha=100)",
    "ElasticNet": "ElasticNet (alpha=100, l1_ratio=0.9998)",
}
MODEL_FILENAMES = {
    "OLS": "linear_regression.json",
    "Ridge": "ridge_model.json",
    "Lasso": "lasso_model.json",
    "ElasticNet": "elasticnet_model.json",
}


def ensure_project_directories() -> None:
    """Create directories used for generated data, models and reports."""
    for path in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_SAMPLES_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        TABLES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
