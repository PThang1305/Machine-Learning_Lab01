"""Factories for the four scikit-learn regression estimators."""
from __future__ import annotations

from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from .config import (
    ALPHA_ELASTICNET,
    ALPHA_LASSO,
    ALPHA_RIDGE,
    ELASTICNET_L1_RATIO,
    MAX_ITER,
    MODEL_ORDER,
)


def build_model(model_name: str):
    """Create one unfitted estimator using the group's agreed parameters."""
    name = model_name.strip()
    factories = {
        "OLS": LinearRegression,
        "Ridge": lambda: Ridge(alpha=ALPHA_RIDGE),
        "Lasso": lambda: Lasso(alpha=ALPHA_LASSO, max_iter=MAX_ITER),
        "ElasticNet": lambda: ElasticNet(
            alpha=ALPHA_ELASTICNET,
            l1_ratio=ELASTICNET_L1_RATIO,
            max_iter=MAX_ITER,
        ),
    }
    if name not in factories:
        raise ValueError(f"Model không được hỗ trợ: {model_name}. Chọn một trong {MODEL_ORDER}.")
    return factories[name]()


def build_models() -> dict[str, object]:
    """Create all four unfitted estimators in the agreed display order."""
    return {name: build_model(name) for name in MODEL_ORDER}
