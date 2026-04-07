# src/geovisxrd/modeling/trainer.py
"""
Model training and evaluation utilities.

Provides train_model() for fitting a named regressor on a train/test split
and computing MAE, RMSE, and R² metrics.  Metrics can optionally be appended
to a CSV file for multi-run tracking.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .model_factory import get_model

from .._logging import get_logger
logger = get_logger(__name__)


def train_model(
    model_name,
    X,
    y,
    test_size=0.2,
    random_state=1,
    return_metrics=True,
    metrics_save_dir=None,
    **model_params
):
    """Train a regression model and evaluate it on a held-out test set.

    Parameters
    ----------
    model_name : str
        Regressor identifier; see get_model() for accepted values.
    X : pd.DataFrame
        Feature matrix (n_samples × n_features).
    y : pd.Series
        Regression target.
    test_size : float, optional
        Fraction of data held out for evaluation (default 0.2).
    random_state : int, optional
        Random seed for reproducibility (default 1).
    return_metrics : bool, optional
        When False, return only the model without metrics (default True).
    metrics_save_dir : str or None, optional
        Directory for appending results to model_metrics.csv;
        pass None to skip (default None).
    **model_params
        Extra keyword arguments forwarded to the model constructor.

    Returns
    -------
    model : fitted estimator
    metrics : dict
        Keys: 'model', 'mae', 'rmse', 'r2'.
        Only returned when return_metrics=True.

    Notes
    -----
    MAE  — mean absolute error; average absolute deviation from the true value.
    RMSE — root mean squared error; penalises large errors more than MAE.
    R²   — coefficient of determination; range (-∞, 1], where 1 is a perfect fit.
    """
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of samples: {len(X)} != {len(y)}"
        )
    if not (0 < test_size < 1):
        raise ValueError(f"test_size must be strictly between 0 and 1, got {test_size}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Training {model_name.upper()} — "
        f"{len(X_train)} train / {len(X_test)} test samples, "
        f"{X.shape[1]} features."
    )
    model = get_model(model_name, **model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5   # sqrt(MSE) = RMSE
    r2   = r2_score(y_test, y_pred)

    metrics = {"model": model_name, "mae": mae, "rmse": rmse, "r2": r2}

    logger.info(f"{model_name.upper()} — MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

    if metrics_save_dir is not None:
        _append_metrics_csv(metrics, metrics_save_dir)

    if return_metrics:
        return model, metrics
    else:
        return model


def _append_metrics_csv(metrics: dict, save_dir: str):
    """Append one row of metrics to <save_dir>/model_metrics.csv."""
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "model_metrics.csv")

    df_new = pd.DataFrame([metrics])

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(csv_path, index=False)
    logger.info(f"Metrics saved to: {csv_path}")
