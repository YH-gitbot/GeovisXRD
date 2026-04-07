# src/geovisxrd/explaining/io.py
"""
SHAP results persistence utilities.

Provides save_shap_results() and load_shap_results() for writing and
reading explainer/SHAP/feature-matrix bundles to/from disk (joblib binary
+ human-readable CSV).
"""

import joblib
import os
from datetime import datetime
import pandas as pd

from .._logging import get_logger
logger = get_logger(__name__)


def save_shap_results(explainer, shap_values, X, save_dir="shap_results", name_prefix="shap"):
    """Save SHAP results to disk as both a binary joblib bundle and a CSV.

    The joblib file stores the full bundle (explainer + shap_values + X) for
    reuse in downstream analysis.  The CSV provides a human-readable table of
    original features alongside their shap_* columns.

    Parameters
    ----------
    explainer : SHAP explainer object
    shap_values : np.ndarray or list of arrays
        SHAP values; shape (n_samples, n_features) for regression.
        For classification models (list), the positive class is used.
    X : pd.DataFrame
        Original feature matrix.
    save_dir : str, optional
        Output directory; created if absent (default 'shap_results').
    name_prefix : str, optional
        Filename prefix, e.g. model name or layer tag (default 'shap').

    Returns
    -------
    str
        Path to the saved .joblib file.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    joblib_filename = f"{name_prefix}_{timestamp}.joblib"
    joblib_path     = os.path.join(save_dir, joblib_filename)

    joblib.dump({"explainer": explainer, "shap_values": shap_values, "X": X}, joblib_path)
    logger.info(f"SHAP bundle (binary) saved to: {joblib_path}")

    # CSV in a csv_data/ subdirectory for easy inspection.
    csv_dir  = os.path.join(save_dir, "csv_data")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{name_prefix}_{timestamp}.csv")

    try:
        # For classification models (list), use the positive class.
        values_arr = (shap_values[1] if len(shap_values) > 1 else shap_values[0]) \
                     if isinstance(shap_values, list) else shap_values

        cols      = X.columns if hasattr(X, "columns") else [f"feature_{i}" for i in range(values_arr.shape[1])]
        shap_cols = [f"shap_{c}" for c in cols]
        df_X      = X.reset_index(drop=True) if hasattr(X, "reset_index") else pd.DataFrame(X)
        df_out    = pd.concat([df_X, pd.DataFrame(values_arr, columns=shap_cols)], axis=1)

        df_out.to_csv(csv_path, index=False)
        logger.info(f"SHAP CSV (features + shap_* columns) saved to: {csv_path}")

    except Exception as e:
        logger.error(f"CSV save failed: {e}")

    return joblib_path


def load_shap_results(filepath):
    """Load a SHAP bundle saved by save_shap_results().

    Parameters
    ----------
    filepath : str
        Path to the .joblib bundle.

    Returns
    -------
    dict
        Keys: 'explainer', 'shap_values', 'X'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Loading SHAP results: {filepath}")
    data = joblib.load(filepath)
    logger.info("SHAP results loaded.")
    return data
