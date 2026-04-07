# src/geovisxrd/pipeline/analysis.py
"""
Standard end-to-end analysis workflow.

Provides run_analysis(), which orchestrates:
  modeling/   → train_model, save_model
  explaining/ → shapexplainer, save_shap_results
  export/     → save_geo_export            (optional)
  plotting/   → plot_all_shap_charts       (optional)

Artifact structure produced under <save_dir>/
─────────────────────────────────────────────
  model/
    <model>_<timestamp>.pkl
  metrics/
    metrics.json
  predictions/
    predictions.csv
  shap/
    <model>_<timestamp>.joblib     ← full bundle: explainer + shap_values + X_shap
    csv_data/
      <model>_<timestamp>.csv      ← human-readable SHAP table
  geo/                             ← only when export_geo=True
  plots/                           ← only when generate_plots=True
  manifest.json                    ← relative-path index of every artifact

The manifest uses relative paths inside save_dir so the output tree is portable.
All artifact paths are also returned as absolute paths in the result dict.
"""

import json
import os

import numpy as np
import pandas as pd

from geovisxrd.modeling.trainer import train_model
from geovisxrd.modeling.save import save_model
from geovisxrd.explaining.explainer import shapexplainer
from geovisxrd.explaining.io import save_shap_results

from geovisxrd._logging import get_logger
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_manifest(save_dir: str, data: dict) -> str:
    """Write manifest.json and return its absolute path."""
    path = os.path.join(save_dir, "manifest.json")
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    return path


def _shap_csv_from_bundle(bundle_path: str) -> str:
    """Derive the CSV path from a SHAP .joblib bundle path.

    save_shap_results() stores the CSV under:
      <bundle_dir>/csv_data/<bundle_stem>.csv
    """
    stem = os.path.basename(bundle_path).replace(".joblib", ".csv")
    return os.path.join(os.path.dirname(bundle_path), "csv_data", stem)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    X,
    y,
    model_name: str = "xgb",
    *,
    save_dir: str,
    run_name: str = None,
    model_params: dict = None,
    shap_sample: int = None,
    export_geo: bool = False,
    generate_plots: bool = False,
) -> dict:
    """Run the standard GeoVisXRD analysis pipeline on a single dataset.

    Sequences: train → evaluate → predict → SHAP → save artifacts →
    write manifest → (optional) geo export → (optional) SHAP plots.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (n_samples × n_features).
    y : pd.Series
        Regression target.
    model_name : str, optional
        Regressor identifier accepted by train_model() (default 'xgb').
    save_dir : str
        Root directory for all outputs.  Sub-directories are created
        automatically.
    run_name : str or None, optional
        Human-readable run identifier recorded in the manifest.
        Defaults to the basename of save_dir.
    model_params : dict or None, optional
        Extra keyword arguments forwarded to train_model().
    shap_sample : int or None, optional
        Number of rows for SHAP computation; None uses all rows.
    export_geo : bool, optional
        Build geo table and save CSV under ``<save_dir>/geo/`` (default False).
    generate_plots : bool, optional
        Generate standard SHAP chart set under ``<save_dir>/plots/``
        (default False).

    Returns
    -------
    dict
        In-memory results and absolute artifact paths:
          'model'            — fitted estimator
          'metrics'          — {'mae', 'rmse', 'r2'}
          'predictions'      — pd.DataFrame with y_true / y_pred columns
          'explainer'        — SHAP explainer object
          'shap_values'      — np.ndarray (n_shap_rows × n_features)
          'X_shap'           — pd.DataFrame subsample used for SHAP
          'save_dir'         — str, absolute save_dir
          'manifest_path'    — str, path to manifest.json
          'model_path'       — str
          'metrics_path'     — str
          'predictions_path' — str
          'shap_bundle'      — str, path to .joblib bundle
          'shap_csv'         — str, path to human-readable CSV
          'geo_df'           — pd.DataFrame   (only when export_geo=True)
          'geo_csv'          — str            (only when export_geo=True)
          'plots_dir'        — str            (only when generate_plots=True)
    """
    if model_params is None:
        model_params = {}

    save_dir = os.path.abspath(save_dir)
    if run_name is None:
        run_name = os.path.basename(save_dir)

    model_dir   = os.path.join(save_dir, "model")
    metrics_dir = os.path.join(save_dir, "metrics")
    preds_dir   = os.path.join(save_dir, "predictions")
    shap_dir    = os.path.join(save_dir, "shap")
    for d in (model_dir, metrics_dir, preds_dir, shap_dir):
        os.makedirs(d, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info(f"run_analysis [{run_name}] — training {model_name.upper()}.")
    model, metrics = train_model(model_name, X, y, **model_params)
    logger.info(
        f"run_analysis [{run_name}] — {model_name.upper()}  "
        f"MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
        f"R²={metrics['r2']:.4f}"
    )

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = save_model(model, model_name, save_dir=model_dir)

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_path = os.path.join(metrics_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    # ── Save predictions ──────────────────────────────────────────────────────
    y_pred   = model.predict(X)
    preds_df = pd.DataFrame(
        {"y_true": y.values, "y_pred": y_pred},
        index=X.index,
    )
    preds_path = os.path.join(preds_dir, "predictions.csv")
    preds_df.to_csv(preds_path, index=True)

    # ── SHAP subsample ────────────────────────────────────────────────────────
    if shap_sample is not None and shap_sample < len(X):
        rng    = np.random.default_rng(42)
        idx    = rng.choice(len(X), size=shap_sample, replace=False)
        X_shap = X.iloc[sorted(idx)].reset_index(drop=True)
    else:
        X_shap = X

    # ── SHAP ──────────────────────────────────────────────────────────────────
    logger.info(
        f"run_analysis [{run_name}] — computing SHAP ({len(X_shap)} rows)."
    )
    explainer, shap_values = shapexplainer(model, X_shap)
    shap_bundle = save_shap_results(
        explainer, shap_values, X_shap,
        save_dir=shap_dir, name_prefix=model_name,
    )
    shap_csv = _shap_csv_from_bundle(shap_bundle)

    # ── Manifest (initial) ────────────────────────────────────────────────────
    def _rel(p):
        return os.path.relpath(p, save_dir)

    manifest = {
        "run_name":   run_name,
        "model_name": model_name,
        "save_dir":   save_dir,
        "artifacts": {
            "model":       _rel(model_path),
            "metrics":     _rel(metrics_path),
            "predictions": _rel(preds_path),
            "shap_bundle": _rel(shap_bundle),
            "shap_csv":    _rel(shap_csv),
        },
    }

    result = {
        # In-memory
        "model":       model,
        "metrics":     metrics,
        "predictions": preds_df,
        "explainer":   explainer,
        "shap_values": shap_values,
        "X_shap":      X_shap,
        # Paths
        "save_dir":         save_dir,
        "model_path":       model_path,
        "metrics_path":     metrics_path,
        "predictions_path": preds_path,
        "shap_bundle":      shap_bundle,
        "shap_csv":         shap_csv,
    }

    # ── Optional: geo export ──────────────────────────────────────────────────
    if export_geo:
        from geovisxrd.export.geo_export import save_geo_export
        geo_dir = os.path.join(save_dir, "geo")
        os.makedirs(geo_dir, exist_ok=True)
        logger.info(f"run_analysis [{run_name}] — building geo table.")
        geo_df, geo_csv, _ = save_geo_export(
            X, y.values, y_pred, shap_values,
            save_dir=geo_dir, name_prefix=model_name,
        )
        manifest["artifacts"]["geo_csv"] = _rel(geo_csv)
        result["geo_df"]  = geo_df
        result["geo_csv"] = geo_csv

    # ── Optional: SHAP plots ──────────────────────────────────────────────────
    if generate_plots:
        from geovisxrd.explaining.io import load_shap_results
        from geovisxrd.plotting.shap_plots import plot_all_shap_charts
        plots_dir = os.path.join(save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        logger.info(f"run_analysis [{run_name}] — generating SHAP plots.")
        try:
            bundle_data = load_shap_results(shap_bundle)
            plot_all_shap_charts(bundle_data, save_dir=plots_dir)
            manifest["artifacts"]["plots_dir"] = _rel(plots_dir)
            result["plots_dir"] = plots_dir
        except Exception as exc:
            logger.warning(f"run_analysis [{run_name}] — plot generation failed: {exc}")

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest_path = _write_manifest(save_dir, manifest)
    result["manifest_path"] = manifest_path
    logger.info(f"run_analysis [{run_name}] — manifest: {manifest_path}")

    return result
