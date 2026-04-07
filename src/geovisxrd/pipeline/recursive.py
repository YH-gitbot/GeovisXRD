# src/geovisxrd/pipeline/recursive.py
"""
Recursive (layered) analysis workflow.

Provides run_recursive_step(), which drills one layer deeper by treating
the SHAP contribution of a chosen feature as the new regression target.

Design principle
----------------
This module adds only the layer-specific logic on top of run_analysis():
  1. Load previous layer's X and shap_values.
  2. Build new_y = SHAP column for target_feature_name.
  3. Call run_analysis() — no analysis logic is re-implemented here.
  4. Optionally run causal discovery + threshold stability.
  5. Patch the layer manifest with recursive metadata.

Input resolution
----------------
``previous`` accepts any of:
  - dict  — in-memory result from run_analysis() or a previous
            run_recursive_step() call (fastest; no disk I/O)
  - str   — path to a manifest.json file
  - str   — path to a layer/run directory (looks for manifest.json inside)
  - str   — path to a .joblib SHAP bundle (legacy fallback)

Artifact structure produced under <parent_dir>/layer<N>_<feature>/
───────────────────────────────────────────────────────────────────
  model/
  metrics/
  predictions/
  shap/
  causal/       ← only when causal_method is given
  threshold/    ← only when causal_method is given
  manifest.json ← standard manifest + layer_num, target_feature,
                  parent_manifest (relative path)
"""

import json
import os

import pandas as pd

from geovisxrd.explaining.io import load_shap_results
from geovisxrd.pipeline.analysis import run_analysis

from geovisxrd._logging import get_logger
logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Input resolution
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_previous(source):
    """Resolve the previous layer's data from any supported source type.

    Returns
    -------
    X : pd.DataFrame
    shap_values : np.ndarray
    parent_dir : str          absolute path to the previous layer's save_dir
    parent_manifest : str or None   absolute path to the parent manifest.json
    """
    if isinstance(source, dict):
        # In-memory result from run_analysis() or run_recursive_step()
        return (
            source["X_shap"],
            source["shap_values"],
            source["save_dir"],
            source.get("manifest_path"),
        )

    if not isinstance(source, str):
        raise TypeError(
            f"'previous' must be a result dict or a str path; got {type(source)}."
        )

    source = os.path.abspath(source)

    # Legacy path: raw .joblib SHAP bundle (no manifest)
    if source.endswith(".joblib"):
        bundle = load_shap_results(source)
        return bundle["X"], bundle["shap_values"], os.path.dirname(source), None

    # Directory → look for manifest inside
    if os.path.isdir(source):
        manifest_path = os.path.join(source, "manifest.json")
    elif source.endswith("manifest.json"):
        manifest_path = source
    else:
        raise ValueError(
            f"Cannot resolve previous layer from: {source}\n"
            "Expected: result dict, manifest.json path, layer directory, "
            "or .joblib SHAP bundle."
        )

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    parent_dir       = manifest["save_dir"]
    shap_bundle_rel  = manifest["artifacts"]["shap_bundle"]
    shap_bundle_abs  = os.path.join(parent_dir, shap_bundle_rel)
    bundle           = load_shap_results(shap_bundle_abs)
    return bundle["X"], bundle["shap_values"], parent_dir, manifest_path


def _auto_layer_num(parent_dir: str) -> int:
    """Return the next available layer number in parent_dir."""
    existing = [
        d for d in os.listdir(parent_dir)
        if d.startswith("layer") and os.path.isdir(os.path.join(parent_dir, d))
    ]
    return len(existing) + 1


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_recursive_step(
    previous,
    target_feature_name: str,
    *,
    layer_num: int = None,
    model_name: str = "xgb",
    causal_method: str = None,
    model_params: dict = None,
    causal_params: dict = None,
) -> dict:
    """Execute one GeoVisXRD recursive analysis step.

    Loads the previous layer's SHAP results, uses SHAP(target_feature_name)
    as the new regression target, runs the standard analysis pipeline, and
    optionally runs causal discovery + threshold stability on the result.

    Parameters
    ----------
    previous : dict | str
        Source of the previous layer's artifacts.  Accepted forms:
          - in-memory result dict from run_analysis() / run_recursive_step()
          - str path to manifest.json
          - str path to a layer/run directory (must contain manifest.json)
          - str path to a .joblib SHAP bundle  (legacy fallback)
    target_feature_name : str
        Feature whose SHAP column becomes the new regression target.
    layer_num : int or None, optional
        Layer index for directory naming.  Auto-increments from the number of
        existing layer directories when None.
    model_name : str, optional
        Regressor identifier (default 'xgb').
    causal_method : str or None, optional
        Causal discovery algorithm: 'pc', 'notears', 'lingam', 'correlation'.
        When None (default), causal analysis is skipped entirely.
    model_params : dict or None, optional
        Extra keyword arguments forwarded to train_model().
    causal_params : dict or None, optional
        Extra keyword arguments forwarded to get_causal_model().

    Returns
    -------
    dict
        Full result dict from run_analysis() with the following extra keys:
          'layer_num'            — int
          'target_feature'       — str
          'layer_dir'            — str, absolute path to this layer's save_dir
          'parent_manifest'      — str or None
          'causal_model'         — fitted model   (only when causal_method given)
          'causal_graph_path'    — str             (only when causal_method given)
          'stability_curve_path' — str             (only when causal_method given)
    """
    if model_params is None:
        model_params = {}
    if causal_params is None:
        causal_params = {}

    # ── Resolve input ─────────────────────────────────────────────────────────
    X, prev_shap, parent_dir, parent_manifest = _resolve_previous(previous)

    if layer_num is None:
        layer_num = _auto_layer_num(parent_dir)

    layer_dir  = os.path.join(parent_dir, f"layer{layer_num}_{target_feature_name}")
    run_name   = f"layer{layer_num}_{target_feature_name}"

    # ── Build new regression target from previous SHAP ────────────────────────
    if target_feature_name not in X.columns:
        raise ValueError(
            f"Feature '{target_feature_name}' not found in X. "
            f"Available: {list(X.columns)}"
        )
    feat_idx = list(X.columns).index(target_feature_name)
    new_y = pd.Series(
        prev_shap[:, feat_idx],
        index=X.index,
        name=f"SHAP_{target_feature_name}",
    )

    logger.info(
        f"Layer {layer_num} — recursive step: target=SHAP({target_feature_name}), "
        f"{len(new_y)} samples, model={model_name}."
    )

    # ── Standard analysis (train + SHAP + save + manifest) ───────────────────
    result = run_analysis(
        X, new_y,
        model_name=model_name,
        save_dir=layer_dir,
        run_name=run_name,
        model_params=model_params,
    )

    # ── Annotate result with recursive metadata ───────────────────────────────
    result["layer_num"]       = layer_num
    result["target_feature"]  = target_feature_name
    result["layer_dir"]       = layer_dir
    result["parent_manifest"] = parent_manifest

    # Patch manifest with recursive context
    with open(result["manifest_path"]) as fh:
        manifest = json.load(fh)
    manifest["layer_num"]      = layer_num
    manifest["target_feature"] = target_feature_name
    if parent_manifest:
        manifest["parent_manifest"] = os.path.relpath(parent_manifest, layer_dir)
    with open(result["manifest_path"], "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    # ── Optional: causal discovery + threshold stability ─────────────────────
    if causal_method is not None:
        from geovisxrd.causal.discovery import get_causal_model
        from geovisxrd.threshold.optimizer import analyze_threshold_stability
        from geovisxrd.plotting.causal_plots import plot_causal_graph_networkx

        causal_dir    = os.path.join(layer_dir, "causal")
        threshold_dir = os.path.join(layer_dir, "threshold")
        os.makedirs(causal_dir, exist_ok=True)
        os.makedirs(threshold_dir, exist_ok=True)

        logger.info(
            f"Layer {layer_num} — causal discovery ({causal_method}) on "
            f"{X.shape[1] + 1} variables."
        )
        combined     = pd.concat([X, new_y.to_frame()], axis=1)
        causal_model = get_causal_model(causal_method, **causal_params)
        causal_model.fit(combined)

        stability_path = os.path.join(threshold_dir, "stability_curve.png")
        _, best_t = analyze_threshold_stability(
            causal_model, save_path=stability_path,
        )
        if best_t is None:
            logger.warning(
                f"Layer {layer_num} — threshold stability returned no result; "
                "using threshold=0.0."
            )
            best_t = 0.0

        causal_graph_path = os.path.join(causal_dir, "causal_graph.png")
        plot_causal_graph_networkx(
            causal_model.get_nx_graph(),
            threshold=best_t,
            title=f"Layer {layer_num} — {target_feature_name} (t={best_t})",
            save_path=causal_graph_path,
        )

        # Update manifest with causal artifact paths
        def _rel(p):
            return os.path.relpath(p, layer_dir)

        with open(result["manifest_path"]) as fh:
            manifest = json.load(fh)
        manifest["artifacts"]["causal_graph"]     = _rel(causal_graph_path)
        manifest["artifacts"]["stability_curve"]  = _rel(stability_path)
        with open(result["manifest_path"], "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)

        result["causal_model"]          = causal_model
        result["causal_graph_path"]     = causal_graph_path
        result["stability_curve_path"]  = stability_path

    logger.info(
        f"Layer {layer_num} — complete. Manifest: {result['manifest_path']}"
    )
    return result
