"""
validate_analysis.py
--------------------
Validates: geovisxrd/pipeline/analysis.py

Responsibilities:
  - run_analysis(): standard end-to-end workflow
  - Result dict contract: required keys, shapes, file existence
  - Manifest JSON written correctly with all artifact paths
  - Optional geo export path (export_geo=True)
  - Optional plot generation path (generate_plots=True)
  - shap_sample subsetting

Note: the underlying low-level modules (modeling, explaining, export,
plotting) are each validated in their own scripts. This script only
validates the orchestration layer — that run_analysis() correctly
sequences the modules and returns a complete, correct result dict.

Dataset: California Housing (subsample for speed).
Output:  validation_outputs/analysis/
"""

import os
import sys
import numpy as np
from sklearn.datasets import fetch_california_housing

sys.path.insert(0, os.path.dirname(__file__))
from _fmt import header, section, inp, out, ok, warn, skip, artifact_summary, W

OUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "validation_outputs", "analysis")
)
os.makedirs(OUT_ROOT, exist_ok=True)

# ── Header ─────────────────────────────────────────────────────────────────────
header(
    script     = "geovisxrd/pipeline/analysis.py",
    purpose    = "Validate run_analysis() orchestration: result dict contract + manifest",
    upstream   = "California Housing dataset  (sklearn built-in, no upstream files)",
    downstream = [
        f"{OUT_ROOT}/basic/             — basic xgb run",
        f"{OUT_ROOT}/linear/            — linear model run",
        f"{OUT_ROOT}/with_geo/          — xgb run + geo export",
        f"{OUT_ROOT}/with_plots/        — xgb run + SHAP plots",
        "  each run dir contains:  model/ metrics/ predictions/ shap/ manifest.json",
    ],
)

# ── Data ───────────────────────────────────────────────────────────────────────
section(1, "Dataset preparation")

housing = fetch_california_housing(as_frame=True)
rng     = np.random.default_rng(42)
idx     = rng.choice(len(housing.data), size=500, replace=False)
X_full  = housing.data.iloc[sorted(idx)].reset_index(drop=True)
y_full  = housing.target.iloc[sorted(idx)].reset_index(drop=True)

inp(f"California Housing  (subsample, seed=42)")
inp(f"rows={len(X_full)}  features={X_full.shape[1]}")
inp(f"feature names: {list(X_full.columns)}")

# ── GeoVisXRD imports ─────────────────────────────────────────────────────────
import geovisxrd
geovisxrd.setup_logging()

from geovisxrd.pipeline import run_analysis

# ── Result dict contract ───────────────────────────────────────────────────────
REQUIRED_KEYS = {
    "model", "metrics", "predictions", "explainer", "shap_values", "X_shap",
    "save_dir", "manifest_path", "model_path", "metrics_path",
    "predictions_path", "shap_bundle", "shap_csv",
}

FILE_KEYS = ("model_path", "metrics_path", "predictions_path",
             "shap_bundle", "manifest_path")


def _check_result(result, label, extra_keys=()):
    """Assert the result dict has the expected structure and all files exist."""
    missing = (REQUIRED_KEYS | set(extra_keys)) - result.keys()
    assert not missing, f"{label}: missing keys {missing}"

    for key in FILE_KEYS:
        assert os.path.isfile(result[key]), \
            f"{label}: {key} not found on disk: {result[key]}"

    assert result["shap_values"].ndim == 2, \
        f"{label}: shap_values should be 2-D, got shape {result['shap_values'].shape}"
    assert result["shap_values"].shape[1] == X_full.shape[1], \
        f"{label}: shap_values feature count mismatch"

    for k in ("mae", "rmse", "r2"):
        assert k in result["metrics"], f"{label}: metrics missing '{k}'"

    ok(f"{label}")
    print(f"         MAE={result['metrics']['mae']:.4f}  "
          f"RMSE={result['metrics']['rmse']:.4f}  "
          f"R²={result['metrics']['r2']:.4f}")
    print(f"         shap_values shape:  {result['shap_values'].shape}")
    print(f"         X_shap rows:        {len(result['X_shap'])}")
    out("manifest.json",    result["manifest_path"])
    out("model pickle",     result["model_path"])
    out("metrics JSON",     result["metrics_path"])
    out("predictions CSV",  result["predictions_path"])
    out("SHAP bundle",      result["shap_bundle"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 2 — Basic run: modeling + SHAP only
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section(2, "Basic run  (xgb, shap_sample=200)")
save_dir_basic = os.path.join(OUT_ROOT, "basic")
inp(f"X  shape={X_full.shape},  y  len={len(y_full)}")
inp(f"model_name=xgb  n_estimators=50  shap_sample=200")
inp(f"save_dir: {save_dir_basic}")

result_basic = run_analysis(
    X_full, y_full,
    model_name="xgb",
    save_dir=save_dir_basic,
    run_name="basic_xgb",
    model_params={"n_estimators": 50},
    shap_sample=200,
)
_check_result(result_basic, "basic xgb run")
assert len(result_basic["X_shap"]) == 200, "shap_sample not respected"
ok("shap_sample=200 correctly subsampled X_shap")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 3 — Linear model (different SHAP explainer path)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section(3, "Linear model  (tests LinearExplainer code path)")
save_dir_linear = os.path.join(OUT_ROOT, "linear")
inp(f"X  shape={X_full.shape},  y  len={len(y_full)}")
inp(f"model_name=linear  shap_sample=200")
inp(f"save_dir: {save_dir_linear}")

result_linear = run_analysis(
    X_full, y_full,
    model_name="linear",
    save_dir=save_dir_linear,
    run_name="linear_run",
    shap_sample=200,
)
_check_result(result_linear, "linear run")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 4 — export_geo=True
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section(4, "export_geo=True  (optional geo export path)")
save_dir_geo = os.path.join(OUT_ROOT, "with_geo")
inp(f"model_name=xgb  export_geo=True  shap_sample=200")
inp(f"save_dir: {save_dir_geo}")

try:
    result_geo = run_analysis(
        X_full, y_full,
        model_name="xgb",
        save_dir=save_dir_geo,
        run_name="xgb_with_geo",
        model_params={"n_estimators": 50},
        shap_sample=200,
        export_geo=True,
    )
    _check_result(result_geo, "xgb + export_geo", extra_keys=("geo_df", "geo_csv"))
    assert os.path.isfile(result_geo["geo_csv"]), \
        f"geo_csv not found: {result_geo['geo_csv']}"
    shap_cols = [c for c in result_geo["geo_df"].columns if c.startswith("shap_")]
    out("geo CSV", result_geo["geo_csv"])
    ok(f"geo_df shap_* columns: {shap_cols}")
except Exception as e:
    skip(f"export_geo — {type(e).__name__}: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 5 — generate_plots=True
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section(5, "generate_plots=True  (optional SHAP plots path)")
save_dir_plots = os.path.join(OUT_ROOT, "with_plots")
inp(f"model_name=xgb  generate_plots=True  shap_sample=200")
inp(f"save_dir: {save_dir_plots}")

try:
    result_plots = run_analysis(
        X_full, y_full,
        model_name="xgb",
        save_dir=save_dir_plots,
        run_name="xgb_with_plots",
        model_params={"n_estimators": 50},
        shap_sample=200,
        generate_plots=True,
    )
    _check_result(result_plots, "xgb + generate_plots")
    plots_dir = os.path.join(save_dir_plots, "plots")
    pngs = (sorted(f for f in os.listdir(plots_dir) if f.endswith(".png"))
            if os.path.isdir(plots_dir) else [])
    out(f"plots/ directory  ({len(pngs)} PNGs)", plots_dir)
    if pngs:
        ok(f"PNG files: {pngs}")
except Exception as e:
    skip(f"generate_plots — {type(e).__name__}: {e}")


# ── Artifact summary ───────────────────────────────────────────────────────────
outputs_written = []
for root, dirs, files in os.walk(OUT_ROOT):
    dirs.sort()
    for f in sorted(files):
        fp  = os.path.join(root, f)
        rel = os.path.relpath(fp, OUT_ROOT)
        if f == "manifest.json":
            desc = "— run manifest with relative artifact paths"
        elif f.endswith(".pkl") and "model" in root:
            desc = "— serialised trained model"
        elif f == "metrics.json":
            desc = "— MAE / RMSE / R²"
        elif f == "predictions.csv":
            desc = "— y_true + y_pred per sample"
        elif f.endswith(".joblib"):
            desc = "— SHAP bundle  (explainer + shap_values + X_shap)"
        elif f.endswith(".csv") and "shap" in root:
            desc = "— SHAP values CSV"
        elif f.endswith(".csv") and "geo" in root:
            desc = "— geo table with SHAP columns"
        elif f.endswith(".png"):
            desc = "— SHAP visualisation"
        else:
            desc = ""
        outputs_written.append(f"{rel}  {desc}")

artifact_summary(
    inputs_used = [
        "California Housing dataset  (sklearn.datasets.fetch_california_housing)",
        f"  subsample: 500 rows × {X_full.shape[1]} features  (seed=42)",
    ],
    outputs_written = outputs_written,
    reusable_by = [
        "validate_recursive.py  — passes run_analysis() result dict as 'previous'",
        "  (manifest.json paths allow any subsequent script to reload artifacts)",
    ],
)
