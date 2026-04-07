"""
validate_explaining.py
----------------------
Validates: geovisxrd/explaining/  +  geovisxrd/plotting/shap_plots.py

Responsibilities:
  - SHAP computation (explaining/explainer.py — shapexplainer)
  - SHAP bundle save/load round-trip (explaining/io.py)
  - Standard SHAP chart set (plotting/shap_plots.py — plot_all_shap_charts)
  - Summary plots: bar, beeswarm, pos/neg ratio
  - Self-dependence plots (SHAP(f) vs raw values of f, LOWESS)
  - Cross-dependence plots (SHAP(f_A) vs raw values of f_B, LOWESS)
  - 3-D interaction plots (SHAP dependence coloured by a third feature)
  - Multi-model dependence comparison

Note: one XGBoost model is trained here as a prerequisite for SHAP computation.
Model training is not the validation target — use validate_modeling.py for that.

Output structure:
  validation_outputs/explaining/
    shap_data/
    plots/
      summary/
      dependence/
        self/
        cross/
        interaction/
      comparison/
"""

import os
import sys
import shutil
import numpy as np
from sklearn.datasets import fetch_california_housing

sys.path.insert(0, os.path.dirname(__file__))
from _fmt import header, section, inp, out, ok, warn, skip, build, artifact_summary, W

# ── Configuration ──────────────────────────────────────────────────────────────
SELF_DEP_FEATURES = [
    "MedInc", "AveOccup", "HouseAge", "Latitude", "Longitude", "AveRooms",
]

CROSS_DEP_PAIRS = [
    ("AveOccup", "MedInc"),
    ("Latitude",  "MedInc"),
    ("HouseAge",  "MedInc"),
    ("MedInc",    "Latitude"),
    ("MedInc",    "AveOccup"),
]

INTERACTION_TRIPLES = [
    ("MedInc",   "MedInc",   "Latitude"),
    ("MedInc",   "MedInc",   "AveOccup"),
    ("AveOccup", "AveOccup", "MedInc"),
    ("Latitude", "Latitude", "Longitude"),
    ("HouseAge", "HouseAge", "MedInc"),
]

COMPARE_MODELS   = ["xgb", "linear"]
COMPARE_FEATURES = ["MedInc", "AveOccup", "Latitude"]

SHAP_SAMPLE = 1000

OUT_ROOT      = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "validation_outputs", "explaining")
)
shap_data_dir = os.path.join(OUT_ROOT, "shap_data")
plots_dir     = os.path.join(OUT_ROOT, "plots")
summary_dir   = os.path.join(plots_dir, "summary")
dep_self_dir  = os.path.join(plots_dir, "dependence", "self")
dep_cross_dir = os.path.join(plots_dir, "dependence", "cross")
dep_3d_dir    = os.path.join(plots_dir, "dependence", "interaction")
compare_dir   = os.path.join(plots_dir, "comparison")

# ── Clear and recreate output directory ───────────────────────────────────────
if os.path.isdir(OUT_ROOT):
    shutil.rmtree(OUT_ROOT)
for d in [shap_data_dir, summary_dir, dep_self_dir, dep_cross_dir,
          dep_3d_dir, compare_dir]:
    os.makedirs(d, exist_ok=True)

# ── Header ─────────────────────────────────────────────────────────────────────
header(
    script     = "geovisxrd/explaining/  +  geovisxrd/plotting/shap_plots.py",
    purpose    = "SHAP computation, bundle IO round-trip, and all SHAP plot types",
    upstream   = "California Housing dataset  (sklearn built-in, no upstream files)",
    downstream = [
        f"{shap_data_dir}/xgb_<ts>.joblib   — SHAP bundle (explainer + values + X)",
        f"{shap_data_dir}/csv_data/xgb_<ts>.csv — human-readable SHAP table",
        f"{plots_dir}/                         — all SHAP visualisations",
    ],
)

# ── Data ───────────────────────────────────────────────────────────────────────
housing = fetch_california_housing(as_frame=True)
X_full  = housing.data
y_full  = housing.target

print(f"\n  Dataset : California Housing")
inp(f"full dataset  rows={len(X_full)}  features={X_full.shape[1]}")

rng = np.random.default_rng(42)
idx = rng.choice(len(X_full), size=min(SHAP_SAMPLE, len(X_full)), replace=False)
X   = X_full.iloc[sorted(idx)].reset_index(drop=True)
y   = y_full.iloc[sorted(idx)].reset_index(drop=True)
inp(f"SHAP subsample  rows={len(X)}  (seed=42, size={SHAP_SAMPLE})")

# ── GeoVisXRD imports ─────────────────────────────────────────────────────────
import geovisxrd
geovisxrd.setup_logging()

from geovisxrd import train_model, save_shap_results
from geovisxrd.explaining.explainer import shapexplainer
from geovisxrd.explaining.io import load_shap_results
from geovisxrd.plotting import (
    plot_all_shap_charts,
    plot_summary_bar,
    plot_beeswarm,
    plot_pos_neg_ratio,
    plot_dependence_2d_lowess,
    plot_dependence_3d_interaction,
)


def _safe(fn, *args, label="", save_path=None, **kwargs):
    """Call a plot function; log and continue on any failure."""
    try:
        fn(*args, save_path=save_path, **kwargs)
        out(label, save_path)
    except Exception as e:
        skip(f"{label} — {type(e).__name__}: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 1 — SHAP computation + bundle IO round-trip
# Validates: explaining/explainer.py, explaining/io.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section(1, "SHAP computation + bundle IO round-trip")
inp(f"California Housing subsample  ({len(X)} rows × {X.shape[1]} features)")

build("Training XGBoost prerequisite  (n_estimators=100)  — not the validation target")
xgb_model, xgb_metrics = train_model("xgb", X_full, y_full, n_estimators=100)
print(f"         MAE={xgb_metrics['mae']:.4f}  R²={xgb_metrics['r2']:.4f}")

xgb_explainer, xgb_shap = shapexplainer(xgb_model, X)
ok(f"SHAP values computed  shape={xgb_shap.shape}")

xgb_bundle_path = save_shap_results(
    xgb_explainer, xgb_shap, X,
    save_dir=shap_data_dir, name_prefix="xgb",
)
out("SHAP bundle (.joblib)", xgb_bundle_path)

xgb_bundle = load_shap_results(xgb_bundle_path)
assert xgb_bundle["shap_values"].shape == xgb_shap.shape, "IO round-trip shape mismatch"
ok("Bundle save/load round-trip  (shape preserved)")

# Derive CSV path (written alongside the joblib by save_shap_results)
xgb_csv_stem = os.path.basename(xgb_bundle_path).replace(".joblib", ".csv")
xgb_csv_path = os.path.join(shap_data_dir, "csv_data", xgb_csv_stem)
if os.path.isfile(xgb_csv_path):
    out("SHAP CSV (human-readable)", xgb_csv_path)

feat_names       = list(X.columns)
model_shap_cache = {"xgb": xgb_shap}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 2 — Summary plots
# Validates: plot_summary_bar, plot_beeswarm, plot_pos_neg_ratio,
#            plot_all_shap_charts (swarm + bar + waterfall)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section(2, "Summary plots")
inp(f"xgb SHAP values  shape={xgb_shap.shape}")

_safe(plot_summary_bar, xgb_shap, feat_names,
      label="xgb_shap_bar.png",
      save_path=os.path.join(summary_dir, "xgb_shap_bar.png"))

_safe(plot_beeswarm, xgb_shap, X,
      label="xgb_shap_beeswarm.png",
      save_path=os.path.join(summary_dir, "xgb_shap_beeswarm.png"))

_safe(plot_pos_neg_ratio, xgb_shap, feat_names,
      label="xgb_pos_neg_ratio.png",
      save_path=os.path.join(summary_dir, "xgb_pos_neg_ratio.png"))

all_charts_dir = os.path.join(summary_dir, "all_charts")
try:
    plot_all_shap_charts(xgb_bundle, save_dir=all_charts_dir)
    out(f"all_charts/ directory  (swarm + bar + waterfall)", all_charts_dir)
except Exception as e:
    skip(f"all_charts — {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 3 — Self-dependence plots (LOWESS)
# Validates: plot_dependence_2d_lowess (self mode)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section(3, f"Self-dependence plots  ({len(SELF_DEP_FEATURES)} features)")
print(f"       SHAP(f) vs raw(f) with LOWESS trend")
inp(f"xgb SHAP values  shape={xgb_shap.shape}")

for feat in SELF_DEP_FEATURES:
    if feat not in feat_names:
        skip(f"{feat} — not in feature set")
        continue
    _safe(plot_dependence_2d_lowess, xgb_shap, X,
          x_feature=feat,
          label=f"dep_{feat}.png",
          save_path=os.path.join(dep_self_dir, f"dep_{feat}.png"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 4 — Cross-dependence plots (LOWESS)
# Validates: plot_dependence_2d_lowess (cross mode)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section(4, f"Cross-dependence plots  ({len(CROSS_DEP_PAIRS)} pairs)")
print(f"       X axis = raw(x_feature),  Y axis = SHAP(y_feature)")
inp(f"xgb SHAP values  shape={xgb_shap.shape}")

for x_feat, y_feat in CROSS_DEP_PAIRS:
    missing = [f for f in [x_feat, y_feat] if f not in feat_names]
    if missing:
        skip(f"({x_feat} → SHAP({y_feat})) — missing features: {missing}")
        continue
    fname = f"cross_{x_feat}_vs_SHAP_{y_feat}.png"
    _safe(plot_dependence_2d_lowess, xgb_shap, X,
          x_feature=x_feat, y_feature=y_feat,
          label=fname,
          save_path=os.path.join(dep_cross_dir, fname))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 5 — 3-D interaction plots
# Validates: plot_dependence_3d_interaction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section(5, f"3-D interaction plots  ({len(INTERACTION_TRIPLES)} triples)")
print(f"       X = raw(x),  Y = SHAP(y),  colour = raw(interaction)")
inp(f"xgb SHAP values  shape={xgb_shap.shape}")

for x_feat, y_feat, int_feat in INTERACTION_TRIPLES:
    missing = [f for f in [x_feat, y_feat, int_feat] if f not in feat_names]
    if missing:
        skip(f"({x_feat}/{y_feat}/{int_feat}) — missing: {missing}")
        continue
    fname = f"3d_{x_feat}_SHAP{y_feat}_color{int_feat}.png"
    _safe(plot_dependence_3d_interaction, xgb_shap, X,
          x_feature=x_feat, y_feature=y_feat, interaction_feature=int_feat,
          label=fname,
          save_path=os.path.join(dep_3d_dir, fname))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 6 — Multi-model dependence comparison
# Validates: plot_dependence_2d_lowess across different model SHAP outputs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section(6, f"Multi-model comparison  (models={COMPARE_MODELS}  features={COMPARE_FEATURES})")
inp(f"xgb SHAP values already in cache")

for model_name in COMPARE_MODELS:
    if model_name in model_shap_cache:
        ok(f"{model_name.upper()} — reusing SHAP values from Section 1 cache")
        continue
    try:
        build(f"Training {model_name.upper()} for comparison  (not validation target)")
        m, met = train_model(model_name, X_full, y_full)
        _, sv  = shapexplainer(m, X)
        model_shap_cache[model_name] = sv
        ok(f"{model_name.upper()}  MAE={met['mae']:.4f}  R²={met['r2']:.4f}  "
           f"SHAP shape={sv.shape}")
    except Exception as e:
        skip(f"{model_name.upper()} — {type(e).__name__}: {e}")

for feat in COMPARE_FEATURES:
    if feat not in feat_names:
        continue
    print(f"\n       Feature: {feat}")
    for model_name, sv in model_shap_cache.items():
        if model_name not in COMPARE_MODELS:
            continue
        fname = f"compare_{model_name}_{feat}.png"
        _safe(plot_dependence_2d_lowess, sv, X,
              x_feature=feat,
              label=fname,
              save_path=os.path.join(compare_dir, fname))


# ── Artifact summary ───────────────────────────────────────────────────────────
png_count = 0
outputs_written = []
for root, dirs, files in os.walk(OUT_ROOT):
    dirs.sort()
    for f in sorted(files):
        fp  = os.path.join(root, f)
        rel = os.path.relpath(fp, OUT_ROOT)
        if f.endswith(".joblib"):
            desc = "— SHAP bundle  (explainer + shap_values + X)"
        elif f.endswith(".csv"):
            desc = "— SHAP values CSV (human-readable)"
        elif f.endswith(".png"):
            desc = "— SHAP visualisation"
            png_count += 1
        else:
            desc = ""
        outputs_written.append(f"{rel}  {desc}")

artifact_summary(
    inputs_used = [
        "California Housing dataset  (sklearn.datasets.fetch_california_housing)",
        f"  full dataset:    {len(X_full)} rows × {X_full.shape[1]} features",
        f"  SHAP subsample:  {len(X)} rows  (seed=42, size={SHAP_SAMPLE})",
        f"  XGBoost model:   built internally as prerequisite (n_estimators=100)",
    ],
    outputs_written = outputs_written,
    reusable_by = [
        "validate_mapping.py  — may reuse geo table built here (if export not available)",
        "(SHAP bundles are self-contained and loadable by any downstream script)",
    ],
)
