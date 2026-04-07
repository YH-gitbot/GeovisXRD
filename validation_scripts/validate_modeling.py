"""
validate_modeling.py
--------------------
Validates: geovisxrd/modeling/

Responsibilities:
  - Multi-model training and evaluation (train_model)
  - Metrics persistence (metrics.json per model)
  - Model serialisation (save_model)
  - Cross-model comparison summary (model_comparison.csv)

SHAP computation and all plotting are intentionally absent.
Those responsibilities belong in validate_explaining.py.

Models attempted (any that fail are skipped gracefully):
  linear  — LinearRegression (OLS)
  rf      — RandomForestRegressor
  xgb     — XGBRegressor
  lgb     — LGBMRegressor
  mlp     — MLPRegressor

Output structure:
  validation_outputs/modeling/
    metrics/
      <model>.json       — MAE / RMSE / R² per model
    models/
      <model>_<ts>.pkl   — serialised trained model per model
    model_comparison.csv — cross-model summary
"""

import os
import sys
import json
import shutil
import pandas as pd
from sklearn.datasets import fetch_california_housing

sys.path.insert(0, os.path.dirname(__file__))
from _fmt import header, section, inp, out, ok, warn, skip, artifact_summary, W

MODEL_LIST = ["linear", "rf", "xgb", "lgb", "mlp"]

OUT_ROOT    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "validation_outputs", "modeling")
)
OUT_METRICS = os.path.join(OUT_ROOT, "metrics")
OUT_MODELS  = os.path.join(OUT_ROOT, "models")

# ── Header ─────────────────────────────────────────────────────────────────────
header(
    script     = "geovisxrd/modeling/",
    purpose    = "Train, evaluate, and serialise all supported regressor types",
    upstream   = "California Housing dataset  (sklearn built-in, no upstream files)",
    downstream = [
        f"{OUT_METRICS}/<model>.json   — MAE / RMSE / R² per model",
        f"{OUT_MODELS}/<model>_<ts>.pkl — serialised trained model",
        f"{OUT_ROOT}/model_comparison.csv — cross-model summary",
    ],
)

# ── Data ───────────────────────────────────────────────────────────────────────
housing = fetch_california_housing(as_frame=True)
X_full  = housing.data
y_full  = housing.target

print(f"\n  Dataset : California Housing")
inp(f"rows={len(X_full)}  features={X_full.shape[1]}")
inp(f"feature names: {list(X_full.columns)}")

# ── GeoVisXRD imports ─────────────────────────────────────────────────────────
import geovisxrd
geovisxrd.setup_logging()

from geovisxrd import train_model, save_model

# ── Clear and recreate output directories ─────────────────────────────────────
if os.path.isdir(OUT_ROOT):
    shutil.rmtree(OUT_ROOT)
for d in [OUT_METRICS, OUT_MODELS]:
    os.makedirs(d, exist_ok=True)

# ── Per-model runner ───────────────────────────────────────────────────────────
def _run_one_model(name, X, y):
    """Train, evaluate, and save one model. Returns a summary row dict."""
    model, metrics = train_model(name, X, y)
    print(f"       MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
          f"R²={metrics['r2']:.4f}")

    metrics_path = os.path.join(OUT_METRICS, f"{name}.json")
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    out(f"metrics JSON ({name})", metrics_path)

    model_path = save_model(model, name, save_dir=OUT_MODELS)
    out(f"model pickle ({name})", model_path)

    return {"model": name, **metrics}


# ── Main loop ──────────────────────────────────────────────────────────────────
section(1, "Per-model training, evaluation, and serialisation")
inp(f"California Housing  (n={len(X_full)}, p={X_full.shape[1]})")

summary_rows  = []
model_paths   = []
metrics_paths = []

for model_name in MODEL_LIST:
    print(f"\n  ── {model_name.upper()} {'─' * (W - len(model_name) - 5)}")
    try:
        row = _run_one_model(model_name, X_full, y_full)
        summary_rows.append(row)
        ok(f"{model_name.upper()} — training and serialisation complete")
    except Exception as e:
        skip(f"{model_name.upper()} — {type(e).__name__}: {e}")

# ── Cross-model comparison summary ────────────────────────────────────────────
section(2, "Cross-model comparison summary")

comparison_csv = None
if summary_rows:
    df_cmp       = pd.DataFrame(summary_rows)
    comparison_csv = os.path.join(OUT_ROOT, "model_comparison.csv")
    df_cmp.to_csv(comparison_csv, index=False)
    print()
    print(df_cmp.to_string(index=False))
    print()
    out("model_comparison.csv", comparison_csv)
    ok(f"{len(summary_rows)} model(s) included in comparison")
else:
    warn("No models succeeded — comparison CSV not written")

# ── Artifact summary ───────────────────────────────────────────────────────────
inputs_used = [
    "California Housing dataset  (sklearn.datasets.fetch_california_housing)",
    f"  {len(X_full)} rows × {X_full.shape[1]} features  (full dataset, no subsample)",
]

outputs_written = []
for root, dirs, files in os.walk(OUT_ROOT):
    dirs.sort()
    for f in sorted(files):
        fp   = os.path.join(root, f)
        rel  = os.path.relpath(fp, OUT_ROOT)
        desc = ""
        if f.endswith(".json") and "metrics" in root:
            desc = "— per-model metrics (MAE / RMSE / R²)"
        elif f.endswith(".pkl"):
            desc = "— serialised trained model"
        elif f == "model_comparison.csv":
            desc = "— cross-model comparison summary"
        outputs_written.append(f"{rel}  {desc}")

artifact_summary(
    inputs_used    = inputs_used,
    outputs_written= outputs_written,
    reusable_by    = ["validate_explaining.py  (loads any .pkl to skip retraining)"],
)
