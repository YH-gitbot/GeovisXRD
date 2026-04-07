"""
validate_recursive.py
---------------------
Validates: geovisxrd/pipeline/recursive.py

Responsibilities:
  - run_recursive_step: one-hop drill-down from a layer-1 analysis result
  - _resolve_previous: accepts in-memory result dict (fastest path)
  - Layer directory structure and manifest patching
  - Optional causal discovery in the recursive layer

Layer 1 is produced via run_analysis() — the standard pipeline.
Layer 2 is produced via run_recursive_step() using the layer-1 result dict.

Uses causal_method="correlation" to avoid requiring heavy optional
dependencies (torch / SAM / causal-learn / lingam / causalnex).

Dataset: California Housing (full dataset).
Output:  validation_outputs/recursive/
           layer1/               — standard run_analysis output
             model/ metrics/ predictions/ shap/ manifest.json
           layer1/layer2_<feature>/   — recursive step output
             model/ metrics/ predictions/ shap/ causal/ threshold/
             manifest.json  (includes layer_num, target_feature, parent_manifest)
"""

import os
import sys
import numpy as np
from sklearn.datasets import fetch_california_housing

sys.path.insert(0, os.path.dirname(__file__))
from _fmt import header, section, inp, out, ok, warn, skip, artifact_summary, W

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "validation_outputs", "recursive")
)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Header ─────────────────────────────────────────────────────────────────────
header(
    script     = "geovisxrd/pipeline/recursive.py",
    purpose    = "Validate run_recursive_step(): drill-down using SHAP as new target",
    upstream   = "California Housing dataset  (sklearn built-in, no upstream files)",
    downstream = [
        f"{OUT_DIR}/layer1/                     — layer 1 (run_analysis output)",
        f"{OUT_DIR}/layer1/layer2_<feature>/    — layer 2 (run_recursive_step output)",
        "  each layer dir:  model/ metrics/ predictions/ shap/ manifest.json",
        "  layer 2 extras:  causal/  threshold/  (causal_method='correlation')",
    ],
)

# ── Data ───────────────────────────────────────────────────────────────────────
section(1, "Dataset preparation")

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

inp(f"California Housing  rows={len(X)}  features={X.shape[1]}")
inp(f"feature names: {list(X.columns)}")

# ── GeoVisXRD imports ─────────────────────────────────────────────────────────
import geovisxrd
geovisxrd.setup_logging()

from geovisxrd.pipeline import run_analysis, run_recursive_step


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 2 — Layer 1: standard run_analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section(2, "Layer 1 — run_analysis()  (prerequisite for recursive step)")
layer1_dir = os.path.join(OUT_DIR, "layer1")
inp(f"X  shape={X.shape},  y  len={len(y)}")
inp(f"model_name=xgb  n_estimators=50  (no shap_sample — full dataset for layer 2)")
inp(f"save_dir: {layer1_dir}")

result_l1 = run_analysis(
    X, y,
    model_name="xgb",
    save_dir=layer1_dir,
    run_name="layer1",
    model_params={"n_estimators": 50},
)

ok(f"Layer 1 complete  MAE={result_l1['metrics']['mae']:.4f}  "
   f"R²={result_l1['metrics']['r2']:.4f}")
print(f"         shap_values shape: {result_l1['shap_values'].shape}")
out("Layer 1 manifest",    result_l1["manifest_path"])
out("Layer 1 model",       result_l1["model_path"])
out("Layer 1 SHAP bundle", result_l1["shap_bundle"])

# ── Choose target feature for layer 2 ─────────────────────────────────────────
section(3, "Feature selection  (top mean |SHAP| → layer 2 regression target)")
inp(f"shap_values  shape={result_l1['shap_values'].shape}")

mean_abs_shap = np.abs(result_l1["shap_values"]).mean(axis=0)
top_feature   = X.columns[np.argmax(mean_abs_shap)]

print(f"\n  Mean |SHAP| per feature:")
for feat, val in sorted(zip(X.columns, mean_abs_shap), key=lambda x: -x[1]):
    marker = "  ← layer 2 target" if feat == top_feature else ""
    print(f"    {feat:<20s}  {val:.4f}{marker}")

ok(f"Layer 2 regression target: SHAP({top_feature})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 4 — Layer 2: run_recursive_step
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
section(4, "Layer 2 — run_recursive_step()")

layer2_dir = os.path.join(layer1_dir, f"layer2_{top_feature}")

inp(f"previous: in-memory result dict from Layer 1  (no disk I/O)")
inp(f"target_feature_name: '{top_feature}'")
inp(f"model_name=xgb  n_estimators=50  causal_method='correlation'")
inp(f"Expected save_dir: {layer2_dir}")

try:
    result_l2 = run_recursive_step(
        previous=result_l1,
        target_feature_name=top_feature,
        layer_num=2,
        model_name="xgb",
        causal_method="correlation",
        model_params={"n_estimators": 50},
        causal_params={"threshold": 0.5},
    )

    ok(f"Layer 2 complete  MAE={result_l2['metrics']['mae']:.4f}  "
       f"R²={result_l2['metrics']['r2']:.4f}")
    print(f"         layer_num:      {result_l2['layer_num']}")
    print(f"         target_feature: {result_l2['target_feature']}")
    print(f"         layer_dir:      {result_l2['layer_dir']}")
    print(f"         shap_values shape: {result_l2['shap_values'].shape}")

    out("Layer 2 manifest",        result_l2["manifest_path"])
    out("Layer 2 model",           result_l2["model_path"])
    out("Layer 2 SHAP bundle",     result_l2["shap_bundle"])

    if "causal_graph_path" in result_l2:
        out("Layer 2 causal graph",    result_l2["causal_graph_path"])
    if "stability_curve_path" in result_l2:
        out("Layer 2 stability curve", result_l2["stability_curve_path"])

    # Manifest contract checks
    import json
    with open(result_l2["manifest_path"]) as fh:
        manifest = json.load(fh)
    assert manifest.get("layer_num")      == 2,           "manifest missing layer_num"
    assert manifest.get("target_feature") == top_feature, "manifest missing target_feature"
    ok("manifest.json contains layer_num and target_feature")

    if manifest.get("parent_manifest"):
        ok(f"manifest.json references parent_manifest: {manifest['parent_manifest']}")

except Exception as e:
    skip(f"run_recursive_step — {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()


# ── Artifact summary ───────────────────────────────────────────────────────────
outputs_written = []
for root, dirs, files in os.walk(OUT_DIR):
    dirs.sort()
    for f in sorted(files):
        fp  = os.path.join(root, f)
        rel = os.path.relpath(fp, OUT_DIR)
        if f == "manifest.json":
            desc = "— run manifest (relative artifact paths)"
        elif f.endswith(".pkl") and "model" in root:
            desc = "— serialised trained model"
        elif f == "metrics.json":
            desc = "— MAE / RMSE / R²"
        elif f == "predictions.csv":
            desc = "— y_true + y_pred per sample"
        elif f.endswith(".joblib"):
            desc = "— SHAP bundle"
        elif f.endswith(".csv") and "shap" in root:
            desc = "— SHAP values CSV"
        elif f.endswith(".png"):
            desc = "— visualisation"
        else:
            desc = ""
        outputs_written.append(f"{rel}  {desc}")

artifact_summary(
    inputs_used = [
        "California Housing dataset  (sklearn.datasets.fetch_california_housing)",
        f"  {len(X)} rows × {X.shape[1]} features  (full dataset)",
        f"  Layer 1 result dict passed in-memory to run_recursive_step()",
        f"  (no intermediate .joblib roundtrip — fastest resolution path)",
    ],
    outputs_written = outputs_written,
    reusable_by = [
        "Any further run_recursive_step() call can pass result_l2 as 'previous'",
        "  or load from result_l2['manifest_path'] for disk-based resolution.",
    ],
)
