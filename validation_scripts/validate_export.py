"""
validate_export.py
------------------
Validates: geovisxrd/export/

Responsibilities:
  - build_geo_table: schema, column set, shape
  - save_geo_export: CSV + GeoPackage output
  - save_qgis_export: QGIS-ready CSV output

Note: model training and SHAP computation are prerequisites for obtaining
valid (X, y_true, y_pred, shap_values) inputs to the export functions.
They are not the validation target here. No plots are generated.

Dataset: California Housing (full dataset — has Latitude & Longitude).
Output:  validation_outputs/export/
"""

import os
import sys
import shutil
from sklearn.datasets import fetch_california_housing

sys.path.insert(0, os.path.dirname(__file__))
from _fmt import header, section, inp, out, ok, warn, skip, build, artifact_summary, W

OUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "validation_outputs", "export")
)
OUT_GEO  = os.path.join(OUT_ROOT, "geo")
OUT_QGIS = os.path.join(OUT_ROOT, "qgis")
OUT_GPKG = os.path.join(OUT_ROOT, "gpkg")

# Clear and recreate output directory on each run.
if os.path.isdir(OUT_ROOT):
    shutil.rmtree(OUT_ROOT)
for d in [OUT_GEO, OUT_QGIS, OUT_GPKG]:
    os.makedirs(d, exist_ok=True)

# ── Header ─────────────────────────────────────────────────────────────────────
header(
    script     = "geovisxrd/export/",
    purpose    = "Build geo table and save geo / QGIS / GeoPackage exports",
    upstream   = "California Housing dataset  (sklearn built-in, no upstream files)",
    downstream = [
        f"{OUT_GEO}/geo_export_<ts>.csv      — geo table with SHAP columns",
        f"{OUT_GPKG}/geo_export_<ts>.gpkg    — GeoPackage (if geopandas available)",
        f"{OUT_QGIS}/qgis_export_<ts>.csv    — QGIS-ready CSV with geometry",
    ],
)

# ── Data ───────────────────────────────────────────────────────────────────────
section(1, "Data + prerequisite model/SHAP  (not the validation target)")

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

inp(f"California Housing  rows={len(X)}  features={X.shape[1]}")
inp(f"Coordinate columns: Latitude={'Latitude' in X.columns}  "
    f"Longitude={'Longitude' in X.columns}")

import geovisxrd
geovisxrd.setup_logging()

from geovisxrd import train_model
from geovisxrd.explaining.explainer import shapexplainer
from geovisxrd import build_geo_table, save_geo_export, save_qgis_export

build("Training XGBoost  (n_estimators=50)  — prerequisite, not validation target")
model, _ = train_model("xgb", X, y, n_estimators=50)
y_pred = model.predict(X)
ok("XGBoost trained and predictions generated")

build("Computing SHAP values  — prerequisite, not validation target")
_, shap_values = shapexplainer(model, X)
ok(f"SHAP values shape={shap_values.shape}")


# ── build_geo_table ───────────────────────────────────────────────────────────
section(2, "build_geo_table — schema and column validation")
inp(f"X  shape={X.shape}")
inp(f"y_true  len={len(y)},  y_pred  len={len(y_pred)}")
inp(f"shap_values  shape={shap_values.shape}")

df_geo   = build_geo_table(X, y.values, y_pred, shap_values)
shap_cols = [c for c in df_geo.columns if c.startswith("shap_")]
ok(f"geo table built  shape={df_geo.shape}")
ok(f"shap_* columns ({len(shap_cols)}): {shap_cols}")

assert shap_cols, "build_geo_table produced no shap_* columns"
assert "Latitude"  in df_geo.columns, "Latitude column missing"
assert "Longitude" in df_geo.columns, "Longitude column missing"
ok("Schema validated: shap_* + Latitude + Longitude all present")

print(f"\n       Columns ({len(df_geo.columns)}):")
for c in df_geo.columns:
    print(f"         {c}")


# ── save_geo_export ───────────────────────────────────────────────────────────
section(3, "save_geo_export — CSV + GeoPackage")
inp(f"geo table  shape={df_geo.shape}")

df_out, csv_path, gpkg_path = save_geo_export(
    X, y.values, y_pred, shap_values,
    save_dir=OUT_GEO,
    name_prefix="geo_export",
)
out("geo export CSV", csv_path)

if gpkg_path and os.path.isfile(gpkg_path):
    import shutil as _sh
    gpkg_dest = os.path.join(OUT_GPKG, os.path.basename(gpkg_path))
    _sh.move(gpkg_path, gpkg_dest)
    gpkg_path = gpkg_dest
    out("GeoPackage (.gpkg)", gpkg_path)
else:
    skip("GeoPackage — not produced  (geopandas may be unavailable)")

assert os.path.isfile(csv_path), f"CSV not written: {csv_path}"
ok("geo_export CSV exists on disk")


# ── save_qgis_export ──────────────────────────────────────────────────────────
section(4, "save_qgis_export — QGIS-ready CSV")
inp(f"geo table  shape={df_geo.shape}")

df_qgis, qgis_csv = save_qgis_export(
    X, y.values, y_pred, shap_values,
    save_dir=OUT_QGIS,
    name_prefix="qgis_export",
)
out("QGIS CSV", qgis_csv)

print(f"\n       Columns ({len(df_qgis.columns)}):")
for c in df_qgis.columns:
    print(f"         {c}")

assert os.path.isfile(qgis_csv), f"QGIS CSV not written: {qgis_csv}"
ok("QGIS CSV exists on disk")


# ── Artifact summary ───────────────────────────────────────────────────────────
outputs_written = []
for root, dirs, files in os.walk(OUT_ROOT):
    dirs.sort()
    for f in sorted(files):
        fp  = os.path.join(root, f)
        rel = os.path.relpath(fp, OUT_ROOT)
        if "geo_export" in f and f.endswith(".csv"):
            desc = "— geo table CSV with SHAP columns (for validate_mapping.py)"
        elif f.endswith(".gpkg"):
            desc = "— GeoPackage for QGIS / GIS tools"
        elif "qgis" in f:
            desc = "— QGIS-ready CSV with geometry column"
        else:
            desc = ""
        outputs_written.append(f"{rel}  {desc}")

artifact_summary(
    inputs_used = [
        "California Housing dataset  (sklearn.datasets.fetch_california_housing)",
        f"  {len(X)} rows × {X.shape[1]} features  (full dataset)",
        "  XGBoost model + SHAP values  (built internally as prerequisites)",
    ],
    outputs_written = outputs_written,
    reusable_by = [
        f"validate_mapping.py  — loads {OUT_GEO}/geo_export_*.csv to skip retraining",
    ],
)
