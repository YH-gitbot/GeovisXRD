"""
validate_mapping.py
-------------------
Validates: geovisxrd/plotting/mapping.py

Responsibilities:
  - plot_shap_single: single-panel SHAP map
  - plot_shap_6panel: six-panel SHAP atlas

Input: geo table from validate_export.py (reused if available).
If no export CSV is found, the geo table is generated from scratch as
a data-preparation step — this is not the validation target.

Dataset: California Housing (full dataset).
Output:  validation_outputs/mapping/
"""

import os
import sys
import glob
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _fmt import (header, section, inp, out, ok, warn, skip,
                  reuse, build, artifact_summary, W)

OUT_ROOT   = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "validation_outputs", "mapping")
)
OUT_SINGLE = os.path.join(OUT_ROOT, "single")
OUT_PANELS = os.path.join(OUT_ROOT, "panels")
EXPORT_GEO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "validation_outputs", "export", "geo")
)

BOUNDARY_FILE = "/Users/shumouren/Downloads/GeoVisXRD/reference/california/boundary/california_boundary.geojson"
BASEMAP_FILE  = "/Users/shumouren/Downloads/GeoVisXRD/reference/california/basemap/california_osm.tif"

# Clear and recreate output directory on each run.
if os.path.isdir(OUT_ROOT):
    shutil.rmtree(OUT_ROOT)
for d in [OUT_SINGLE, OUT_PANELS]:
    os.makedirs(d, exist_ok=True)

# ── Header ─────────────────────────────────────────────────────────────────────
header(
    script     = "geovisxrd/plotting/mapping.py",
    purpose    = "Render single-panel and six-panel SHAP maps for California Housing",
    upstream   = [
        f"{EXPORT_GEO}/geo_export_*.csv  (from validate_export.py)  [REUSE if present]",
        "  or  California Housing + XGBoost + SHAP  [BUILD if CSV not found]",
        f"  {BOUNDARY_FILE}",
        f"  {BASEMAP_FILE}",
    ],
    downstream = [
        f"{OUT_SINGLE}/map_shap_single_<feature>.png — single-panel SHAP map",
        f"{OUT_PANELS}/map_shap_6panel.png           — six-panel SHAP atlas",
    ],
)

# ── GeoVisXRD imports ─────────────────────────────────────────────────────────
import geovisxrd
geovisxrd.setup_logging()

from geovisxrd.plotting.mapping import plot_shap_single, plot_shap_6panel


# ── Section 1 — Prepare geo table ─────────────────────────────────────────────
section(1, "Geo table acquisition  (reuse or build)")

csv_candidates = glob.glob(os.path.join(EXPORT_GEO, "geo_export_*.csv"))
geo_csv_source = None

if csv_candidates:
    geo_csv_source = sorted(csv_candidates)[-1]
    reuse(f"Existing geo export CSV  (skipping model/SHAP rebuild)", geo_csv_source)
    import pandas as pd
    df_geo = pd.read_csv(geo_csv_source)
    ok(f"Loaded  shape={df_geo.shape}")
else:
    build(
        "No upstream geo export CSV found — generating geo table from scratch.\n"
        "  Run validate_export.py first to skip this step."
    )
    from sklearn.datasets import fetch_california_housing
    from geovisxrd import train_model, build_geo_table
    from geovisxrd.explaining.explainer import shapexplainer

    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    inp(f"California Housing  rows={len(X)}  features={X.shape[1]}")

    model, _ = train_model("xgb", X, y, n_estimators=50)
    y_pred   = model.predict(X)
    _, shap_values = shapexplainer(model, X)
    df_geo   = build_geo_table(X, y.values, y_pred, shap_values)
    ok(f"Geo table built from scratch  shape={df_geo.shape}")

inp(f"geo table  shape={df_geo.shape}")
inp(f"columns: {list(df_geo.columns)}")

LAT = "Latitude"
LON = "Longitude"

if LAT not in df_geo.columns or LON not in df_geo.columns:
    raise RuntimeError(
        f"Coordinate columns '{LAT}'/'{LON}' not in geo table. "
        f"Available: {list(df_geo.columns)}"
    )

shap_cols = [c for c in df_geo.columns if c.startswith("shap_")]
if not shap_cols:
    raise RuntimeError("No shap_* columns found in geo table.")

ranked_shap_cols = sorted(
    shap_cols, key=lambda c: df_geo[c].abs().mean(), reverse=True
)[:6]

print(f"\n  Top {len(ranked_shap_cols)} SHAP columns by mean |SHAP|:")
for c in ranked_shap_cols:
    print(f"    {c:<30s}  mean|SHAP|={df_geo[c].abs().mean():.4f}")

# ── Section 2 — Check reference files ─────────────────────────────────────────
section(2, "Reference file availability")

for label, path in [("Boundary GeoJSON", BOUNDARY_FILE),
                    ("Basemap TIFF",     BASEMAP_FILE)]:
    if os.path.isfile(path):
        inp(f"{label}", path)
    else:
        warn(f"{label} not found — maps may render without overlay\n"
             f"             path checked: {path}")


# ── Section 3 — plot_shap_single ──────────────────────────────────────────────
section(3, "plot_shap_single — single-panel SHAP map")

top_col    = ranked_shap_cols[0]
single_png = os.path.join(OUT_SINGLE, f"map_shap_single_{top_col}.png")

inp(f"geo table  shape={df_geo.shape}")
inp(f"SHAP column: {top_col}  (top by mean |SHAP|)")

try:
    plot_shap_single(
        df_geo,
        shap_col=top_col,
        lat_col=LAT,
        lon_col=LON,
        boundary_path=BOUNDARY_FILE,
        basemap_path=BASEMAP_FILE,
        basemap_alpha=0.6,
        title=f"California Housing — SHAP: {top_col}",
        save_path=single_png,
    )
    out("Single-panel SHAP map", single_png)
except Exception as e:
    skip(f"plot_shap_single — {type(e).__name__}: {e}")


# ── Section 4 — plot_shap_6panel ──────────────────────────────────────────────
section(4, "plot_shap_6panel — six-panel SHAP atlas")

atlas_png = os.path.join(OUT_PANELS, "map_shap_6panel.png")

inp(f"geo table  shape={df_geo.shape}")
inp(f"SHAP columns ({len(ranked_shap_cols)}): {ranked_shap_cols}")

try:
    plot_shap_6panel(
        df_geo,
        shap_vars=ranked_shap_cols,
        lat_col=LAT,
        lon_col=LON,
        boundary_path=BOUNDARY_FILE,
        basemap_path=BASEMAP_FILE,
        basemap_alpha=0.6,
        title="California Housing — SHAP Atlas",
        save_path=atlas_png,
    )
    out("Six-panel SHAP atlas", atlas_png)
except Exception as e:
    skip(f"plot_shap_6panel — {type(e).__name__}: {e}")


# ── Artifact summary ───────────────────────────────────────────────────────────
if geo_csv_source:
    inputs_used = [
        f"geo export CSV  [REUSE]  → {geo_csv_source}",
        f"  (produced by validate_export.py — no rebuild performed)",
    ]
else:
    inputs_used = [
        "California Housing dataset  [BUILD]  (generated geo table from scratch)",
        "  Run validate_export.py first to reuse its output instead.",
    ]
inputs_used += [
    f"Boundary GeoJSON: {BOUNDARY_FILE}",
    f"Basemap TIFF:     {BASEMAP_FILE}",
]

outputs_written = []
for root, dirs, files in os.walk(OUT_ROOT):
    dirs.sort()
    for f in sorted(files):
        fp  = os.path.join(root, f)
        rel = os.path.relpath(fp, OUT_ROOT)
        if "single" in rel:
            desc = "— single-panel SHAP map"
        elif "6panel" in f:
            desc = "— six-panel SHAP atlas"
        else:
            desc = ""
        outputs_written.append(f"{rel}  {desc}")

if not outputs_written:
    outputs_written = ["(none — all map plots were skipped)"]

artifact_summary(
    inputs_used     = inputs_used,
    outputs_written = outputs_written,
    reusable_by     = ["(map outputs are terminal — no downstream scripts consume them)"],
)
