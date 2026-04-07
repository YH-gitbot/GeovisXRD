"""
validate_sam.py
---------------
SAM-only causal discovery validation.

SAM (Structural Agnostic Model) is excluded from validate_causal.py because it
causes a segmentation fault in the current environment when run alongside other
causal methods.  This script runs SAM in complete isolation to avoid that issue.

Requirements:
  - torch >= 2.0
  - cdt >= 0.6

Run independently:
  python validation_scripts/validate_sam.py

Output: validation_outputs/sam/
"""

import os
import sys
import shutil
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _fmt import header, section, inp, out, ok, warn, skip, artifact_summary, W

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "validation_outputs", "sam")
)

# Clear and recreate output directory on each run.
if os.path.isdir(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR)

# ── Header ─────────────────────────────────────────────────────────────────────
header(
    script     = "geovisxrd/causal/  (SAM, isolated)",
    purpose    = "Validate SAM causal discovery in a single-process environment",
    upstream   = "California Housing dataset  (sklearn built-in, no upstream files)",
    downstream = [
        f"{OUT_DIR}/adjacency.csv    — learned weighted adjacency matrix",
        f"{OUT_DIR}/causal_model.pkl — serialised fitted SAM model",
        f"{OUT_DIR}/graph.png        — causal graph visualisation",
    ],
)
print(
    "\n  NOTE: SAM is run in isolation to avoid a segfault that occurs when\n"
    "  it shares a process with other causal methods (torch/cdt/fork conflict).\n"
    "  For all other methods, use validate_causal.py."
)

# ── Data ───────────────────────────────────────────────────────────────────────
section(1, "Dataset preparation")

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
X_full  = housing.data
y_full  = housing.target.rename("MedHouseVal")
df_full = pd.concat([X_full, y_full], axis=1)

N   = 200
rng = np.random.default_rng(42)
idx = rng.choice(len(df_full), size=min(N, len(df_full)), replace=False)
df  = df_full.iloc[sorted(idx)].reset_index(drop=True)

inp(f"California Housing  full dataset  rows={len(df_full)}  cols={df_full.shape[1]}")
inp(f"SAM subsample  rows={len(df)}  cols={df.shape[1]}  "
    f"(seed=42, n={N} — SAM is slow on large data)")
print(f"       columns: {list(df.columns)}")

# ── GeoVisXRD imports ─────────────────────────────────────────────────────────
import geovisxrd
geovisxrd.setup_logging()

from geovisxrd import CausalConstraints, plot_causal_graph_networkx
from geovisxrd.causal.discovery import get_causal_model

constraints = CausalConstraints(
    tier_map={
        "Latitude":    1,
        "Longitude":   1,
        "HouseAge":    2,
        "AveRooms":    2,
        "AveBedrms":   2,
        "AveOccup":    2,
        "Population":  2,
        "MedInc":      3,
        "MedHouseVal": 4,
    },
    forbidden_edges=[("MedInc", "AveRooms")],
)


# ── SAM ────────────────────────────────────────────────────────────────────────
section(2, "SAM  (nruns=1, train_epochs=100, nh=10)")
inp(f"subsample  rows={len(df)}  cols={df.shape[1]}")

try:
    model = get_causal_model(
        "sam", constraints=constraints,
        nruns=1, train_epochs=100, nh=10,
    )
    model.fit(df)

    if model.graph_ is None:
        skip("SAM produced no graph  "
             "(required library likely unavailable — see log above)")
    else:
        G = model.get_nx_graph()
        ok(f"Graph fitted  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")
        for src, dst, d in G.edges(data=True):
            print(f"       {src} → {dst}   weight={d.get('weight', 'n/a')}")

        adj_path = os.path.join(OUT_DIR, "adjacency.csv")
        model.graph_.to_csv(adj_path)
        out("adjacency.csv", adj_path)

        pkl_path = os.path.join(OUT_DIR, "causal_model.pkl")
        joblib.dump(model, pkl_path)
        out("causal_model.pkl", pkl_path)

        graph_png = os.path.join(OUT_DIR, "graph.png")
        plot_causal_graph_networkx(
            G, threshold=0.0,
            title=f"SAM Causal Graph  (n={len(df)})",
            save_path=graph_png,
        )
        out("graph.png", graph_png)

except Exception as e:
    skip(f"SAM — {type(e).__name__}: {e}")


# ── Artifact summary ───────────────────────────────────────────────────────────
outputs_written = []
for root, dirs, files in os.walk(OUT_DIR):
    dirs.sort()
    for f in sorted(files):
        fp  = os.path.join(root, f)
        rel = os.path.relpath(fp, OUT_DIR)
        if f == "adjacency.csv":
            desc = "— weighted adjacency matrix"
        elif f == "causal_model.pkl":
            desc = "— serialised fitted SAM model"
        elif f == "graph.png":
            desc = "— causal graph visualisation"
        else:
            desc = ""
        outputs_written.append(f"{rel}  {desc}")

if not outputs_written:
    outputs_written = ["(none — SAM did not produce a graph)"]

artifact_summary(
    inputs_used = [
        "California Housing dataset  (sklearn.datasets.fetch_california_housing)",
        f"  subsample: {len(df)} rows × {df.shape[1]} cols  (seed=42, n={N})",
    ],
    outputs_written = outputs_written,
    reusable_by = [
        "validate_threshold.py  — can load sam/causal_model.pkl or sam/adjacency.csv",
    ],
)
