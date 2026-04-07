"""
validate_causal.py
------------------
Validates: geovisxrd/causal/

Responsibilities:
  True causal discovery algorithms that learn causal structure from data.
  Each produces a directed graph of hypothesised causal relationships.

    pc      — PC constraint-based algorithm (requires causal-learn)
    notears — NOTEARS differentiable structure learning (requires causalnex)
    lingam  — DirectLiNGAM, ICA-based (requires lingam)

  SAM is excluded from the default flow due to environment instability;
  use validation_scripts/validate_sam.py to test it in isolation.

  Threshold segmentation and Jaccard stability analysis of the learned
  graphs are validated separately in validate_threshold.py.

Note: causal graph visualisation uses plotting/causal_plots.py
(plot_causal_graph_networkx) as a natural output step.

Output structure:
  validation_outputs/causal/
    <method>/
      adjacency.csv      — learned adjacency matrix
      causal_model.pkl   — serialised model object
      graph.png          — causal graph visualisation
"""

import os
import sys
import shutil
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _fmt import (header, section, part, inp, out, ok, warn, skip,
                  artifact_summary, W)

# ── Configuration ──────────────────────────────────────────────────────────────
CAUSAL_METHODS = [
    {"name": "pc",      "params": {"alpha": 0.05}},
    {"name": "notears", "params": {"threshold": 0.1}},
    {"name": "lingam",  "params": {}},
]

METHOD_N = {
    "pc":      1000,
    "notears": 500,
    "lingam":  1000,
}

OUT_CAUSAL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "validation_outputs", "causal")
)

if os.path.isdir(OUT_CAUSAL):
    shutil.rmtree(OUT_CAUSAL)
os.makedirs(OUT_CAUSAL, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────────
from sklearn.datasets import fetch_california_housing

housing  = fetch_california_housing(as_frame=True)
X_full   = housing.data
y_full   = housing.target.rename("MedHouseVal")
df_full  = pd.concat([X_full, y_full], axis=1)

# ── Header ─────────────────────────────────────────────────────────────────────
header(
    script     = "geovisxrd/causal/",
    purpose    = "Fit causal discovery models and save adjacency + graph artifacts",
    upstream   = "California Housing dataset  (sklearn built-in, no upstream files)",
    downstream = [
        f"{OUT_CAUSAL}/<method>/adjacency.csv    — learned weighted adjacency matrix",
        f"{OUT_CAUSAL}/<method>/causal_model.pkl — serialised fitted causal model",
        f"{OUT_CAUSAL}/<method>/graph.png        — causal graph visualisation",
        "  ↳ consumed by: validate_threshold.py",
    ],
)

print(f"\n  Dataset : California Housing  (joined with target)")
inp(f"rows={df_full.shape[0]}  cols={df_full.shape[1]}")
inp(f"columns: {list(df_full.columns)}")

# ── GeoVisXRD imports ─────────────────────────────────────────────────────────
import geovisxrd
geovisxrd.setup_logging()

from geovisxrd import CausalConstraints
from geovisxrd.causal.discovery import get_causal_model
from geovisxrd import calculate_jaccard_similarity, plot_causal_graph_networkx

# ── Domain-knowledge constraints ──────────────────────────────────────────────
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

forbidden_all = constraints.get_forbidden_edges(list(df_full.columns))
print(f"\n  Constraints: {len(constraints.tier_map)} variables  "
      f"|  {len(forbidden_all)} total forbidden edges")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Part 1 — Causal discovery methods
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
part("PART 1 — CAUSAL DISCOVERY METHODS")

discovered = {}   # method_name → nx.DiGraph (for pairwise Jaccard below)

for cfg in CAUSAL_METHODS:
    method     = cfg["name"]
    params     = cfg["params"]
    n          = METHOD_N.get(method, 500)
    method_dir = os.path.join(OUT_CAUSAL, method)
    os.makedirs(method_dir, exist_ok=True)

    print(f"\n── {method.upper()}  {'─' * (W - len(method) - 4)}")
    print(f"     subsample n={n}  |  params={params}")

    try:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(df_full), size=min(n, len(df_full)), replace=False)
        df  = df_full.iloc[sorted(idx)].reset_index(drop=True)
        inp(f"subsample  rows={len(df)}  cols={df.shape[1]}  (seed=42)")

        model = get_causal_model(method, constraints=constraints, **params)
        model.fit(df)

        if model.graph_ is None:
            skip(f"{method.upper()} — fit produced no graph "
                 "(required library likely unavailable; see log above)")
            continue

        G = model.get_nx_graph()
        discovered[method] = G

        ok(f"Graph fitted  nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")
        for src, dst, d in G.edges(data=True):
            print(f"       {src} → {dst}   weight={d.get('weight', 'n/a')}")

        violations = [(s, d) for s, d in G.edges() if (s, d) in forbidden_all]
        if violations:
            warn(f"Forbidden edge violations: {violations}")
        else:
            ok("No forbidden edge violations")

        adj_path = os.path.join(method_dir, "adjacency.csv")
        model.graph_.to_csv(adj_path)
        out("adjacency.csv", adj_path)

        pkl_path = os.path.join(method_dir, "causal_model.pkl")
        joblib.dump(model, pkl_path)
        out("causal_model.pkl", pkl_path)

        graph_png = os.path.join(method_dir, "graph.png")
        plot_causal_graph_networkx(
            G, threshold=0.0,
            title=f"{method.upper()} Causal Graph  (n={n})",
            save_path=graph_png,
        )
        out("graph.png", graph_png)

    except Exception as e:
        skip(f"{method.upper()} — {type(e).__name__}: {e}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Part 2 — Pairwise Jaccard similarity
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
part("PART 2 — PAIRWISE JACCARD SIMILARITY BETWEEN DISCOVERED GRAPHS")

if len(discovered) >= 2:
    methods = list(discovered.keys())
    inp(f"{len(methods)} successfully discovered graphs: {methods}")
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            ma, mb  = methods[i], methods[j]
            j_score = calculate_jaccard_similarity(discovered[ma], discovered[mb])
            print(f"  Jaccard({ma}, {mb}) = {j_score:.4f}")
else:
    skip(f"Pairwise Jaccard — fewer than 2 graphs discovered "
         f"({len(discovered)} available)")


# ── Artifact summary ───────────────────────────────────────────────────────────
outputs_written = []
for root, dirs, files in os.walk(OUT_CAUSAL):
    dirs.sort()
    for f in sorted(files):
        fp  = os.path.join(root, f)
        rel = os.path.relpath(fp, OUT_CAUSAL)
        if f == "adjacency.csv":
            desc = "— weighted adjacency matrix"
        elif f == "causal_model.pkl":
            desc = "— serialised fitted causal model"
        elif f == "graph.png":
            desc = "— causal graph visualisation"
        else:
            desc = ""
        outputs_written.append(f"{rel}  {desc}")

artifact_summary(
    inputs_used = [
        "California Housing dataset  (sklearn.datasets.fetch_california_housing)",
        f"  full dataset: {df_full.shape[0]} rows × {df_full.shape[1]} cols",
        f"  per-method subsamples: pc/lingam n={METHOD_N['pc']},  "
        f"notears n={METHOD_N['notears']}  (seed=42)",
    ],
    outputs_written = outputs_written,
    reusable_by = [
        f"validate_threshold.py  — loads {OUT_CAUSAL}/<method>/causal_model.pkl "
        "or adjacency.csv",
    ],
)
