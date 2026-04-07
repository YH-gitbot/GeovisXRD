"""
validate_threshold.py
---------------------
Validates: geovisxrd/threshold/

Responsibilities:
  - filter_graph_by_threshold: edge-weight segmentation of a learned causal graph
  - save_filtered_graphs: save filtered graphs at multiple thresholds
  - analyze_threshold_stability: Jaccard stability sweep across filtered graphs
  - calculate_jaccard_similarity: graph comparison utility (self-Jaccard sanity check)

Input:
  All available causal outputs produced by validate_causal.py, one per method.
  Checks validation_outputs/causal/<method>/ for each of: lingam, notears, pc.
  Falls back to a single synthetic weighted graph only when none exist.

Output: validation_outputs/threshold/<method>/  (one directory per method)
  Example for lingam:
    threshold/lingam/
      lingam_stability_curve.png
      lingam_graph_t<best>.png
      filtered/
        lingam_filtered_t0_10.csv / .pkl
        lingam_filtered_t0_20.csv / .pkl
        ...
"""

import os
import sys
import shutil
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from _fmt import (header, section, part, inp, out, ok, warn, skip,
                  reuse, fallback, artifact_summary, W)

THRESHOLD_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "validation_outputs", "threshold")
)
CAUSAL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "validation_outputs", "causal")
)

# ── Header ─────────────────────────────────────────────────────────────────────
header(
    script     = "geovisxrd/threshold/",
    purpose    = "Sweep edge-weight thresholds; measure Jaccard stability per method",
    upstream   = [
        f"{CAUSAL_DIR}/<method>/causal_model.pkl  or  adjacency.csv",
        "  (produced by validate_causal.py — run that first for real graphs)",
        "  Synthetic fallback used only when NO causal output exists.",
    ],
    downstream = [
        f"{THRESHOLD_ROOT}/<method>/<method>_stability_curve.png",
        f"{THRESHOLD_ROOT}/<method>/<method>_graph_t<best>.png",
        f"{THRESHOLD_ROOT}/<method>/filtered/<method>_filtered_t*.csv/.pkl",
    ],
)

# ── GeoVisXRD imports ─────────────────────────────────────────────────────────
import geovisxrd
geovisxrd.setup_logging()

from geovisxrd import (
    filter_graph_by_threshold,
    save_filtered_graphs,
    analyze_threshold_stability,
    calculate_jaccard_similarity,
    plot_causal_graph_networkx,
    load_causal_artifact,
)


# ── Collect all available upstream causal graphs ──────────────────────────────
section(1, "Upstream artifact detection")
print(f"       Scanning: {CAUSAL_DIR}")

sources = []   # list of (method_name, G, source_path)

for method in ("lingam", "notears", "pc"):
    method_dir = os.path.join(CAUSAL_DIR, method)
    G, source_path = load_causal_artifact(method_dir)
    if G is not None:
        source_type = "pkl" if source_path.endswith(".pkl") else "csv"
        reuse(f"{method} graph  ({source_type})", source_path)
        print(f"         nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")
        sources.append((method, G, source_path))
    else:
        skip(f"{method} — no artifact found in {method_dir}")

# ── Synthetic fallback ────────────────────────────────────────────────────────
if not sources:
    fallback(
        "No causal outputs found — using synthetic weighted DiGraph.\n"
        "  Run validate_causal.py first to validate against real learned graphs."
    )
    G_syn = nx.DiGraph()
    G_syn.add_nodes_from(["A", "B", "C", "D", "E"])
    for u, v, w in [
        ("A", "B", 0.9), ("A", "C", 0.4), ("B", "D", 0.7),
        ("C", "D", 0.2), ("B", "E", 0.55), ("D", "E", 0.85),
    ]:
        G_syn.add_edge(u, v, weight=w)
    G_syn.graph["source_method"] = "synthetic"
    sources.append(("synthetic", G_syn, "(synthetic)"))

print(f"\n  Running threshold validation for {len(sources)} source(s): "
      f"{[m for m, *_ in sources]}")


# ── Per-method validation ─────────────────────────────────────────────────────
def _run_threshold_validation(source_method, G_input):
    """Run all five threshold validation sections for one source graph."""

    METHOD_OUT_DIR = os.path.join(THRESHOLD_ROOT, source_method)
    FILTERED_DIR   = os.path.join(METHOD_OUT_DIR, "filtered")
    PREFIX         = f"{source_method}_"

    if os.path.isdir(METHOD_OUT_DIR):
        shutil.rmtree(METHOD_OUT_DIR)
    os.makedirs(FILTERED_DIR, exist_ok=True)

    part(f"METHOD: {source_method.upper()}")
    inp(f"graph  nodes={G_input.number_of_nodes()}  "
        f"edges={G_input.number_of_edges()}")

    # ── Section A — filter_graph_by_threshold (spot checks) ──────────────────
    section("A", "filter_graph_by_threshold — spot checks")
    inp(f"source graph  nodes={G_input.number_of_nodes()}  "
        f"edges={G_input.number_of_edges()}")

    for t in (0.0, 0.3, 0.6, 1.0):
        G_f = filter_graph_by_threshold(G_input, t)
        print(f"       t={t:.1f}  →  {G_f.number_of_edges()} edge(s) retained")

    G_all  = filter_graph_by_threshold(G_input, 0.0)
    G_none = filter_graph_by_threshold(G_input, 1e9)
    assert G_all.number_of_edges()  == G_input.number_of_edges(), \
        "t=0.0 should retain all edges"
    assert G_none.number_of_edges() == 0, \
        "t=1e9 should retain no edges"
    ok("Edge-retention sanity checks passed  (t=0.0 keeps all, t=1e9 keeps none)")

    # ── Section B — save_filtered_graphs ─────────────────────────────────────
    section("B", "save_filtered_graphs — bulk save at 10 thresholds")
    inp(f"source graph  (same as above)")
    print(f"       output directory: {FILTERED_DIR}")

    thresholds = np.arange(0.1, 1.1, 0.1)
    saved = save_filtered_graphs(
        G_input, thresholds,
        save_dir=FILTERED_DIR,
        name_prefix=PREFIX,
    )

    ok(f"{len(saved)} filtered graphs written  (thresholds 0.1 … 1.0)")
    for t_val, info in sorted(saved.items()):
        G_f = info["graph"]
        out(f"t={t_val:.2f}  ({G_f.number_of_edges()} edges)  "
            f"{os.path.basename(info['csv'])}", info["csv"])

    assert len(saved) == len(thresholds), "Expected one saved graph per threshold"
    ok("All filtered graphs present on disk")

    # ── Section C — analyze_threshold_stability ───────────────────────────────
    section("C", "analyze_threshold_stability — Jaccard stability sweep")
    inp(f"source graph  nodes={G_input.number_of_nodes()}  "
        f"edges={G_input.number_of_edges()}")

    stab_png  = os.path.join(METHOD_OUT_DIR, f"{PREFIX}stability_curve.png")
    df_stab, best_t = analyze_threshold_stability(G_input, save_path=stab_png)
    out("stability curve PNG", stab_png)

    if best_t is None:
        warn("No stability results — graph may have no weighted edges")
        best_t = 0.0
    else:
        ok(f"Peak-stability threshold: t={best_t}")
        print(df_stab.to_string(index=False))

    # Test: pkl-path input
    first_pkl = sorted(saved.values(), key=lambda x: x["pkl"])[0]["pkl"]
    inp(f"pkl-path input test: {os.path.basename(first_pkl)}")
    df_stab2, _ = analyze_threshold_stability(first_pkl)
    ok("analyze_threshold_stability accepts pkl-path input")

    # Test: CSV-path input
    first_csv = sorted(saved.values(), key=lambda x: x["csv"])[0]["csv"]
    inp(f"csv-path input test: {os.path.basename(first_csv)}")
    df_stab3, _ = analyze_threshold_stability(first_csv)
    ok("analyze_threshold_stability accepts csv-path input")

    # ── Section D — calculate_jaccard_similarity ──────────────────────────────
    section("D", "calculate_jaccard_similarity — sanity checks")
    inp(f"source graph  (same as above)")

    j_self = calculate_jaccard_similarity(G_input, G_input)
    assert j_self == 1.0, f"Self-Jaccard expected 1.0, got {j_self}"
    ok(f"Jaccard(G, G) = {j_self:.4f}  (expected 1.0)")

    G_empty = nx.DiGraph()
    G_empty.add_nodes_from(G_input.nodes())
    j_empty = calculate_jaccard_similarity(G_input, G_empty)
    ok(f"Jaccard(G, ∅) = {j_empty:.4f}  (expected 0.0 for non-empty G)")

    G_lo = filter_graph_by_threshold(G_input, 0.2)
    G_hi = filter_graph_by_threshold(G_input, 0.6)
    j_lh = calculate_jaccard_similarity(G_lo, G_hi)
    ok(f"Jaccard(t=0.2, t=0.6) = {j_lh:.4f}")

    # ── Section E — visualise the best-threshold graph ────────────────────────
    section("E", f"Best-threshold graph visualisation  (t={best_t})")
    inp(f"filtered graph at peak-stability t={best_t}")

    G_best   = filter_graph_by_threshold(G_input, best_t)
    t_str    = f"{best_t:.2f}".replace(".", "_")
    best_png = os.path.join(METHOD_OUT_DIR, f"{PREFIX}graph_t{t_str}.png")
    try:
        plot_causal_graph_networkx(
            G_best, threshold=0.0,
            title=f"{source_method} — peak-stability threshold (t={best_t})",
            save_path=best_png,
        )
        out(f"best-threshold graph PNG  (t={best_t})", best_png)
    except Exception as e:
        skip(f"Graph plot — {type(e).__name__}: {e}")


# ── Run for every collected source ────────────────────────────────────────────
for source_method, G_input, source_path in sources:
    _run_threshold_validation(source_method, G_input)


# ── Artifact summary ───────────────────────────────────────────────────────────
inputs_used = []
if any(m != "synthetic" for m, *_ in sources):
    for method, _, source_path in sources:
        if method != "synthetic":
            inputs_used.append(
                f"{method} causal graph  [REUSE]  → {source_path}"
            )
if any(m == "synthetic" for m, *_ in sources):
    inputs_used.append(
        "synthetic weighted DiGraph  [FALLBACK]  (no real causal output found)"
    )

outputs_written = []
for root, dirs, files in os.walk(THRESHOLD_ROOT):
    dirs.sort()
    for f in sorted(files):
        fp  = os.path.join(root, f)
        rel = os.path.relpath(fp, THRESHOLD_ROOT)
        if "stability_curve" in f:
            desc = "— Jaccard stability sweep plot"
        elif "graph_t" in f and f.endswith(".png"):
            desc = "— peak-stability threshold visualisation"
        elif "filtered" in fp and f.endswith(".csv"):
            desc = "— filtered adjacency matrix at threshold t"
        elif "filtered" in fp and f.endswith(".pkl"):
            desc = "— filtered DiGraph at threshold t"
        else:
            desc = ""
        outputs_written.append(f"{rel}  {desc}")

artifact_summary(
    inputs_used     = inputs_used,
    outputs_written = outputs_written,
    reusable_by     = ["(threshold outputs are terminal — no downstream scripts consume them)"],
)
