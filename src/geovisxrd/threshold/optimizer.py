# src/geovisxrd/threshold/optimizer.py
"""
Causal-graph threshold segmentation and Jaccard stability utilities.

All functions in this module operate on learned weighted causal graphs
(nx.DiGraph with edge 'weight' attributes, fitted causal model objects,
or weighted adjacency matrix DataFrames).  They do NOT operate on raw
feature data.

Typical workflow
----------------
1. Fit a causal discovery model (PC / NOTEARS / LiNGAM / SAM).
2. Obtain its weighted nx.DiGraph via model.get_nx_graph().
3. Call filter_graph_by_threshold(G, t) to produce edge-filtered graphs.
4. Call save_filtered_graphs(G, thresholds, save_dir) to persist them.
5. Call analyze_threshold_stability(G) to find the most stable threshold.
"""

import os
import joblib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from ..causal.io import load_causal_graph
from .._logging import get_logger
logger = get_logger(__name__)


# ==============================================================================
# Graph comparison
# ==============================================================================

def calculate_jaccard_similarity(graph_a: nx.DiGraph, graph_b: nx.DiGraph) -> float:
    """Compute the Jaccard similarity between two directed graphs.

    Similarity is computed over edge sets (node labels, not positions).

    Parameters
    ----------
    graph_a, graph_b : nx.DiGraph

    Returns
    -------
    float
        Jaccard index in [0, 1]; 1.0 = identical edge sets.
    """
    edges_a = set(graph_a.edges())
    edges_b = set(graph_b.edges())
    if not edges_a and not edges_b:
        return 1.0
    intersection = len(edges_a & edges_b)
    union        = len(edges_a | edges_b)
    return intersection / union if union > 0 else 0.0


# ==============================================================================
# Edge-weight filtering
# ==============================================================================

def filter_graph_by_threshold(G: nx.DiGraph, threshold: float) -> nx.DiGraph:
    """Return a new DiGraph retaining only edges with |weight| >= threshold.

    Node set is preserved; only edges are filtered.

    Parameters
    ----------
    G : nx.DiGraph
        Weighted causal graph (edges carry a 'weight' attribute).
    threshold : float
        Minimum absolute edge weight required to keep an edge.

    Returns
    -------
    nx.DiGraph
        Filtered graph with the same nodes as G.
    """
    G_f = nx.DiGraph()
    G_f.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1.0)
        if abs(w) >= threshold:
            G_f.add_edge(u, v, **d)
    return G_f


# ==============================================================================
# Bulk save of filtered graphs
# ==============================================================================

def save_filtered_graphs(
    G: nx.DiGraph,
    thresholds,
    save_dir: str,
    name_prefix: str = "",
) -> dict:
    """Sweep thresholds, filter G at each step, and save the results.

    For each threshold value t, writes:
      <save_dir>/<prefix>filtered_t<t>.csv  — weighted adjacency matrix
      <save_dir>/<prefix>filtered_t<t>.pkl  — filtered nx.DiGraph (joblib)

    The filename prefix is resolved in order:
      1. ``name_prefix`` if explicitly provided (non-empty).
      2. ``G.graph["source_method"]`` if present (set by ``load_causal_artifact``).
      3. ``"unknown_"`` as a fallback.

    Parameters
    ----------
    G : nx.DiGraph
        Weighted causal graph to filter.
    thresholds : array-like
        Sequence of threshold values.
    save_dir : str
        Directory in which to write output files.
    name_prefix : str, optional
        Override prefix prepended to every output filename, e.g. ``"lingam_"``.

    Returns
    -------
    dict[float, dict]
        ``{t: {"csv": csv_path, "pkl": pkl_path, "graph": G_filtered}}``
    """
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    # Resolve prefix: explicit arg > graph metadata > fallback
    if not name_prefix:
        method = G.graph.get("source_method", "unknown")
        name_prefix = f"{method}_"

    for t in thresholds:
        t_val  = round(float(t), 4)
        t_str  = f"{t_val:.2f}".replace(".", "_")
        G_f    = filter_graph_by_threshold(G, t_val)

        adj_df   = nx.to_pandas_adjacency(G_f)
        csv_path = os.path.join(save_dir, f"{name_prefix}filtered_t{t_str}.csv")
        pkl_path = os.path.join(save_dir, f"{name_prefix}filtered_t{t_str}.pkl")

        adj_df.to_csv(csv_path)
        joblib.dump(G_f, pkl_path)

        results[t_val] = {"csv": csv_path, "pkl": pkl_path, "graph": G_f}
        logger.info(
            f"Saving filtered graph (method={name_prefix.rstrip('_')}, t={t_val:.2f}): "
            f"{G_f.number_of_edges()} edge(s) → {csv_path}"
        )

    return results


# ==============================================================================
# Stability sweep
# ==============================================================================

def _resolve_graph_input(data_input):
    """Normalise any supported input type into a weighted nx.DiGraph.

    Delegates all path and model loading to ``causal.io.load_causal_graph``.
    The threshold module owns only the stability-sweep logic; artifact loading
    semantics live in causal/io.py.

    Accepted types
    --------------
    nx.DiGraph
        Returned as-is (no copy).
    pd.DataFrame
        Converted to nx.DiGraph via nx.from_pandas_adjacency.
    Everything else
        Forwarded to load_causal_graph().

    Returns
    -------
    nx.DiGraph
        Weighted directed graph.
    """
    if isinstance(data_input, nx.DiGraph):
        return data_input

    if isinstance(data_input, pd.DataFrame):
        return nx.from_pandas_adjacency(data_input, create_using=nx.DiGraph)

    return load_causal_graph(data_input)


def analyze_threshold_stability(
    graph_input,
    threshold_range=np.arange(0.1, 1.1, 0.1),
    save_path=None,
):
    """Sweep edge-weight thresholds and compute Jaccard stability between steps.

    Operates on a learned weighted causal graph — not on raw feature data.
    At each consecutive pair of thresholds (t_i, t_{i+1}), the Jaccard
    similarity between the two filtered graphs is computed.  The threshold
    with the highest Jaccard score is the most stable segmentation point.

    Parameters
    ----------
    graph_input : str | nx.DiGraph | BaseCausalDiscovery | pd.DataFrame
        Source of the weighted causal graph:
          - str path to a .pkl (joblib) or .csv (adjacency matrix) file
          - nx.DiGraph with edge 'weight' attributes
          - Fitted causal model object (uses model.graph_)
          - pd.DataFrame treated as a weighted adjacency matrix
    threshold_range : array-like, optional
        Sequence of edge-weight thresholds to sweep (default 0.1 → 1.0).
    save_path : str or None, optional
        If provided, save the stability curve PNG to this path.

    Returns
    -------
    df_res : pd.DataFrame
        Columns: 'Threshold', 'Jaccard'.
    best_t : float or None
        Threshold with the highest Jaccard stability score.
        None when fewer than two threshold steps produce results.
    """
    raw_graph = _resolve_graph_input(graph_input)

    logger.info(
        f"Threshold stability sweep "
        f"(steps={len(threshold_range)}, "
        f"range={float(threshold_range[0]):.1f}–{float(threshold_range[-1]):.1f})."
    )

    results    = []
    prev_graph = None

    for th in threshold_range:
        th_val        = round(float(th), 4)
        current_graph = filter_graph_by_threshold(raw_graph, th_val)

        if prev_graph is not None:
            j_score = calculate_jaccard_similarity(prev_graph, current_graph)
            results.append({"Threshold": th_val, "Jaccard": j_score})
            logger.info(
                f"  t={th_val:.2f} vs t={round(th_val - 0.1, 2):.2f}  "
                f"Jaccard={j_score:.4f}"
            )

        prev_graph = current_graph

    df_res  = pd.DataFrame(results)
    best_th = None

    plt.figure(figsize=(7, 6))
    plt.title("Causal Graph — Edge-Weight Threshold Stability", fontsize=12)
    plt.xlabel("Edge-weight threshold (t)")
    plt.ylabel("Jaccard similarity (consecutive steps)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)

    if not df_res.empty:
        plt.plot(
            df_res["Threshold"], df_res["Jaccard"],
            marker="o", color="#1f77b4", linewidth=2, label="Jaccard",
        )
        best_th = df_res.loc[df_res["Jaccard"].idxmax(), "Threshold"]
        plt.axvline(
            x=best_th, color="red", linestyle="--",
            label=f"Peak stability (t={best_th})",
        )
        plt.legend()
    else:
        logger.warning(
            "Stability sweep produced no results — "
            "threshold range may be too narrow or graph has no weighted edges."
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
        logger.info(f"Stability curve saved: {save_path}")

    plt.close()
    return df_res, best_th
