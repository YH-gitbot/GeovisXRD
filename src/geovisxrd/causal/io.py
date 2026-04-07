# src/geovisxrd/causal/io.py
"""
Causal graph I/O — loading and normalisation.

This module owns all causal artifact loading semantics so that neither
the threshold module nor validation scripts need to implement their own
graph-loading logic.

Public API
----------
load_causal_graph(path_or_obj) → nx.DiGraph
    Normalise any supported representation of a causal graph into a
    weighted nx.DiGraph.

load_causal_artifact(method_dir) → (nx.DiGraph | None, str | None)
    Load a causal graph from a method output directory produced by the
    causal discovery pipeline.

Supported input types for load_causal_graph
-------------------------------------------
str (.pkl)          joblib-serialised nx.DiGraph or fitted causal model
str (.csv)          weighted adjacency matrix (index = row labels)
nx.DiGraph          returned as-is (shallow copy)
BaseCausalDiscovery fitted model — calls model.get_nx_graph()
pd.DataFrame        treated as a weighted adjacency matrix

Artifact directory convention (validate_causal.py outputs)
-----------------------------------------------------------
  validation_outputs/causal/<method>/
    causal_model.pkl   — joblib-serialised fitted BaseCausalDiscovery
    adjacency.csv      — weighted adjacency matrix (CSV)
    graph.png          — visualisation (not loaded by this module)

load_causal_artifact checks causal_model.pkl first (richer: preserves
model metadata and edge weights), then falls back to adjacency.csv.
"""

import os
import joblib
import pandas as pd
import networkx as nx

from .._logging import get_logger
logger = get_logger(__name__)


def load_causal_graph(path_or_obj) -> nx.DiGraph:
    """Normalise any supported causal graph representation into a DiGraph.

    Parameters
    ----------
    path_or_obj : str | nx.DiGraph | BaseCausalDiscovery | pd.DataFrame
        - str ending in ``.pkl``: joblib-loaded object (DiGraph or fitted model)
        - str ending in ``.csv``: adjacency matrix CSV (index_col=0)
        - nx.DiGraph: returned as a copy
        - object with ``get_nx_graph()`` method: calls it (fitted causal model)
        - object with ``graph_`` attribute (pd.DataFrame): converts adjacency matrix
        - pd.DataFrame: treated as weighted adjacency matrix

    Returns
    -------
    nx.DiGraph
        Weighted directed graph.  Self-loops are stripped.

    Raises
    ------
    FileNotFoundError  if a path is given but the file does not exist.
    ValueError         if the input type is unsupported.
    """
    if isinstance(path_or_obj, str):
        path = path_or_obj
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Causal graph file not found: {path}")
        if path.endswith(".csv"):
            df = pd.read_csv(path, index_col=0)
            return _adj_df_to_digraph(df, source=path)
        # Assume pickle (joblib)
        obj = joblib.load(path)
        return load_causal_graph(obj)   # recurse on the loaded object

    if isinstance(path_or_obj, nx.DiGraph):
        G = path_or_obj.copy()
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        return G

    if hasattr(path_or_obj, "get_nx_graph"):
        G = path_or_obj.get_nx_graph()
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        return G

    if hasattr(path_or_obj, "graph_") and isinstance(
        getattr(path_or_obj, "graph_", None), pd.DataFrame
    ):
        return _adj_df_to_digraph(path_or_obj.graph_)

    if isinstance(path_or_obj, pd.DataFrame):
        return _adj_df_to_digraph(path_or_obj)

    raise ValueError(
        f"Unsupported input type for load_causal_graph: {type(path_or_obj)}. "
        "Accepted: str path (.pkl/.csv), nx.DiGraph, fitted causal model, "
        "or adjacency pd.DataFrame."
    )


def load_causal_artifact(method_dir: str):
    """Load a causal graph from a method output directory.

    Checks for ``causal_model.pkl`` first (preserves full edge-weight
    information from the fitted model), then falls back to
    ``adjacency.csv``.

    Parameters
    ----------
    method_dir : str
        Path to a method subdirectory, e.g.
        ``validation_outputs/causal/lingam/``.

    Returns
    -------
    G : nx.DiGraph or None
        Loaded graph, or None if no recognisable artifact is found.
    source_path : str or None
        Absolute path of the file that was loaded, or None.
    """
    method_name = os.path.basename(os.path.normpath(method_dir))

    # 1. Prefer the serialised model pickle
    pkl_path = os.path.join(method_dir, "causal_model.pkl")
    if os.path.isfile(pkl_path):
        try:
            G = load_causal_graph(pkl_path)
            if G.number_of_nodes() > 0:
                G.graph["source_method"] = method_name
                logger.info(f"Loaded causal graph (method={method_name}) from {pkl_path}")
                return G, pkl_path
        except Exception as exc:
            logger.warning(f"Failed to load causal pkl {pkl_path}: {exc}")

    # 2. Fall back to the adjacency CSV
    csv_path = os.path.join(method_dir, "adjacency.csv")
    if os.path.isfile(csv_path):
        try:
            G = load_causal_graph(csv_path)
            if G.number_of_nodes() > 0:
                G.graph["source_method"] = method_name
                logger.info(f"Loaded causal graph (method={method_name}) from {csv_path}")
                return G, csv_path
        except Exception as exc:
            logger.warning(f"Failed to load causal csv {csv_path}: {exc}")

    return None, None


# ==============================================================================
# Internal helper
# ==============================================================================

def _adj_df_to_digraph(df: pd.DataFrame, source: str = "") -> nx.DiGraph:
    """Convert a weighted adjacency DataFrame to a DiGraph."""
    G = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    return G
