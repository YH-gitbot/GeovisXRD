# src/geovisxrd/causal/discovery.py

import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from abc import ABC, abstractmethod
from typing import Dict, Optional, Set, Tuple

from .._logging import get_logger
logger = get_logger(__name__)


# ==============================================================================
# 1. Domain-Knowledge Constraints
# ==============================================================================

class CausalConstraints:
    """
    Tier-based causal ordering constraints.

    Each variable is assigned an integer tier; edges are only permitted
    from lower tier → higher tier  (strict: tier[src] < tier[dst]).

    Tier example
    ------------
    geo / root variables          → tier 1
    environmental / structural    → tier 2
    socio-economic                → tier 3
    target / leaf variable        → tier 4

    Parameters
    ----------
    tier_map : dict[str, int], optional
        Explicit mapping of variable name → integer tier.
        Edges where tier[src] >= tier[dst] are forbidden automatically.
    forbidden_edges : list[tuple[str, str]], optional
        Additional (src, dst) pairs that are always forbidden, regardless
        of tier.
    geo_vars : list[str], optional
        Convenience shortcut — assigns each variable to tier 1.
    target_vars : list[str], optional
        Convenience shortcut — assigns each variable to tier 4.
    """

    def __init__(
        self,
        tier_map: Optional[Dict[str, int]] = None,
        forbidden_edges=None,
        geo_vars=None,
        target_vars=None,
    ):
        self.tier_map: Dict[str, int] = dict(tier_map or {})

        # Convenience shortcuts — do not overwrite explicit tier assignments
        for v in (geo_vars or []):
            self.tier_map.setdefault(v, 1)
        for v in (target_vars or []):
            self.tier_map.setdefault(v, 4)

        self.forbidden_edges: Set[Tuple[str, str]] = set(
            tuple(e) for e in (forbidden_edges or [])
        )

    # ------------------------------------------------------------------
    # Forbidden-edge computation
    # ------------------------------------------------------------------
    def get_forbidden_edges(self, columns) -> Set[Tuple[str, str]]:
        """
        Return the complete set of forbidden (src, dst) pairs for the given
        columns, combining tier rules with explicit overrides.
        """
        forbidden = set(self.forbidden_edges)
        for src in columns:
            for dst in columns:
                if src == dst:
                    continue
                st = self.tier_map.get(src)
                dt = self.tier_map.get(dst)
                if st is not None and dt is not None and st >= dt:
                    forbidden.add((src, dst))
        return forbidden

    def get_forbidden_mask(self, columns) -> np.ndarray:
        """
        Return a boolean (n×n) ndarray; True = forbidden edge.
        Axis 0 = source index, axis 1 = destination index.
        """
        n = len(columns)
        idx = {c: i for i, c in enumerate(columns)}
        mask = np.zeros((n, n), dtype=bool)

        for src in columns:
            for dst in columns:
                if src == dst:
                    continue
                st = self.tier_map.get(src)
                dt = self.tier_map.get(dst)
                if st is not None and dt is not None and st >= dt:
                    mask[idx[src], idx[dst]] = True

        for src, dst in self.forbidden_edges:
            if src in idx and dst in idx:
                mask[idx[src], idx[dst]] = True

        return mask

    def _is_tier_violation(self, src: str, dst: str) -> bool:
        """True if the edge src→dst violates the tier ordering."""
        st, dt = self.tier_map.get(src), self.tier_map.get(dst)
        return st is not None and dt is not None and st >= dt

    # ------------------------------------------------------------------
    # Post-hoc application — safety net used after fitting
    # ------------------------------------------------------------------
    def apply(self, adj_df: pd.DataFrame) -> pd.DataFrame:
        """Zero out all forbidden edges in an adjacency DataFrame."""
        adj = adj_df.copy()
        cols = list(adj.columns)
        forbidden = self.get_forbidden_edges(cols)

        tier_rm = explicit_rm = 0
        for src, dst in forbidden:
            if src in adj.index and dst in adj.columns and adj.loc[src, dst] != 0:
                adj.loc[src, dst] = 0
                if self._is_tier_violation(src, dst):
                    tier_rm += 1
                else:
                    explicit_rm += 1

        logger.info(
            f"Constraints applied: {tier_rm + explicit_rm} forbidden edge(s) removed "
            f"(tier rules: {tier_rm}, explicit overrides: {explicit_rm})."
        )
        return adj

    def apply_nx(self, G: nx.DiGraph) -> nx.DiGraph:
        """Remove forbidden edges from a NetworkX DiGraph (in-place)."""
        cols = list(G.nodes())
        forbidden = self.get_forbidden_edges(cols)

        tier_rm = explicit_rm = 0
        for src, dst in forbidden:
            if G.has_edge(src, dst):
                G.remove_edge(src, dst)
                if self._is_tier_violation(src, dst):
                    tier_rm += 1
                else:
                    explicit_rm += 1

        logger.info(
            f"Constraints applied: {tier_rm + explicit_rm} forbidden edge(s) removed "
            f"(tier rules: {tier_rm}, explicit overrides: {explicit_rm})."
        )
        return G


# ==============================================================================
# 2. Bootstrap Stability Filter
# ==============================================================================

class BootstrapStabilityFilter:
    """Run any causal discovery model on bootstrap resamples and retain stable edges.

    An edge is retained only if it appears in more than ``threshold`` fraction
    of bootstrap runs.

    Parameters
    ----------
    model_factory : callable
        Zero-argument callable returning a fresh BaseCausalDiscovery instance,
        e.g. ``lambda: NOTEARSDiscovery(constraints=c)``.
    n_bootstrap : int, optional
        Number of bootstrap runs (default 20).
    threshold : float, optional
        Minimum edge frequency required to retain an edge (default 0.6).
    """

    def __init__(self, model_factory, n_bootstrap: int = 20, threshold: float = 0.6):
        self.model_factory = model_factory
        self.n_bootstrap   = n_bootstrap
        self.threshold     = threshold

    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the bootstrap ensemble and return a stable adjacency matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Input feature matrix; rows are resampled with replacement.

        Returns
        -------
        pd.DataFrame
            Binary adjacency matrix (float 0/1) of stable edges.
        """
        cols   = list(data.columns)
        n      = len(cols)
        idx    = {c: i for i, c in enumerate(cols)}
        counts = np.zeros((n, n))

        logger.info(
            f"Bootstrap stability: running {self.n_bootstrap} iterations "
            f"(threshold={self.threshold})."
        )

        for run in range(self.n_bootstrap):
            sample = data.sample(n=len(data), replace=True, random_state=run)
            model  = self.model_factory()
            try:
                model.fit(sample)
            except Exception as e:
                logger.warning(f"Bootstrap run {run} failed: {e}")
                continue
            adj = model.graph_
            if adj is None:
                continue
            for src in cols:
                for dst in cols:
                    if src != dst and src in adj.index and dst in adj.columns:
                        if adj.loc[src, dst] != 0:
                            counts[idx[src], idx[dst]] += 1

        freq        = counts / self.n_bootstrap
        stable_mask = freq > self.threshold
        n_seen      = int((counts > 0).sum())
        n_stable    = int(stable_mask.sum())
        n_filtered  = n_seen - n_stable

        logger.info(
            f"Bootstrap stability complete: {n_stable} stable edge(s) kept, "
            f"{n_filtered} removed (threshold={self.threshold})."
        )

        return pd.DataFrame(stable_mask.astype(float), index=cols, columns=cols)


# ==============================================================================
# 3. Helper Functions
# ==============================================================================

def _matrix_to_nx(adj_df):
    """Convert adjacency matrix DataFrame to a weighted NetworkX DiGraph."""
    G = nx.DiGraph()
    G.add_nodes_from(adj_df.columns)
    for col in adj_df.columns:
        for row in adj_df.index:
            weight = adj_df.loc[row, col]
            if weight != 0:
                G.add_edge(row, col, weight=weight)
    return G


# ==============================================================================
# 4. Abstract Base Class
# ==============================================================================

class BaseCausalDiscovery(ABC):
    """Abstract base class for all causal discovery algorithms.

    Parameters
    ----------
    model_name : str, optional
        Short name used in log messages and saved files.
    constraints : CausalConstraints or None, optional
        When provided, forbidden edges are enforced *during* fitting
        (algorithm-specific) and a final safety pass removes any remaining
        violations afterwards.
    **kwargs
        Additional parameters stored in self.params and forwarded to
        algorithm-specific constructors.
    """

    def __init__(self, model_name="unknown", constraints=None, **kwargs):
        self.graph_      = None
        self.nx_graph_   = None
        self.params      = kwargs
        self.model_name  = model_name
        self.constraints = constraints

    @abstractmethod
    def fit(self, data):
        pass

    # ------------------------------------------------------------------
    # Final safety pass — called at the end of each fit()
    # ------------------------------------------------------------------
    def _apply_constraints(self):
        """Post-processing safety pass: remove any remaining forbidden edges.

        Subclasses enforce constraints during training where possible;
        this final pass catches anything that slips through.
        """
        if self.constraints is None:
            return

        if self.graph_ is not None:
            self.graph_    = self.constraints.apply(self.graph_)
            self.nx_graph_ = _matrix_to_nx(self.graph_)
        elif self.nx_graph_ is not None:
            self.nx_graph_ = self.constraints.apply_nx(self.nx_graph_)
            self.graph_    = nx.to_pandas_adjacency(self.nx_graph_)

    def _log_final_edge_count(self):
        if self.graph_ is not None:
            n = int((self.graph_.values != 0).sum())
            logger.info(f"{self.model_name.upper()} — final graph: {n} edge(s).")

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def get_graph(self):
        return self.graph_

    def get_nx_graph(self):
        if self.nx_graph_ is None:
            if self.graph_ is not None:
                self.nx_graph_ = _matrix_to_nx(self.graph_)
            else:
                raise RuntimeError("Model not fitted. Call fit() first.")

        self.nx_graph_.graph['model_name'] = self.model_name
        self.nx_graph_.graph['params']     = str(self.params)
        return self.nx_graph_

    def save(self, path):
        if self.graph_ is None:
            raise ValueError("Model not fitted — nothing to save.")
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        if path.endswith('.csv'):
            self.graph_.to_csv(path)
        elif path.endswith('.pkl'):
            with open(path, 'wb') as f:
                graph_to_save = self.nx_graph_ if self.nx_graph_ else self.get_nx_graph()
                pickle.dump(graph_to_save, f)

        logger.info(f"Causal model saved to: {path}")


# ==============================================================================
# 5. Concrete Models
# ==============================================================================

class CorrelationBaseline(BaseCausalDiscovery):
    """Raw-data Pearson correlation skeleton.

    Builds an undirected edge skeleton from pairwise Pearson correlations
    on the raw feature matrix.  This is a lightweight data-association
    baseline, NOT a causal discovery algorithm and NOT part of the
    threshold module (geovisxrd/threshold/).

    The threshold module operates on the *edge weights of a learned causal
    graph*, not on raw data correlations.  Use CorrelationBaseline only
    as a fast, dependency-free comparison baseline.
    """

    def __init__(self, threshold=0.5, constraints=None, **kwargs):
        super().__init__(model_name="correlation", constraints=constraints, **kwargs)
        self.threshold = threshold

    def fit(self, data):
        logger.info(
            f"Fitting CorrelationBaseline (Pearson |r| > {self.threshold})."
        )
        corr = data.corr().abs()
        adj  = (corr > self.threshold).astype(int)
        np.fill_diagonal(adj.values, 0)
        self.graph_ = adj
        self._apply_constraints()
        self._log_final_edge_count()


class SAMDiscovery(BaseCausalDiscovery):
    """Structural Agnostic Model (cdt library).

    Tier constraints are applied to the learned graph immediately after
    SAM's predict() call — before the adjacency matrix is finalised —
    so forbidden edges never propagate to downstream analysis.
    """

    def __init__(self, constraints=None, **kwargs):
        super().__init__(model_name="sam", constraints=constraints, **kwargs)

    def fit(self, data):
        logger.info("Fitting SAM (Structural Agnostic Model).")
        try:
            from cdt.causality.graph import SAM
            model = SAM(**self.params)
            G = model.predict(data)

            # Filter forbidden edges from the predicted graph before finalising.
            if self.constraints is not None:
                cols      = list(data.columns)
                forbidden = self.constraints.get_forbidden_edges(cols)
                tier_rm = explicit_rm = 0
                for src, dst in list(G.edges()):
                    if (src, dst) in forbidden:
                        G.remove_edge(src, dst)
                        if self.constraints._is_tier_violation(src, dst):
                            tier_rm += 1
                        else:
                            explicit_rm += 1
                logger.info(
                    f"SAM constraints applied: {tier_rm + explicit_rm} edge(s) removed "
                    f"(tier: {tier_rm}, explicit: {explicit_rm})."
                )

            self.nx_graph_ = G
            self.graph_    = nx.to_pandas_adjacency(G)
            self._log_final_edge_count()
        except Exception as e:
            logger.error(f"SAM fitting failed: {e}")


class NOTEARSDiscovery(BaseCausalDiscovery):
    """NOTEARS — differentiable acyclic structure learning (causalnex).

    Tier constraints are injected as ``tabu_edges`` into the NOTEARS
    optimiser, so forbidden edges are fixed to zero in the weight matrix
    throughout optimisation — not just removed afterwards.
    """

    def __init__(self, threshold=0.1, constraints=None, **kwargs):
        super().__init__(model_name="notears", constraints=constraints, **kwargs)
        self.threshold = threshold

    def fit(self, data):
        logger.info(f"Fitting NOTEARS (w_threshold={self.threshold}).")
        try:
            from causalnex.structure.notears import from_pandas

            # Collect user-supplied tabu_edges without mutating self.params.
            user_tabu    = list(self.params.get("tabu_edges", []))
            extra_params = {k: v for k, v in self.params.items() if k != "tabu_edges"}

            tabu_edges = list(set(user_tabu))
            if self.constraints is not None:
                forbidden     = self.constraints.get_forbidden_edges(list(data.columns))
                tier_tabu     = [(s, d) for s, d in forbidden
                                 if self.constraints._is_tier_violation(s, d)]
                explicit_tabu = [(s, d) for s, d in forbidden
                                 if not self.constraints._is_tier_violation(s, d)]
                tabu_edges    = list(set(tabu_edges) | forbidden)
                logger.info(
                    f"NOTEARS tabu edges: {len(tier_tabu)} tier-based + "
                    f"{len(explicit_tabu)} explicit."
                )

            sm = from_pandas(
                data,
                w_threshold=self.threshold,
                tabu_edges=tabu_edges if tabu_edges else None,
                **extra_params,
            )
            adj = nx.adjacency_matrix(sm).todense()
            self.graph_    = pd.DataFrame(adj, index=data.columns, columns=data.columns)
            self.nx_graph_ = _matrix_to_nx(self.graph_)
            # Safety pass — should find nothing if tabu_edges worked correctly.
            self._apply_constraints()
            self._log_final_edge_count()
        except Exception as e:
            logger.error(f"NOTEARS fitting failed: {e}")


class LiNGAMDiscovery(BaseCausalDiscovery):
    """Linear Non-Gaussian Acyclic Model.

    Tier constraints are encoded as a ``prior_knowledge`` matrix passed
    to DirectLiNGAM, so forbidden paths are excluded during the ICA-based
    causal ordering search — not just filtered from the output.

    prior_knowledge convention (lingam library):
        -1  unknown
         0  no direct path from variable i to variable j (Xi -/-> Xj)
         1  direct path exists
    """

    def __init__(self, constraints=None, **kwargs):
        super().__init__(model_name="lingam", constraints=constraints, **kwargs)

    def fit(self, data):
        logger.info("Fitting LiNGAM (DirectLiNGAM).")
        try:
            import lingam

            cols = list(data.columns)
            n    = len(cols)
            idx  = {c: i for i, c in enumerate(cols)}

            prior = None
            if self.constraints is not None:
                prior     = np.full((n, n), -1, dtype=int)
                forbidden = self.constraints.get_forbidden_edges(cols)
                tier_count = explicit_count = 0
                for src, dst in forbidden:
                    if src in idx and dst in idx:
                        prior[idx[src], idx[dst]] = 0   # no path src → dst
                        if self.constraints._is_tier_violation(src, dst):
                            tier_count += 1
                        else:
                            explicit_count += 1
                logger.info(
                    f"LiNGAM prior knowledge: {tier_count} tier + "
                    f"{explicit_count} explicit forbidden path(s) applied."
                )

            model = lingam.DirectLiNGAM(prior_knowledge=prior, **self.params)
            model.fit(data)
            self.graph_    = pd.DataFrame(
                model.adjacency_matrix_, index=data.columns, columns=data.columns
            )
            self.nx_graph_ = _matrix_to_nx(self.graph_)
            # Safety pass
            self._apply_constraints()
            self._log_final_edge_count()
        except Exception as e:
            logger.error(f"LiNGAM fitting failed: {e}")


class PCDiscovery(BaseCausalDiscovery):
    """PC constraint-based algorithm (causal-learn).

    Tier constraints are injected via ``BackgroundKnowledge`` so that
    forbidden edge orientations are blocked during the PC orientation
    phase, not only removed after the CPDAG is built.
    """

    def __init__(self, alpha=0.05, constraints=None, **kwargs):
        super().__init__(model_name="pc", constraints=constraints, **kwargs)
        self.alpha = alpha

    def fit(self, data):
        logger.info(f"Fitting PC algorithm (alpha={self.alpha}).")
        try:
            from causallearn.search.ConstraintBased.PC import pc
            from causallearn.utils.cit import fisherz

            bk = None
            if self.constraints is not None:
                try:
                    from causallearn.graph.GraphNode import GraphNode
                    from causallearn.utils.BackgroundKnowledge import BackgroundKnowledge

                    cols      = list(data.columns)
                    forbidden = self.constraints.get_forbidden_edges(cols)
                    nodes     = {c: GraphNode(c) for c in cols}
                    bk        = BackgroundKnowledge()
                    tier_count = explicit_count = 0
                    for src, dst in forbidden:
                        if src in nodes and dst in nodes:
                            bk.add_forbidden_by_node(nodes[src], nodes[dst])
                            if self.constraints._is_tier_violation(src, dst):
                                tier_count += 1
                            else:
                                explicit_count += 1
                    logger.info(
                        f"PC BackgroundKnowledge: {tier_count} tier + "
                        f"{explicit_count} explicit forbidden orientation(s) applied."
                    )
                except ImportError:
                    logger.warning(
                        "PC BackgroundKnowledge unavailable — "
                        "constraints applied post-hoc only."
                    )
                    bk = None

            cols = list(data.columns)
            cg = pc(
                data.to_numpy(),
                self.alpha,
                indep_test=fisherz,
                background_knowledge=bk,
                node_names=cols,
                verbose=False,
            )
            self.graph_    = pd.DataFrame(
                cg.G.graph, index=cols, columns=cols
            )
            self.nx_graph_ = _matrix_to_nx(self.graph_)
            # Safety pass
            self._apply_constraints()
            self._log_final_edge_count()
        except Exception as e:
            logger.error(f"PC fitting failed: {e}")


# ==============================================================================
# 6. Factory Function
# ==============================================================================

def get_causal_model(method="sam", constraints=None, **kwargs):
    """Instantiate a causal discovery model by name.

    Parameters
    ----------
    method : str, optional
        Algorithm identifier: 'pc', 'notears', 'lingam', 'sam', or
        'correlation' / 'correlation_baseline' (raw-data Pearson baseline,
        NOT the threshold module).  Default 'sam'.
    constraints : CausalConstraints or None, optional
        Domain-knowledge constraints; see CausalConstraints for details.
    **kwargs
        Additional parameters forwarded to the algorithm constructor.

    Returns
    -------
    BaseCausalDiscovery
        Unfitted causal discovery model instance.

    Examples
    --------
    >>> from geovisxrd.causal.discovery import CausalConstraints, get_causal_model
    >>> c = CausalConstraints(
    ...     tier_map={"Latitude": 1, "Longitude": 1, "MedInc": 3, "HousePrice": 4},
    ...     forbidden_edges=[("MedInc", "AveRooms")],
    ... )
    >>> model = get_causal_model("lingam", constraints=c)
    >>> model.fit(df)
    """
    method = method.lower()
    # Deprecated alias: "threshold" used to refer to CorrelationBaseline.
    # The threshold module now means causal-graph edge-weight segmentation.
    if method == "threshold":
        warnings.warn(
            "causal_method='threshold' is deprecated and will be removed in a "
            "future release.  Use causal_method='correlation' instead.  "
            "'threshold' now refers to edge-weight segmentation of a learned "
            "causal graph (geovisxrd.threshold), not raw-data correlation.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "correlation"
    # "correlation" / "correlation_baseline" — raw-data Pearson skeleton.
    # NOT the threshold module; use only as a lightweight comparison baseline.
    if method in ("correlation", "correlation_baseline"):
        return CorrelationBaseline(constraints=constraints, **kwargs)
    if method == "sam":     return SAMDiscovery(constraints=constraints, **kwargs)
    if method == "notears": return NOTEARSDiscovery(constraints=constraints, **kwargs)
    if method == "lingam":  return LiNGAMDiscovery(constraints=constraints, **kwargs)
    if method == "pc":      return PCDiscovery(constraints=constraints, **kwargs)
    raise ValueError(f"Unknown causal model: '{method}'. "
                     f"Supported: 'pc', 'notears', 'lingam', 'sam', 'correlation'.")
