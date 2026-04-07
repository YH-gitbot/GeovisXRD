from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("geovisxrd")
except PackageNotFoundError:
    __version__ = "2.0.0"  # fallback version if package metadata is not found

__author__ = "Yuanhan"

import logging
from ._logging import setup_logging, _suppress_third_party

# Standard library practice: libraries add NullHandler so logs are silent
# unless the application calls setup_logging() or configures the root logger.
logging.getLogger("geovisxrd").addHandler(logging.NullHandler())

# Raise the log floor for noisy third-party libraries unconditionally.
# This only sets a level floor — no handlers are installed.
_suppress_third_party()

from .modeling import train_model, get_model, save_model

from .explaining import shapexplainer
from .explaining.io import save_shap_results, load_shap_results

from .causal.discovery import get_causal_model, CausalConstraints, BootstrapStabilityFilter
from .causal.io import load_causal_graph, load_causal_artifact

from .plotting import (
    plot_summary_bar, plot_beeswarm, plot_pos_neg_ratio,
    plot_dependence_2d_lowess, plot_dependence_3d_interaction,
    plot_causal_graph_networkx, plot_all_shap_charts,
)

from .threshold.optimizer import (
    analyze_threshold_stability,
    filter_graph_by_threshold,
    save_filtered_graphs,
    calculate_jaccard_similarity,
)

from .pipeline import run_analysis, run_recursive_step

from .export.geo_export import build_geo_table, to_geodataframe, save_geo_export, save_qgis_export
from .plotting.mapping import plot_shap_single, plot_shap_6panel

__all__ = [
    "setup_logging",
    "train_model", "get_model", "save_model",
    "shapexplainer",
    "save_shap_results", "load_shap_results",
    "plot_summary_bar", "plot_beeswarm", "plot_pos_neg_ratio",
    "plot_dependence_2d_lowess", "plot_dependence_3d_interaction",
    "plot_causal_graph_networkx", "plot_all_shap_charts",
    "get_causal_model", "CausalConstraints", "BootstrapStabilityFilter",
    "load_causal_graph", "load_causal_artifact",
    "analyze_threshold_stability", "filter_graph_by_threshold",
    "save_filtered_graphs", "calculate_jaccard_similarity",
    "run_analysis", "run_recursive_step",
    "build_geo_table", "to_geodataframe", "save_geo_export", "save_qgis_export",
    "plot_shap_single", "plot_shap_6panel",
]