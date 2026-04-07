# src/geovisxrd/plotting/shap_plots.py
"""
SHAP visualisation utilities.

Provides standalone plotting functions for SHAP analysis outputs:
  - plot_summary_bar            — feature importance bar chart
  - plot_beeswarm               — beeswarm (swarm) summary plot
  - plot_dependence_2d_lowess   — 2-D dependence with LOWESS trend line
  - plot_dependence_3d_interaction — 3-D dependence with colour interaction term
  - plot_pos_neg_ratio          — positive/negative SHAP contribution ratio per feature
  - plot_all_shap_charts        — standard chart set (swarm, bar, waterfall) from a SHAP bundle
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess

from .._logging import get_logger
logger = get_logger(__name__)

plt.rcParams.update({'font.size': 12})

# =========================================================
# 1. Summary Bar Plot (Overall Importance Bar Chart)
# =========================================================
def plot_summary_bar(shap_values, feature_names, save_path=None, **kwargs):
    """Plot a SHAP feature importance bar chart.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values; shape (n_samples, n_features).
    feature_names : list of str
        Feature names corresponding to columns of shap_values.
    save_path : str or None, optional
        Path to save the PNG; figure is returned without saving if None.
    **kwargs
        Extra keyword arguments forwarded to shap.summary_plot()
        (e.g. max_display=10).

    Returns
    -------
    matplotlib.Figure
    """
    logger.info("Plotting SHAP importance bar chart.")
    plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False, **kwargs)
    fig = plt.gcf()
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP importance bar chart saved to: {save_path}")
    return fig

# =========================================================
# 2. Beeswarm Plot
# =========================================================
def plot_beeswarm(shap_values, X, feature_names=None, save_path=None, **kwargs):
    """Plot a SHAP beeswarm (swarm) summary plot.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values; shape (n_samples, n_features).
    X : pd.DataFrame or np.ndarray
        Feature matrix used as the data source for the plot.
    feature_names : list of str or None, optional
        Feature names; auto-detected from X.columns when None.
    save_path : str or None, optional
        Path to save the PNG; figure is returned without saving if None.
    **kwargs
        Extra keyword arguments forwarded to shap.plots.beeswarm()
        (e.g. max_display=20).

    Returns
    -------
    matplotlib.Figure
    """
    logger.info("Plotting SHAP beeswarm plot.")

    if feature_names is None:
        feature_names = X.columns if hasattr(X, "columns") else [f"F{i}" for i in range(X.shape[1])]

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.mean(shap_values),
        data=X.values if hasattr(X, "values") else X,
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, show=False, **kwargs)
    fig = plt.gcf()

    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP beeswarm plot saved to: {save_path}")

    return fig


# =========================================================
# 3. Dependence Plot
# =========================================================
def plot_dependence_2d_lowess(shap_values, X,
                              x_feature,       # X-axis: feature values
                              y_feature=None,  # Y-axis: SHAP values (defaults to x_feature)
                              xlim=None, ylim=None, save_path=None,
                              lower=0.01, upper=0.99):
    """Plot a 2-D SHAP dependence scatter with a LOWESS trend line.

    Outliers beyond the [lower, upper] quantile range of x_feature are excluded.
    When x_feature and y_feature differ, the plot shows a cross-dependence view.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values; shape (n_samples, n_features).
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    x_feature : str or int
        Feature for the X axis (raw values).
    y_feature : str, int, or None, optional
        Feature whose SHAP value is plotted on Y; defaults to x_feature.
    xlim : tuple or None, optional
        X-axis limits (min, max).
    ylim : tuple or None, optional
        Y-axis limits (min, max).
    save_path : str or None, optional
        Path to save the PNG.
    lower : float, optional
        Lower quantile for outlier removal (default 0.01).
    upper : float, optional
        Upper quantile for outlier removal (default 0.99).

    Returns
    -------
    matplotlib.Figure or None
        Figure is returned when save_path is None; None otherwise.
    """
    if y_feature is None:
        y_feature = x_feature

    logger.info(f"Plotting 2D LOWESS dependence: X={x_feature}, Y=SHAP({y_feature}).")

    if isinstance(X, pd.DataFrame):
        try:
            x_data    = X[x_feature].values
            col_idx_y = X.columns.get_loc(y_feature)
            y_shap    = shap_values[:, col_idx_y]
        except (KeyError, ValueError) as e:
            # get_loc() raises ValueError for missing labels; [] raises KeyError.
            logger.error(f"Feature not found: {e}")
            return
    else:
        x_data    = X[:, int(x_feature)]
        y_shap    = shap_values[:, int(y_feature)]
        x_feature = f"Feature {x_feature}"
        y_feature = f"Feature {y_feature}"

    # Remove X-axis outliers; Y follows the same mask.
    q_low, q_high = np.quantile(x_data, lower), np.quantile(x_data, upper)
    mask    = (x_data >= q_low) & (x_data <= q_high)
    x_clean = x_data[mask]
    y_clean = y_shap[mask]

    fig, ax = plt.subplots(dpi=150, figsize=(6, 4))
    ax.axhline(0, linestyle='--', color='black')
    ax.scatter(x_clean, y_clean, c="#2196F3", s=12, edgecolors="white", lw=0.3, alpha=0.7)

    try:
        smoothed = lowess(y_clean, x_clean, frac=0.25, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", lw=2, label='Trend')
    except Exception as e:
        logger.warning(f"LOWESS calculation failed: {e}")

    if ylim: ax.set_ylim(ylim)
    if xlim: ax.set_xlim(xlim)
    ax.set_xlabel(str(x_feature))
    ax.set_ylabel(f"SHAP value for\n{y_feature}")
    if x_feature != y_feature:
        ax.set_title(f"Cross-Dependence: {x_feature} vs SHAP({y_feature})", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"2D LOWESS dependence plot saved to: {save_path}")
    else:
        return fig


def plot_dependence_3d_interaction(shap_values, X,
                                   x_feature,                # X-axis: feature values
                                   y_feature=None,           # Y-axis: SHAP values (defaults to x_feature)
                                   interaction_feature=None, # Colour: feature values
                                   xlim=None, ylim=None, save_path=None):
    """Plot a 3-D SHAP dependence scatter with a colour interaction term.

    X axis shows raw feature values, Y axis shows SHAP values, and an optional
    third feature drives the point colour for interaction exploration.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values; shape (n_samples, n_features).
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    x_feature : str or int
        Feature for the X axis (raw values).
    y_feature : str, int, or None, optional
        Feature whose SHAP value is plotted on Y; defaults to x_feature.
    interaction_feature : str, int, or None, optional
        Feature used to colour points; no colour mapping when None.
    xlim : tuple or None, optional
        X-axis limits (min, max).
    ylim : tuple or None, optional
        Y-axis limits (min, max).
    save_path : str or None, optional
        Path to save the PNG.

    Returns
    -------
    matplotlib.Figure or None
        Figure is returned when save_path is None; None otherwise.
    """
    if y_feature is None:
        y_feature = x_feature

    logger.info(
        f"Plotting 3D interaction dependence: "
        f"X={x_feature}, Y=SHAP({y_feature}), colour={interaction_feature}."
    )

    try:
        if isinstance(X, pd.DataFrame):
            x_data    = X[x_feature].values
            col_idx_y = X.columns.get_loc(y_feature)
            y_data    = shap_values[:, col_idx_y]
            c_data    = X[interaction_feature].values if interaction_feature else None
        else:
            x_data    = X[:, int(x_feature)]
            y_data    = shap_values[:, int(y_feature)]
            c_data    = X[:, int(interaction_feature)] if interaction_feature else None
            x_feature = f"Feature {x_feature}"
            y_feature = f"Feature {y_feature}"
            if interaction_feature:
                interaction_feature = f"Feature {interaction_feature}"
    except KeyError as e:
        logger.error(f"Feature not found: {e}")
        return

    fig, ax = plt.subplots(dpi=150, figsize=(6, 4))

    if c_data is not None:
        # Colour interaction uses SHAP's red-blue palette.
        vmin    = np.nanpercentile(c_data, 1)
        vmax    = np.nanpercentile(c_data, 99)
        scatter = ax.scatter(x_data, y_data, c=c_data,
                             cmap=shap.plots.colors.red_blue,
                             vmin=vmin, vmax=vmax,
                             s=12, edgecolors="white", lw=0.3, alpha=1.0)
        plt.colorbar(scatter, ax=ax).set_label(interaction_feature, fontsize=10)
    else:
        ax.scatter(x_data, y_data, c="#2196F3",
                   s=12, edgecolors="white", lw=0.3, alpha=0.7)

    ax.axhline(0, linestyle='--', color='black')
    if ylim: ax.set_ylim(ylim)
    if xlim: ax.set_xlim(xlim)
    ax.set_xlabel(str(x_feature))
    ax.set_ylabel(f"SHAP value for\n{y_feature}")
    title = f"Dependence: {x_feature}"
    if x_feature != y_feature:
        title += f" (SHAP: {y_feature})"
    ax.set_title(title, fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"3D interaction dependence plot saved to: {save_path}")
    else:
        return fig


# =========================================================
# 4. Positive/Negative Ratio Plot
# =========================================================
def plot_pos_neg_ratio(shap_values, feature_names, save_path=None, color_palette=None):
    """Plot positive/negative SHAP contribution ratio per feature.

    Shows the fraction of samples where each feature pushes the prediction
    up (positive) versus down (negative).

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values; shape (n_samples, n_features).
    feature_names : list of str
        Feature names corresponding to columns of shap_values.
    save_path : str or None, optional
        Path to save the PNG.
    color_palette : list of str or None, optional
        Two-element list of hex colours [positive, negative].
        Defaults to ['#9acffa', '#aeb5ba'].

    Returns
    -------
    matplotlib.Axes
    """
    logger.info("Plotting SHAP positive/negative ratio chart.")

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    pos_neg_ratios = {}

    for column in shap_df.columns:
        positive_count = (shap_df[column] > 0).sum()
        negative_count = (shap_df[column] < 0).sum()
        total = positive_count + negative_count
        if total > 0:
            pos_neg_ratios[column] = {
                'Positive': positive_count / total,
                'Negative': negative_count / total
            }

    ratios_df = pd.DataFrame(pos_neg_ratios).T
    filtered_df = ratios_df[ratios_df['Positive'] > 0].sort_values('Positive', ascending=True)

    if color_palette is None:
        color_palette = ['#9acffa', '#aeb5ba']

    fig, ax = plt.subplots(figsize=(10, max(6, len(filtered_df)*0.3)))
    filtered_df.plot(kind='barh', stacked=True, color=color_palette, ax=ax, width=0.8, edgecolor='none')

    ax.set_title('Positive and Negative Ratios per Feature')
    ax.set_xlabel('Ratio')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend(['Positive', 'Negative'], loc='lower right', frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()
        logger.info(f"SHAP positive/negative ratio chart saved to: {save_path}")

    return ax


# =========================================================
# 5. Standard SHAP chart set (swarm + bar + waterfall)
# =========================================================
def plot_all_shap_charts(shap_data_dict, save_dir="shap_plots"):
    """Generate the standard SHAP chart set: swarm, bar, and waterfall plots.

    Parameters
    ----------
    shap_data_dict : dict
        Bundle returned by load_shap_results(); must contain keys
        'explainer', 'shap_values', 'X'.
    save_dir : str
        Output directory for saved PNG files.
    """
    os.makedirs(save_dir, exist_ok=True)

    shap_values = shap_data_dict["shap_values"]
    X           = shap_data_dict["X"]
    explainer   = shap_data_dict["explainer"]

    # For classification models, take the positive class (index 1) or the first class.
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # Extract the base value from the explainer for the waterfall plot.
    base_value = 0
    if hasattr(explainer, "expected_value"):
        val = explainer.expected_value
        if isinstance(val, (np.ndarray, list)):
            base_value = val.item() if np.size(val) == 1 else val[0]
        else:
            base_value = val

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=X.values if hasattr(X, "values") else X,
        feature_names=X.columns if hasattr(X, "columns") else [f"Feat {i}" for i in range(shap_values.shape[1])]
    )

    logger.info(f"Generating SHAP plots — output directory: {save_dir}")

    # Swarm plot: feature direction and magnitude
    logger.info("Plotting swarm plot (shap_summary_dot.png).")
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(save_dir, "shap_summary_dot.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # Bar plot: global feature importance ranking
    logger.info("Plotting importance bar chart (shap_summary_bar.png).")
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig(os.path.join(save_dir, "shap_summary_bar.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # Waterfall plot: single-sample decision breakdown (sample 0)
    logger.info("Plotting waterfall plot for sample 0 (shap_waterfall_sample0.png).")
    try:
        plt.figure()
        shap.plots.waterfall(explanation[0], show=False)
        plt.savefig(os.path.join(save_dir, "shap_waterfall_sample0.png"), bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        logger.error(f"Waterfall plot failed (possible SHAP version incompatibility): {e}")
        plt.close()

    logger.info(f"All SHAP plots saved to: {save_dir}")
