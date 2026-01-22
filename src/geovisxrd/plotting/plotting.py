import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from statsmodels.nonparametric.smoothers_lowess import lowess

# Global font settings
plt.rcParams.update({'font.size': 12})

# =========================================================
# 1. Summary Bar Plot (Overall Importance Bar Chart)
# =========================================================
def plot_summary_bar(shap_values, feature_names, save_path=None, **kwargs):
    """
    Plot SHAP importance bar chart.
    **kwargs: additional parameters passed to shap.summary_plot (e.g., max_display=10)
    """
    print(f"INFO: Plotting Summary Bar ({save_path if save_path else 'display'})...")
    plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, plot_type="bar", show=False, **kwargs)
     
    fig = plt.gcf()  
    
    # Save if path provided; otherwise return object for further modification
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()  # Close to release memory
    
    return fig

# =========================================================
# 2. Beeswarm Plot
# =========================================================
def plot_beeswarm(shap_values, X, feature_names=None, save_path=None, **kwargs):
    """
    Plot SHAP beeswarm plot.
    """
    print(f"INFO: Plotting Beeswarm ({save_path if save_path else 'display'})...")
    
    if feature_names is None:
        feature_names = X.columns if hasattr(X, "columns") else [f"F{i}" for i in range(X.shape[1])]

    # Construct Explanation object
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.mean(shap_values), 
        data=X.values if hasattr(X, "values") else X,
        feature_names=feature_names
    )

    plt.figure(figsize=(10, 8))
    
    # Pass kwargs to shap.plots.beeswarm (e.g., max_display=20)
    shap.plots.beeswarm(explanation, show=False, **kwargs)
    
    fig = plt.gcf()
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        
    return fig


# =========================================================
# 3. Dependence Plot 
# =========================================================
def plot_dependence_2d_lowess(shap_values, X, 
                              x_feature,           # [Dimension 1] X-axis (feature values)
                              y_feature=None,      # [Dimension 2] Y-axis (SHAP values) - defaults to x_feature
                              xlim=None, ylim=None, save_path=None, 
                              lower=0.01, upper=0.99):
    """
    2D dependence plot + red line (Lowess) + outlier removal
    Supports free combination of X and Y (SHAP) axes.
    """
    # 1. Default Y-axis = X-axis (standard dependence plot)
    if y_feature is None:
        y_feature = x_feature
        
    print(f"INFO: Plotting 2D Lowess: X={x_feature}, Y=SHAP({y_feature}) ...")
    
    # 2. Get data
    if isinstance(X, pd.DataFrame):
        try:
            # Get X-axis data (Raw Value)
            x_data = X[x_feature].values
            # Get Y-axis data (SHAP Value)
            col_idx_y = X.columns.get_loc(y_feature)
            y_shap = shap_values[:, col_idx_y]
        except KeyError as e:
            print(f"ERROR: Feature not found: {e}")
            return
    else:
        # NumPy indexing mode
        x_data = X[:, int(x_feature)]
        y_shap = shap_values[:, int(y_feature)]
        # Update display names
        x_feature = f"Feature {x_feature}"
        y_feature = f"Feature {y_feature}"

    # 3. Remove outliers (filter X-axis only, Y follows)
    q_low = np.quantile(x_data, lower)
    q_high = np.quantile(x_data, upper)
    mask = (x_data >= q_low) & (x_data <= q_high)
    
    x_clean = x_data[mask]
    y_clean = y_shap[mask]

    # 4. Plotting
    fig, ax = plt.subplots(dpi=150, figsize=(6, 4))
    
    # Zero baseline
    ax.axhline(0, linestyle='--', color='black')
    
    # Blue scatter points
    ax.scatter(x_clean, y_clean, c="#2196F3", s=12, edgecolors="white", lw=0.3, alpha=0.7)
    
    # Red smoothing line (Lowess)
    try:
        smoothed = lowess(y_clean, x_clean, frac=0.25, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", lw=2, label='Trend')
    except Exception as e:
        print(f"WARNING: Lowess calculation failed: {e}")

    # Axis limits
    if ylim: ax.set_ylim(ylim)
    if xlim: ax.set_xlim(xlim)
    
    # Auto-generate labels
    ax.set_xlabel(str(x_feature))
    ax.set_ylabel(f"SHAP value for\n{y_feature}")
    
    # Title (show if X and Y differ)
    if x_feature != y_feature:
        ax.set_title(f"Cross-Dependence: {x_feature} vs SHAP({y_feature})", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   - Saved: {save_path}")
    else:
        return fig


def plot_dependence_3d_interaction(shap_values, X, 
                                   x_feature,               # [Dimension 1] X-axis: feature values
                                   y_feature=None,          # [Dimension 2] Y-axis: SHAP values (defaults to x_feature)
                                   interaction_feature=None,# [Dimension 3] Color: feature values
                                   xlim=None, ylim=None, save_path=None):
    """
    3D dependence plot + color interaction
    X, Y, Color are completely independent dimensions.
    """
    # 1. Default Y-axis = X-axis
    if y_feature is None:
        y_feature = x_feature

    print(f"INFO: Plotting 3D Interaction: X={x_feature}, Y=SHAP({y_feature}), Color={interaction_feature} ...")
    
    # 2. Get data
    try:
        if isinstance(X, pd.DataFrame):
            # X-axis data
            x_data = X[x_feature].values
            # Y-axis data (SHAP)
            col_idx_y = X.columns.get_loc(y_feature)
            y_data = shap_values[:, col_idx_y]
            # Color data
            if interaction_feature:
                c_data = X[interaction_feature].values
            else:
                c_data = None
        else:
            # NumPy mode
            x_data = X[:, int(x_feature)]
            y_data = shap_values[:, int(y_feature)]
            c_data = X[:, int(interaction_feature)] if interaction_feature else None
            # Update names
            x_feature = f"Feature {x_feature}"
            y_feature = f"Feature {y_feature}"
            if interaction_feature: interaction_feature = f"Feature {interaction_feature}"

    except KeyError as e:
        print(f"ERROR: Invalid feature name: {e}")
        return

    # 3. Plotting
    fig, ax = plt.subplots(dpi=150, figsize=(6, 4))
    
    # Plot scatter points
    if c_data is not None:
        # With color interaction: use red_blue colormap (SHAP default style)
        vmin = np.nanpercentile(c_data, 1)
        vmax = np.nanpercentile(c_data, 99)
        
        scatter = ax.scatter(x_data, y_data, c=c_data, 
                             cmap=shap.plots.colors.red_blue,
                             vmin=vmin, vmax=vmax, 
                             s=12, edgecolors="white", lw=0.3, alpha=1.0)
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(interaction_feature, fontsize=10)
    else:
        # No color interaction: default blue
        ax.scatter(x_data, y_data, c="#2196F3", 
                   s=12, edgecolors="white", lw=0.3, alpha=0.7)

    # 4. Decoration
    ax.axhline(0, linestyle='--', color='black')
    
    if ylim: ax.set_ylim(ylim)
    if xlim: ax.set_xlim(xlim)
    
    ax.set_xlabel(str(x_feature))
    ax.set_ylabel(f"SHAP value for\n{y_feature}")
    
    # Title
    title = f"Dependence: {x_feature}"
    if x_feature != y_feature:
        title += f" (SHAP: {y_feature})"
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"INFO: Saved: {save_path}")
    else:
        return fig


# =========================================================
# 4. Positive/Negative Ratio Plot
# =========================================================
def plot_pos_neg_ratio(shap_values, feature_names, save_path=None, color_palette=None):
    """
    Plot positive/negative contribution ratio chart
    """
    print(f"INFO: Plotting Ratio Plot ({save_path if save_path else 'display'})...")
    
    # Data processing
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
        
    return ax

# =========================================================
# 5. Causal plotting utils
# =========================================================
import networkx as nx

def plot_causal_graph_networkx(nx_graph, threshold=0.0, title="Causal Graph", 
                               save_path=None, show=False):
    """
    Plot causal graph.
    New parameter: show (bool) - whether to display the plot window. 
    Recommended to set to False for batch processing.
    """
    print(f"INFO: Plotting causal graph (Threshold={threshold})...")
    
    # 1. Filter weak edges
    filtered_edges = []
    for u, v, d in nx_graph.edges(data=True):
        weight = d.get('weight', 0)
        if abs(weight) >= threshold:
            filtered_edges.append((u, v, weight))
    
    filtered_graph = nx.DiGraph()
    filtered_graph.add_nodes_from(nx_graph.nodes())
    filtered_graph.add_weighted_edges_from(filtered_edges)

    # 2. Plotting
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(filtered_graph, seed=42)
    
    nx.draw(
        filtered_graph, pos, 
        with_labels=True, 
        node_size=3000, 
        node_color='lightblue',
        font_size=10, 
        font_weight='bold', 
        edge_color='gray',
        arrowsize=20
    )
    
    edge_labels = {(u, v): f"{w:.2f}" for u, v, w in filtered_edges}
    nx.draw_networkx_edge_labels(
        filtered_graph, pos, 
        edge_labels=edge_labels,
        font_color='red'
    )

    plt.title(f"{title} (Threshold={threshold})")
    
    # 3. Save
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"INFO: Plot saved: {save_path}")
        
    if show:
        plt.show()  # Blocking, wait for user to close
    else:
        plt.close()  # Non-blocking, release memory and continue
    
    return filtered_edges
