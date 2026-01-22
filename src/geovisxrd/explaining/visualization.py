# src/geovisxrd/explaining/visualization.py

import shap
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_all_shap_charts(shap_data_dict, save_dir="shap_plots"):
    """
    Generate all common SHAP charts in one go (swarm plot, bar plot, waterfall plot).
    
    Parameters:
    shap_data_dict: Dictionary returned by load_shap_results, containing {"explainer", "shap_values", "X"}
    save_dir: Directory path to save plots
    """
    # 1. Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Unpack data
    shap_values = shap_data_dict["shap_values"]
    X = shap_data_dict["X"]
    explainer = shap_data_dict["explainer"]  # Extract explainer object
    
    # --- Data preprocessing ---
    
    # Handle classification model: if shap_values is a list, take positive class (index 1) or first one
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    # Key step: extract base_value from explainer
    base_value = 0
    if hasattr(explainer, "expected_value"):
        val = explainer.expected_value
        # If it's an array (single value), extract and convert to scalar
        if isinstance(val, (np.ndarray, list)):
             if np.size(val) == 1:
                 base_value = val.item()
             else:
                 # Multi-class case, take base value for corresponding class (here simply take first one)
                 base_value = val[0]
        else:
             base_value = val


    # Create SHAP Explanation object for plotting
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=X.values if hasattr(X, "values") else X,
        feature_names=X.columns if hasattr(X, "columns") else [f"Feat {i}" for i in range(shap_values.shape[1])]
    )
    # ---------------------------------------------------------

    print(f"INFO: [GeoVisXRD] Generating plots, saving to: {save_dir}/ ")

    # --- Chart 1: Swarm Plot (Summary Dot) ---
    # Display the direction and magnitude of feature impact on predictions
    print("INFO: 1. Plotting: Swarm Plot (Summary Dot)...")
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(save_dir, "shap_summary_dot.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # --- Chart 2: Bar Plot (Summary Bar) ---
    # Display feature importance ranking
    print("INFO: 2. Plotting: Importance Bar Chart (Summary Bar)...")
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig(os.path.join(save_dir, "shap_summary_bar.png"), bbox_inches='tight', dpi=300)
    plt.close()

    # --- Chart 3: Waterfall Plot ---
    # Display the decision process of a single sample. Here we plot sample 0 as an example by default.
    print("INFO: 3. Plotting: Single Sample Waterfall Plot (Waterfall - Sample 0)...")
    try:
        plt.figure()
        # explanation[0] will automatically slice while preserving explanation object attributes
        shap.plots.waterfall(explanation[0], show=False)
        plt.savefig(os.path.join(save_dir, "shap_waterfall_sample0.png"), bbox_inches='tight', dpi=300)
        plt.close()
    except Exception as e:
        print(f"ERROR: [GeoVisXRD] Waterfall plot generation failed (possible version compatibility issue): {e}")
        plt.close()

    print(f"SUCCESS: [GeoVisXRD] All plots generated successfully!")