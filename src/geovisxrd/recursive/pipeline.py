import os
import pandas as pd
from geovisxrd.modeling.trainer import train_model
from geovisxrd.modeling.save import save_model
from geovisxrd.explaining.explainer import shapexplainer
from geovisxrd.explaining.io import save_shap_results, load_shap_results
from geovisxrd.causal.discovery import get_causal_model
from geovisxrd.threshold.optimizer import analyze_threshold_stability
from geovisxrd.plotting.plotting import plot_causal_graph_networkx

def run_recursive_step(layer_num, previous_shap_path, target_feature_name):
    """
    Execute GeoVisXRD recursive analysis.
    Modification: Automatically create result folder in the same directory as input file.
    """
    # 1. Automatically locate the directory containing the input file
    # If path is "test_outputs/shap/xgb_data/shap_xxx.joblib"
    # input_dir will become "test_outputs/shap/xgb_data"
    input_dir = os.path.dirname(os.path.abspath(previous_shap_path))
    
    # 2. Define new subfolder name and path
    new_folder_name = f"layer_{layer_num}_{target_feature_name}"
    step_dir = os.path.join(input_dir, new_folder_name)
    
    # 3. Ensure directory exists
    os.makedirs(step_dir, exist_ok=True)

    # --- Business logic below ---
    print(f"INFO: [GeoVisXRD] Starting layer {layer_num} analysis: {target_feature_name} ---")
    print(f"INFO: Results will be saved to: {step_dir}")

    # Load data
    saved_data = load_shap_results(previous_shap_path)
    X = saved_data["X"]
    prev_shap_values = saved_data["shap_values"]
    
    # Extract target SHAP values
    feat_idx = list(X.columns).index(target_feature_name)
    new_y = pd.Series(prev_shap_values[:, feat_idx], index=X.index, name=f"SHAP_{target_feature_name}")

    # Train model and save (to new step_dir)
    model, _ = train_model("xgb", X, new_y, n_estimators=100, max_depth=6)
    save_model(model, f"layer{layer_num}_xgb", save_dir=os.path.join(step_dir, "models"))
    
    explainer, shap_values = shapexplainer(model, X, save_path=os.path.join(step_dir, "shap_summary.png"))
    current_shap_path = save_shap_results(explainer, shap_values, X, 
                                          save_dir=os.path.join(step_dir, "shap_data"), 
                                          name_prefix=f"layer{layer_num}")

    # Causal discovery (to new step_dir)
    sam_params = {"nruns": 1, "train_epochs": 100, "nh": 10}
    combined_df = pd.concat([X, new_y.to_frame()], axis=1)
    sam_model = get_causal_model("sam", **sam_params)
    sam_model.fit(combined_df)
    
    _, best_t = analyze_threshold_stability(sam_model, save_path=os.path.join(step_dir, "stability_curve.png"))
    
    plot_causal_graph_networkx(sam_model.get_nx_graph(), threshold=best_t, 
                               title=f"Layer {layer_num} Causal (t={best_t})", 
                               save_path=os.path.join(step_dir, "final_causal_graph.png"))

    print(f"INFO: [GeoVisXRD] Analysis complete. Next layer data path: {current_shap_path} ---")
    return current_shap_path
