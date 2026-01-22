# src/geovisxrd/shap_explain.py

import shap
import matplotlib.pyplot as plt
import numpy as np     
from tqdm import tqdm

def _get_smart_explainer(model, X):

    model_class_name = type(model).__name__.lower()
    
    print(f"INFO: [GeoVisXRD] Detected model type: {type(model).__name__} ---")

    tree_keywords = ['forest', 'xgb', 'lgbm', 'tree', 'boosting', 'gradientboost']
    if any(k in model_class_name for k in tree_keywords):
        try:
            print("INFO: [GeoVisXRD] Using TreeExplainer (for tree models)...")
            return shap.TreeExplainer(model)
        except Exception as e:
            print(f"ERROR: TreeExplainer failed ({e}), trying fallback...")

    linear_keywords = ['linear', 'ridge', 'lasso', 'elastic']
    if any(k in model_class_name for k in linear_keywords):
        try:
            print("INFO: [GeoVisXRD] Using LinearExplainer (for linear models)...")
            return shap.LinearExplainer(model, X)
        except Exception:
            pass

    print("INFO: [GeoVisXRD] Model type not recognized, using generic Explainer (may be slow)...")
    background_data = X
    if len(X) > 100:
        background_data = shap.sample(X, 100)
        
    return shap.Explainer(model.predict, background_data)


def shapexplainer(model, X, plot_type="summary", save_path=None, batch_size=1000):
    
    # 1. Create explainer
    explainer = _get_smart_explainer(model, X)
    print(f"INFO: [GeoVisXRD] Computing SHAP values ({len(X)} samples)...")
    
    # 2. Compute SHAP values in batches
    shap_values_list = []
    
    for i in tqdm(range(0, len(X), batch_size), desc="SHAP computation progress", unit="batch"):
        
        batch_X = X.iloc[i : i + batch_size]
        
        # Check if TreeExplainer. LinearExplainer doesn't need check_additivity parameter
        if isinstance(explainer, shap.TreeExplainer):
            batch_shap = explainer.shap_values(batch_X, check_additivity=False)
        else:
            if hasattr(explainer, "shap_values"):
                batch_shap = explainer.shap_values(batch_X)
            else:
                shap_result = explainer(batch_X)
                batch_shap = shap_result.values
        
        shap_values_list.append(batch_shap)
    
    shap_values = np.concatenate(shap_values_list, axis=0)
    
    print("INFO: [GeoVisXRD] Computation complete, generating plot...")
    
    plt.figure(figsize=(10, 6)) 
    
    if plot_type == "summary":
        shap.summary_plot(shap_values, X, show=False)
    elif plot_type == "bar":
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    
    # 4. Save or display
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"INFO: [GeoVisXRD] Plot saved to: {save_path}")
    else:
        print("ERROR: [GeoVisXRD] No save path provided, cannot save")
        
    return explainer, shap_values
