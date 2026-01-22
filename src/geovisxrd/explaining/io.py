# src/geovisxrd/explaining/io.py

import joblib
import os
from datetime import datetime
import pandas as pd

def save_shap_results(explainer, shap_values, X, save_dir="shap_results", name_prefix="shap"):

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    joblib_filename = f"{name_prefix}_{timestamp}.joblib"
    joblib_path = os.path.join(save_dir, joblib_filename)
    
    data_to_save = {
        "explainer": explainer,    
        "shap_values": shap_values,
        "X": X                    
    }
    
    joblib.dump(data_to_save, joblib_path)
    print(f"INFO: [GeoVisXRD] Complete data (binary) saved to: {joblib_path}")

    # Create subdirectory for CSV files
    csv_dir = os.path.join(save_dir, "csv_data")
    os.makedirs(csv_dir, exist_ok=True)

    csv_filename = f"{name_prefix}_{timestamp}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    
    try:
        # SHAP values can be list (classification) or array (regression)
        values_arr = shap_values
        if isinstance(shap_values, list):
            # If list, typically take index 1 (positive class)
            values_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Determine column names
        cols = X.columns if hasattr(X, "columns") else [f"feature_{i}" for i in range(values_arr.shape[1])]
        
        # Create DataFrame
        df_shap = pd.DataFrame(values_arr, columns=cols)
        
        # Save to CSV
        df_shap.to_csv(csv_path, index=False)
        print(f"INFO: [GeoVisXRD] SHAP values table (CSV) saved to: {csv_path}")
        
    except Exception as e:
        # Print specific error for debugging
        print(f"ERROR: [GeoVisXRD] CSV save failed: {e}")

    # Return binary file path for later plotting
    return joblib_path


def load_shap_results(filepath):
    """
    Load binary SHAP data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ERROR: [GeoVisXRD] File not found: {filepath}")
    
    print(f"INFO: [GeoVisXRD] Loading SHAP data: {filepath} ...")
    data = joblib.load(filepath)
    
    # Return the entire dictionary containing explainer, shap_values, X, etc.
    return data
