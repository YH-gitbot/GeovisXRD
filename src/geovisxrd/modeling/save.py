# src/geovisxrd/save.py

import os
import joblib
from datetime import datetime


def save_model(model, model_name: str, save_dir: str = "models"):
    '''
    Save the trained model to disk with a timestamped filename.
    Args:
        model: Trained model instance
        model_name (str): Name of the model
        save_dir (str): Directory to save the model files
    Returns:
        filepath (str): Path to the saved model file
    '''
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)

    joblib.dump(model, filepath)

    print(f"Save {filename} to {filepath}")
    return filepath
