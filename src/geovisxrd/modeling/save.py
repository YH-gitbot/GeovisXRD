# src/geovisxrd/modeling/save.py

import os
import joblib
from datetime import datetime

from .._logging import get_logger
logger = get_logger(__name__)


def save_model(model, model_name: str, save_dir: str = "models"):
    """Save a trained model to disk with a timestamped filename.

    Parameters
    ----------
    model      : trained model instance
    model_name : str — short name used in the filename (e.g. 'xgb')
    save_dir   : str — output directory (created if absent)

    Returns
    -------
    str : absolute path to the saved .pkl file
    """
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{model_name}_{timestamp}.pkl"
    filepath = os.path.join(save_dir, filename)

    joblib.dump(model, filepath)

    logger.info(f"Model saved to: {filepath}")
    return filepath
