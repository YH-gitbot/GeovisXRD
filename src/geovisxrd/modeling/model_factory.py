from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

def get_model(model_name: str, **params):
    """
    used to get model instance by name 
    Args:
        model_name (str): model name
        **params: model parameters
    Returns:
        model instance      
        
    """
    model_name = model_name.lower()

    model_dict = {
        "random_forest": RandomForestRegressor,
        "rf": RandomForestRegressor,
        "mlp": MLPRegressor,
        "xgboost": XGBRegressor,
        "xgb": XGBRegressor,
        "linear": LinearRegression,
        "ols": LinearRegression,
        "lgb": LGBMRegressor,
        "lightgbm": LGBMRegressor,
    }

    if model_name not in model_dict:
        raise ValueError(f"unknown model: {model_name}")

    return model_dict[model_name](**params)
