from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

def get_model(model_name: str, **params):
    """Return a configured regressor instance by name.

    Parameters
    ----------
    model_name : str
        Model identifier (case-insensitive).
        Accepted values: 'rf'/'random_forest', 'xgb'/'xgboost',
        'lgb'/'lightgbm', 'mlp', 'linear'/'ols'.
    **params
        Extra keyword arguments forwarded to the model constructor.

    Returns
    -------
    Scikit-learn compatible regressor instance.
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
        raise ValueError(
            f"Unknown model name: '{model_name}'. "
            f"Supported: {sorted(model_dict.keys())}"
        )

    # Stability defaults for repeated validation / pipeline runs.
    # All can be overridden by the caller via **params.
    if model_name in ("rf", "random_forest"):
        # n_jobs=1 avoids multiprocessing instability across repeated runs;
        # random_state=42 ensures reproducibility.
        params.setdefault("n_estimators", 100)
        params.setdefault("n_jobs", 1)
        params.setdefault("random_state", 42)

    if model_name == "mlp":
        # max_iter=300 reduces convergence warnings seen in validation runs.
        params.setdefault("max_iter", 300)
        params.setdefault("random_state", 42)

    # Silence XGBoost and LightGBM training output by default.
    # Callers can override by passing verbosity= / verbose= explicitly.
    #
    # HOW THIS INTERACTS WITH _suppress_third_party() IN _logging.py:
    # There are two independent noise sources for these libraries:
    #
    #   1. C++ native output — training-progress lines such as
    #        "[0] train-rmse:0.500"  (XGBoost)
    #        "[1]\ttraining's rmse: 0.456"  (LightGBM)
    #      These are written directly to stdout/stderr by the C++ core and
    #      CANNOT be suppressed via Python logging.  They are silenced here
    #      via verbosity=0 (XGBoost) and verbose=-1 (LightGBM).
    #
    #   2. Python-level log records — emitted through the libraries' own
    #      Python loggers ("xgboost", "lightgbm").  These ARE routed through
    #      Python logging and are silenced by _suppress_third_party() raising
    #      those loggers' effective level to WARNING.
    #
    # Both mechanisms are necessary; neither alone is sufficient.
    if model_name in ("xgb", "xgboost"):
        params.setdefault("verbosity", 0)
        params.setdefault("random_state", 42)
    if model_name in ("lgb", "lightgbm"):
        params.setdefault("verbose", -1)
        params.setdefault("random_state", 42)

    return model_dict[model_name](**params)
