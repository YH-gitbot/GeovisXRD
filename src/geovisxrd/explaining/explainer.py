# src/geovisxrd/explaining/explainer.py
"""
SHAP computation utilities.

Provides:
  select_shap_explainer() — choose a SHAP explainer for a given model
  shapexplainer()         — compute SHAP values in batches and return them

No figures are generated here.  All visualisation lives in geovisxrd.plotting.
Typical usage after this refactor:

    explainer, shap_values = shapexplainer(model, X, model_family="tree")
    # then plot explicitly:
    from geovisxrd.plotting import plot_summary_bar
    plot_summary_bar(shap_values, list(X.columns), save_path="shap_bar.png")
"""

import warnings
import shap
import numpy as np
from tqdm import tqdm

from .._logging import get_logger
logger = get_logger(__name__)


def select_shap_explainer(model, X, model_family="auto"):
    """Return a SHAP explainer appropriate for the given model.

    Parameters
    ----------
    model : fitted estimator
    X : pd.DataFrame
        Feature matrix; used as background data for the generic fallback.
    model_family : {"auto", "tree", "linear", "generic"}
        Which explainer family to use.

        "auto"    — infer from the model's class name (default).
        "tree"    — force shap.TreeExplainer (XGBoost, LightGBM, RF, …).
        "linear"  — force shap.LinearExplainer (LinearRegression, Ridge, …).
        "generic" — force the kernel-based shap.Explainer (slow; works for
                    any predict-compatible model including MLP).

    Returns
    -------
    shap.Explainer (or subclass)
    """
    logger.info(f"Model type: {type(model).__name__}  |  model_family='{model_family}'.")

    if model_family == "tree":
        logger.info("Using TreeExplainer (explicit).")
        return shap.TreeExplainer(model)

    if model_family == "linear":
        logger.info("Using LinearExplainer (explicit).")
        return shap.LinearExplainer(model, X)

    if model_family == "generic":
        logger.info("Using generic Explainer (explicit).")
        background = shap.sample(X, 100) if len(X) > 100 else X
        return shap.Explainer(model.predict, background)

    # model_family == "auto": infer from class name
    name = type(model).__name__.lower()

    tree_keywords    = ["forest", "xgb", "lgbm", "tree", "boosting", "gradientboost"]
    linear_keywords  = ["linear", "ridge", "lasso", "elastic"]

    if any(k in name for k in tree_keywords):
        try:
            logger.info("Auto-selected TreeExplainer.")
            return shap.TreeExplainer(model)
        except Exception as e:
            logger.error(f"TreeExplainer failed: {e}. Falling back to generic Explainer.")

    if any(k in name for k in linear_keywords):
        try:
            logger.info("Auto-selected LinearExplainer.")
            return shap.LinearExplainer(model, X)
        except Exception as e:
            logger.warning(f"LinearExplainer failed: {e}. Falling back to generic Explainer.")

    logger.info("Auto-selected generic Explainer (class name not recognised; may be slow).")
    background = shap.sample(X, 100) if len(X) > 100 else X
    return shap.Explainer(model.predict, background)


def shapexplainer(model, X, model_family="auto", batch_size=1000,
                  # ── deprecated parameters ──────────────────────────────────
                  plot_type=None, save_path=None):
    """Compute SHAP values for a fitted model.

    Parameters
    ----------
    model : fitted estimator
        Tree, linear, or any sklearn-compatible model.
    X : pd.DataFrame
        Feature matrix (n_samples × n_features).
    model_family : {"auto", "tree", "linear", "generic"}, optional
        Passed directly to select_shap_explainer().  When the model family is
        known upstream (e.g. after calling train_model("xgb", …)), passing
        "tree" avoids class-name inference (default "auto").
    batch_size : int, optional
        Number of samples per SHAP computation batch (default 1000).

    Deprecated parameters (emit a warning, have no effect)
    -------------------------------------------------------
    plot_type : str or None
        Formerly used to select the SHAP summary plot type.
        Plotting has moved to geovisxrd.plotting — call plot_summary_bar()
        or plot_beeswarm() explicitly after this function.
    save_path : str or None
        Formerly used to save a SHAP summary plot.
        Plotting has moved to geovisxrd.plotting.

    Returns
    -------
    explainer : shap.Explainer (or subclass)
    shap_values : np.ndarray, shape (n_samples, n_features)
    """
    if plot_type is not None:
        warnings.warn(
            "shapexplainer(): 'plot_type' is deprecated and has no effect. "
            "Plotting has moved to geovisxrd.plotting — call plot_summary_bar() "
            "or plot_beeswarm() explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
    if save_path is not None:
        warnings.warn(
            "shapexplainer(): 'save_path' is deprecated and has no effect. "
            "Plotting has moved to geovisxrd.plotting — call plot_summary_bar() "
            "or plot_beeswarm() explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )

    explainer = select_shap_explainer(model, X, model_family=model_family)
    logger.info(f"Computing SHAP values — {len(X)} samples, batch_size={batch_size}.")

    shap_values_list = []

    # NOTE — tqdm writes directly to sys.stderr and is not routed through the
    # Python logging module, so _suppress_third_party() has no effect on it.
    # To silence the bar globally, set tqdm.tqdm.disable = True before calling.
    for i in tqdm(range(0, len(X), batch_size),
                  desc="SHAP batches", unit="batch", disable=None):
        batch_X = X.iloc[i : i + batch_size]

        # TreeExplainer accepts check_additivity; other explainers do not.
        if isinstance(explainer, shap.TreeExplainer):
            batch_shap = explainer.shap_values(batch_X, check_additivity=False)
        elif hasattr(explainer, "shap_values"):
            batch_shap = explainer.shap_values(batch_X)
        else:
            batch_shap = explainer(batch_X).values

        shap_values_list.append(batch_shap)

    shap_values = np.concatenate(shap_values_list, axis=0)
    logger.info(f"SHAP values computed — shape: {shap_values.shape}.")

    return explainer, shap_values
