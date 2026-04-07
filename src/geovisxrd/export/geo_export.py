# src/geovisxrd/export/geo_export.py
"""
Standardised geographic export for GeoVisXRD results.

Column contract (in order)
--------------------------
[lat_col]      latitude or y-coordinate   (present only when found)
[lon_col]      longitude or x-coordinate  (present only when found)
y_true         ground-truth target values
y_pred         model predictions
residual       y_true - y_pred
abs_residual   |residual|
shap_{f1}      SHAP contribution of feature f1
shap_{f2}      SHAP contribution of feature f2
...

This format is directly usable in:
  - pandas / geopandas
  - QGIS (load CSV with lat/lon, or open GeoPackage)
  - any GIS software that accepts GeoJSON / GPKG
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime

from .._logging import get_logger
logger = get_logger(__name__)

# ==============================================================================
# Coordinate keyword patterns (case-insensitive substring match)
# ==============================================================================
_LAT_KEYWORDS = ["latitude", "y_coord", "northing"]
_LON_KEYWORDS = ["longitude", "x_coord", "easting"]
# TODO: substring matching is order-dependent and may match unintended columns
#       (e.g. "my_latitude" or "latitude_raw"). Consider exact-match preference
#       or user-supplied override before relying on this in production.


# ==============================================================================
# 1. Internal helpers
# ==============================================================================

def _detect_coord_cols(columns):
    """
    Auto-detect latitude and longitude column names from a list of column names.
    Returns (lat_col, lon_col), either of which may be None if not found.
    """
    lat_col = lon_col = None
    for c in columns:
        cl = c.lower()
        if lat_col is None and any(k in cl for k in _LAT_KEYWORDS):
            lat_col = c
        if lon_col is None and any(k in cl for k in _LON_KEYWORDS):
            lon_col = c
    return lat_col, lon_col


def _format_shap_columns(shap_values, feature_names) -> pd.DataFrame:
    """
    Build a DataFrame of SHAP values with 'shap_' prefixed column names.

    Parameters
    ----------
    shap_values  : np.ndarray (n_samples, n_features) or list of arrays.
                   For classification models that return a list, index 1
                   (positive class) is used when available.
    feature_names : list[str] — original feature names (without prefix)

    Returns
    -------
    pd.DataFrame with columns ['shap_f1', 'shap_f2', ...]
    """
    if isinstance(shap_values, list):
        # Classification: take positive class; else take first class
        arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        arr = shap_values

    shap_cols = [f"shap_{f}" for f in feature_names]
    return pd.DataFrame(arr, columns=shap_cols)


# ==============================================================================
# 2. Core assembly function
# ==============================================================================

def build_geo_table(
    X,
    y_true,
    y_pred,
    shap_values,
    lat_col=None,
    lon_col=None,
) -> pd.DataFrame:
    """
    Assemble the standardised geo-export table from model outputs.

    Causal tier context
    -------------------
    Geographic columns (latitude, longitude) are placed first because they
    are root variables — all other values are derived from or influenced by
    location.  The target and error columns come next, followed by per-feature
    SHAP contributions.

    Parameters
    ----------
    X           : pd.DataFrame  — original feature matrix (n_samples × n_features)
    y_true      : array-like    — ground-truth target values
    y_pred      : array-like    — model predictions
    shap_values : np.ndarray or list — SHAP values (n_samples × n_features)
    lat_col     : str | None    — latitude column name in X; auto-detected if None
    lon_col     : str | None    — longitude column name in X; auto-detected if None

    Returns
    -------
    pd.DataFrame with columns:
        [lat_col, lon_col,] y_true, y_pred, residual, abs_residual, shap_f1, …
    """
    feature_names = list(X.columns)

    # --- Auto-detect coordinates when not explicitly provided ---
    if lat_col is None or lon_col is None:
        auto_lat, auto_lon = _detect_coord_cols(feature_names)
        lat_col = lat_col or auto_lat
        lon_col = lon_col or auto_lon

    # --- Coordinates block: only columns that actually exist in X ---
    coord_cols = [c for c in [lat_col, lon_col] if c is not None and c in X.columns]
    df_coords = X[coord_cols].reset_index(drop=True) if coord_cols else pd.DataFrame()

    # --- Prediction & residual block ---
    y_true_arr   = np.asarray(y_true).ravel()
    y_pred_arr   = np.asarray(y_pred).ravel()
    residual     = y_true_arr - y_pred_arr
    abs_residual = np.abs(residual)

    df_pred = pd.DataFrame({
        "y_true":       y_true_arr,
        "y_pred":       y_pred_arr,
        "residual":     residual,
        "abs_residual": abs_residual,
    })

    # --- SHAP block: shap_{feature} columns ---
    df_shap = _format_shap_columns(shap_values, feature_names)

    # --- Assemble ---
    df_out = pd.concat([df_coords, df_pred, df_shap], axis=1)

    coord_info = f"coords=({lat_col}, {lon_col})" if coord_cols else "no coordinates detected"
    logger.info(
        f"Geo table built — "
        f"{len(df_out)} rows, {coord_info}, "
        f"{len(feature_names)} shap_* columns"
    )
    return df_out


# ==============================================================================
# 3. GeoDataFrame converter (requires geopandas + shapely)
# ==============================================================================

def to_geodataframe(df, lat_col="Latitude", lon_col="Longitude", crs="EPSG:4326"):
    """
    Convert a geo-export DataFrame to a GeoDataFrame with Point geometry.

    The resulting GeoDataFrame can be:
      - saved as GeoPackage (.gpkg) for QGIS
      - saved as GeoJSON for web mapping
      - used directly with geopandas for spatial analysis

    Parameters
    ----------
    df      : pd.DataFrame — output of build_geo_table()
    lat_col : str          — latitude column name  (default "Latitude")
    lon_col : str          — longitude column name (default "Longitude")
    crs     : str          — coordinate reference system (default "EPSG:4326" = WGS84)

    Returns
    -------
    geopandas.GeoDataFrame
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        raise ImportError(
            "geopandas and shapely are required for GeoDataFrame export.\n"
            "Install with: pip install geopandas shapely"
        )

    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(
            f"Coordinate columns '{lat_col}' and/or '{lon_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Point(x=longitude, y=latitude) — standard GIS convention
    geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs)

    logger.info(f"GeoDataFrame created — CRS={crs}, {len(gdf)} features")
    return gdf


# ==============================================================================
# 4. One-stop export function
# ==============================================================================

def save_geo_export(
    X,
    y_true,
    y_pred,
    shap_values,
    save_dir="geo_export",
    name_prefix="geo",
    lat_col=None,
    lon_col=None,
    export_gpkg=False,
    crs="EPSG:4326",
):
    """
    Build the standardised geo table and persist it to disk.

    Always writes a CSV.  Optionally writes a GeoPackage (.gpkg) that can be
    opened directly in QGIS or read with geopandas.read_file().

    Parameters
    ----------
    X            : pd.DataFrame — feature matrix (with coordinate columns if available)
    y_true       : array-like   — ground-truth target values
    y_pred       : array-like   — model predictions
    shap_values  : np.ndarray   — SHAP contribution matrix (n_samples × n_features)
    save_dir     : str          — output directory (created if absent)
    name_prefix  : str          — filename prefix (e.g. model name)
    lat_col      : str | None   — latitude column name; auto-detected from X if None
    lon_col      : str | None   — longitude column name; auto-detected from X if None
    export_gpkg  : bool         — also save .gpkg when coordinate columns are present
    crs          : str          — CRS for GeoPackage (default "EPSG:4326")

    Returns
    -------
    df        : pd.DataFrame    — the assembled geo table
    csv_path  : str             — path to the saved CSV
    gpkg_path : str | None      — path to the GeoPackage, or None if not exported
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Build standardised table ---
    df = build_geo_table(
        X, y_true, y_pred, shap_values,
        lat_col=lat_col, lon_col=lon_col,
    )

    # --- Save CSV ---
    csv_filename = f"{name_prefix}_{timestamp}.csv"
    csv_path     = os.path.join(save_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    logger.info(f"Geo CSV saved to: {csv_path}")

    # --- Optionally save GeoPackage ---
    gpkg_path = None
    if export_gpkg:
        # Re-detect coordinates from the assembled table (handles renamed columns)
        auto_lat, auto_lon = _detect_coord_cols(list(df.columns))
        _lat = lat_col or auto_lat
        _lon = lon_col or auto_lon

        if _lat and _lon and _lat in df.columns and _lon in df.columns:
            gdf           = to_geodataframe(df, lat_col=_lat, lon_col=_lon, crs=crs)
            gpkg_filename = f"{name_prefix}_{timestamp}.gpkg"
            gpkg_path     = os.path.join(save_dir, gpkg_filename)
            gdf.to_file(gpkg_path, driver="GPKG")
            logger.info(f"GeoPackage saved to: {gpkg_path}")
        else:
            logger.warning(
                "export_gpkg=True but no coordinate columns "
                "found — GeoPackage not written."
            )

    return df, csv_path, gpkg_path


# ==============================================================================
# 5. QGIS-compatible export  (matches reference/shap_mapping_qgis_muenchen.py)
# ==============================================================================

def save_qgis_export(
    X,
    y_true,
    y_pred,
    shap_values,
    save_dir="geo_export",
    name_prefix="qgis",
    lat_col=None,
    lon_col=None,
    contributions_col=None,
):
    """
    Export a CSV whose column naming matches the QGIS reference script convention.

    Column layout
    -------------
    Latitude            geographic root variable (WGS84)
    Longitude           geographic root variable (WGS84)
    y_true              ground-truth target
    y_pred              model prediction
    residual            y_true - y_pred
    abs_residual        |residual|
    total_contributions per-row SHAP magnitude (∑|shap_i|), used as circle-size
                        proxy in QGIS; replace with a real contributions column
                        by passing contributions_col=<col_name>
    {f1}_shap           SHAP contribution of feature f1  ← suffix convention
    {f2}_shap           SHAP contribution of feature f2
    ...

    The suffix convention ({feature}_shap) is the one used by the QGIS script.
    The prefix convention (shap_{feature}) is used by save_geo_export / build_geo_table.
    Both formats can coexist; this function produces the QGIS-ready variant.

    Parameters
    ----------
    X                : pd.DataFrame — feature matrix
    y_true           : array-like   — ground-truth target values
    y_pred           : array-like   — model predictions
    shap_values      : np.ndarray   — SHAP matrix (n_samples × n_features)
    save_dir         : str          — output directory
    name_prefix      : str          — filename prefix
    lat_col          : str | None   — latitude column; auto-detected if None
    lon_col          : str | None   — longitude column; auto-detected if None
    contributions_col: str | None   — column in X to use as 'total_contributions';
                                      if None, uses ∑|shap_i| as a proxy

    Returns
    -------
    df       : pd.DataFrame — assembled QGIS-ready table
    csv_path : str          — path to the saved CSV
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_names = list(X.columns)

    # --- Coordinate detection ---
    if lat_col is None or lon_col is None:
        auto_lat, auto_lon = _detect_coord_cols(feature_names)
        lat_col = lat_col or auto_lat
        lon_col = lon_col or auto_lon

    coord_cols = [c for c in [lat_col, lon_col] if c is not None and c in X.columns]
    df_coords  = X[coord_cols].reset_index(drop=True) if coord_cols else pd.DataFrame()

    # --- Prediction & residual ---
    y_true_arr   = np.asarray(y_true).ravel()
    y_pred_arr   = np.asarray(y_pred).ravel()
    residual     = y_true_arr - y_pred_arr
    abs_residual = np.abs(residual)

    df_pred = pd.DataFrame({
        "y_true":       y_true_arr,
        "y_pred":       y_pred_arr,
        "residual":     residual,
        "abs_residual": abs_residual,
    })

    # --- SHAP block with *suffix* naming: {feature}_shap ---
    if isinstance(shap_values, list):
        arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        arr = np.asarray(shap_values)

    suffix_cols = [f"{f}_shap" for f in feature_names]
    df_shap     = pd.DataFrame(arr, columns=suffix_cols)

    # --- total_contributions: real column or ∑|shap_i| proxy ---
    if contributions_col and contributions_col in X.columns:
        contributions = X[contributions_col].reset_index(drop=True).values
    else:
        # Sum of absolute SHAP values per row — indicates overall feature importance
        # at that location; used as circle-size driver in QGIS, matching reference logic
        contributions = np.abs(arr).sum(axis=1)

    df_contrib = pd.DataFrame({"total_contributions": contributions})

    # --- Assemble in QGIS reference column order ---
    df_out = pd.concat([df_coords, df_pred, df_contrib, df_shap], axis=1)

    # --- Save ---
    csv_filename = f"{name_prefix}_{timestamp}.csv"
    csv_path     = os.path.join(save_dir, csv_filename)
    df_out.to_csv(csv_path, index=False)

    coord_info = f"coords=({lat_col}, {lon_col})" if coord_cols else "no coordinates"
    logger.info(
        f"QGIS export saved → {csv_path}\n"
        f"      {len(df_out)} rows, {coord_info}, "
        f"{len(feature_names)} SHAP columns ({feature_names[0]}_shap … format)"
    )
    return df_out, csv_path
