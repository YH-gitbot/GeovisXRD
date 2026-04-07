# src/geovisxrd/plotting/mapping.py
"""
QGIS-style SHAP atlas for GeoVisXRD geo-export tables.

Public API
----------
plot_shap_single(df, shap_col, *, ...)
    Single-panel SHAP map; creates its own figure.

plot_shap_6panel(df_or_path, *, ...)
    Multi-panel atlas (3 × 2 grid); delegates per-panel drawing to the
    private helper _draw_single_panel().

Both public functions share the same internal drawing logic via
_draw_single_panel(), which handles only actual rendering (no figure
creation, no file I/O).

Boundary policy
---------------
* boundary_path provided → loaded via geopandas (EPSG:4326); raises on failure.
* boundary_path not provided → boundary drawing skipped; no misleading legend shown.

Basemap policy
--------------
* basemap_path provided → loaded locally; no internet access used.
    Supported formats:
    - .mbtiles  — MBTiles tile store (sqlite3 + mercantile + Pillow)
    - .tif/.tiff — GeoTIFF (rasterio); extent auto-converted to WGS-84
    - .png/.jpg  — plain image (matplotlib)
  Raises on load failure.
* basemap_path not provided → basemap drawing skipped.

All coordinates are kept in WGS-84 (EPSG:4326).  No region-specific assumptions
are embedded in this module.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .._logging import get_logger
logger = get_logger(__name__)


# ==============================================================================
# Data helpers
# ==============================================================================

def _load(df_or_path):
    """Accept a DataFrame or a path string; always return a DataFrame."""
    if isinstance(df_or_path, str):
        if not os.path.exists(df_or_path):
            raise FileNotFoundError(f"File not found: {df_or_path}")
        return pd.read_csv(df_or_path)
    return df_or_path.copy()


def _require_cols(df, *cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required column(s) not found in the geo table: {missing}\n"
            f"Available: {list(df.columns)}"
        )


# ==============================================================================
# Boundary drawing — separated into load once / draw many
# ==============================================================================

def _extract_rings(geom, rings):
    """Recursively extract coordinate rings from a GeoJSON geometry dict.

    Appends each ring (list of [lon, lat] pairs) to *rings* in place.
    Handles Polygon, MultiPolygon, GeometryCollection, Feature, FeatureCollection.
    """
    if geom is None:
        return
    gtype = geom.get("type", "")
    if gtype == "Polygon":
        for ring in geom.get("coordinates", []):
            rings.append(ring)
    elif gtype == "MultiPolygon":
        for polygon in geom.get("coordinates", []):
            for ring in polygon:
                rings.append(ring)
    elif gtype == "GeometryCollection":
        for sub in geom.get("geometries", []):
            _extract_rings(sub, rings)
    elif gtype == "Feature":
        _extract_rings(geom.get("geometry"), rings)
    elif gtype == "FeatureCollection":
        for feat in geom.get("features", []):
            _extract_rings(feat, rings)


def _load_boundary(path):
    """Load a boundary file and return a list of coordinate rings (load once).

    GeoJSON / JSON — parsed via stdlib ``json``; no shapely dependency.
    Other formats (shapefile, GPKG, …) — loaded via geopandas, reprojected to
    EPSG:4326, then converted to ring lists via ``__geo_interface__``.

    Returns
    -------
    list[list]  flat list of rings; each ring is a list of [lon, lat] pairs.

    Raises
    ------
    RuntimeError  if the file cannot be loaded or contains no drawable geometry.
    ImportError   if a non-GeoJSON format is requested but geopandas is absent.
    """
    ext = os.path.splitext(path)[1].lower()
    rings = []

    # ── GeoJSON: pure stdlib path, no shapely ──────────────────────────────
    if ext in (".geojson", ".json"):
        try:
            import json as _json
            with open(path, encoding="utf-8") as f:
                data = _json.load(f)
        except Exception as exc:
            msg = f"Failed to read GeoJSON boundary file '{path}': {exc}"
            logger.error(msg)
            raise RuntimeError(msg) from exc
        _extract_rings(data, rings)

    else:
        # ── Other formats: geopandas path ─────────────────────────────────
        try:
            import geopandas as gpd
        except ImportError as exc:
            msg = ("geopandas is required to load non-GeoJSON boundary files. "
                   "Install with: pip install geopandas")
            logger.error(msg)
            raise ImportError(msg) from exc

        try:
            gdf = gpd.read_file(path)
        except Exception as exc:
            msg = f"Failed to load boundary file '{path}': {exc}"
            logger.error(msg)
            raise RuntimeError(msg) from exc

        try:
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            elif gdf.crs.to_epsg() != 4326:
                gdf = gdf.to_crs("EPSG:4326")
        except Exception as exc:
            msg = f"CRS reprojection failed for boundary file '{path}': {exc}"
            logger.error(msg)
            raise RuntimeError(msg) from exc

        for geom in gdf.geometry:
            if geom is not None and not geom.is_empty:
                _extract_rings(geom.__geo_interface__, rings)

    if not rings:
        msg = f"Boundary file '{path}' contains no drawable geometries."
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info(f"Boundary loaded: {path} ({len(rings)} rings).")
    return rings


def _draw_boundary_rings(ax, rings, color, linewidth, zorder=4):
    """Draw pre-loaded boundary rings onto *ax* (draw many times, no I/O)."""
    for ring in rings:
        if not ring:
            continue
        xs = [c[0] for c in ring]
        ys = [c[1] for c in ring]
        ax.plot(xs, ys, color=color, linewidth=linewidth, zorder=zorder,
                solid_capstyle="round")


# ==============================================================================
# Basemap drawing — load once / draw many
# ==============================================================================

def _load_basemap(path):
    """Load a basemap file into memory (load once).

    Returns a dict describing the loaded data:
      {"type": "mbtiles", "zoom": int,  "tiles": {(x, tms_y): PIL.Image}}
      {"type": "geotiff", "array": np.ndarray, "extent": [W,E,S,N]}
      {"type": "image",   "array": np.ndarray}

    Raises
    ------
    RuntimeError  if the file cannot be loaded.
    ImportError   if a required optional library is missing.
    """
    ext = os.path.splitext(path)[1].lower()

    # ── MBTiles ────────────────────────────────────────────────────────────
    if ext == ".mbtiles":
        try:
            import sqlite3
            import io
            from PIL import Image
        except ImportError as exc:
            msg = ("mercantile and Pillow are required for MBTiles basemaps. "
                   "Install with: pip install mercantile pillow")
            logger.error(msg)
            raise ImportError(msg) from exc

        try:
            con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            zoom_levels = [r[0] for r in con.execute(
                "SELECT DISTINCT zoom_level FROM tiles ORDER BY zoom_level DESC"
            ).fetchall()]
            if not zoom_levels:
                raise RuntimeError(f"No tiles found in MBTiles file: {path}")
            z = zoom_levels[0]

            rows = con.execute(
                "SELECT tile_column, tile_row, tile_data FROM tiles WHERE zoom_level=?", (z,)
            ).fetchall()
            con.close()

            tile_cache = {}
            for x, tms_y, data in rows:
                tile_cache[(x, tms_y)] = Image.open(io.BytesIO(data)).convert("RGBA")

            logger.info(f"Basemap loaded: {path} (MBTiles z={z}, {len(tile_cache)} tiles in memory).")
            return {"type": "mbtiles", "zoom": z, "tiles": tile_cache}

        except (ImportError, RuntimeError):
            raise
        except Exception as exc:
            msg = f"Failed to load MBTiles basemap '{path}': {exc}"
            logger.error(msg)
            raise RuntimeError(msg) from exc

    # ── GeoTIFF ────────────────────────────────────────────────────────────
    if ext in (".tif", ".tiff"):
        try:
            import rasterio
            import numpy as np
        except ImportError as exc:
            msg = "rasterio is required for GeoTIFF basemaps. Install with: pip install rasterio"
            logger.error(msg)
            raise ImportError(msg) from exc

        try:
            with rasterio.open(path) as src:
                data = src.read()          # (bands, H, W)
                bounds = src.bounds
                crs = src.crs

            # Convert to (H, W, bands) uint8 for imshow
            if data.dtype != np.uint8:
                lo, hi = data.min(), data.max()
                data = ((data - lo) / (hi - lo + 1e-10) * 255).astype(np.uint8)
            arr = np.transpose(data, (1, 2, 0))  # (H, W, bands)

            # Convert extent to WGS-84 if needed (e.g. contextily GeoTIFFs use EPSG:3857)
            if crs is not None and crs.to_epsg() != 4326:
                try:
                    from pyproj import Transformer
                    t = Transformer.from_crs(crs.to_epsg(), 4326, always_xy=True)
                    w, s = t.transform(bounds.left,  bounds.bottom)
                    e, n = t.transform(bounds.right, bounds.top)
                    extent = [w, e, s, n]
                except Exception:
                    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            else:
                extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

            logger.info(f"Basemap loaded: {path} (GeoTIFF {arr.shape[1]}×{arr.shape[0]} px).")
            return {"type": "geotiff", "array": arr, "extent": extent}

        except (ImportError, RuntimeError):
            raise
        except Exception as exc:
            msg = f"Failed to load GeoTIFF basemap '{path}': {exc}"
            logger.error(msg)
            raise RuntimeError(msg) from exc

    # ── Plain image (PNG, JPEG, …) ─────────────────────────────────────────
    try:
        import matplotlib.image as mpimg
        arr = mpimg.imread(path)
        logger.info(f"Basemap loaded: {path} (image {arr.shape[1]}×{arr.shape[0]} px).")
        return {"type": "image", "array": arr}
    except Exception as exc:
        msg = f"Failed to load basemap file '{path}': {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc


def _draw_basemap(ax, basemap, alpha=1.0, zorder=1):
    """Draw a pre-loaded basemap onto *ax* (draw many times, no file I/O).

    *basemap* must be the dict returned by _load_basemap().
    Called after scatter data so axes limits are already set; the basemap
    appears behind all data layers via zorder=1.
    """
    import numpy as np

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    btype = basemap["type"]

    if btype == "mbtiles":
        try:
            import mercantile
            from PIL import Image
        except ImportError as exc:
            msg = "mercantile and Pillow are required to render MBTiles basemaps."
            logger.error(msg)
            raise ImportError(msg) from exc

        west, east, south, north = xlim[0], xlim[1], ylim[0], ylim[1]
        z = basemap["zoom"]
        tile_cache = basemap["tiles"]

        tiles = list(mercantile.tiles(west, south, east, north, zooms=z))
        if not tiles:
            logger.warning("No MBTiles tiles intersect the current axes extent.")
            return

        min_x = min(t.x for t in tiles)
        max_x = max(t.x for t in tiles)
        min_y = min(t.y for t in tiles)
        max_y = max(t.y for t in tiles)
        tile_px = 256

        canvas = Image.new("RGBA",
                           ((max_x - min_x + 1) * tile_px,
                            (max_y - min_y + 1) * tile_px),
                           (255, 255, 255, 0))
        for tile in tiles:
            tms_y = (2 ** z - 1) - tile.y
            img = tile_cache.get((tile.x, tms_y))
            if img is None:
                continue
            canvas.paste(img, ((tile.x - min_x) * tile_px,
                               (tile.y - min_y) * tile_px))

        ul = mercantile.bounds(mercantile.Tile(min_x, min_y, z))
        lr = mercantile.bounds(mercantile.Tile(max_x, max_y, z))
        img_extent = [ul.west, lr.east, lr.south, ul.north]

        ax.imshow(np.array(canvas), extent=img_extent, aspect="auto",
                  zorder=zorder, alpha=alpha, origin="upper")

    elif btype == "geotiff":
        ax.imshow(basemap["array"], extent=basemap["extent"],
                  aspect="auto", zorder=zorder, alpha=alpha)

    elif btype == "image":
        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        ax.imshow(basemap["array"], extent=extent,
                  aspect="auto", zorder=zorder, alpha=alpha)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# ==============================================================================
# Styling constants
# ==============================================================================

# Diverging color palette: deep cyan-blue → red
_QGIS_COLORS = [
    "#417eaf",   # deep cyan-blue  (most negative)
    "#dce6f0",   # light cyan-blue
    "#f7ece9",   # light pink/cream (near zero)
    "#ea9f86",   # light red/orange
    "#b93b31",   # deep red        (most positive)
]

# Contribution-to-size tiers (scaled to matplotlib pt²)
_DEFAULT_SIZE_RANGES = [
    (0,    100,  12),
    (100,  300,  45),
    (300,  500, 110),
    (500,  900, 260),
    (900,  1e9, 550),
]

_BOUNDARY_COLOR = "#000000"
_BOUNDARY_WIDTH = 0.8
_BG_POINT_COLOR = "#b8d4e8"


# ==============================================================================
# Plot helpers
# ==============================================================================

def _format_var_name(col: str, label_map=None) -> str:
    """Convert a SHAP column name to a human-readable label.

    Strips 'shap_' prefix or '_shap' suffix, replaces underscores with spaces,
    and applies title case.  An optional *label_map* dict can override labels
    for specific stripped or original column names.
    """
    name = col.replace("shap_", "").replace("_shap", "")
    if label_map:
        result = label_map.get(name) or label_map.get(col)
        if result:
            return result
    return name.replace("_", " ").title()


def _fmt_break(v):
    """Format a SHAP break value adaptively."""
    if v == 0.0:
        return "0"
    a = abs(v)
    if a >= 0.01:
        return f"{v:.2f}"
    elif a >= 0.0001:
        return f"{v:.4f}"
    else:
        return f"{v:.2e}"


def _compute_shap_breaks(values, n_classes=5):
    """Asymmetric quantile class breaks (6 break-points → 5 classes)."""
    neg = values[values < 0]
    pos = values[values >= 0]
    if len(neg) > 0 and len(pos) > 0:
        neg_s = sorted(neg)
        pos_s = sorted(pos)
        nlen, plen = len(neg_s), len(pos_s)
        breaks = [
            neg_s[int(nlen * 0.01)],
            neg_s[nlen // 2],
            0.0,
            pos_s[plen // 2],
            pos_s[min(int(plen * 0.95), plen - 1)],
            pos_s[min(int(plen * 0.99), plen - 1)],
        ]
    else:
        q = np.linspace(0, 100, n_classes + 1)
        breaks = [float(np.percentile(values, qi)) for qi in q]
    return breaks


def _marker_sizes(contributions, size_ranges=None):
    """Discrete-tier marker sizes from contribution values."""
    if size_ranges is None:
        size_ranges = _DEFAULT_SIZE_RANGES
    sizes = np.zeros(len(contributions))
    for low, high, s in size_ranges:
        sizes[(contributions >= low) & (contributions < high)] = s
    return sizes


def _style_map_axes(ax):
    """Remove all chart decorations; keep a thin outer frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#333333")


def _add_north_arrow(ax):
    """Place a simple north arrow in the upper-right corner."""
    ax.annotate(
        "", xy=(0.92, 0.92), xycoords="axes fraction",
        xytext=(0.92, 0.82), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
        annotation_clip=True,
    )
    ax.text(0.92, 0.94, "N", ha="center", va="bottom", fontsize=8,
            fontweight="bold", transform=ax.transAxes, color="black")


def _draw_size_legend(ax, size_ranges=None):
    """Contributions circle-size legend."""
    if size_ranges is None:
        size_ranges = _DEFAULT_SIZE_RANGES

    tiers = list(reversed(size_ranges))
    y_pos = np.linspace(0.82, 0.12, len(tiers))
    max_s = max(s for _, _, s in tiers)
    scale = 600.0 / max_s

    for (low, high, s), y in zip(tiers, y_pos):
        def _fmt(v):
            return str(int(v)) if v == int(v) else f"{v:.2f}"
        label = f"≥{_fmt(low)}" if high >= 1e8 else f"{_fmt(low)}–{_fmt(high)}"
        ax.scatter(0.20, y, s=s * scale, c="black", linewidths=0.5, zorder=2)
        ax.text(0.38, y, label, va="center", ha="left", fontsize=10)

    ax.text(0.15, 1.0, "Contributions", ha="left", va="top",
            fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _draw_boundary_legend(ax, boundary_color):
    """Data boundary line legend."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot([0.05, 0.38], [0.5, 0.5], color=boundary_color, linewidth=2.0)
    ax.text(0.45, 0.5, "Data Boundary", va="center", ha="left", fontsize=10)
    ax.text(0.08, 0.92, "Boundaries", ha="left", va="top",
            fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.axis("off")


# ==============================================================================
# Private panel drawing (layer A)
# ==============================================================================

def _draw_single_panel(
    ax,
    df,
    shap_col,
    *,
    lat_col="Latitude",
    lon_col="Longitude",
    contributions_col="total_contributions",
    boundary_rings=None,
    basemap=None,
    basemap_alpha=1.0,
    size_ranges=None,
    colors=None,
    opacity=0.6,
    show_background_points=True,
    bg_color=_BG_POINT_COLOR,
    boundary_color=_BOUNDARY_COLOR,
    boundary_lw=_BOUNDARY_WIDTH,
    north_arrow=True,
    panel_label=None,
    label_map=None,
):
    """Render one SHAP panel onto an existing Axes.

    This is the shared private drawing function used by both plot_shap_single()
    and plot_shap_6panel().  It handles only rendering — no figure creation,
    no save/show, no file I/O, no logging.

    Both boundary and basemap must be pre-loaded by the caller; this function
    only draws — it never reads files.

    Layer draw order (bottom → top, controlled by zorder):
      zorder=1 — local basemap (drawn after scatter so axes limits are set;
                 appears behind all data layers via zorder=1)
      zorder=2 — boundary outline (pre-loaded rings, if provided)
      zorder=3 — dense tiny background points
      zorder=4 — contribution-sized, SHAP-classified circles

    Parameters
    ----------
    ax                  : matplotlib Axes
    df                  : pd.DataFrame      — geo table with lat/lon and SHAP columns
    shap_col            : str               — SHAP column to visualise
    lat_col / lon_col   : str               — coordinate column names
    contributions_col   : str               — marker-size column; falls back to
                                              sum(abs(SHAP)) across all SHAP columns
    boundary_rings      : list[list] | None — pre-loaded rings from _load_boundary();
                                              None skips boundary drawing
    basemap             : dict | None       — pre-loaded basemap from _load_basemap();
                                              None skips basemap drawing
    basemap_alpha       : float             — basemap opacity 0–1
    size_ranges         : list[tuple]       — [(low, high, pt²), …]
    colors              : list[str]         — 5-element diverging hex palette
    opacity             : float             — SHAP circle alpha
    show_background_points : bool           — draw tiny neutral background dots
    bg_color            : str               — background dot hex colour
    boundary_color      : str               — boundary line colour
    boundary_lw         : float             — boundary line width
    north_arrow         : bool              — add north arrow
    panel_label         : str | None        — prefix prepended to the title, e.g. "(a)"
    label_map           : dict | None       — {stripped_name: display_label} overrides
    """
    if colors is None:
        colors = _QGIS_COLORS
    if size_ranges is None:
        size_ranges = _DEFAULT_SIZE_RANGES

    lons = df[lon_col].values
    lats = df[lat_col].values

    # Contributions → normalised marker sizes
    all_shap = [c for c in df.columns if c.startswith("shap_") or c.endswith("_shap")]
    if contributions_col in df.columns:
        contrib = df[contributions_col].values
    elif all_shap:
        contrib = df[all_shap].abs().sum(axis=1).values
    else:
        contrib = np.full(len(df), size_ranges[2][0] + 1)

    contrib_min, contrib_max = contrib.min(), contrib.max()
    if contrib_max > contrib_min:
        contrib_norm = (contrib - contrib_min) / (contrib_max - contrib_min) * 1000
    else:
        contrib_norm = np.full_like(contrib, 500.0)
    sizes = _marker_sizes(contrib_norm, size_ranges)
    order = np.argsort(sizes)

    _style_map_axes(ax)

    # Boundary overlay (zorder=2) — draw from pre-loaded rings, no file I/O
    if boundary_rings is not None:
        _draw_boundary_rings(ax, boundary_rings, boundary_color, boundary_lw, zorder=2)

    # Background footprint points (zorder=3)
    if show_background_points:
        ax.scatter(lons, lats, s=1.5, c=bg_color, alpha=0.22, linewidths=0, zorder=3)

    # SHAP colour classification
    values = df[shap_col].values
    breaks = _compute_shap_breaks(values)
    class_idx = np.clip(np.digitize(values, breaks[1:-1]), 0, len(colors) - 1)
    pt_colors = np.array(colors)[class_idx]

    # SHAP-classified circles (small sizes first, large on top; zorder=4)
    ax.scatter(
        lons[order], lats[order],
        c=pt_colors[order], s=sizes[order],
        alpha=opacity, linewidths=0, zorder=4,
    )

    # Local basemap (behind all other layers via zorder=1; drawn after data
    # so axes limits are already set from the scatter calls above)
    if basemap is not None:
        _draw_basemap(ax, basemap, alpha=basemap_alpha, zorder=1)

    # SHAP colour legend
    for ci, color in enumerate(colors):
        ax.scatter([], [], c=color, s=40,
                   label=f"{_fmt_break(breaks[ci])}–{_fmt_break(breaks[ci+1])}")
    ax.legend(
        title="SHAP value", fontsize=6, title_fontsize=7,
        loc="lower right", framealpha=0.88, edgecolor="#cccccc",
        markerscale=1.0, handletextpad=0.3, borderpad=0.4,
    )

    # Panel title
    var_label = _format_var_name(shap_col, label_map=label_map)
    title_str = f"{panel_label} {var_label}" if panel_label else var_label
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=4, loc="left")

    if north_arrow:
        _add_north_arrow(ax)


# ==============================================================================
# Public single-panel API (layer B)
# ==============================================================================

def plot_shap_single(
    df,
    shap_col,
    *,
    lat_col="Latitude",
    lon_col="Longitude",
    contributions_col="total_contributions",
    boundary_path=None,
    basemap_path=None,
    basemap_alpha=1.0,
    size_ranges=None,
    colors=None,
    opacity=0.6,
    show_background_points=True,
    bg_color=_BG_POINT_COLOR,
    boundary_color=_BOUNDARY_COLOR,
    boundary_lw=_BOUNDARY_WIDTH,
    north_arrow=True,
    label_map=None,
    figsize=(9, 8),
    title=None,
    save_path=None,
):
    """Plot a single SHAP variable as a standalone map figure.

    Creates its own figure and axes; delegates all rendering to
    _draw_single_panel().

    Parameters
    ----------
    df                  : pd.DataFrame  — geo table with lat/lon and SHAP columns
    shap_col            : str           — SHAP column to visualise
    lat_col / lon_col   : str           — coordinate column names
    contributions_col   : str           — marker-size column; falls back to
                                          sum(abs(SHAP)) across all SHAP columns
    boundary_path       : str | None    — local boundary file (GeoJSON, shapefile,
                                          GPKG, …); raises on load failure
    basemap_path        : str | None    — local raster basemap file (GeoTIFF, PNG, …);
                                          raises on load failure; no internet access used
    basemap_alpha       : float         — basemap opacity 0–1
    size_ranges         : list[tuple]   — [(low, high, pt²), …]
    colors              : list[str]     — 5-element diverging hex palette
    opacity             : float         — SHAP circle alpha
    show_background_points : bool       — draw tiny neutral background dots
    bg_color            : str           — background dot hex colour
    boundary_color      : str           — boundary line colour
    boundary_lw         : float         — boundary line width
    north_arrow         : bool          — add north arrow
    label_map           : dict | None   — {stripped_name: display_label} overrides
    figsize             : tuple         — (width, height) in inches
    title               : str | None    — override the axes title; uses feature name if None
    save_path           : str | None    — PNG output path; plt.show() if None

    Returns
    -------
    fig : matplotlib Figure
    """
    _require_cols(df, lat_col, lon_col, shap_col)

    # Load boundary and basemap once; each loader logs its own success message.
    boundary_rings = _load_boundary(boundary_path) if boundary_path is not None else None
    if boundary_rings is None:
        logger.info("No boundary_path provided — boundary overlay skipped.")

    basemap_data = _load_basemap(basemap_path) if basemap_path is not None else None
    if basemap_data is None:
        logger.info("No basemap_path provided — basemap skipped.")

    logger.info(f"Rendering single-panel SHAP map for '{shap_col}'.")
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    _draw_single_panel(
        ax, df, shap_col,
        lat_col=lat_col,
        lon_col=lon_col,
        contributions_col=contributions_col,
        boundary_rings=boundary_rings,
        basemap=basemap_data,
        basemap_alpha=basemap_alpha,
        size_ranges=size_ranges,
        colors=colors,
        opacity=opacity,
        show_background_points=show_background_points,
        bg_color=bg_color,
        boundary_color=boundary_color,
        boundary_lw=boundary_lw,
        north_arrow=north_arrow,
        panel_label=None,
        label_map=label_map,
    )
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=4, loc="left")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"Single-panel SHAP map saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)
    return fig


# ==============================================================================
# Public multi-panel API (layer C)
# ==============================================================================

def plot_shap_6panel(
    df_or_path,
    shap_vars=None,
    lat_col="Latitude",
    lon_col="Longitude",
    contributions_col="total_contributions",
    boundary_path=None,
    basemap_path=None,
    basemap_alpha=1.0,
    size_ranges=None,
    colors=None,
    ncols=2,
    opacity=0.6,
    show_background_points=True,
    bg_color=_BG_POINT_COLOR,
    boundary_color=_BOUNDARY_COLOR,
    boundary_lw=_BOUNDARY_WIDTH,
    north_arrow=True,
    save_path=None,
    title=None,
    label_map=None,
):
    """Six-panel (3 × 2) GIS-style SHAP atlas.

    Handles layout, variable iteration, and figure composition only.
    Per-panel rendering is fully delegated to _draw_single_panel().

    Accepts both SHAP column naming conventions:
      - prefix:  shap_{feature}   (save_geo_export / build_geo_table output)
      - suffix:  {feature}_shap   (save_qgis_export output)

    Parameters
    ----------
    df_or_path        : pd.DataFrame or str — geo table or path to CSV
    shap_vars         : list[str] | None    — SHAP columns to plot (≤ 6);
                                              auto-selects first 6 if None
    lat_col / lon_col : str                 — coordinate column names
    contributions_col : str                 — marker-size column; falls back to
                                              sum(abs(SHAP)) if absent
    boundary_path     : str | None          — local boundary file; raises on failure;
                                              boundary legend omitted when None
    basemap_path      : str | None          — local raster basemap file; raises on failure;
                                              no internet access used
    basemap_alpha     : float               — basemap opacity 0–1
    size_ranges       : list[tuple] | None  — [(low, high, pt²), …]
    colors            : list[str] | None    — 5-element diverging hex palette
    ncols             : int                 — grid columns (default 2 → 3 × 2)
    opacity           : float               — SHAP circle alpha (default 0.6)
    show_background_points : bool           — draw tiny neutral background dots
    bg_color          : str                 — background dot hex colour
    boundary_color    : str                 — boundary line colour
    boundary_lw       : float               — boundary line width
    north_arrow       : bool                — add north arrow to each panel
    save_path         : str | None          — PNG output path; plt.show() if None
    title             : str | None          — overall figure suptitle
    label_map         : dict | None         — {stripped_name: display_label} overrides

    Returns
    -------
    fig : matplotlib Figure
    """
    df = _load(df_or_path)
    logger.info(f"Geo table loaded — {len(df)} rows, {df.shape[1]} columns.")
    _require_cols(df, lat_col, lon_col)

    if colors is None:
        colors = _QGIS_COLORS
    if size_ranges is None:
        size_ranges = _DEFAULT_SIZE_RANGES

    all_shap = [c for c in df.columns if c.startswith("shap_") or c.endswith("_shap")]
    if not all_shap:
        raise ValueError(
            "No SHAP columns found. Expected columns starting with 'shap_' "
            "or ending with '_shap'."
        )
    if shap_vars is None:
        shap_vars = all_shap[:6]
    logger.info(
        f"SHAP columns selected for atlas ({len(shap_vars)}): "
        f"{shap_vars}"
    )

    # Load boundary and basemap once before the panel loop — each logs its own success.
    boundary_rings = _load_boundary(boundary_path) if boundary_path is not None else None
    if boundary_rings is None:
        logger.info("No boundary_path provided — boundary overlay skipped.")

    basemap_data = _load_basemap(basemap_path) if basemap_path is not None else None
    if basemap_data is None:
        logger.info("No basemap_path provided — basemap skipped.")

    logger.info(f"Rendering {len(shap_vars)}-panel SHAP atlas.")

    n_panels = len(shap_vars)
    nrows    = int(np.ceil(n_panels / ncols))
    labels   = list("abcdefghijklmnopqrstuvwxyz")

    fig = plt.figure(figsize=(7.5 * ncols, 6.5 * nrows + 2.2), facecolor="white")
    gs  = fig.add_gridspec(
        nrows + 1, ncols,
        height_ratios=[6.5] * nrows + [2.2],
        hspace=0.30, wspace=0.12,
    )

    for i, col in enumerate(shap_vars):
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found — skipping panel ({labels[i]}).")
            continue

        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        _draw_single_panel(
            ax, df, col,
            lat_col=lat_col,
            lon_col=lon_col,
            contributions_col=contributions_col,
            boundary_rings=boundary_rings,
            basemap=basemap_data,
            basemap_alpha=basemap_alpha,
            size_ranges=size_ranges,
            colors=colors,
            opacity=opacity,
            show_background_points=show_background_points,
            bg_color=bg_color,
            boundary_color=boundary_color,
            boundary_lw=boundary_lw,
            north_arrow=north_arrow,
            panel_label=f"({labels[i]})",
            label_map=label_map,
        )

    # Hide unused grid cells
    for j in range(n_panels, nrows * ncols):
        fig.add_subplot(gs[j // ncols, j % ncols]).set_visible(False)

    # Legend row: always show size legend; show boundary legend only if relevant
    _draw_size_legend(fig.add_subplot(gs[nrows, 0]), size_ranges)
    if boundary_rings is not None and ncols > 1:
        _draw_boundary_legend(fig.add_subplot(gs[nrows, 1]), boundary_color)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.005)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        logger.info(f"6-panel SHAP atlas saved to: {save_path}")
    else:
        plt.show()
    plt.close(fig)
    return fig
