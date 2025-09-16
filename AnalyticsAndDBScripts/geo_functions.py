import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.spatial import cKDTree
from shapely.ops import transform
from shapely import wkt
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
from shapely import vectorized
import pyproj
import matplotlib.pyplot as plt

# Helper functions for geospatial data processing
def _ensure_geometry(series):
    """
    Ensure a Pandas/GeoPandas column contains shapely geometries.

    If the first element is a WKT string, the entire column is parsed with
    `shapely.wkt.loads`. If the series is empty or already contains shapely
    geometries, it is returned unchanged.

    Parameters
    ----------
    series : pandas.Series
        Column expected to contain either WKT strings or shapely geometries.

    Returns
    -------
    pandas.Series
        Series of shapely geometries (or the original empty series).
    """
    if series.empty:
        return series
    first = series.iloc[0]
    if isinstance(first, str):
        return series.apply(wkt.loads)
    return series

_HAS_VECT = hasattr(vectorized, "covers")
def _covers_poly(poly, X, Y):
    """
    Vectorized polygon cover test.

    Uses `shapely.vectorized.covers` when available; otherwise falls back to a
    chunked Python loop that treats a point as inside if it is within or on the
    boundary.

    Parameters
    ----------
    poly : shapely.geometry.Polygon
        Polygon to test against (in the same CRS as X/Y).
    X, Y : ndarray
        1-D arrays of x and y coordinates (same length).

    Returns
    -------
    ndarray
        Boolean mask of length `len(X)` where True means covered by `poly`.
    """
    if _HAS_VECT:
        return vectorized.covers(poly, X, Y)
    mask = np.zeros_like(X, dtype=bool)
    for i in range(0, X.size, 10000):
        xs = X[i:i+10000]; ys = Y[i:i+10000]
        mask[i:i+10000] = np.array([poly.contains(Point(x,y)) or poly.touches(Point(x,y)) for x,y in zip(xs,ys)])
    return mask

def _normalize_grid_units(arr, grid_epsg, utm_units='m'):
    """
    Normalize grid coordinates to match the CRS units.

    For UTM-like CRSs (e.g., EPSG:269##, 326##), the standard unit is meters.
    If your incoming grid coordinates are in US survey feet, set `utm_units='ft'`
    to convert x,y columns to meters.

    Parameters
    ----------
    arr : ndarray, shape (N, 3)
        Grid samples with columns [x, y, z]. Will be copied and cast to float.
    grid_epsg : int
        EPSG code of the grid CRS (used for intent; no lookup performed here).
    utm_units : {'m','ft'}, optional
        Units of `arr` x,y inputs. 'm' = meters (no-op). 'ft' converts to meters.

    Returns
    -------
    ndarray
        Copy of `arr` with x,y in meters (z unchanged).
    """
    arr = np.asarray(arr, float).copy()
    try:
        crs = pyproj.CRS.from_user_input(grid_epsg)
        units = [(ai.unit_name or "").lower() for ai in crs.axis_info]
        is_meter_crs = any(("metre" in u) or ("meter" in u) for u in units)
    except Exception:
        is_meter_crs = True
    if is_meter_crs and utm_units.lower().startswith("ft"):
        arr[:, :2] *= 0.3048006096012192
    return arr

def _project_geoms(df, geo_col, src_epsg, dst_epsg):
    """
    Project shapely geometries from one CRS to another (once).

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        Table containing a geometry column.
    geo_col : str
        Column name containing shapely geometries (or WKT—use `_ensure_geometry` first).
    src_epsg, dst_epsg : int
        Source and destination EPSG codes.

    Returns
    -------
    pandas.Series
        Series of geometries in `dst_epsg`. If CRSs are equal, returns the original column.
    """
    if src_epsg == dst_epsg:
        return df[geo_col]
    tr = pyproj.Transformer.from_crs(src_epsg, dst_epsg, always_xy=True).transform
    return df[geo_col].apply(lambda g: transform(tr, g))

def _sample_points_for_geom(geom, line_points=50, poly_grid_target=200):
    """
    Generate sample points for a geometry in its native CRS.

    - Point: returns the point coordinate.
    - LineString/MultiLineString: returns `line_points` samples evenly spaced by
      arc-length on each component line.
    - Polygon/MultiPolygon: builds a roughly square mesh inside each polygon
      envelope with ~`poly_grid_target` nodes, then keeps the nodes covered by
      the polygon. Falls back to a representative point for tiny/thin polygons.

    Parameters
    ----------
    geom : shapely geometry
        Input geometry (Point/LineString/MultiLineString/Polygon/MultiPolygon).
    line_points : int, optional
        Number of points sampled along each line component (min 2).
    poly_grid_target : int, optional
        Approximate number of pre-filter grid nodes per polygon (before cover test).

    Returns
    -------
    ndarray, shape (M, 2)
        Sample points as [x, y].

    Notes
    -----
    Uses vectorized cover checks when available for speed.
    """
    if isinstance(geom, Point):
        return np.array([[geom.x, geom.y]])
    if isinstance(geom, (LineString, MultiLineString)):
        lines = [geom] if isinstance(geom, LineString) else list(geom.geoms)
        out = []
        for ln in lines:
            if ln.is_empty or ln.length == 0:
                continue
            d = np.linspace(0.0, ln.length, max(2, line_points))
            pts = np.array([ln.interpolate(di).coords[0] for di in d])
            out.append(pts)
        if out:
            return np.vstack(out)
        c = geom.centroid
        return np.array([[c.x, c.y]])
    if isinstance(geom, (Polygon, MultiPolygon)):
        polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
        out = []
        for poly in polys:
            if poly.is_empty:
                continue
            minx, miny, maxx, maxy = poly.bounds
            if not np.isfinite([minx, miny, maxx, maxy]).all() or maxx <= minx or maxy <= miny:
                continue
            w, h = maxx - minx, maxy - miny
            aspect = w / max(h, 1e-12)
            nx = max(1, int(np.sqrt(poly_grid_target * max(aspect, 1e-6))))
            ny = max(1, poly_grid_target // max(nx, 1))
            xs = np.linspace(minx, maxx, nx, dtype=float)
            ys = np.linspace(miny, maxy, ny, dtype=float)
            XX, YY = np.meshgrid(xs, ys)
            X = XX.ravel(); Y = YY.ravel()
            mask = _covers_poly(poly, X, Y)
            if mask.any():
                out.append(np.column_stack([X[mask], Y[mask]]))
            else:
                c = poly.representative_point()
                out.append(np.array([[c.x, c.y]]))
        if out:
            return np.vstack(out)
        c = geom.centroid
        return np.array([[c.x, c.y]])
    c = geom.centroid
    return np.array([[c.x, c.y]])

def build_interpolator(arr, method='linear', assume_structured=False):
    """
    Build a reusable interpolator for scattered or structured 2D data.

    Parameters
    ----------
    arr : ndarray, shape (N, 3)
        Sample points as columns [x, y, z] in a single planar CRS (e.g., UTM).
    method : {'nearest','linear','cubic'}, optional
        Interpolation scheme:
          - 'nearest' : very fast via `cKDTree`.
          - 'linear'  : piecewise-linear via `LinearNDInterpolator`.
          - 'cubic'   : smooth surface via `CloughTocher2DInterpolator` (slower).
    assume_structured : bool, optional
        If True, attempt to detect a full rectilinear grid and build a
        `RegularGridInterpolator` (fast). Falls back to scattered methods if the
        grid is not perfectly regular/full.

    Returns
    -------
    (callable, object or None)
        A function `f(XY)` that maps Mx2 query points to interpolated z values,
        and an auxiliary object (e.g., KDTree) when applicable.

    Raises
    ------
    ValueError
        If `method` is not one of the supported values.
    """
    arr = np.asarray(arr, float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("arr must be (N,3) with columns [x,y,z]")
    x, y, z = arr[:,0], arr[:,1], arr[:,2]

    if assume_structured:
        ux = np.unique(x); uy = np.unique(y)
        N = arr.shape[0]; nx, ny = ux.size, uy.size
        if nx * ny == N and np.unique(arr[:, :2], axis=0).shape[0] == N:
            ux = np.sort(ux); uy = np.sort(uy)
            idx = np.lexsort((x, y))
            xs, ys, zs = x[idx], y[idx], z[idx]
            try:
                X = xs.reshape(ny, nx); Y = ys.reshape(ny, nx)
                if np.allclose(X[0], ux, rtol=0, atol=1e-9) and np.allclose(Y[:, 0], uy, rtol=0, atol=1e-9):
                    Zg = zs.reshape(ny, nx)
                    rgi = RegularGridInterpolator(
                        (uy, ux), Zg, method='linear',
                        bounds_error=False, fill_value=np.nan
                    )
                    return (lambda XY: rgi(np.column_stack([XY[:,1], XY[:,0]]))), None
            except ValueError:
                pass

    if method == 'nearest':
        tree = cKDTree(np.column_stack([x, y]))
        def f(XY):
            XY = np.asarray(XY, float)
            if XY.ndim != 2 or XY.shape[1] != 2:
                raise ValueError("XY must be (M,2)")
            try:
                dist, idx = tree.query(XY, k=1, workers=-1)
            except TypeError:
                dist, idx = tree.query(XY, k=1)
            return z[idx]
        return f, tree

    if method == 'linear':
        interp = LinearNDInterpolator(np.column_stack([x, y]), z, fill_value=np.nan)
        return (lambda XY: interp(np.asarray(XY, float))), None

    if method == 'cubic':
        interp = CloughTocher2DInterpolator(np.column_stack([x, y]), z, fill_value=np.nan)
        return (lambda XY: interp(np.asarray(XY, float))), None

    raise ValueError("method must be 'nearest', 'linear', or 'cubic'")

# Useful geologic functions
def calculate_tvd(df, target_md):
    """
    Linearly interpolate TVD at a target measured depth (MD).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns 'MD_FT' and 'TVD_FT' and be sorted by 'MD_FT'.
    target_md : float
        Measured depth at which to compute TVD.

    Returns
    -------
    float or None
        Interpolated TVD at `target_md`, or None if target is outside data range.
    """

    # Check if target MD is outside the range of survey points
    if target_md < df['MD_FT'].min() or target_md > df['MD_FT'].max():
        return None

    # Check if target MD is exactly at a survey point
    if target_md in df['MD_FT'].values:
        return df[df['MD_FT'] == target_md]['TVD_FT'].values[0]

    # Find the two bracketing points
    lower_point = df[df['MD_FT'] < target_md].iloc[-1]
    upper_point = df[df['MD_FT'] > target_md].iloc[0]

    # If target MD falls between two survey points, perform linear interpolation
    md_lower, tvd_lower = lower_point['MD_FT'], lower_point['TVD_FT']
    md_upper, tvd_upper = upper_point['MD_FT'], upper_point['TVD_FT']
    return tvd_lower + ((tvd_upper - tvd_lower) / (md_upper - md_lower)) * (target_md - md_lower)

# Function to interpolate values from a grid to a set of points
def sample_xyz(
    df,
    file_name,
    arr,
    df_epsg,
    grid_epsg,
    id_col,
    geo_col,
    sample_method='linear',
    line_points=50,
    poly_grid_target=200,
    sample_strategy='across',
    assume_structured=False,
    utm_units='m'
):
    """
    Sample a 2D grid (x,y,z) at points derived from geometries.

    Workflow:
      1) Reproject `df[geo_col]` from `df_epsg` to `grid_epsg` once.
      2) Build a reusable interpolator from `arr` (structured fast-path if requested).
      3) Generate sample points (centroid or “across”) for each geometry.
      4) Interpolate all points in one batch and aggregate per geometry.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        Table containing an identifier and a geometry column.
    file_name : str
        Identifier recorded in the output for provenance.
    arr : ndarray, shape (N, 3)
        Grid samples with columns [x, y, z] in `grid_epsg` units.
    df_epsg : int
        EPSG of `df[geo_col]`.
    grid_epsg : int
        EPSG of the grid (and of `arr`).
    id_col : str
        Name of the identifier column to carry through.
    geo_col : str
        Name of the geometry column.
    sample_method : {'nearest','linear','cubic'}, optional
        Interpolation method (see `build_interpolator`).
    line_points : int, optional
        Samples per line component (used when `sample_strategy='across'`).
    poly_grid_target : int, optional
        Approximate pre-filter grid nodes per polygon.
    sample_strategy : {'across','centroid'}, optional
        - 'centroid': sample only at geometry centroid (fastest).
        - 'across'  : sample many points along/inside geometry and aggregate.
    assume_structured : bool, optional
        If True, try a structured-grid fast path.
    utm_units : {'m','ft'}, optional
        Units of x,y in `arr`. If 'ft', they are converted to meters before use.

    Returns
    -------
    pandas.DataFrame
        One row per input geometry with columns:
        [id_col, geo_col, file_name, epsg, method, centroid_x, centroid_y,
         samples_total, samples_valid, sampled_z_mean, sampled_z_std,
         sampled_z_min, sampled_z_max].

    Notes
    -----
    - Keep `arr` and `grid_epsg` consistent (same CRS/units) or set `utm_units` accordingly.
    - For very large jobs, consider chunking the interpolation or reducing sampling density.
    """
    df = df.copy()
    df[geo_col] = _ensure_geometry(df[geo_col])
    df[geo_col] = _project_geoms(df, geo_col, df_epsg, grid_epsg)

    # normalize units if grid is UTM-like
    arr = _normalize_grid_units(arr, grid_epsg, utm_units=utm_units)

    interp, _aux = build_interpolator(arr, method=sample_method, assume_structured=assume_structured)
    per_counts = []
    all_pts = []
    for g in df[geo_col].values:
        if sample_strategy == 'centroid':
            c = g.centroid
            XY = np.array([[c.x, c.y]])
        else:
            XY = _sample_points_for_geom(g, line_points=line_points, poly_grid_target=poly_grid_target)
        per_counts.append(XY.shape[0])
        all_pts.append(XY)
    all_XY = np.vstack(all_pts)
    all_vals = interp(all_XY)
    out = []
    start = 0
    for (idx, row), n in zip(df.iterrows(), per_counts):
        vals = all_vals[start:start+n]; start += n
        valid = np.isfinite(vals)
        if not np.any(valid):
            agg = dict(
                samples_total=int(n),
                samples_valid=0,
                sampled_z_mean=np.nan,
                sampled_z_std=np.nan,
                sampled_z_min=np.nan,
                sampled_z_max=np.nan
            )
        else:
            v = vals[valid]
            agg = dict(
                samples_total=int(n),
                samples_valid=int(valid.sum()),
                sampled_z_mean=float(np.mean(v)),
                sampled_z_std=float(np.std(v)),
                sampled_z_min=float(np.min(v)),
                sampled_z_max=float(np.max(v))
            )
        c = row[geo_col].centroid
        epsg_out = pyproj.CRS.from_user_input(grid_epsg)
        epsg_out = epsg_out.to_epsg() or str(epsg_out)
        out.append({
            id_col: row[id_col],
            geo_col: row[geo_col],
            'file_name': file_name,
            'epsg': epsg_out,
            'method': sample_method,
            'centroid_x': float(c.x),
            'centroid_y': float(c.y),
            **agg
        })
    return pd.DataFrame(out)

# Mapping function using Matplotlib and scipy.interpolate.griddata
def plot_heatmap_and_histogram(arr, file_name=None, grid_resolution=100, color_map='jet', z_min=None, z_max=None):
    """
    Visualize a scattered grid as a heatmap plus a histogram of Z.

    Parameters
    ----------
    arr : ndarray, shape (N, 3)
        Input points [x, y, z] (planar CRS recommended).
    file_name : str, optional
        Label used in plot titles.
    grid_resolution : int, optional
        Number of nodes along each axis of the visualization grid.
    color_map : str, optional
        Matplotlib colormap name for the heatmap.
    z_min, z_max : float or None, optional
        Color limits for the heatmap. None = auto.
    """
    # Extract x, y, and z from the array
    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]

    # Create grid
    xi = np.linspace(min(x), max(x), grid_resolution)
    yi = np.linspace(min(y), max(y), grid_resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate z values onto grid
    zi = griddata((x, y), z, (xi, yi), method='linear')
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot heatmap
    contour = axs[0].contourf(xi, yi, zi, 100, cmap=color_map, vmin=z_min, vmax=z_max)
    fig.colorbar(contour, ax=axs[0], label='Z Value')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    title = 'Heatmap of Z values'
    if file_name:
        title += f' for {file_name}'
    axs[0].set_title(title)
    
    # Plot histogram
    axs[1].hist(z, bins=30, color="dodgerblue", edgecolor="k")
    title = 'Histogram of Z values'
    if file_name:
        title += f' from {file_name}'
    axs[1].set_title(title)
    axs[1].set_xlabel('Z Value')
    axs[1].set_ylabel('Count')
    axs[1].grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

# Save heatmaps as an image file and extract max and min x-y coordinates values
def save_heatmap_as_image(arr, image_file_name, grid_resolution=100, color_map='jet', z_min=None, z_max=None):
    """
    Render and save a heatmap image from scattered (x,y,z) data.

    Parameters
    ----------
    arr : ndarray, shape (N, 3)
        Input points [x, y, z] (planar CRS recommended).
    image_file_name : str
        Output image path (format inferred by matplotlib, e.g., .png, .jpg).
    grid_resolution : int, optional
        Number of nodes along each axis of the visualization grid.
    color_map : str, optional
        Matplotlib colormap name for the heatmap.
    z_min, z_max : float or None, optional
        Color limits for the heatmap. None = auto.

    Returns
    -------
    tuple
        (min_y, max_y, min_x, max_x) computed from the input points.
    """
    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]

    # Create grid
    xi = np.linspace(min(x), max(x), grid_resolution)
    yi = np.linspace(min(y), max(y), grid_resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate z values onto grid
    zi = griddata((x, y), z, (xi, yi), method='linear')
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xi, yi, zi, 100, cmap=color_map, vmin=z_min, vmax=z_max)
    plt.axis('off')
    plt.savefig(image_file_name, bbox_inches='tight', pad_inches=0)
    plt.close()

    x, y = arr[:, 0], arr[:, 1]
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    return min_y, max_y, min_x, max_x
