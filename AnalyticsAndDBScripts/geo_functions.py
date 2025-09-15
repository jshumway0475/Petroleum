import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import griddata
from functools import partial
from shapely.ops import transform
from shapely import wkt
from shapely.geometry import Point, LineString, MultiLineString, Polygon, MultiPolygon
import pyproj
import matplotlib.pyplot as plt

# Helper functions for geospatial data processing
def _ensure_geometry(series):
    if isinstance(series.iloc[0], str):
        return series.apply(wkt.loads)
    return series

def _to_lonlat_transformer(src_epsg):
    # Returns a callable that transforms (x,y) from src_epsg -> EPSG:4269 (lon, lat)
    return pyproj.Transformer.from_crs(src_epsg, 4269, always_xy=True).transform

def _utm_to_lonlat_for_arr(utm_epsg, arr_xy_units_ft):
    tr = pyproj.Transformer.from_crs(utm_epsg, 4269, always_xy=True)
    def convert(arr, units='ft'):
        if units == 'ft':
            x_m = arr[:, 0] * 0.3048
            y_m = arr[:, 1] * 0.3048
        else:
            x_m = arr[:, 0]
            y_m = arr[:, 1]
        lon, lat = tr.transform(x_m, y_m)
        return np.asarray(lon), np.asarray(lat), arr[:, 2]
    return convert

def _sample_points_for_geom(geom, line_points=50, poly_grid_target=200):
    """
    Returns a list of shapely Points in EPSG:4269 to sample within/along the geometry.
    line_points: number of points along each line geometry
    poly_grid_target: target count of grid points (before in-polygon filtering) for polygons
    """
    pts = []
    if isinstance(geom, Point):
        pts = [geom]

    elif isinstance(geom, (LineString, MultiLineString)):
        lines = [geom] if isinstance(geom, LineString) else list(geom.geoms)
        for ln in lines:
            if ln.is_empty or ln.length == 0:
                continue
            n = max(2, line_points)
            dists = np.linspace(0.0, ln.length, n)
            pts.extend([ln.interpolate(d) for d in dists])

    elif isinstance(geom, (Polygon, MultiPolygon)):
        polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
        for poly in polys:
            if poly.is_empty:
                continue
            minx, miny, maxx, maxy = poly.bounds
            if not np.isfinite([minx, miny, maxx, maxy]).all() or maxx <= minx or maxy <= miny:
                continue

            # Build a reasonably square grid with ~poly_grid_target points
            # nx * ny ~= poly_grid_target
            aspect = (maxx - minx) / max(1e-12, (maxy - miny))
            nx = int(np.sqrt(poly_grid_target * max(aspect, 1e-6)))
            ny = max(1, int(poly_grid_target // max(nx, 1)))
            nx = max(1, nx)

            xs = np.linspace(minx, maxx, nx)
            ys = np.linspace(miny, maxy, ny)
            # Use cell centers when possible
            if nx > 1: xs = (xs[:-1] + xs[1:]) / 2.0
            if ny > 1: ys = (ys[:-1] + ys[1:]) / 2.0

            grid = [Point(x, y) for x in xs for y in ys]
            inside = [p for p in grid if poly.contains(p) or poly.touches(p)]
            # If the polygon is tiny/thin and filtering removed all, fall back to representative point
            if not inside:
                inside = [poly.representative_point()]
            pts.extend(inside)
    else:
        # Unknown/other geometry types: fall back to centroid
        pts = [geom.centroid]

    # Filter out any non-finite or empty points just in case
    clean = []
    for p in pts:
        if p and np.isfinite([p.x, p.y]).all():
            clean.append(p)
    if not clean:
        clean = [geom.centroid]
    return clean

# Useful geologic functions
def calculate_tvd(df, target_md):
    """
    Calculate the True Vertical Depth (TVD) at a given measured depth (MD) using linear interpolation between survey
    points with survey data provided in a Pandas DataFrame.
    
    :param df: Pandas DataFrame sorted by 'md' containing 'md' and 'TVD' for each survey point.
    :param target_md: The measured depth at which to calculate TVD.
    :return: The calculated TVD.
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
    epsg,
    id_col,
    geo_col,
    sample_method,
    input_type,
    utm_epsg=26914,
    utm_units='ft',
    line_points=50,
    poly_grid_target=200,
    sample_strategy='across'
):
    """
    df: pandas dataframe with a geometry column
    arr: numpy array with columns [lon, lat, z] if input_type==1; or [x, y, z] in UTM if input_type==2
    epsg: EPSG of df[geo_col]
    id_col: identifier column
    geo_col: geometry column (Shapely or WKT)
    sample_method: 'nearest' | 'linear' | 'cubic'
    input_type: 1=lon,lat,z ; 2=UTM x,y,z with utm_epsg & utm_units
    line_points: number of samples along line geometries
    poly_grid_target: target gridpoint count (pre-filter) used to sample polygon interiors
    sample_strategy: "across" to sample multiple points & aggregate, "centroid" to sample only at centroid
    """
    # Ensure shapely geometries
    df = df.copy()
    df[geo_col] = _ensure_geometry(df[geo_col])

    # Reproject geometries to EPSG:4269 for interpolation domain
    if epsg != 4269:
        project = partial(_to_lonlat_transformer(epsg))
        df[geo_col] = df[geo_col].apply(lambda g: transform(project, g))

    # Prepare interpolation cloud (lon, lat, z)
    if input_type == 1:
        lon, lat, z = arr[:, 0], arr[:, 1], arr[:, 2]
    elif input_type == 2:
        lon, lat, z = _utm_to_lonlat_for_arr(utm_epsg, utm_units)(arr, units=utm_units)
    else:
        raise ValueError("input_type must be 1 or 2")

    out_rows = []

    for idx, row in df.iterrows():
        geom = row[geo_col]
        c = geom.centroid

        if sample_strategy == "centroid":
            xs = np.array([c.x])
            ys = np.array([c.y])
        else:
            pts = _sample_points_for_geom(geom, line_points=line_points, poly_grid_target=poly_grid_target)
            xs = np.array([p.x for p in pts], dtype=float)
            ys = np.array([p.y for p in pts], dtype=float)

        sampled_vals = griddata((lon, lat), z, (xs, ys), method=sample_method, fill_value=np.nan)

        valid = ~np.isnan(sampled_vals)
        n_total = sampled_vals.size
        n_valid = int(valid.sum())

        if n_valid == 0:
            agg_mean = float('nan')
            agg_std = float('nan')
            agg_min = float('nan')
            agg_max = float('nan')
        else:
            vals = sampled_vals[valid]
            agg_mean = float(np.nanmean(vals))
            agg_std  = float(np.nanstd(vals))
            agg_min  = float(np.nanmin(vals))
            agg_max  = float(np.nanmax(vals))

        out_rows.append({
            id_col: row[id_col],
            geo_col: geom,
            'file_name': file_name,
            'epsg': epsg,
            'method': sample_method,
            'input_type': input_type,
            'centroid_longitude': float(c.x),
            'centroid_latitude': float(c.y),
            'samples_total': int(n_total),
            'samples_valid': int(n_valid),
            'sampled_z_mean': agg_mean,
            'sampled_z_std': agg_std,
            'sampled_z_min': agg_min,
            'sampled_z_max': agg_max,
        })

    df_out = pd.DataFrame(out_rows)
    return df_out

# Mapping function using Matplotlib and scipy.interpolate.griddata
def plot_heatmap_and_histogram(arr, file_name=None, grid_resolution=100, color_map='jet', z_min=None, z_max=None):
    """
    Plot heatmap using x, y, z values alongside a histogram of z values.
    arr: A numpy array with shape (n, 3) where first column is x, second is y, and third is z.
    file_name: Name of the file, used for titles.
    grid_resolution: The resolution of the grid onto which data will be interpolated.
    color_map: The matplotlib color map to use for the heatmap.
    z_min: Lower limit for Z values in the heatmap.
    z_max: Upper limit for Z values in the heatmap.
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
    Generate and save a heatmap based on x, y, z values to an image file.
    
    arr: A numpy array with shape (n, 3) where first column is x (longitude), 
         second is y (latitude), and third is z.
    image_file_name: Name of the image file to save the heatmap.
    grid_resolution: The resolution of the grid onto which data will be interpolated.
    color_map: The matplotlib color map to use for the heatmap.
    z_min: Lower limit for Z values in the heatmap.
    z_max: Upper limit for Z values in the heatmap.
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
    min_long, max_long = min(x), max(x)
    min_lat, max_lat = min(y), max(y)
    return min_lat, max_lat, min_long, max_long
