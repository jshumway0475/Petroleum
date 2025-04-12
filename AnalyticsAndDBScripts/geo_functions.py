import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import griddata
from functools import partial
from shapely.ops import transform
from shapely import wkt
import pyproj
import matplotlib.pyplot as plt

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
def sample_xyz(df, file_name, arr, epsg, id_col, geo_col, sample_method, input_type, utm_epsg=26914, utm_units='ft'):
    '''
    df: pandas dataframe containing the points to be sampled
    arr: numpy array to be sampled
    epsg: epsg code of the projection of the points
    id_col: name of the column containing the id of the points like a Well ID or API number
    geo_col: name of the column containing the geometry of the points
    sample_method: method to be used for interpolation ('nearest', 'linear', 'cubic')
    input_type: type of input data (1: long, lat, z; 2: x, y, z). Type 2 is used when the data is in UTM coordinates
    utm_epsg: Used in cases where input_type is 2, the EPSG code of the x, y coordinates of the raw data
    utm_units: Used in cases where input_type is 2, the units of the x, y coordiates of the UTM projection ('ft' or 'm')
    '''
    # If the data in geo_col is not already a Shapely geometry, convert it
    if isinstance(df[geo_col].iloc[0], str):
        df[geo_col] = df[geo_col].apply(wkt.loads)

    # Create Transformer for coordinate transformations
    transformer = pyproj.Transformer.from_crs(epsg, 4269, always_xy=True)

    # Reproject to EPSG:4269 if it's not already in that projection
    if epsg != 4269:
        project = partial(transformer.transform)
        df[geo_col] = df[geo_col].apply(lambda geom: transform(project, geom))

    # Calculate centroid
    df['centroid'] = df[geo_col].apply(lambda geom: geom.centroid)
    
    # Extract x (longitude) and y (latitude) from the centroid
    df['longitude'] = df['centroid'].apply(lambda point: point.x)
    df['latitude'] = df['centroid'].apply(lambda point: point.y)

    # Depending on input_type adjust the columns
    if input_type == 1:
        long, lat, z = arr[:, 0], arr[:, 1], arr[:, 2]
    elif input_type == 2:
        transformer_utm = pyproj.Transformer.from_crs(utm_epsg, 4269, always_xy=True)
        if utm_units == 'ft':
            x, y, z = arr[:, 0] * 0.3048, arr[:, 1] * 0.3048, arr[:, 2]
        else:
            x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
        long, lat = transformer_utm.transform(x, y)

    # Using griddata for interpolation
    df['sampled_z'] = griddata((long, lat), z, (df['longitude'].values, df['latitude'].values), method=sample_method, fill_value=np.nan)

    # Create new columns to store the values
    df['file_name'] = file_name
    df['epsg'] = epsg
    df['method'] = sample_method
    df['input_type'] = input_type

    # Create the output dataframe
    df_out = df[[id_col, geo_col, 'file_name', 'epsg', 'method', 'input_type', 'longitude', 'latitude', 'sampled_z']]
    
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
