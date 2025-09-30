import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection, Polygon
import pyproj
import math


# Helper function to convert Shapely geometries to WKT strings
def geom_to_wkt(value):
    if isinstance(value, (Point, LineString, MultiLineString, Polygon)):
        return wkt.dumps(value)
    return value


# Helper function to convert WKT strings to Shapely geometries
def wkt_to_geom(value):
    if isinstance(value, str):
        try:
            return wkt.loads(value)
        except (ValueError, TypeError):
            return value
    return value


# Function used to determine if a geometry has at least two unique coordinates
def has_at_least_two_unique_coords(geometry):
    '''
    Args:
    - geometry: Shapely geometry object, however the function will convert a WKT string to a geometry object if needed.

    Returns:
    - True if geometry has at least two unique coordinates.
    '''
    # Check if geometry is a string and try to convert it using wkt.loads
    if isinstance(geometry, str):
        try:
            geometry = wkt.loads(geometry)
        except Exception as e:
            print(f"Error converting WKT string to geometry: {e}")
            return False

    if isinstance(geometry, Point):
        return False  # Points by definition have only one coordinate

    if isinstance(geometry, LineString):
        if geometry.is_empty or len(set(geometry.coords)) < 2:
            return False

    elif isinstance(geometry, MultiLineString):
        if geometry.is_empty:
            return False

        # Collect all coordinates from all LineStrings within the MultiLineString
        all_coords = [coord for line in geometry.geoms for coord in line.coords]
        if len(set(all_coords)) < 2:
            return False

    else:
        return False  # If geometry is not Point, LineString, or MultiLineString, return False

    return True  # If all checks passed, return True


# Function to add points at a minimum frequency of 100 ft along the lateral line
def interpolate_points(geometry, distance_ft=100):
    '''
    Args:
    - geometry: Shapely geometry object projected to EPSG: 6579 (units in meters)
    - distance_ft: Distance in feet between interpolated points. Default is 100 ft.
    Returns:
    - geometry: Shapely geometry object with interpolated points
    '''
    # Convert feet to the same unit as the line length
    distance = distance_ft / 3.281

    # Function to handle interpolation for individual LineString
    def interpolate_line_string(line):
        # Check if line only consists of 2 points and if so, leave as is
        if len(line.coords) == 2:
            return line
        else:
            num_points = max(2, int(line.length / distance) + 1)  # Ensure at least 2 points
            points = [line.interpolate(dist * distance) for dist in range(num_points)]
            return LineString(points)

    # Check if geometry is a MultiLineString
    if isinstance(geometry, MultiLineString):
        # Interpolate each LineString inside the MultiLineString using geoms attribute
        lines = [interpolate_line_string(line) for line in geometry.geoms]
        return MultiLineString(lines)
    elif isinstance(geometry, LineString):
        return interpolate_line_string(geometry)
    else:
        return geometry  # For other types like Point, simply return as is


# Function to determine if a polygon is a rectangle
def rectangle_conformity(geometry):
    '''
    Args:
    - geometry: Shapely Polygon geometry object
    Returns:
    - rectangle_conformity: Ratio of intersection area to minimum rotated rectangle area
    - rectangle_conformity will be close to 1 for geometries that closely conform to a
      rectangle and lower for those that don't
    '''
    rect = geometry.minimum_rotated_rectangle
    intersect = geometry.intersection(rect)
    if rect.area == 0:  # Avoid division by zero
        return 0
    return intersect.area / rect.area


# Function extracts the endpoints of a LineString or MultiLineString
def get_endpoints(geom):
    '''
    Args:
    - geom: Shapely geometry object
    Returns:
    - tuple of start and end coordinates
    '''
    if geom.is_empty:
        return None, None
    if geom.geom_type == 'LineString':
        if len(geom.coords) < 2:
            return None, None
        return geom.coords[0], geom.coords[-1]
    elif geom.geom_type == 'MultiLineString':
        line_strings = list(geom.geoms)
        longest = max(line_strings, key=lambda part: part.length)
        return longest.coords[0], longest.coords[-1]
    else:
        raise ValueError('Unhandled geometry type: ' + repr(geom.geom_type))


# Convert MultiLineString or GeometryCollection to LineString
def get_longest_linestring(geometry):
    '''
    Args:
    - geometry: Shapely geometry object, which could be a LineString, MultiLineString, or GeometryCollection
    Returns:
    - LineString: The longest LineString geometry object found within the input geometry.
    '''
    if isinstance(geometry, LineString):
        return geometry
    elif isinstance(geometry, MultiLineString):
        return max(geometry.geoms, key=lambda x: x.length)
    elif isinstance(geometry, GeometryCollection):
        # Filter geometries to only include LineStrings and MultiLineStrings
        linestrings = [geom for geom in geometry.geoms if isinstance(geom, (LineString, MultiLineString))]
        # Flatten MultiLineStrings into LineStrings
        all_linestrings = []
        for geom in linestrings:
            if isinstance(geom, LineString):
                all_linestrings.append(geom)
            elif isinstance(geom, MultiLineString):
                all_linestrings.extend(geom.geoms)
        # Return the longest LineString if any exist
        if all_linestrings:
            return max(all_linestrings, key=lambda x: x.length)
        else:
            raise ValueError("No LineString or MultiLineString found in GeometryCollection")
    else:
        raise ValueError("Unhandled geometry type")


# Function to calculate azimuth relative to North (0 degrees)
def calculate_azimuth(line, reverse=False):
    '''
    Args:
    - line: Shapely LineString or MultiLineString geometry object
    - reverse: Reverse order of geometry coordinates. Default is False.
    Returns:
    - azimuth: Azimuth in degrees relative to North (0 degrees)
    '''
    # Check if line is empty
    if line.is_empty:
        return None

    # Reverse the line if needed
    if reverse:
        line = LineString([(y, x) for x, y in line.coords])

    # Extract endpoints based on geometry type
    if line.geom_type == 'LineString':
        if len(line.coords) < 2:
            return None
        start_point, end_point = line.coords[0], line.coords[-1]
    elif line.geom_type == 'MultiLineString':
        line_strings = list(line.geoms)
        longest = max(line_strings, key=lambda part: part.length)
        start_point, end_point = longest.coords[0], longest.coords[-1]
    else:
        raise ValueError('Unhandled geometry type: ' + repr(line.geom_type))

    # Calculate azimuth
    x1, y1, x2, y2 = *start_point, *end_point
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    azimuth = (angle + 360) % 360  # normalize to (0, 360)

    return azimuth


# Function to determine the relative direction of an adjacent line with respect to a reference line
def determine_relative_direction(theta_ref, theta_adj):
    """
    Determine the relative direction of an adjacent line with respect to a reference line.

    Parameters:
    theta_ref (float): Azimuth of the reference line, ranging from 0° to 180°.
    theta_adj (float): Azimuth of the line between the centroids, ranging from 0° to 360°.

    Returns:
    str: Relative direction of the adjacent line ("RIGHT", "LEFT", or "IN-LINE").

    This function normalizes north-south orientations to always face north and east-west orientations to always face east.
    This means that a well to the east of a north-south facing line will always be considered "RIGHT" and a well to the west
    of a north-south facing line will always be considered "LEFT". Similarly, a well to the north of an east-west facing line
    will always be considered "RIGHT" and a well to the south of an east-west facing line will always be considered "LEFT".
    """  # noqa
    # Correct theta_ref to use a -90° to 90° range
    theta_ref = theta_ref - 180 if theta_ref > 90 else theta_ref

    # Calculate and normalize the relative angle
    delta_theta = theta_adj - theta_ref
    delta_theta = delta_theta - 360 if delta_theta > 180 else (delta_theta + 360 if delta_theta < -180 else delta_theta)

    # Special checks for edge cases and determine the relative position
    if delta_theta == 0 or delta_theta == 180 or delta_theta == -180:
        return "IN-LINE"
    elif delta_theta > 0:
        return "RIGHT"
    else:
        return "LEFT"


# Function to optimize the buffer around the surface location needed to clip deviation from surface to bottomhole
def optimize_buffer(
    df,
    geo_col,
    sfc_lat_col,
    sfc_long_col,
    epsg=4326,
    start_buffer=500.0,
    max_buffer=1500.0,
    max_iter=20,
    buffer_distance_ft=5280,
    rec_conformity_threshold=0.5
    ):
    '''
    Args:
    - df (DataFrame): DataFrame containing the following columns:
        - geometry (LineString or MultiLineString or WKT string): The geometry column representing
          the well survey in xy coordinates
        - surface latitude (float): The latitude of the surface location
        - surface longitude (float): The longitude of the surface location
    - geo_col (str): The name of the geometry column
    - sfc_lat_col (str): The name of the surface latitude column
    - sfc_long_col (str): The name of the surface longitude column
    - start_buffer (float): The starting radial buffer distance around the sfc_loc in feet
    - max_buffer (float): The maximum radial buffer distance around the sfc_loc in feet
    - max_iter (int): The maximum number of iterations on the radial buffer size to perform
    - buffer_distance_ft (float): The buffer distance around the LateralLine in feet
    - rec_conformity_threshold (float): The minimum rectangle conformity threshold to for optimization
      of the surface location buffer
    Returns:
    - df (DataFrame): DataFrame with the following columns:
        - geometry (LineString or MultiLineString): The geometry column representing the well survey in xy coordinates
        - surface latitude (float): The latitude of the surface location
        - surface longitude (float): The longitude of the surface location
        - sfc_loc (Point): The surface location as a Shapely Point geometry object
        - optimal_buffer (float): The optimal buffer distance around the surface location in feet
        - optimal_conformity (float): The rectangle conformity of the optimal buffer
        - clipped_lateral_geometry (LineString or MultiLineString): The clipped lateral geometry
        - lateral_geometry_buffer (Polygon): The buffer around the clipped lateral geometry
    Example Use:
    - new_df = optimize_buffer(df, geo_col='Geometry', sfc_lat_col='Latitude', sfc_long_col='Longitude', buffer_distance_ft=5280)
    '''  # noqa
    # Check if geometry is a string and try to convert it using wkt.loads
    df[geo_col] = df[geo_col].apply(wkt.loads) if isinstance(df[geo_col].iloc[0], str) else df[geo_col]

    # Filter dataframe to only include LineString and MultiLineString geometries
    df = df[df[geo_col].apply(lambda x: isinstance(x, LineString) or isinstance(x, MultiLineString))]

    # Create geometry objects from lat/long coordinates
    df['sfc_loc'] = df.apply(lambda x: Point(x[sfc_long_col], x[sfc_lat_col]), axis=1)

    # Reproject geometries to EPSG:6579
    gdf1 = gpd.GeoDataFrame(df, geometry=geo_col, crs=f"EPSG:{epsg}")
    gdf2 = gpd.GeoDataFrame(df, geometry='sfc_loc', crs=f"EPSG:{epsg}")
    target_crs = pyproj.CRS("EPSG:6579")
    gdf1_projected = gdf1.to_crs(target_crs)
    gdf2_projected = gdf2.to_crs(target_crs)
    df[geo_col] = gdf1_projected.geometry
    df['sfc_loc'] = gdf2_projected.geometry

    # Check for invalid geometries
    df['geometry_is_valid'] = df[geo_col].apply(lambda x: x.is_valid)
    df['sfc_loc_is_valid'] = df['sfc_loc'].apply(lambda x: x.is_valid)

    # Drop invalid geometries
    invalid_rows = df[(df['geometry_is_valid'] == False) | (df['sfc_loc_is_valid'] == False)].index
    df = df.drop(invalid_rows)

    # Simplify MultiLineString geometries
    df[geo_col] = df[geo_col].apply(lambda x: x.simplify(tolerance=0.1) if isinstance(x, MultiLineString) else x)

    # Calculate the step
    step = (max_buffer - start_buffer) / max_iter

    # Apply cleaning, filtering, and interpolation before the main logic
    mask = df[geo_col].apply(has_at_least_two_unique_coords)
    df = df[mask]
    df.loc[:, geo_col] = df[geo_col].apply(interpolate_points)

    optimal_buffers = []
    optimal_conformities = []
    clipped_lateral_geometries = []
    lat_geo_buffers = []

    for _, row in df.iterrows():
        sfc_loc_buffer_ft = start_buffer
        iterations = 0
        max_conformity = rec_conformity_threshold
        optimal_buffer = start_buffer

        # Handle case where lateral_geometry is a LineString or MultiLineString
        lateral_geometry = row[geo_col].difference(row['sfc_loc'].buffer(sfc_loc_buffer_ft / 3.281))

        # Create a buffer around the LateralLine
        lat_geo_buffer = lateral_geometry.buffer(buffer_distance_ft / 3.281, cap_style=2)

        rect_conf = rectangle_conformity(lat_geo_buffer)

        if rect_conf > max_conformity:
            max_conformity = rect_conf
            optimal_buffer = sfc_loc_buffer_ft

        sfc_loc_buffer_ft += step
        iterations += 1

        # Get end points of resulting clipped LateralLine
        start, end = get_endpoints(lateral_geometry)
        if start is None or end is None:
            optimal_buffers.append(None)
            optimal_conformities.append(None)
            clipped_lateral_geometries.append(None)
            lat_geo_buffers.append(None)
            continue

        # Create a LineString from the endpoints
        end_line = LineString([start, end])

        # Create a buffer around the end_line
        end_line_buffer = end_line.buffer(buffer_distance_ft / 3.281, cap_style=2)

        optimal_buffers.append(optimal_buffer)
        optimal_conformities.append(max_conformity)
        clipped_lateral_geometries.append(lateral_geometry)
        lat_geo_buffers.append(end_line_buffer)

    df['optimal_buffer'] = optimal_buffers
    df['optimal_conformity'] = optimal_conformities
    df['clipped_lateral_geometry'] = clipped_lateral_geometries
    df['lateral_geometry_buffer'] = lat_geo_buffers

    return df


# Function to prepare the DataFrame for distance calculations. For use after the optimize_buffer function and before the calculate_distance function.  # noqa
def prep_df_distance(df, well_id_col):
    '''
    This function is for use after the optimize_buffer function. It takes the output of the optimize_buffer function and
    prepares the dataframe for distance calculations.
    Args:
    - df (DataFrame): DataFrame resulting from the optimize_buffer function
    - well_id_col (str): The name of the well ID column
    Returns:
    - df (DataFrame): DataFrame with the following columns:
        - WellID (str): The well ID
        - optimal_buffer (float): The optimal buffer distance around the surface location in feet
        - optimal_conformity (float): The rectangle conformity of the optimal buffer
        - clipped_lateral_geometry (LineString): The clipped lateral geometry
        - lateral_geometry_buffer (Polygon): The buffer around the clipped lateral geometry
    Example Use:
    - clean_df = prep_df_distance(new_df, well_id_col='WellID')
    '''
    # Filter out rows with NaN in 'clipped_lateral_geometry' or 'lateral_geometry_buffer'
    filtered_df = df.dropna(subset=['clipped_lateral_geometry', 'lateral_geometry_buffer'])

    # Ensure 'clipped_lateral_geometry' and 'lateral_geometry_buffer' are shapely geometries
    filtered_df['clipped_lateral_geometry'] = filtered_df['clipped_lateral_geometry'].apply(wkt_to_geom)
    filtered_df['lateral_geometry_buffer'] = filtered_df['lateral_geometry_buffer'].apply(wkt_to_geom)

    # Convert to GeoDataFrames
    gdf_lines = gpd.GeoDataFrame(filtered_df, geometry='clipped_lateral_geometry')
    gdf_buffers = gpd.GeoDataFrame(filtered_df, geometry='lateral_geometry_buffer')

    # Set index names for joining
    gdf_lines.index.name = 'index_left'
    gdf_buffers.index.name = 'index_right'

    # Create spatial index
    gdf_buffers.sindex

    # Find intersecting wells
    intersecting_wells = gpd.sjoin(gdf_lines, gdf_buffers, predicate='intersects', how='inner')

    # Remove wells that intersect with themselves
    intersecting_wells = (
        intersecting_wells[intersecting_wells[f'{well_id_col}_left'] != intersecting_wells[f'{well_id_col}_right']]
    )

    # Aggregate intersecting well IDs
    intersecting_well_ids = intersecting_wells.groupby('index_left')[f'{well_id_col}_right'].apply(list)

    # Merge intersecting well IDs into the original DataFrame
    new_df = df.merge(
        intersecting_well_ids.rename('Intersecting_Well_IDs'),
        how='left',
        left_index=True,
        right_index=True
    )

    # Explode Intersecting_Well_IDs for one-to-many mapping
    exploded_df = new_df.explode('Intersecting_Well_IDs').dropna(subset=['Intersecting_Well_IDs'])

    # Merge to get lateral lines and buffers from neighboring wells
    merged_df = pd.merge(
        exploded_df[[well_id_col, 'Intersecting_Well_IDs', 'clipped_lateral_geometry', 'lateral_geometry_buffer']],
        new_df[[well_id_col, 'clipped_lateral_geometry', 'lateral_geometry_buffer']],
        left_on='Intersecting_Well_IDs',
        right_on=well_id_col,
        suffixes=('', '_from_neighbor')
    )

    # Rename columns for clarity
    result_df = merged_df.rename(columns={
        'Intersecting_Well_IDs': 'neighboring_WellID',
        'clipped_lateral_geometry_from_neighbor': 'clipped_neighbor_lateral_geometry',
        'lateral_geometry_buffer_from_neighbor': 'neighbor_lateral_geometry_buffer'
    })

    # Drop redundant columns
    result_df.drop(columns=[f'{well_id_col}_from_neighbor'], inplace=True)

    # Rename well_id_col to Well_ID
    result_df.rename(columns={well_id_col: 'WellID'}, inplace=True)

    return result_df


# Function to calculate distances between two lines
def calculate_distance(row, min_distance_ft=100.0):
    '''
    This function assumes that the geometries are projected to EPSG:6579 with units in meters
    Args:
    - row: Row from a DataFrame resulting from the prep_df_distance function
    - min_distance_ft: The minimum distance in feet between the two lines to be considered in calculations. Default is 100 ft.
    Returns:
    - min_distance_ft: The minimum distance in feet between the two lines
    - median_distance_ft: The median distance in feet between the two lines
    - max_distance_ft: The maximum distance in feet between the two lines
    - avg_distance_ft: The average distance in feet between the two lines
    - intersection_fraction: The fraction of the neighboring lateral that intersects with the buffer of the reference lateral
    - relative_position: The relative position of the neighboring lateral with respect to the reference lateral
    Example Use:
    - clean_df[['MinDistance', 'MedianDistance', 'MaxDistance', 'AvgDistance', 'neighbor_IntersectionFraction', 'RelativePosition']] = clean_df.apply(calculate_distance, axis=1, result_type='expand')
    '''  # noqa
    # Identify the well's lateral and the adjacent lateral
    reference_geometry = wkt_to_geom(row['clipped_lateral_geometry'])
    neighbor_geometry = wkt_to_geom(row['clipped_neighbor_lateral_geometry'])

    # Identify buffers around each lateral
    reference_buffer = wkt_to_geom(row['lateral_geometry_buffer'])
    neighbor_buffer = wkt_to_geom(row['neighbor_lateral_geometry_buffer'])

    # Clean up geometries
    reference_geometry = get_longest_linestring(reference_geometry)
    neighbor_geometry = get_longest_linestring(neighbor_geometry)

    # Calculate the intersection of each line with the buffer of the other line
    reference_intersection = reference_geometry.intersection(neighbor_buffer)
    neighbor_intersection = neighbor_geometry.intersection(reference_buffer)

    # Handle GeometryCollection types by extracting the longest LineString
    reference_intersection = get_longest_linestring(reference_intersection)
    neighbor_intersection = get_longest_linestring(neighbor_intersection)

    neighbor_intersection_length_ft = 0.0  # Initialize the intersection length to 0

    # Check if either intersection is empty after handling GeometryCollections
    if reference_intersection.is_empty or neighbor_intersection.is_empty:
        return None, None, None, None, None, None

    else:
        neighbor_intersection_length_ft = neighbor_intersection.length * 3.281  # Convert from meters to feet

        # Calculate the total length of neighbor_geometry
        total_neighbor_length_ft = neighbor_geometry.length * 3.281  # Convert from meters to feet

        # Calculate the percentage of neighbor_geometry that intersects with the buffer of reference_geometry
        intersection_fraction = (
            (neighbor_intersection_length_ft / total_neighbor_length_ft) if total_neighbor_length_ft != 0 else 0
        )

    if reference_intersection.geom_type == 'Point':
        distances = [reference_geometry.distance(reference_intersection) * 3.281]
    else:
        if reference_intersection.geom_type == 'LineString':
            lines = [reference_intersection]
        elif reference_intersection.geom_type == 'MultiLineString':
            lines = [get_longest_linestring(reference_intersection)]
        else:
            raise ValueError(f"Unhandled geometry type: {reference_intersection.geom_type}")

        distances = [neighbor_geometry.distance(Point(x, y)) * 3.281 for line in lines for x, y in line.coords]

        if not distances:
            return None, None, None, None, None, None

        avg_distance = np.mean(distances)
        std_dev = np.std(distances)

        filtered_distances = (
            [d for d in distances if avg_distance - std_dev <= d <= avg_distance + std_dev and d >= min_distance_ft]
        )

        if not filtered_distances:
            return None, None, None, None, None, None

        # Normalize the azimuth of reference_geometry to a 180-degree system
        azimuth_lateral = calculate_azimuth(reference_geometry, reverse=True) % 180

        # Get the centroids of the two lines
        lat_centroid = reference_geometry.centroid
        neighbor_centroid = neighbor_geometry.centroid

        # Create a LineString geometry from the centroids
        centroid_line = LineString([lat_centroid, neighbor_centroid])

        # Keep the azimuth of the centroid line in a 360-degree system
        azimuth_centroid_line = calculate_azimuth(centroid_line, reverse=True)

        # Calculate the relative azimuth between reference_geometry and the centroid line
        relative_position = determine_relative_direction(azimuth_lateral, azimuth_centroid_line)

        return
        min(filtered_distances),
        np.median(filtered_distances),
        max(filtered_distances),
        np.mean(filtered_distances),
        intersection_fraction,
        relative_position


# Function to determine parent-child relationship
def parent_child(day_diff, min_days_threshold):
    '''
    Determine the parent-child relationship between two wells based on the difference in their first production dates.
    Args:
    - day_diff (int): Difference in first production dates between the reference well (well in question) and an adjacent well.
                      A positive value indicates the reference well started production after the adjacent well.
    - min_days_threshold (int): Threshold in days to determine if wells are co-completed. If the absolute difference in first 
                                production dates is less than or equal to this value, the wells are considered co-completed.
    Returns:
    - 'Co': if wells are co-completed (within min_days_threshold).
    - 'P': if the reference well is the parent (started production earlier).
    - 'C': if the reference well is the child (started production later).

    Example Use:
    # Example DataFrame 'spacing_df'
    spacing_df = pd.DataFrame({
        'WellID': [1, 2, 3, 4],
        'neighboring_WellID': [2, 3, 4, 1],
        'DayDiff': [10, -15, 30, -5]
    })
    - spacing_df['Relationship'] = spacing_df.apply(lambda x: parent_child(x['DayDiff'], 30), axis=1)
      Here, spacing_df is a DataFrame with wells data, and 'DayDiff' is the column indicating the difference in days 
      between the first production of the well in question and its adjacent well.

    - spacing_df['Relationship'] = spacing_df['DayDiff'].apply(lambda x: parent_child(x, co_completed_threshold))
      In this example, spacing_df is a DataFrame where 'DayDiff' represents the day difference to the nearest 
      neighboring well's first production date. 'co_completed_threshold' is a predefined threshold value.
    '''  # noqa

    if abs(day_diff.days) <= min_days_threshold:
        return 'Co'
    return 'P' if day_diff.days < 0 else 'C'


# Function to find the closest distance and resulting relationship between wells over time
def get_min_distance_rows(df, id_col, position_col, date_col, distance_col):
    '''
    Process a DataFrame to find the closest distance and resulting relationship between wells over time.
    Args:
    - df (DataFrame): DataFrame containing well spacing information.
    - id_col (str): Name of the column representing the well ID.
    - position_col (str): Name of the column representing the relative position.
    - date_col (str): Name of the column representing the date.
    - distance_col (str): Name of the column representing the average distance.
    Returns:
    - DataFrame: Processed DataFrame with minimum average distance and related information.
    Example Use:
    - updated_df = get_min_distance_rows(spacing_df, 'WellID', 'RelativePosition', 'Date', 'AvgDistance')
    '''
    df = df.sort_values(by=[id_col, position_col, date_col])
    min_distance_col = 'min_' + distance_col
    df[min_distance_col] = df.groupby([id_col, position_col])[distance_col].transform('cummin')
    df[distance_col] = df[distance_col].mask(df[distance_col] != df[min_distance_col])
    df.sort_values(by=[id_col, position_col, date_col, min_distance_col], inplace=True)
    df.dropna(subset=[distance_col], inplace=True)
    df.drop_duplicates(subset=[id_col, position_col, date_col], keep='first', inplace=True)

    return df.drop(columns=min_distance_col)


# Function to process data from well distance calculations to determine relationships and closest distances between wells  # noqa
def parent_child_processing(
    spacing_df,
    well_df,
    co_completed_threshold,
    id_col,
    position_col,
    date_col,
    distance_col,
    neighbor_date_col,
    scenario_name
    ):
    '''
    Process well data to determine relationships and distances between wells.
    Args:
    - spacing_df (DataFrame): DataFrame containing well spacing information.
    - well_df (DataFrame): DataFrame containing well information.
    - co_completed_threshold (int): Threshold in days to determine if wells are co-completed.
    - id_col (str): Name of the column representing the well ID.
    - position_col (str): Name of the column representing the relative position.
    - date_col (str): Name of the column representing the date.
    - distance_col (str): Name of the column representing the average distance.
    - neighbor_date_col (str): Name of the column representing the neighbor's first production date.
    Returns:
    - DataFrame: Processed DataFrame with well relationships and distances.
        Example Use:
    # Example DataFrame 'spacing_df'
    spacing_df = pd.DataFrame({
        'WellID': [1, 2, 3, 4],
        'neighboring_WellID': [2, 3, 4, 1],
        'DayDiff': [10, -15, 30, -5]
    })
    # Example DataFrame 'well_df'
    well_df = pd.DataFrame({
        'WellID': [1, 2, 3, 4],
        'API_UWI_Unformatted': ['1001', '1002', '1003', '1004'],
        'Basin': ['BasinA', 'BasinB', 'BasinC', 'BasinD'],
        'Interval': ['Interval1', 'Interval2', 'Interval3', 'Interval4'],
        'FirstProdDate': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01']
    })
    - closest_wells = process_well_data(spacing_df, well_df, co_completed_threshold, 'WellID', 'RelativePosition', 'FirstProdDate', 'AvgDistance', 'neighbor_FirstProdDate', 'ENVERUS')
    '''  # noqa
    # Apply the parent_child function to the spacing_df dataframe
    spacing_df['Relationship'] = spacing_df.apply(
        lambda x: parent_child(x[date_col] - x[neighbor_date_col],
        co_completed_threshold),
        axis=1
    )

    # Clean up dataframe to find the closest distance and resulting relationship between wells over time
    spacing_df['Date'] = np.maximum(spacing_df[date_col], spacing_df[neighbor_date_col])
    spacing_df = get_min_distance_rows(spacing_df, id_col, position_col, 'Date', distance_col)

    # Pivot closest wells to see distance and relationship on each side of the reference well
    pivoted_spacing_df = spacing_df.pivot(
        index=[id_col, 'Date', date_col],
        columns=position_col,
        values=[distance_col, 'Relationship']
    ).reset_index()

    # Flatten the MultiIndex columns and rename for clarity
    pivoted_spacing_df.columns = ['_'.join(col).rstrip('_') for col in pivoted_spacing_df.columns.values]
    pivoted_spacing_df.rename(columns={f'{distance_col}_LEFT': 'ClosestHzDistance_Left',
                                       f'{distance_col}_RIGHT': 'ClosestHzDistance_Right',
                                       'Relationship_LEFT': 'LEFT_Relationship',
                                       'Relationship_RIGHT': 'RIGHT_Relationship'}, inplace=True)
    pivoted_spacing_df = pivoted_spacing_df.sort_values(by=[id_col, 'Date']).reset_index(drop=True)
    columns_to_fill = ['ClosestHzDistance_Left', 'ClosestHzDistance_Right', 'LEFT_Relationship', 'RIGHT_Relationship']
    for col in columns_to_fill:
        pivoted_spacing_df[col] = pivoted_spacing_df.groupby(id_col)[col].ffill()

    pivoted_spacing_df['LEFT_Relationship'] = pivoted_spacing_df['LEFT_Relationship'].fillna('')
    pivoted_spacing_df['RIGHT_Relationship'] = pivoted_spacing_df['RIGHT_Relationship'].fillna('')
    pivoted_spacing_df['Relationship'] = (
        pivoted_spacing_df['LEFT_Relationship'] + '|' + pivoted_spacing_df['RIGHT_Relationship']
    )

    pivoted_spacing_df = (
        pivoted_spacing_df[pivoted_spacing_df['Date'] >= pivoted_spacing_df[date_col]].reset_index(drop=True)
    )
    pivoted_spacing_df['ClosestHzDistance'] = (
        pivoted_spacing_df[['ClosestHzDistance_Left', 'ClosestHzDistance_Right']].min(axis=1)
    )
    pivoted_spacing_df.drop(columns=[date_col, 'LEFT_Relationship', 'RIGHT_Relationship'], inplace=True)

    closest_wells = well_df.merge(
        pivoted_spacing_df,
        left_on=[id_col, date_col],
        right_on=[id_col, 'Date'],
        how='outer',
        indicator=True
    ).drop_duplicates()
    closest_wells['Date'] = closest_wells['Date'].fillna(closest_wells[date_col])
    closest_wells = closest_wells.drop(columns=[date_col, '_merge'])
    closest_wells['Relationship'] = closest_wells['Relationship'].fillna('S')
    closest_wells.sort_values(by=[id_col, 'Date'], inplace=True)
    closest_wells = closest_wells[
        [id_col, 'Date', 'Relationship', 'ClosestHzDistance', 'ClosestHzDistance_Left', 'ClosestHzDistance_Right']
    ].copy()

    closest_wells['ScenarioName'] = scenario_name
    closest_wells['UpdateDate'] = pd.Timestamp.now()

    float_cols = ['ClosestHzDistance', 'ClosestHzDistance_Left', 'ClosestHzDistance_Right']
    for col in float_cols:
        closest_wells[col] = closest_wells[col].astype(float)

    return closest_wells


# Function to calculate distances between verical wells
def calc_vertical_distance(gdf, buffer_radius, id_col, geo_col, date_col, source_epsg=4326):
    '''
    Args:
        gdf (GeoDataFrame): A GeoDataFrame of geometries, including a 'WellID' column.
        buffer_radius (float): The radius to buffer each geometry by, in feet.
        id_col (str): The name of the column containing the well IDs.
        geo_col (str): The name of the column containing the geometries.
        date_col (str): The name of the column containing the first production date.
        source_epsg (int): The EPSG code of the geometries.
    Returns:
        GeoDataFrame: The original GeoDataFrame with new columns for intersecting WellIDs and distances.
    '''

    # Ensure the geometries are in the correct CRS and convert them to EPSG:6579
    if gdf.crs is None or gdf.crs.to_epsg() != source_epsg:
        gdf = gdf.to_crs(epsg=source_epsg)
    gdf = gdf.to_crs(epsg=6579)

    # Find intersecting geometries
    intersections = []
    for index, row in gdf.iterrows():
        geometry = row[geo_col]
        centroid = geometry.centroid if geometry.geom_type != 'Point' else geometry
        buffer = centroid.buffer(buffer_radius / 3.281)  # Convert feet to meters
        intersecting_wellids = gdf[gdf.intersects(buffer)][id_col].tolist()
        if row[id_col] in intersecting_wellids:
            intersecting_wellids.remove(row[id_col])
        intersections.append(intersecting_wellids)

    gdf[f'neighboring_{id_col}'] = intersections

    # Explode the DataFrame
    exploded_df = gdf.explode(f'neighboring_{id_col}')
    exploded_df = exploded_df.merge(
        gdf[[id_col, geo_col, date_col]],
        left_on=f'neighboring_{id_col}',
        right_on=id_col,
        suffixes=('', '_neighboring')
    )
    exploded_df.drop(columns=[f'{id_col}_neighboring'], inplace=True)

    # Calculate distances
    exploded_df['distance_FT'] = (
        exploded_df[geo_col].centroid.distance(exploded_df[f'{geo_col}_neighboring'].centroid) * 3.281
    )

    return exploded_df
