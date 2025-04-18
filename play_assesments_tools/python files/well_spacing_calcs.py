import numpy as np
import geopandas as gpd
import pandas as pd
from config.config_loader import get_config
import AnalyticsAndDBScripts.sql_connect as sql
import AnalyticsAndDBScripts.sql_schemas as schema
import AnalyticsAndDBScripts.well_spacing as ws
from concurrent.futures import as_completed
from multiprocessing import Lock, Manager
from loky import get_reusable_executor, wrap_non_picklable_objects
import gc
import warnings

# Ignore warnings
warnings.filterwarnings(action='ignore')

# Path to the config file
config_path = '/app/conduit/config/analytics_config.yaml'

# Load credentials for SQL
sql_creds_dict = get_config('credentials', 'sql1_sa', path=config_path)

# Add db_name to the dictionary
sql_creds_dict['db_name'] = 'Analytics'

# Load parameters
params = get_config('well_spacing', path=config_path)
fit_groups_config = params['fit_groups']
projection = params['final_projection']
min_lat_length = params['minimum_lateral_length']
update_date = pd.Timestamp.now()
day_offset = params['day_offset']

# Function to create sql statements
def create_statement(config, group_name, min_lat_length, day_offset=0):
    # Function to extract basin list from config
    def get_basins(config, group_name):
        return next((group['basins'] for group in config if group['name'] == group_name), None)

    # Functions to create sql statements
    def create_statement_inclusive(config, group_name, min_lat_length, day_offset):
        basin_sql = "', '".join(get_basins(config, group_name))
        return f'''
        WITH MinUpdateDate AS (
            SELECT      MIN(S.UpdateDate) AS MinDate
            FROM        dbo.WELL_SPACING S
            INNER JOIN  dbo.WELL_HEADER W ON S.WellID = W.WellID
            WHERE       W.Trajectory = 'HORIZONTAL' 
            AND         W.FirstProdDate >= '2003-01-01' 
            AND         W.LateralLength_FT > {min_lat_length}
            AND         W.Basin IN ('{basin_sql}')
        ),
        FilteredWells AS (
            SELECT      W.WellID, W.API_UWI_Unformatted, W.Basin, W.FirstProdDate, W.LateralLength_FT, W.Geometry.STAsText() AS Geometry, 
                        W.Geometry.STSrid AS EPSGCode, W.Latitude, W.Longitude, W.Latitude_BH, W.Longitude_BH, U.MinDate
            FROM        dbo.WELL_HEADER W
            CROSS JOIN	MinUpdateDate U
            WHERE       W.Trajectory = 'HORIZONTAL' 
            AND         W.FirstProdDate >= '2003-01-01' 
            AND         W.LateralLength_FT > {min_lat_length}
            AND         W.Geometry IS NOT NULL
            AND         W.Basin IN ('{basin_sql}')
        )
        SELECT      * 
        FROM        FilteredWells
        WHERE       MinDate IS NULL OR DATEADD(day, {day_offset}, MinDate) <= CAST(GETDATE() AS DATE)
        '''

    # Function to create SQL statement for basins not in any group
    def create_statement_exclusive(config, min_lat_length, day_offset):
        all_basins = (basin for group in config for basin in group['basins'])
        all_basins_sql = "', '".join(all_basins)
        return f'''
        WITH MinUpdateDate AS (
            SELECT      MIN(S.UpdateDate) AS MinDate
            FROM        dbo.WELL_SPACING S
            LEFT JOIN   dbo.WELL_HEADER W ON S.WellID = W.WellID
            WHERE       W.Trajectory = 'HORIZONTAL' 
            AND         W.FirstProdDate >= '2003-01-01' 
            AND         W.LateralLength_FT > {min_lat_length}
            AND         (W.Basin NOT IN ('{all_basins_sql}') OR W.Basin IS NULL)
        ),
        FilteredWells AS (
            SELECT      W.WellID, W.API_UWI_Unformatted, W.Basin, W.FirstProdDate, W.LateralLength_FT, W.Geometry.STAsText() AS Geometry, 
                        W.Geometry.STSrid AS EPSGCode, W.Latitude, W.Longitude, W.Latitude_BH, W.Longitude_BH, U.MinDate
            FROM        dbo.WELL_HEADER W
            CROSS JOIN	MinUpdateDate U
            WHERE       W.Trajectory = 'HORIZONTAL' 
            AND         W.FirstProdDate >= '2003-01-01' 
            AND         W.LateralLength_FT > {min_lat_length}
            AND         W.Geometry IS NOT NULL
            AND         (W.Basin NOT IN ('{all_basins_sql}') OR W.Basin IS NULL)
        )
        SELECT      * 
        FROM        FilteredWells
        WHERE       MinDate IS NULL OR DATEADD(day, {day_offset}, MinDate) <= CAST(GETDATE() AS DATE)
        '''
    
    if group_name == 'OTHER':
        return create_statement_exclusive(config, min_lat_length, day_offset)
    else:
        return create_statement_inclusive(config, group_name, min_lat_length, day_offset)
        
# Execute query and store results in a dataframe
def load_data(creds, statement):
    engine = sql.sql_connect(
        username=creds['username'], 
        password=creds['password'], 
        db_name=creds['db_name'], 
        server_name=creds['servername'], 
        port=creds['port']
    )
    try:
        df = pd.read_sql(statement, engine)
    finally:
        engine.dispose()
    return df
    
# Function to process each basin
@wrap_non_picklable_objects
def process_data(args):
    config, group_name, projection, min_lat_length, day_offset, update_date, sql_creds_dict, lock = args
    try:
        # Load data from SQL Server
        statement = create_statement(config, group_name, min_lat_length, day_offset)
        print(f"Loading data for fit_group {group_name}")
        df = load_data(sql_creds_dict, statement)

        if df.empty:
            print(f"No data for fit_group {group_name}")
            return
    
        # Apply optimize_buffer function to dataframe
        df = ws.optimize_buffer(df, geo_col='Geometry', sfc_lat_col='Latitude', sfc_long_col='Longitude', buffer_distance_ft=params['buffer_distance'])

        # Clean dataframe and prep for distance calculations
        df = ws.prep_df_distance(df, well_id_col='WellID')

        # Apply calculations to the dataframe
        df_cols = ['MinDistance', 'MedianDistance', 'MaxDistance', 'AvgDistance', 'neighbor_IntersectionFraction', 'RelativePosition']
        df[df_cols] = df.apply(ws.calculate_distance, axis=1, result_type='expand')

        # Add a few columns to the dataframe
        df['Projection'] = projection
        df['UpdateDate'] = update_date

        # Drop rows with null values and compute the dataframe
        df = df.dropna()

        def split_dataframe_generator(df, chunk_size):
            nrows = df.shape[0]
            for i in range(0, nrows, chunk_size):
                yield df.loc[i:i + chunk_size - 1]

        # Columns that contain spatial data
        geometry_columns = ['clipped_lateral_geometry', 'lateral_geometry_buffer', 'clipped_neighbor_lateral_geometry', 'neighbor_lateral_geometry_buffer']

        for chunk in split_dataframe_generator(df, 500000):
            # Reproject geometries from EPSG:6579 to defined projection
            for col in geometry_columns:
                gdf = gpd.GeoDataFrame(chunk, geometry=col, crs='EPSG:6579')
                gdf = gdf.to_crs(projection)
                chunk[col] = gdf.geometry
            chunk = chunk.map(ws.geom_to_wkt)
            sql.load_data_to_sql(chunk, sql_creds_dict, schema.well_spacing_stage, lock)
            del chunk
            gc.collect()

        # Move data from dbo.WELL_SPACING_STAGE to dbo.WELL_SPACING and drop dbo.WELL_SPACING_STAGE
        sql.execute_stored_procedure(sql_creds_dict, 'sp_InsertFromStagingToWellSpacing', lock)

        print(f"Processing fit_group {group_name} complete")

    except Exception as e:
        print(f"Error processing fit_group {group_name}: {e}")
        gc.collect()
        
# Calculate well spacing calcuations and load data into Axia_Anaytics
fit_group_list = list(group['name'] for group in fit_groups_config) + ['OTHER']

with Manager() as manager:
    lock = manager.Lock()
    args_list = ((fit_groups_config, group, projection, min_lat_length, day_offset, update_date, sql_creds_dict, lock) for group in fit_group_list)

    with get_reusable_executor(max_workers=1) as executor:
        futures = (executor.submit(process_data, args) for args in args_list)

        for future in as_completed(futures):
            try:
                future.result()  # Get the result to catch any exceptions
            except Exception as exc:
                print(f'Generated an exception: {exc}')
                gc.collect()
