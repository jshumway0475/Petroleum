from config.config_loader import get_config
import AnalyticsAndDBScripts.geo_functions as geo
import AnalyticsAndDBScripts.sql_connect as sql
import AnalyticsAndDBScripts.sql_schemas as schema
from sqlalchemy.exc import SQLAlchemyError
import logging
from sqlalchemy import BigInteger, text
import numpy as np
import pandas as pd
import io
import os
from shapely import wkt
import geopandas as gpd

# Path to the config file
config_path = '/app/conduit/config/analytics_config.yaml'

# Load configs
sql_creds_dict = get_config('credentials', 'sql1_sa', path=config_path)
grid_config = get_config('geology', path=config_path)

# Add db_name to sql_creds_dict
sql_creds_dict['db_name'] = 'Analytics'

# Add grid files to a dictionary of arrays
for grid in grid_config:
    with open(grid['path'], 'r') as f:
        content = f.read()
        array = np.loadtxt(io.StringIO(content), delimiter=grid['delimiter'])
        grid['array'] = array

def sql_statement(config):
    interval = config['interval']

    # Check if interval is a list
    if type(interval) is list:
        interval = "', '".join([str(i) for i in interval])
    sql = f'''
    SELECT      W.WellID, W.CurrentCompletionID, C.Interval, C.DataSource, W.Geometry.STAsText() AS Geometry
    FROM        dbo.WELL_HEADER W
    INNER JOIN  dbo.COMPLETION_HEADER C
    ON          W.CurrentCompletionID = C.CompletionID
    WHERE       C.Interval IN ('{interval}')
    AND         C.DataSource = 'ENVERUS'
    '''
    return sql

sampled_dfs = []

for grid in grid_config:
    # Execute query and store results in a dataframe
    sql_engine = sql.sql_connect(
        username=sql_creds_dict['username'], 
        password=sql_creds_dict['password'], 
        db_name=sql_creds_dict['db_name'], 
        server_name=sql_creds_dict['servername'], 
        port=sql_creds_dict['port']
    )
    try:
        well_df = pd.read_sql(sql_statement(grid), sql_engine).drop_duplicates()
    finally:
        sql_engine.dispose()

    # Convert geometry to a shapely object
    well_df['Geometry'] = well_df['Geometry'].apply(wkt.loads)
    well_gdf = gpd.GeoDataFrame(well_df, geometry='Geometry', crs="EPSG:4326")

    # Filter dataframe to only include valid geometries
    valid_geometries = well_gdf['Geometry'].notnull() & ~well_gdf['Geometry'].is_empty
    well_gdf = well_gdf[valid_geometries]
    df_out = geo.sample_xyz(
        df=well_gdf, 
        file_name=grid['name'], 
        arr=grid['array'], 
        epsg=grid['epsg'], 
        id_col='WellID', 
        geo_col='Geometry', 
        sample_method='linear', 
        input_type=grid['type']
    ).drop_duplicates()
    df_out.loc[:, 'destination_column'] = grid['destination_column']
    sampled_dfs.append(df_out)

# Merge all dataframes
df = pd.concat(sampled_dfs, axis=0)

# Check for duplicates
duplicates = df[df.duplicated(subset=['WellID', 'file_name', 'destination_column'], keep=False)]  

# Handle duplicates
if not duplicates.empty:
    print("Duplicates found in the DataFrame based on 'WellID' and 'destination_column'.")
    print(duplicates)  # Optionally, print the duplicates for debugging
    raise ValueError("Execution stopped due to duplicate entries.")

# Continue with other operations if no duplicates are found
print("No duplicates found. Continuing with further operations.")

# Pivot df where WellID is the index, the columns are the destination columns, and the values are the sampled values
df = df.pivot_table(index='WellID', columns='destination_column', values='sampled_z', aggfunc='sum').reset_index()

# Add a DataSource column
df['DataSource'] = 'ENVERUS'

# Convert NaN to None for proper database insertion
df = df.where(pd.notnull(df), None)

# Prepare staging table schema
staging_tbl_name = 'GEO_STAGE'
temp_schema = schema.dataframe_to_table_schema(df, staging_tbl_name, {'WellID': BigInteger})

# Load well_df into dbo.FORECAST_STAGE table in SQL Server
sql.load_data_to_sql(df, sql_creds_dict, temp_schema)

# Prepare the update query
query = sql.generate_update_query(
    df=df,
    dest_table_name='dbo.COMPLETION_HEADER',
    source_table_name=f'dbo.{staging_tbl_name}',
    join_columns=['WellID', 'DataSource']
)

# Set up logging
logging.basicConfig(level=logging.INFO)

try:
    # Connect to the database
    sql_engine = sql.sql_connect(
        username=sql_creds_dict['username'], 
        password=sql_creds_dict['password'], 
        db_name=sql_creds_dict['db_name'], 
        server_name=sql_creds_dict['servername'], 
        port=sql_creds_dict['port']
    )

    # Execute update and drop operations
    with sql_engine.connect() as connection:
        with connection.begin():
            query_text = text(query)
            connection.execute(query_text)
            logging.info("Update successfully executed.")    
            drop_query = text(f"DROP TABLE dbo.{staging_tbl_name}")
            connection.execute(drop_query)
            logging.info(f"Staging table dbo.{staging_tbl_name} dropped successfully.")

except SQLAlchemyError as e:
    logging.error(f"An error occurred: {e}")

finally:
    # Dispose of the engine explicitly in case of pooled connections
    if 'sql_engine' in locals():
        sql_engine.dispose()
        logging.info("Database connection closed.")
