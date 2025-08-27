import os
import pandas as pd
import numpy as np
from config.config_loader import get_config
import AnalyticsAndDBScripts.sql_connect as sql
import AnalyticsAndDBScripts.sql_schemas as schema
import AnalyticsAndDBScripts.well_spacing as ws
import gc
from multiprocessing import Pool, Lock

# Path to the config file
config_path = os.getenv("CONFIG_PATH")

# Load credentials for SQL connection
sql_creds_dict = get_config('credentials', 'sql1_sa', path=config_path)

# Add db_name to the dictionary
sql_creds_dict['db_name'] = 'Analytics'

# Load parameters
params = get_config('well_spacing', path=config_path)

# Define the chunk size
chunk_size = 500000

# SQL statement to get unique instances of DataSource
statement = '''
SELECT		DISTINCT S.WellID, W.DataSource
FROM		dbo.WELL_SPACING S
INNER JOIN	dbo.WELL_HEADER W 
ON			S.WellID = W.WellID
INNER JOIN	dbo.COMPLETION_HEADER C 
ON			W.CurrentCompletionID = C.CompletionID

UNION ALL

SELECT		DISTINCT S.WellID, O.DataSource
FROM		dbo.WELL_SPACING S
INNER JOIN	dbo.WELL_OVERRIDE O 
ON			S.WellID = O.WellID
'''

# Execute query and store results in a dataframe
engine = sql.sql_connect(
    username=sql_creds_dict['username'], 
    password=sql_creds_dict['password'], 
    db_name=sql_creds_dict['db_name'], 
    server_name=sql_creds_dict['servername'], 
    port=sql_creds_dict['port']
)
try:
    unique_df = pd.read_sql(statement, engine)
finally:
    engine.dispose()

# Create a list of unique instance of DataSource
data_source_list = unique_df['DataSource'].unique().tolist()

# Statement to load data from SQL Server into a pandas dataframe
intersection_fraction = params['intersection_fraction']
comparison_field = 'Interval'

# Function to create SQL statements for use in the script
def create_sql_statement(data_source, intersection_fraction, comparison_field):
    statement1 = f'''
    SELECT		W.WellID, W.API_UWI_Unformatted, W.FitGroup, 
                W.Basin, C.Interval, W.FirstProdDate
    FROM		dbo.WELL_HEADER W
    INNER JOIN	dbo.COMPLETION_HEADER C
    ON			W.CurrentCompletionID = C.CompletionID
    WHERE		W.FirstProdDate IS NOT NULL
    AND			UPPER(W.Trajectory) = 'HORIZONTAL'
    AND			W.Geometry IS NOT NULL
    AND			C.DataSource = '{data_source}'
    '''
    statement2 = f'''
    WITH WELL_CTE AS ({statement1})
    SELECT      S.WellID, S.neighboring_WellID, S.MinDistance, S.MedianDistance, S.AvgDistance, 
                S.Neighbor_IntersectionFraction, S.RelativePosition, W.FirstProdDate, 
                W2.FirstProdDate AS neighbor_FirstProdDate, 
                DATEDIFF(DAY, W.FirstProdDate, W2.FirstProdDate) AS DaysToNeighborFirstProd
    FROM        dbo.WELL_SPACING AS S
    INNER JOIN  WELL_CTE AS W
    ON          S.WellID = W.WellID
    INNER JOIN  WELL_CTE AS W2
    ON          S.neighboring_WellID = W2.WellID
    WHERE       S.neighbor_IntersectionFraction >= {intersection_fraction}
    AND 	   	W.Basin = W2.Basin
    AND 	   	W.{comparison_field} = W2.{comparison_field}
    '''
    return statement1, statement2

# Lock for database operations
db_lock = Lock()

# Function to process data for each unique instance of Basin and DataSource
def process_spacing_data(chunk):
    try:
        data_source = chunk['DataSource']
        statement1, statement2 = create_sql_statement(data_source, intersection_fraction, comparison_field)
        engine = sql.sql_connect(
            username=sql_creds_dict['username'], 
            password=sql_creds_dict['password'], 
            db_name=sql_creds_dict['db_name'], 
            server_name=sql_creds_dict['servername'], 
            port=sql_creds_dict['port']
        )
        try:
            well_df = pd.read_sql(statement1, engine)
            spacing_df = pd.read_sql(statement2, engine)
        finally:
            engine.dispose()

        # Run parent_child_processing function
        closest_wells = ws.parent_child_processing(
            spacing_df, 
            well_df, 
            co_completed_threshold=params['co_completed_threshold'], 
            id_col='WellID', 
            position_col='RelativePosition', 
            date_col='FirstProdDate', 
            distance_col='AvgDistance', 
            neighbor_date_col='neighbor_FirstProdDate', 
            scenario_name=data_source
        )

        # Convert NaN to None for proper database insertion
        closest_wells = closest_wells.where(pd.notnull(closest_wells), None)

        # Load well_df into dbo.PARENT_CHILD_STAGE table in SQL Server
        sql.load_data_to_sql(closest_wells, sql_creds_dict, schema.parent_child_stage, lock=db_lock)

        # Free up memory used by dataframes
        del well_df, spacing_df, closest_wells
        gc.collect()
    except Exception as e:
        print(f"Error processing chunk: {e}")

# Use multiprocessing to process chunks in parallel
if __name__ == '__main__':
    engine = sql.sql_connect(
        username=sql_creds_dict['username'], 
        password=sql_creds_dict['password'], 
        db_name=sql_creds_dict['db_name'], 
        server_name=sql_creds_dict['servername'], 
        port=sql_creds_dict['port']
    )
    try:
        for chunk in pd.read_sql(statement, engine, chunksize=chunk_size):
            unique_tasks = chunk.drop_duplicates(subset=['DataSource']).to_dict('records')
            
            with Pool() as pool:
                pool.map(process_spacing_data, unique_tasks)
    finally:
        engine.dispose()

    # Move data from dbo.PARENT_CHILD_STAGE to dbo.PARENT_CHILD and drop dbo.PARENT_CHILD_STAGE
    sql.execute_stored_procedure(sql_creds_dict, 'sp_InsertFromStagingToParentChild', lock=db_lock)
