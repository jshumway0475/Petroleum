import numpy as np
import pandas as pd
import multiprocessing
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client, Lock, as_completed, TimeoutError
from dask import delayed, compute
from config.config_loader import get_config
import AnalyticsAndDBScripts.sql_connect as sql
import AnalyticsAndDBScripts.sql_schemas as schema
import AnalyticsAndDBScripts.prod_fcst_functions as fcst
import os
import warnings
import logging
import time
from sqlalchemy.exc import OperationalError

# Ignore warnings
warnings.filterwarnings(action='ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set environment variables
os.environ["PYTENSOR_FLAGS"] = "floatX=float64,optimizer_excluding=constant_folding"

# Path to the config file
config_path = '/app/conduit/config/analytics_config.yaml'

# Load credentials for SQL server
sql_creds_dict = get_config('credentials', 'sql1_sa', path=config_path)

# Load parameters for the script
params_list = get_config('decline_curve', path=config_path)

# bifurcate the parameters
arps_params = next((item for item in params_list if item['name'] == 'arps_parameters'), None)
bourdet_params = next((item for item in params_list if item['name'] == 'bourdet_outliers'), None)
changepoint_params = next((item for item in params_list if item['name'] == 'detect_changepoints'), None)
b_estimate_params = next((item for item in params_list if item['name'] == 'estimate_b'), None)
smoothing_params = next((item for item in params_list if item['name'] == 'smoothing'), None)
method_params = next((item for item in params_list if item['name'] == 'method'), None)
segment_params = next((item for item in params_list if item['name'] == 'fit_segment'), None)

# Create distinct dictionaries for each database
sql_aries_creds_dict = sql_creds_dict.copy()
sql_aries_creds_dict['db_name'] = 'Analytics_Aries'
sql_creds_dict['db_name'] = 'Analytics'

# Define parameters
value_col = 'Value'
fit_segment = changepoint_params['fit_segment']
fit_method = method_params['setting']
trials = method_params['trials']
use_advi = method_params['use_advi']
fit_months = method_params['fit_months']
manual_analyst = method_params['manual_analyst']
ta_offset_months = method_params['ta_offset_mos']
new_data_months = method_params['new_data_mos']
log_folder = method_params['log_folder']
fit_population = method_params['fit_population']

# Initialize optional parameters for well_list and fit_group
well_list = method_params.get('well_list', [])
fit_group = method_params.get('fit_group', None)

# Define batch_size based on fit_method
if fit_method == 'monte_carlo':
    batch_size = 50
else:
    batch_size = 50000

# Load parameters from config file
def_dict = arps_params['terminal_decline']
dei_dict1 = arps_params['initial_decline']
min_q_dict = arps_params['abandonment_rate']
default_b_dict = arps_params['b_factor']

# Define columns for output dataframe
param_df_cols = [
    'WellID', 'Measure', 'fit_months', 'fit_type', 'fit_segment', 'StartDate', 
    'StartMonth', 'Q_guess', 'Q3', 'Dei', 'b_factor', 'R_squared', 'RMSE', 'MAE'
]

# Function to create sql query to get wells that need to be forecasted
def create_statement_wells(population, manual_analyst, ta_offset_mos=12, new_data_mos=3, well_list=[], fit_group=None):
    if population == 'all':
        statement = '''
        SELECT		WellID, Measure, LastProdDate
        FROM		dbo.vw_FORECAST
        WHERE		CumulativeProduction > 0
        AND			(Analyst != ? OR Analyst IS NULL)
        AND			LastProdDate > DATEADD(month, -?, GETDATE())
        AND			(DATEDIFF(month, LastProdDate, DateCreated) >= ? OR DateCreated IS NULL)
        ORDER BY	WellID, PHASE_INT
        '''
        params = (manual_analyst, ta_offset_mos, new_data_mos)
        return statement, params
    
    elif population == 'well_list':
        if not well_list:
            raise ValueError("well_list is empty, but population is set to 'well_list'.")
        sql_list = ', '.join(['?' for _ in well_list])
        statement = f'''
        SELECT		WellID, Measure, LastProdDate
        FROM		dbo.vw_FORECAST
        WHERE		CumulativeProduction > 0
        AND			(Analyst != ? OR Analyst IS NULL)
        AND			WellID IN ({sql_list})
        ORDER BY	WellID, PHASE_INT
        '''
        params = (manual_analyst,) + tuple(well_list)
        return statement, params
    
    elif population == 'fit_group':
        if not fit_group:
            raise ValueError("fit_group is not specified, but population is set to 'fit_group'.")
        statement = '''
        SELECT		F.WellID, F.Measure, F.LastProdDate
        FROM		dbo.WELL_HEADER W
        INNER JOIN  dbo.vw_FORECAST F
        ON          F.WellID = W.WellID
        WHERE		F.CumulativeProduction > 0
        AND			(F.Analyst != ? OR F.Analyst IS NULL)
        AND			W.FitGroup = ?
        ORDER BY	F.WellID, F.PHASE_INT
        '''
        params = (manual_analyst, fit_group)
        return statement, params
    
    # If population doesn't match any valid option
    else:
        raise ValueError("Invalid population type. Choose from 'all', 'well_list', or 'fit_group'.")

# Query to get production data from SQL Server
def create_statement(id, measure, last_prod_date, cadence='MONTHLY', fit_months=60):
    divisor_dict = {'DAILY': 1.0, 'MONTHLY': 30.42}
    statement = '''
    SELECT      WellID, Measure, Date, 
                Value / COALESCE(NULLIF(ProducingDays, 0), ?) AS Value
    FROM        dbo.PRODUCTION
    WHERE       Cadence = ?
    AND         WellID = ?
    AND         Measure = ?
    AND         Value > 0
    AND         Value IS NOT NULL
    AND			Date >= DATEADD(month, -?, ?)
    AND         SourceRank = 1
    ORDER BY    WellID, Date
    '''

    params = (divisor_dict[cadence], cadence, id, measure, fit_months, last_prod_date.strftime('%Y-%m-%d'))

    return statement, params

# Execute query and store results in a dataframe
def load_data(creds, statement_params):
    engine = sql.sql_connect(
        username=creds['username'], 
        password=creds['password'], 
        db_name=creds['db_name'], 
        server_name=creds['servername'], 
        port=creds['port']
    )

    statement, params = statement_params
    
    try:
        df = pd.read_sql(statement, engine, params=params)
    finally:
        engine.dispose()
    return df

# Function to handle OperationalError retries when loading data from SQL Server
def load_data_with_retry(creds, statement_params, retries=5, delay=5):
    for attempt in range(retries):
        try:
            return load_data(creds, statement_params)
        except OperationalError as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise

# Function to add producing months to prod_df
def calc_months_producing(group):
    min_date = group['Date'].min()
    group['MonthsProducing'] = group['Date'].map(lambda x: fcst.MonthDiff(min_date, x))
    return group

# Remove outliers from production data
def apply_bourdet_outliers(group, date_col, value_col):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x = group[date_col].values
        y = group[value_col].values
        y_new, x_new = fcst.bourdet_outliers(y, x, L=bourdet_params['smoothing_factor'], xlog=False, ylog=True, z_threshold=bourdet_params['z_threshold'], min_array_size=bourdet_params['min_array_size'])
        group = group[group[date_col].isin(x_new)]
        group[value_col] = y_new
    return group

# Function to create b_dict from b_factor_df
def create_b_dict(b_low, b_avg, b_high, min_b=0.5, max_b=1.4):
    # Handle nulls and enforce boundaries for b_low and b_high
    b_low = max(b_low if pd.notnull(b_low) else min_b, min_b)
    b_high = min(max(b_high if pd.notnull(b_high) else max_b, b_low * 1.1), max_b)
    
    # Ensure b_avg is between b_low and b_high, and adjust if it's null or out of bounds
    if pd.isnull(b_avg) or b_avg < b_low or b_avg > b_high:
        b_avg = (b_low + b_high) / 2  # Midpoint if b_avg is not usable
    
    # Prepare the final dictionary with rounded values to maintain relative differences
    return {
        'min': round(b_low, 4),
        'guess': round(b_avg, 4),
        'max': round(b_high, 4)
    }

# Alternative function that leverages Markov Chain Monte Carlo (MCMC) sampling for parameter estimation
def fit_arps_curve(
        property_id, 
        phase, 
        b_dict, 
        dei_dict, 
        def_dict, 
        min_q_dict, 
        prod_df_cleaned, 
        value_col, 
        method='curve_fit',
        use_advi=False, 
        trials=1000, 
        fit_segment='all', 
        smoothing_factor=smoothing_params['factor']
    ):
    # Function to add the terminal decline rate to the dei_dict
    def dict_coalesce(dei_dict, def_dict):
        return dei_dict.get('min', def_dict[phase])

    # Filter the dataframe to only include the rows for the property_id and phase being analyzed and filter out any rows with 0 or NaN values
    df = prod_df_cleaned[
        (prod_df_cleaned['WellID'] == property_id) & 
        (prod_df_cleaned[value_col] > 0) &
        (prod_df_cleaned['Measure'] == phase)
    ].sort_values(by='Date')

    # Identify the fit group for the property_id
    df['month_int'] = df['Date'].rank(method='dense', ascending=True)
    min_length = 12  # Minimum length of production data desired for fitting

    # First, check if the entire DataFrame meets the minimum length requirement
    if len(df) <= min_length:
        df_selected = df
    else:
        unique_segments = sorted(df['segment'].unique())
        df_selected = pd.DataFrame()  # Initialize an empty DataFrame for the selected data
        
        if fit_segment == 'first':
            segment_index = 0
            df_selected = df[df['segment'] == unique_segments[segment_index]]
            
            # Add data from the next segment until the minimum length is reached
            while len(df_selected) < min_length and segment_index + 1 < len(unique_segments):
                segment_index += 1
                next_segment_df = df[df['segment'] == unique_segments[segment_index]]
                df_selected = pd.concat([df_selected, next_segment_df])
            
        elif fit_segment == 'last':
            segment_index = len(unique_segments) - 1
            df_selected = df[df['segment'] == unique_segments[segment_index]]
            
            # Add data from the previous segment until the minimum length is reached
            while len(df_selected) < min_length and segment_index - 1 >= 0:
                segment_index -= 1
                prev_segment_df = df[df['segment'] == unique_segments[segment_index]]
                df_selected = pd.concat([prev_segment_df, df_selected])
        if len(df_selected) < min_length:
            df_selected = df

    # Create df and remove the first row from the dataframe for noise reduction
    df = df_selected.reset_index(drop=True).iloc[1:]

    # Prepare the data for fitting
    arr_length = len(df)
    t_act = df['Date'].rank(method='min', ascending=True).to_numpy()
    q_act = df[value_col].to_numpy()
    start_date = df['Date'].min()
    start_month = df['month_int'].min()
    Qi_guess = np.max(q_act, initial=0)
    Dei_init = dei_dict['guess']
    Dei_min = dict_coalesce(dei_dict, def_dict)
    Dei_max = dei_dict['max']
    b_guess = b_dict['guess']

    # Ensure bounds are valid
    b_min = min(b_dict['min'], b_dict['max'])
    b_max = max(b_dict['min'], b_dict['max'])

    def auto_fit1(method=method):
        bounds = ((Qi_guess*0.9, Dei_min, b_min), (Qi_guess, Dei_max, b_max))
        initial_guess = [Qi_guess, Dei_init, b_guess]
        config_optimize_qi_dei_b = {
            'optimize': ['Qi', 'Dei', 'b'],
            'fixed': {'Def': def_dict[phase]}
        }
        optimized_params = fcst.perform_curve_fit(t_act, q_act, initial_guess, bounds, config_optimize_qi_dei_b, method=method, trials=trials, use_advi=use_advi)
        qi_fit, Dei_fit, b_fit = optimized_params
        # Fitting the curve
        q_pred = fcst.varps_decline(1, 1, qi_fit, Dei_fit, def_dict[phase], b_fit, t_act, 0, 0)[3]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r_squared, rmse, mae = fcst.calc_goodness_of_fit(q_act, q_pred)

        return [
            property_id, phase, arr_length, 'auto_fit_1', fit_segment, start_date, start_month, 
            Qi_guess, qi_fit, Dei_fit, b_fit, r_squared, rmse, mae
        ]
    
    def auto_fit2(method=method):
        initial_guess = [Dei_init]
        bounds = ((Dei_min, Dei_max))
        config_optimize_dei = {
            'optimize': ['Dei'],
            'fixed': {'Qi': Qi_guess, 'b': b_guess, 'Def': def_dict[phase]}
        }
        optimized_params = fcst.perform_curve_fit(t_act, q_act, initial_guess, bounds, config_optimize_dei, method=method, trials=trials, use_advi=use_advi)
        Dei_fit = optimized_params[0]
        # Fitting the curve
        q_pred = fcst.varps_decline(1, 1, Qi_guess, Dei_fit, def_dict[phase], b_dict['guess'], t_act, 0, 0)[3]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r_squared, rmse, mae = fcst.calc_goodness_of_fit(q_act, q_pred)

        return [
            property_id, phase, arr_length, 'auto_fit_2', fit_segment, start_date, start_month, 
            Qi_guess, Qi_guess, Dei_fit, b_guess, r_squared, rmse, mae
        ]
    
    def auto_fit3():      
        return [
            property_id, phase, arr_length, 'auto_fit_3', fit_segment, start_date, start_month, 
            Qi_guess, Qi_guess, max(Dei_init, def_dict[phase]), b_guess, np.nan, np.nan, np.nan
        ]
    
    # Case to handle forecasts with less than 3 months of production
    if (Qi_guess < min_q_dict[phase]) | (arr_length < 3.0):
        result = auto_fit3()
    # Case to handle forecasts with more than 2 months and less than 7 months of production
    elif arr_length < 7.0:
        try:
            result = auto_fit2()
        except Exception as e:
            print(f"Failed auto_fit2 with error {e}, falling back to auto_fit3")
            result = auto_fit3()
    else:
        # Apply 3-month rolling average to q_act for smoothing smoothing_factor times
        q_act_series = pd.Series(q_act)
        
        if smoothing_factor > 0:
            for i in range(smoothing_factor):
                q_act_series = q_act_series.rolling(window=3, min_periods=1).mean()
        
        q_act = q_act_series.to_numpy()
        Qi_guess = np.max(q_act, initial=0)
        try:
            result = auto_fit1()
        except Exception as e1:
            try:
                print(f"Failed auto_fit1 with error {e1}, trying auto_fit2")
                result = auto_fit2()
            except Exception as e2:
                print(f"Failed auto_fit2 with error {e2}, falling back to auto_fit3")
                result = auto_fit3()
    
    return result

# Function to process production data
def auto_forecast(wellid, measure, last_prod_date, sql_creds_dict, value_col, bourdet_params, changepoint_params, b_estimate_params, dei_dict1, default_b_dict, method, use_advi, smoothing_factor):
    # Load production data
    prod_df = load_data_with_retry(sql_creds_dict, create_statement(wellid, measure, last_prod_date, fit_months=fit_months))

    # Check if prod_df is empty
    if prod_df.empty:
        return [wellid, measure, 0, 'no_data', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    
    # Add MonthsProducing column
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        prod_df = prod_df.groupby(['WellID', 'Measure']).apply(calc_months_producing)
        prod_df.reset_index(inplace=True, drop=True)
    
    # Apply Bourdet outliers
    BOURDET_OUTLIERS = bourdet_params['setting']
    if BOURDET_OUTLIERS:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            grouped = prod_df.groupby(['WellID', 'Measure'])
            prod_df_cleaned = grouped.apply(apply_bourdet_outliers, 'MonthsProducing', value_col)
            prod_df_cleaned.reset_index(inplace=True, drop=True)
    else:
        prod_df_cleaned = prod_df.copy()

    # Segment the time series data using changepoint detection
    DETECT_CHANGEPOINTS = changepoint_params['setting']
    if DETECT_CHANGEPOINTS:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cp_penalty = changepoint_params['penalty']
            prod_df_cleaned = prod_df_cleaned.groupby(['WellID']).apply(fcst.detect_changepoints, 'WellID', value_col, 'Date', cp_penalty)
            prod_df_cleaned.reset_index(inplace=True, drop=True)
    else:
        prod_df_cleaned['segment'] = 0

    # Estimate b factor
    ESTIMATE_B_FACTOR = b_estimate_params['setting']
    if ESTIMATE_B_FACTOR:
        results = fcst.b_factor_diagnostics(prod_df_cleaned, value_col, 'MonthsProducing')
        b_dict = create_b_dict(results['b_low'], results['b_avg'], results['b_high'])
    else:
        b_dict = default_b_dict[measure]

    # Fit Arps forecast to production data
    result = fit_arps_curve(wellid, measure, b_dict, dei_dict1, def_dict, min_q_dict, prod_df_cleaned, value_col, method, use_advi, trials, fit_segment, smoothing_factor)

    return result

# Function to apply the auto_forecast function to each row in the dataframe
def auto_forecast_partition(df, sql_creds_dict, value_col, bourdet_params, changepoint_params, b_estimate_params, dei_dict1, default_b_dict, method, use_advi, smoothing_factor): 
    # Apply auto_forecast_wrapper to each row in the dataframe
    results = df.apply(
        lambda row: pd.Series(auto_forecast(
            row['WellID'],
            row['Measure'],
            row['LastProdDate'],
            sql_creds_dict,
            value_col,
            bourdet_params,
            changepoint_params,
            b_estimate_params,
            dei_dict1,
            default_b_dict,
            method,
            use_advi,
            smoothing_factor
        )).tolist(), axis=1
    )
    return pd.DataFrame(results.values.tolist(), index=df.index, columns=param_df_cols)

def load_processed_chunks(folder_path):
    file_path = os.path.join(folder_path, 'processed_chunks.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=None)
        return set(df[0].astype(int))  # Convert to a set of integers
    else:
        return set()

def save_processed_chunk(chunk_index, folder_path):
    file_path = os.path.join(folder_path, 'processed_chunks.csv')
    # Append the chunk index to the CSV
    df = pd.DataFrame([chunk_index])
    df.to_csv(file_path, mode='a', header=False, index=False)

def main(client, folder_path, batch_size=batch_size, retries=5, delay=5):
    # Configure Lock
    sql_lock = Lock("sql_lock", client=client)

    # Track processed chunks to avoid reprocessing on restart
    processed_chunks = load_processed_chunks(folder_path)

    logging.info(f'Dask dashboard available at: {client.dashboard_link}')
    while True:
        # Create SQL statement dynamically based on the population type
        if fit_population == 'well_list':
            sql_statement = create_statement_wells(fit_population, manual_analyst, ta_offset_months, new_data_months, well_list=well_list)
        elif fit_population == 'fit_group':
            sql_statement = create_statement_wells(fit_population, manual_analyst, ta_offset_months, new_data_months, fit_group=fit_group)
        else:
            # Default to 'all' population
            sql_statement = create_statement_wells(fit_population, manual_analyst, ta_offset_months, new_data_months)

        # Load data into fcst_df and convert to dask dataframe
        fcst_df = load_data(sql_creds_dict, sql_statement)
        rows_fetched = len(fcst_df)
        logging.info(f'Fetched {rows_fetched} rows')

        if fcst_df.empty:
            logging.info('No data to process.')
            break

        # Calculate the number of splits based on batch_size
        num_splits = int(np.ceil(rows_fetched / batch_size))

        # Split fcst_df into smaller dataframes based on batch_size
        fcst_chunks = np.array_split(fcst_df, num_splits)
        delayed_tasks, chunk_indices = [], []
        for i, fcst_chunk in enumerate(fcst_chunks):
            if i in processed_chunks:
                logging.info(f"Skipping already-processed chunk {i+1}/{num_splits}")
                continue
            logging.info(f"Enqueuing chunk {i+1}/{num_splits} with {len(fcst_chunk)} rows")
            task = delayed(auto_forecast_partition)(
                fcst_chunk,
                sql_creds_dict,
                value_col,
                bourdet_params,
                changepoint_params,
                b_estimate_params,
                dei_dict1,
                default_b_dict,
                fit_method,
                use_advi,
                smoothing_params['factor']
            )
            delayed_tasks.append(task)
            chunk_indices.append(i)

        # Compute all tasks in parallel and stream results as they finish
        if delayed_tasks:
            # 1) submit
            futures = client.compute(delayed_tasks)
            fut2idx = {f: idx for f, idx in zip(futures, chunk_indices)}

            # 2) consume in arrival order
            for finished in as_completed(futures):
                i = fut2idx[finished]
                try:
                    res = finished.result()
                except Exception as e:
                    logging.error(f"Chunk {i+1} failed in Dask: {e}")
                    client.cancel(finished)
                    continue

                # 3) write to SQL with retries
                for attempt in range(retries):
                    try:
                        param_df = pd.DataFrame(res, columns=param_df_cols)
                        param_df['Units'] = param_df['Measure'].map({'OIL': 'BBL', 'GAS': 'MCF', 'WATER': 'BBL'})
                        param_df['Def'] = param_df['Measure'].map(def_dict)
                        param_df['Qabn'] = param_df['Measure'].map(min_q_dict)
                        param_df[['Q1','Q2','t1','t2']] = None
                        param_df['DateCreated'] = pd.to_datetime('today')
                        param_df.rename(columns={'fit_type': 'Analyst'}, inplace=True)
                        cols = ['WellID', 'Measure', 'Units', 'StartDate', 'Q1', 'Q2', 'Q3', 'Qabn', 'Dei', 'b_factor', 'Def', 't1', 't2', 'Analyst', 'DateCreated']
                        param_df = param_df[cols].where(pd.notnull(param_df), None)
                        with sql_lock:
                            sql.load_data_to_sql(param_df, sql_creds_dict, schema.forecast_stage)
                            sql.execute_stored_procedure(sql_creds_dict, 'sp_InsertFromStagingToForecast')
                            sql.execute_stored_procedure(sql_aries_creds_dict, 'sp_UpdateInsertFromForecastToACECONOMIC')

                        processed_chunks.add(i)
                        save_processed_chunk(i, folder_path)
                        logging.info(f"Chunk {i+1}/{num_splits} written successfully.")
                        break
                    except OperationalError as e:
                        logging.error(f"Chunk {i+1} write attempt {attempt+1} failed: {e}")
                        if attempt < retries - 1:
                            time.sleep(delay)
                    except Exception as e:
                        logging.error(f"Unexpected error writing chunk {i+1}: {e}")
                        break

                # 4) free memory on both driver and worker
                del res, param_df
                client.cancel(finished)

            logging.info("All chunks processed; exiting.")
            break

    # If all chunks are processed, delete the processed_chunks.csv file
    progress_file = os.path.join(folder_path, 'processed_chunks.csv')
    if os.path.exists(progress_file):
        os.remove(progress_file)
        logging.info(f"Deleted progress file: {progress_file}")

if __name__ == "__main__":
    # Check if the start method is already set
    if multiprocessing.get_start_method(allow_none=True) != 'fork':
        multiprocessing.set_start_method('fork')

    # Initialize LocalCluster and Client
    cluster = LocalCluster(
        n_workers=8, 
        threads_per_worker=2, 
        memory_limit='2GB',
        memory_target_fraction=0.6,
        memory_spill_fraction=0.7,
        dashboard_address='0.0.0.0:8787'
    )
    client = Client(cluster)
    logging.info(f"Dask dashboard running at {cluster.dashboard_link}")

    try:
        main(client, folder_path=log_folder)
        logging.info("Processing complete. Dashboard still available until you Ctrl-C.")
        time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        cluster.close()
        logging.info("Dask client and cluster shut down successfully.")
