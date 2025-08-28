import os, io, re, tempfile, psutil, ctypes
import numpy as np
import faulthandler, signal, sys, gc

# Set environment variables
flags = os.environ.get("PYTENSOR_FLAGS", "floatX=float64,optimizer_excluding=constant_folding")
flags = re.sub(r"(?:^|,)compiledir=[^,]*", "", flags)  # strip any existing compiledir
driver_compiledir = tempfile.mkdtemp(prefix=f"pytensor-driver-{os.getpid()}-")
os.environ["PYTENSOR_FLAGS"] = f"{flags},compiledir={driver_compiledir}".lstrip(",")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
faulthandler.enable()

def _trap(sig, frame):
    faulthandler.dump_traceback()
    sys.exit(255)
for _s in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_s, _trap)
    except Exception:
        pass

import atexit, shutil
atexit.register(lambda: shutil.rmtree(driver_compiledir, ignore_errors=True))

import pandas as pd
import multiprocessing
from dask.distributed import LocalCluster, Client, Lock, as_completed
from dask import delayed
from dask import config as dask_config
from config.config_loader import get_config
import AnalyticsAndDBScripts.sql_connect as sql
import AnalyticsAndDBScripts.sql_schemas as schema
import AnalyticsAndDBScripts.prod_fcst_functions as fcst
import warnings
import logging
import time
from typing import Optional
from sqlalchemy.exc import OperationalError
import hashlib

# Ignore warnings
warnings.filterwarnings(action='ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the config file
config_path = os.getenv("CONFIG_PATH")

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
method_params = next((item for item in params_list if item['name'] == 'method'), None) or {}

# Create distinct dictionaries for each database
sql_aries_creds_dict = sql_creds_dict.copy()
sql_aries_creds_dict['db_name'] = 'Analytics_Aries'
sql_creds_dict['db_name'] = 'Analytics'

# Helper function to ensure each well has a valid fit_method
def normalize_fit_method(m: str, default: str = 'curve_fit') -> str:
    """Return a valid fit method, falling back to default if NULL/NaN."""
    if m is None or (isinstance(m, float) and pd.isna(m)):
        return default
    m = str(m).strip().lower()
    return m if m in {'curve_fit','monte_carlo','differential_evolution'} else default

# Read method config
default_fit_method = normalize_fit_method(method_params.get('setting', 'curve_fit'), 'curve_fit')
general_params = method_params.get('general', {}) or {}
mc_config = method_params.get('monte_carlo', {}) or {}

# Define parameters
value_col = 'Value'
fit_segment = changepoint_params['fit_segment']
trials = general_params['trials']
use_advi = bool(mc_config['use_advi'])
save_trace = bool(mc_config['save_trace'])
fit_months = general_params['fit_months']
manual_analyst = general_params['manual_analyst']
ta_offset_months = general_params['ta_offset_mos']
new_data_months = general_params['new_data_mos']
log_folder = general_params['log_folder']
fit_population = general_params['fit_population']

# Initialize optional parameters for well_list and fit_group
well_list = general_params.get('well_list', [])
fit_group = general_params.get('fit_group', None)

# Load parameters from config file
def_dict = arps_params['terminal_decline']
dei_dict1 = arps_params['initial_decline']
min_q_dict = arps_params['abandonment_rate']
default_b_dict = arps_params['b_factor']

# Define columns for output dataframe
param_df_cols = [
    'WellID', 'Measure', 'fit_months', 'fit_type', 'fit_segment', 'StartDate', 
    'StartMonth', 'Q_guess', 'Q3', 'Dei', 'b_factor', 'R_squared', 'RMSE', 'MAE',
    'TraceBlob'
]

# Return freed pages to the OS (helps steady-state RSS).
def trim_memory():
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# Function to convert trace to bytes
def serialize_trace_to_bytes(trace, max_samples: int = 400) -> Optional[bytes]:
    """
    Serialize ONLY posterior draws for Qi, Dei, Def, b to a compressed NPZ bytes object.
    Returns None if unavailable. Keeps memory/storage tiny while allowing P10/P50/P90 later.
    """
    if trace is None:
        return None
    try:
        import arviz as az
        if not hasattr(trace, "posterior"):
            try:
                trace = az.from_pymc(trace)
            except Exception:
                return None
        post = trace.posterior
        def _flat(name):
            if name not in post:
                return None
            x = post[name].stack(sample=("chain","draw")).values
            return np.asarray(x, dtype=np.float64).ravel()
        qi  = _flat("Qi"); dei = _flat("Dei"); dff = _flat("Def"); b = _flat("b")
        if any(v is None for v in (qi, dei, dff, b)):
            return None
        # simple thinning to cap size
        n = len(qi)
        if n == 0:
            return None
        keep = min(max_samples, n)
        step = max(1, n // keep)
        sl = slice(None, None, step)
        buf = io.BytesIO()
        # store as float64 for numeric safety
        np.savez_compressed(buf, Qi=qi[sl], Dei=dei[sl], Def=dff[sl], b=b[sl])
        return buf.getvalue()
    except Exception as e:
        logging.warning(f"Param-draw serialization failed: {e}")
        return None

# Function to create sql query to get wells that need to be forecasted
def create_statement_wells(population, manual_analyst, ta_offset_mos=12, new_data_mos=3, well_list=None, fit_group=None):
    if well_list is None:
        well_list = []  
    if population == 'all':
        statement = '''
        SELECT		F.WellID, F.Measure, F.LastProdDate,
                    COALESCE(W.FitMethod, ?) AS FitMethod
        FROM		dbo.WELL_HEADER W
        INNER JOIN  dbo.vw_FORECAST F
        ON          F.WellID = W.WellID
        WHERE		F.CumulativeProduction > 0
        AND			(F.Analyst != ? OR F.Analyst IS NULL)
        AND			F.LastProdDate > DATEADD(month, -?, GETDATE())
        AND			(DATEDIFF(month, F.LastProdDate, F.DateCreated) >= ? OR F.DateCreated IS NULL)
        ORDER BY	F.WellID, F.PHASE_INT
        '''
        params = (default_fit_method, manual_analyst, ta_offset_mos, new_data_mos)
        return statement, params
    
    elif population == 'well_list':
        if not well_list:
            raise ValueError("well_list is empty, but population is set to 'well_list'.")
        sql_list = ', '.join(['?' for _ in well_list])
        statement = f'''
        SELECT		F.WellID, F.Measure, F.LastProdDate,
                    COALESCE(W.FitMethod, ?) AS FitMethod
        FROM		dbo.WELL_HEADER W
        INNER JOIN  dbo.vw_FORECAST F
        ON          F.WellID = W.WellID
        WHERE		F.CumulativeProduction > 0
        AND			(F.Analyst != ? OR F.Analyst IS NULL)
        AND			F.WellID IN ({sql_list})
        ORDER BY	F.WellID, F.PHASE_INT
        '''
        params = (default_fit_method, manual_analyst, *well_list)
        return statement, params
    
    elif population == 'fit_group':
        if not fit_group:
            raise ValueError("fit_group is not specified, but population is set to 'fit_group'.")
        statement = '''
        SELECT		F.WellID, F.Measure, F.LastProdDate,
                    COALESCE(W.FitMethod, ?) AS FitMethod
        FROM		dbo.WELL_HEADER W
        INNER JOIN  dbo.vw_FORECAST F
        ON          F.WellID = W.WellID
        WHERE		F.CumulativeProduction > 0
        AND			(F.Analyst != ? OR F.Analyst IS NULL)
        AND			W.FitGroup = ?
        ORDER BY	F.WellID, F.PHASE_INT
        '''
        params = (default_fit_method, manual_analyst, fit_group)
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
        y_new, x_new = fcst.bourdet_outliers(
            y, 
            x, 
            L=bourdet_params['smoothing_factor'], 
            xlog=False, 
            ylog=True, 
            z_threshold=bourdet_params['z_threshold'], 
            min_array_size=bourdet_params['min_array_size']
        )
        mask = group[date_col].isin(x_new)
        group = group.loc[mask].copy()
        group.loc[:, value_col] = y_new
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
        use_advi=use_advi, 
        trials=1000, 
        fit_segment='all', 
        smoothing_factor=smoothing_params['factor'],
        save_trace=save_trace
    ):
    """
    Fit an ARPS decline curve on a (property_id, phase) slice with
    robust pre-processing and multiple fitting strategies.

    Parameters
    ----------
    property_id : hashable
        Well/asset identifier to slice the input frame.
    phase : {'OIL','GAS','WATER'}
        Phase label expected in `prod_df_cleaned['Measure']`.
    b_dict : dict
        {'min': float, 'guess': float, 'max': float} bounds/guess for b-factor.
    dei_dict : dict
        {'guess': float, 'max': float, 'min': (optional)} for initial decline.
        The effective lower bound is max(dei_dict.get('min', Def), Def).
    def_dict : dict
        Terminal decline (Def) by phase, e.g. {'OIL': 0.08, ...}. Def is fixed.
    min_q_dict : dict
        Abandonment rate by phase; used for short-series fallback screening.
    prod_df_cleaned : pd.DataFrame
        Must contain columns: ['WellID','Measure','Date','MonthsProducing', value_col, 'segment'].
    value_col : str
        Name of the rate column to fit (e.g., 'OilRate').
    method : {'curve_fit','monte_carlo','differential_evolution'}, default 'curve_fit'
        Fitting backend. The 'monte_carlo' backend uses PyMC with NUTS/ADVI.
        If 'monte_carlo' is selected, an internal warm-start via SciPy is used and,
        when that warm-start yields Dei ≈ Def, sampling is skipped by design.
    use_advi : bool
        If True and method == 'monte_carlo', run ADVI before sampling.
    trials : int
        Iterations/draws parameter forwarded to the chosen backend.
    fit_segment : {'all','first','last'}
        Which contiguous segment(s) to use, with intelligent padding to reach a
        minimum length of 12 points when possible.
    smoothing_factor : int
        For non-MC methods only: number of 3-point moving-average passes applied
        after head-trim and before thinning.
    save_trace : bool
        If True and method == 'monte_carlo', return the PyMC/ArviZ trace (InferenceData)
        alongside the result list.

    Returns
    -------
    result : list
        [PropertyID, Phase, N_points_used, 'auto_fit_{1|2|3}', segment_choice, start_date, start_month,
         Qi_guess, Qi_fit, Dei_fit, b_fit, R2, RMSE, MAE]
        For auto_fit_3 (fallback), the goodness-of-fit metrics are NaN.
    trace : arviz.InferenceData, optional
        Only returned when save_trace=True and method=='monte_carlo'.
        Contains posterior draws for Qi, Dei, Def (fixed), b, and derived vars
        (e.g., 'log_q_model') as set up in the modeling function.

    Notes
    -----
    • Pre-processing: drops ≤2 head outliers via MAD test; thins tail (keep_head=36, stride=2).
    • Bounds: Dei_min is enforced as max(Def, dei_dict.get('min', Def)); b_guess is clipped to [b_min, b_max].
    • Early-exit: If the SciPy warm-start in 'monte_carlo' finds Dei ≈ Def, MCMC is skipped and
      the point estimate is returned quickly (trace will be None unless save_trace=True, in which
      case you'll receive None because no sampling occurred).
    • To compute P10/P50/P90 bands from a returned trace, stack ('chain','draw') and
      evaluate arps_q_np per draw, then take np.quantile over draws.
    """
    # Function to add the terminal decline rate to the dei_dict
    def dict_coalesce(dei_dict, def_dict):
        return max(float(dei_dict.get('min', def_dict[phase])), float(def_dict[phase]))

    # Filter the dataframe to only include the rows for the property_id and phase being analyzed and filter out any rows with 0 or NaN values
    df = prod_df_cleaned[
        (prod_df_cleaned['WellID'] == property_id) & 
        (prod_df_cleaned[value_col] > 0) &
        (prod_df_cleaned['Measure'] == phase)
    ].sort_values(by='Date')

    # Identify the fit group for the property_id
    df['month_int'] = df['MonthsProducing']
    min_length = 12  # Minimum length of production data desired for fitting

    # First, check if the entire DataFrame meets the minimum length requirement
    if len(df) <= min_length:
        df_selected = df
    else:
        unique_segments = sorted(df['segment'].unique())
        df_selected = pd.DataFrame()
        
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

    # --- Precompute values for a safe fallback to auto_fit_3 ---
    df0 = df_selected.reset_index(drop=True)
    arr_length0 = len(df0)
    q0 = np.clip(df0[value_col].to_numpy(), 1e-9, None)
    Qi_guess0 = float(np.max(q0)) if q0.size else 0.0
    start_date0 = df0['Date'].min() if not df0.empty else pd.NaT
    start_month0 = float(df0['MonthsProducing'].min()) if not df0.empty else np.nan
    Dei_init = float(dei_dict['guess'])
    b_guess = float(b_dict['guess'])

    # Noise reduction: drop first row only if we still have ≥2 points
    df = df0
    if len(df) > 1:
        df = df.iloc[1:]
    # If the segment is too short after dropping the first row, fallback to auto_fit_3
    if len(df) < 2:
        return [
            property_id, phase, arr_length0, 'auto_fit_3', fit_segment,
            start_date0,
            (int(start_month0) if not np.isnan(start_month0) else np.nan),
            Qi_guess0, Qi_guess0,
            max(Dei_init, def_dict[phase]), b_guess,
            np.nan, np.nan, np.nan
        ]

    # Prepare time and production arrays
    def thin_tail(t, q, keep_head=36, stride=2, keep_last=True):
        """Keep first `keep_head` points; after that, take every `stride`-th."""
        n = len(t)
        if n <= keep_head or stride <= 1:
            idx = np.arange(n, dtype=int)
        else:
            head = np.arange(keep_head, dtype=int)
            tail = keep_head + np.arange(0, n - keep_head, stride, int)
            idx = np.concatenate([head, tail])
            if keep_last and (n - 1) != idx[-1]:
                idx = np.append(idx, n - 1)
        return t[idx], q[idx]
    
    q_all = np.clip(df[value_col].to_numpy(), 1e-9, None)

    # --- Trim head outliers (high OR low) vs. local median (up to 2 points) ---
    def drop_head_outliers(t, q, k=4, z=3.0, max_drop=2):
        """
        Drop up to max_drop leading points while |q0 - median(head window)| > z * 1.4826 * MAD.
        """
        dropped = 0
        n = len(q)
        while (n - dropped) > (k + 1) and dropped < max_drop:
            w = q[dropped+1 : dropped+1+k]
            med = np.median(w)
            mad = np.median(np.abs(w - med)) + 1e-12
            if np.abs(q[dropped] - med) > z * 1.4826 * mad:
                dropped += 1
            else:
                break
        return t[dropped:], q[dropped:], dropped

    # Head-trim by MAD (drop at most 2 leading points); then drop the same rows in df
    # Use q_all for trimming; we don’t need time for that step
    t_dummy = np.arange(len(q_all), dtype=float)  # placeholder
    _, _, n_head_dropped = drop_head_outliers(t_dummy, q_all)
    if n_head_dropped:
        df = df.iloc[n_head_dropped:].reset_index(drop=True)

    # Build time axis, dense rank of Date, zero-based
    t_act = (df['MonthsProducing'] - df['MonthsProducing'].iloc[0]).to_numpy(dtype=float)
    q_act = np.clip(df[value_col].to_numpy(), 1e-9, None)

    # Start metadata from the first kept row (no assumptions about gaps)
    start_date = df['Date'].iloc[0]
    start_month = int(df['MonthsProducing'].iloc[0])

    # Smooth (but not for MC) after head trim
    arr_fit_length = len(t_act)
    if (method != 'monte_carlo') and (smoothing_factor > 0) and (arr_fit_length >= 3):
        s = pd.Series(q_act, copy=False)
        for _ in range(smoothing_factor):
            s = s.rolling(window=3, min_periods=1).mean()
        q_act = s.to_numpy(copy=False)
    t_act, q_act = thin_tail(t_act, q_act, keep_head=36, stride=2)
    Qi_guess = float(np.max(q_act, initial=0))
    Dei_min = float(dict_coalesce(dei_dict, def_dict))  # ensures Dei_min >= Def
    Dei_max = float(dei_dict['max'])
    if Dei_max <= Dei_min:
        Dei_max = Dei_min * 1.05
    Dei_init = float(np.clip(Dei_init, Dei_min, Dei_max))

    # Ensure bounds are valid
    b_min = float(min(b_dict['min'], b_dict['max']))
    b_max = float(max(b_dict['min'], b_dict['max']))
    if b_max <= b_min:  # just in case
        b_max = b_min * 1.1
    b_guess = float(np.clip(b_guess, b_min, b_max))

    def _run_fit(initial_guess, bounds, config, method):
        """
        Always return (params, success, trace). `trace` is None unless
        method == 'monte_carlo' and save_trace is True.
        """
        out = fcst.perform_curve_fit(
            t_act, 
            q_act, 
            initial_guess, 
            bounds, 
            config,
            method=method, 
            trials=trials, 
            use_advi=use_advi,
            include_deterministics=False,
            log_likelihood=False,
            return_trace=save_trace
        )
        if method == 'monte_carlo':
            params, success, _trace = out
            return np.array(params), success, (_trace if save_trace else None)
        else:
            params, success = out
            return np.array(params), success, None

    def auto_fit1(method=method):
        # Slightly tighter Qi bounds to stabilize on short series
        bounds = ((Qi_guess*0.8, Dei_min, b_min), (Qi_guess*1.1, Dei_max, b_max))
        initial_guess = [Qi_guess, Dei_init, b_guess]
        config_optimize_qi_dei_b = {
            'optimize': ['Qi', 'Dei', 'b'],
            'fixed': {'Def': def_dict[phase]}
        }
        optimized_params, ok, trace_local = _run_fit(initial_guess, bounds, config_optimize_qi_dei_b, method)
        if not ok:
            raise RuntimeError("auto_fit1 failed")
        qi_fit, Dei_fit, b_fit = optimized_params
        # Fitting the curve
        q_pred = fcst.varps_decline(1, 1, qi_fit, Dei_fit, def_dict[phase], b_fit, t_act, 0, 0)[3]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r_squared, rmse, mae = fcst.calc_goodness_of_fit(q_act, q_pred)

        return ([
            property_id, phase, arr_fit_length, 'auto_fit_1', fit_segment, start_date, start_month, 
            Qi_guess, qi_fit, Dei_fit, b_fit, r_squared, rmse, mae
        ], trace_local)

    def auto_fit2(method=method):
        initial_guess = [Dei_init]
        bounds = ((Dei_min,), (Dei_max,))
        config_optimize_dei = {
            'optimize': ['Dei'],
            'fixed': {'Qi': Qi_guess, 'b': b_guess, 'Def': def_dict[phase]}
        }
        optimized_params, ok, trace_local = _run_fit(initial_guess, bounds, config_optimize_dei, method)
        if not ok:
            raise RuntimeError("auto_fit2 failed")
        Dei_fit = float(optimized_params[0])
        # Fitting the curve
        q_pred = fcst.varps_decline(1, 1, Qi_guess, Dei_fit, def_dict[phase], b_guess, t_act, 0, 0)[3]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r_squared, rmse, mae = fcst.calc_goodness_of_fit(q_act, q_pred)

        return ([
            property_id, phase, arr_fit_length, 'auto_fit_2', fit_segment, start_date, start_month, 
            Qi_guess, Qi_guess, Dei_fit, b_guess, r_squared, rmse, mae
        ], trace_local)

    def auto_fit3():      
        return ([
            property_id, phase, arr_fit_length, 'auto_fit_3', fit_segment, start_date, start_month, 
            Qi_guess, Qi_guess, max(Dei_init, def_dict[phase]), b_guess, np.nan, np.nan, np.nan
        ], None)
    
    # Case to handle forecasts with less than 3 months of production
    if (Qi_guess < min_q_dict[phase]) or (arr_fit_length < 3.0):
        result, trace = auto_fit3()
    # Case to handle forecasts with more than 2 months and less than 7 months of production
    elif arr_fit_length < 7.0:
        try:
            result, trace = auto_fit2()
        except Exception as e:
            print(f"Failed auto_fit2 with error {e}, falling back to auto_fit3")
            result, trace = auto_fit3()
    else:
        try:
            result, trace = auto_fit1()
        except Exception as e1:
            try:
                print(f"Failed auto_fit1 with error {e1}, trying auto_fit2")
                result, trace = auto_fit2()
            except Exception as e2:
                print(f"Failed auto_fit2 with error {e2}, falling back to auto_fit3")
                result, trace = auto_fit3()
    # If requested, return the trace alongside the result list (only MC produces one)
    if save_trace and method == 'monte_carlo':
        return result, trace
    return result

# Function to process production data
def auto_forecast(
        wellid, 
        measure, 
        last_prod_date, 
        sql_creds_dict, 
        value_col, 
        bourdet_params, 
        changepoint_params, 
        b_estimate_params, 
        dei_dict1, 
        default_b_dict, 
        method, 
        use_advi, 
        smoothing_factor,
        save_trace
    ):
    # Load production data
    prod_df = load_data_with_retry(sql_creds_dict, create_statement(wellid, measure, last_prod_date, fit_months=fit_months))

    # Check if prod_df is empty
    if prod_df.empty:
        return [wellid, measure, 0, 'no_data', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None]
    
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
        try:
            results = fcst.b_factor_diagnostics(prod_df_cleaned, value_col, 'MonthsProducing')
        except Exception:
            results = None
        b_dict = create_b_dict(results['b_low'], results['b_avg'], results['b_high']) if results else default_b_dict[measure]
    else:
        b_dict = default_b_dict[measure]

    # Fit Arps forecast to production data
    fit_out = fit_arps_curve(
        wellid, 
        measure, 
        b_dict, 
        dei_dict1, 
        def_dict, 
        min_q_dict, 
        prod_df_cleaned, 
        value_col, 
        method, 
        use_advi, 
        trials, 
        fit_segment, 
        smoothing_factor,
        save_trace=save_trace
    )

    # When saving trace (MC only), serialize InferenceData ->, store compact param draws NPZ.
    if save_trace and method == 'monte_carlo':
        result, trace = fit_out
        draws_saved = int(mc_config.get('draws_saved', 400) or 400)
        trace_bytes = serialize_trace_to_bytes(trace, max_samples=draws_saved)
        del trace
        gc.collect()
        return result + [trace_bytes]
    return fit_out + [None]

# Function to apply the auto_forecast function to each row in the dataframe
def auto_forecast_partition(
        df, 
        sql_creds_dict, 
        value_col, 
        bourdet_params, 
        changepoint_params, 
        b_estimate_params, 
        dei_dict1, 
        default_b_dict, 
        default_method, 
        use_advi, 
        smoothing_factor,
        save_trace
    ): 
    def method_for(row):
        return normalize_fit_method(row.get('FitMethod'), default_method)

    # Apply auto_forecast_wrapper to each row in the dataframe
    results = df.apply(
        lambda row: auto_forecast(
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
            method_for(row), 
            use_advi, 
            smoothing_factor, 
            save_trace
        ),
        axis=1
    )
    out = pd.DataFrame(list(results), index=df.index, columns=param_df_cols)

    gc.collect(); trim_memory()
    return out

def chunk_fingerprint(df: pd.DataFrame) -> str:
    # Use stable ordering and a minimal identity (WellID, Measure)
    keys = tuple(zip(df['WellID'].tolist(), df['Measure'].tolist()))
    h = hashlib.sha1(repr(keys).encode('utf-8')).hexdigest()[:16]
    return h

def load_processed_chunks(folder_path):
    file_path = os.path.join(folder_path, 'processed_chunks.csv')
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=None, dtype=str)
        return set(df[0].astype(str))  # set of fingerprints
    else:
        return set()

def save_processed_chunk(chunk_id: str, folder_path):
    file_path = os.path.join(folder_path, 'processed_chunks.csv')
    df = pd.DataFrame([chunk_id])
    df.to_csv(file_path, mode='a', header=False, index=False)

def main(client, folder_path, retries=5, delay=5, reset_every=6, mem_threshold_gb=9.0):
    # Configure Lock
    sql_lock = Lock("sql_lock", client=client)

    processed_since_restart = 0
    wrote_any = False

    # Simple memory helpers
    def rss_gb():
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)
    def driver_trim():
        gc.collect()
        # glibc trim if available (Linux)
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    def worker_trim():
        """Run on workers to aggressively return memory to OS."""
        import gc, ctypes
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    def hard_restart(reason: str):
        nonlocal sql_lock
        # clear any local caches you hold (e.g., PyMC model cache)
        try:
            fcst._MODEL_CACHE.clear()
            gc.collect()
        except Exception:
            pass
        driver_trim()
        client.run(worker_trim)
        logging.info(f"Restarting Dask cluster to shed memory ({reason})...")
        client.restart()
        sql_lock = Lock("sql_lock", client=client)
        # Optional: re-register the worker plugin after restart
        try:
            client.register_worker_plugin(PytensorCompiledir(), name="pytensor-compiledir", replace=True)
        except Exception:
            pass
        driver_trim()
        logging.info("Restart complete.")

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

        # Ensure FitMethod exists and has a default
        if 'FitMethod' not in fcst_df.columns:
            fcst_df['FitMethod'] = default_fit_method
        fcst_df['FitMethod'] = fcst_df['FitMethod'].fillna(default_fit_method)

        # Split into MC vs other, then chunk each with its own batch size
        mc_df = fcst_df[fcst_df['FitMethod'] == 'monte_carlo']
        other_df = fcst_df[fcst_df['FitMethod'] != 'monte_carlo']

        def split_by_size(df, bs):
            if df.empty:
                return []
            n = int(np.ceil(len(df) / bs))
            return np.array_split(df, n)

        mc_chunks = split_by_size(mc_df, 50) # monte_carlo fits in batches of 50
        other_chunks = split_by_size(other_df, 50000) # other fits in batches of 50000

        # Final list of chunks to enqueue
        fcst_chunks = mc_chunks + other_chunks
        total_chunks = len(fcst_chunks)

        delayed_tasks, chunk_indices = [], []
        skipped = 0
        for i, fcst_chunk in enumerate(fcst_chunks):
            chunk_id = chunk_fingerprint(fcst_chunk)
            if chunk_id in processed_chunks:
                skipped += 1
                continue
            logging.info(f"Enqueuing chunk {i+1}/{total_chunks} with {len(fcst_chunk)} rows")
            task = delayed(auto_forecast_partition)(
                fcst_chunk,
                sql_creds_dict,
                value_col,
                bourdet_params,
                changepoint_params,
                b_estimate_params,
                dei_dict1,
                default_b_dict,
                default_fit_method,
                use_advi,
                smoothing_params['factor'],
                save_trace
            )
            delayed_tasks.append(task)
            chunk_indices.append((i, chunk_id))

        if skipped:
            logging.info(f"Skipped {skipped}/{total_chunks} already-processed chunk(s).")

        # If there is nothing left to run in this batch, exit the outer loop.
        if not delayed_tasks:
            logging.info("All chunks in this batch are already processed. Exiting.")
            break

        # Compute all tasks in parallel and stream results as they finish
        if delayed_tasks:
            # 1) submit
            futures = client.compute(delayed_tasks)
            fut2idx = {f: idx for f, idx in zip(futures, chunk_indices)}

            # 2) consume in arrival order
            need_restart = False
            for finished in as_completed(futures):
                i, chunk_id = fut2idx[finished]
                try:
                    res = finished.result()
                except Exception as e:
                    logging.error(f"Chunk {i+1} failed in Dask: {e}")
                    finished.release()
                    continue

                # 3) write to SQL with retries
                for attempt in range(retries):
                    try:
                        param_df = res
                        param_df['Units'] = param_df['Measure'].map({'OIL': 'BBL', 'GAS': 'MCF', 'WATER': 'BBL'})
                        param_df['Def'] = param_df['Measure'].map(def_dict)
                        param_df['Qabn'] = param_df['Measure'].map(min_q_dict)
                        param_df[['Q1','Q2','t1','t2']] = None
                        param_df['DateCreated'] = pd.to_datetime('today')
                        param_df.rename(columns={'fit_type': 'Analyst'}, inplace=True)
                        cols = [
                            'WellID', 
                            'Measure', 
                            'Units', 
                            'StartDate', 
                            'Q1', 
                            'Q2', 
                            'Q3', 
                            'Qabn', 
                            'Dei', 
                            'b_factor', 
                            'Def', 
                            't1', 
                            't2', 
                            'Analyst', 
                            'DateCreated', 
                            'TraceBlob'
                        ]
                        param_df = param_df[cols].where(pd.notnull(param_df), None)
                        # Ensure blob column is bytes/object dtype for SQLAlchemy
                        if 'TraceBlob' in param_df.columns:
                            param_df['TraceBlob'] = param_df['TraceBlob'].astype(object)
                        with sql_lock:
                            sql.load_data_to_sql(param_df, sql_creds_dict, schema.forecast_stage)
                            sql.execute_stored_procedure(sql_creds_dict, 'sp_InsertFromStagingToForecast')

                        processed_chunks.add(chunk_id)
                        save_processed_chunk(chunk_id, folder_path)
                        logging.info(f"Chunk {i+1}/{total_chunks} written successfully.")
                        wrote_any = True
                        break
                    except OperationalError as e:
                        logging.error(f"Chunk {i+1} write attempt {attempt+1} failed: {e}")
                        if attempt < retries - 1:
                            time.sleep(delay)
                    except Exception as e:
                        logging.error(f"Unexpected error writing chunk {i+1}: {e}")
                        break

                # 4) free memory on both driver and worker
                try:
                    del res
                    if 'param_df' in locals():
                        del param_df
                except Exception:
                    pass
                finished.release()
                gc.collect()
                driver_trim()
                client.run(worker_trim)

                # track progress & check memory each finished chunk
                processed_since_restart += 1
                cur_rss = rss_gb()
                if (processed_since_restart >= reset_every) or (cur_rss > mem_threshold_gb):
                    need_restart = True
                    logging.info(
                        f"Marking batch for restart: processed_since_restart={processed_since_restart}, "
                        f"driver_rss={cur_rss:.2f} GB (threshold={mem_threshold_gb} GB)"
                    )

            try:
                fcst._MODEL_CACHE.clear()
                gc.collect()
            except Exception:
                pass

            # all futures from this batch are completed now
            del futures, fut2idx
            gc.collect()
            driver_trim()
            client.run(worker_trim)

            if need_restart:
                before = rss_gb()
                hard_restart(
                    f"processed_since_restart={processed_since_restart}, "
                    f"driver_rss(before)={before:.2f} GB"
                )
                after = rss_gb()
                logging.info(f"Driver RSS after restart: {after:.2f} GB")
                processed_since_restart = 0

            logging.info("Batch complete.")
            # continue outer while-loop to pick up next SQL batch
            continue

    # Run ACECONOMIC update ONCE at the end (after all batches)
    if wrote_any:
        logging.info("Running sp_UpdateInsertFromForecastToACECONOMIC...")
        for attempt in range(retries):
            try:
                with sql_lock:
                    sql.execute_stored_procedure(sql_aries_creds_dict, 'sp_UpdateInsertFromForecastToACECONOMIC')
                logging.info("ACECONOMIC update completed.")
                break
            except OperationalError as e:
                logging.error(f"Final ACECONOMIC update attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
            except Exception as e:
                logging.error(f"Unexpected error in final ACECONOMIC update: {e}")
                break

    # If all chunks are processed, delete the processed_chunks.csv file
    progress_file = os.path.join(folder_path, 'processed_chunks.csv')
    if os.path.exists(progress_file):
        os.remove(progress_file)
        logging.info(f"Deleted progress file: {progress_file}")

if __name__ == "__main__":
    # Check if the start method is already set
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    # Configure Dask worker memory behavior (target/spill) via config
    dask_config.set({
    "distributed.worker.memory.target": 0.65,
    "distributed.worker.memory.spill": 0.75,
    "distributed.worker.memory.pause": 0.85,
    "distributed.worker.memory.terminate": 0.95,
    })

    # Initialize LocalCluster and Client
    cluster = LocalCluster(
        n_workers=10, 
        threads_per_worker=1, 
        memory_limit='2GB',
        local_directory=f"/tmp/dask-{os.getpid()}",
        dashboard_address='0.0.0.0:8787',
        processes=True
    )
    client = Client(cluster)

    # Give every worker its own PyTensor compiledir
    from dask.distributed import WorkerPlugin
    class PytensorCompiledir(WorkerPlugin):
        def setup(self, worker):
            import os, re, tempfile, faulthandler, signal, pathlib
            faulthandler.enable()
            # Log traces into the worker’s local directory
            _wd = pathlib.Path(getattr(worker, "local_directory", "/tmp"))
            _wd.mkdir(parents=True, exist_ok=True)
            _log = open(str(_wd / f'faulthandler-worker-{os.getpid()}.log'), 'a', buffering=1)
            faulthandler.register(signal.SIGTERM, file=_log, all_threads=True, chain=True)
            faulthandler.register(signal.SIGINT,  file=_log, all_threads=True, chain=True)
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
            os.environ.setdefault("PYTHONFAULTHANDLER", "1")
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")
            os.environ.setdefault("MALLOC_ARENA_MAX", "2")
            flags = os.environ.get("PYTENSOR_FLAGS", "floatX=float64,optimizer_excluding=constant_folding")
            flags = re.sub(r"(?:^|,)compiledir=[^,]*", "", flags)
            self.worker_compiledir = tempfile.mkdtemp(prefix=f"pytensor-w{os.getpid()}-")
            os.environ["PYTENSOR_FLAGS"] = f"{flags},compiledir={self.worker_compiledir}".lstrip(",")

        def teardown(self, worker):
            import shutil
            shutil.rmtree(self.worker_compiledir, ignore_errors=True)
    client.register_worker_plugin(PytensorCompiledir(), name="pytensor-compiledir")
    logging.info(f"Dask dashboard running at {cluster.dashboard_link}")

    try:
        main(client, folder_path=log_folder)
        logging.info("Processing complete.")
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        cluster.close()
        logging.info("Dask client and cluster shut down successfully.")
