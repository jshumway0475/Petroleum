import numpy as np
import ruptures as rpt
from petbox import dca
import statsmodels.api as sm
import scipy.stats
import pytensor.tensor as pt
import pymc as pm
from scipy.optimize import curve_fit, differential_evolution
from functools import partial
import numbers
from typing import Tuple, Dict
import gc


# Create MonthDiff function to calculate time difference in months
def MonthDiff(BaseDate, StartDate):
    '''
    Args:
    - BaseDate is the date from which the time difference is calculated
    - StartDate is the date for which the time difference is calculated
    Returns:
    - An integer representing the number of months between the two dates
    '''
    BaseDate = np.datetime64(BaseDate, 'M')
    StartDate = np.datetime64(StartDate, 'M')
    MonthDiff = int(((StartDate - (BaseDate - np.timedelta64(1, 'M'))) / np.timedelta64(1, 'M')) - 1)
    return MonthDiff


# Function to calculate forecasted volumes using Arps decline equations
def arps_decline(UID, phase, Qi, Dei, Def, b, t, prior_cum=0, prior_t=0):
    '''
    Args:
    - UID is a unique identifier for the well such as API, must be a number
    - phase is 1 = oil, 2 = gas, or 3 = water
    - Qi is the initial production rate typically in bbl/day or Mcf/day
    - Dei is the initial effective annual decline rate
    - Def is the final effective annual decline rate at which point the decline becomes exponential
    - b is the b-factor used in hyperbolic or harmonic decline equations
    - t is the time as a month integer
    - prior_cum is the cumulative amount produced before the start of the decline calcuations
    - prior_t is an integer representing the final month from a previous decline segment
    Returns:
    - A numpy array containing the following columns:
        - UID: Unique identifier for the well such as API
        - phase: 1 = oil, 2 = gas, or 3 = water
        - t: Time in months
        - q: Production rate
        - De_t: Effective annual decline rate
        - Np: Cumulative production
    '''
    # Calculations to determine decline type
    if np.isclose(Dei, Def):
        Type = 'exp'
    elif Dei > Def and b == 1:
        Type = 'har'
        Dn = Dei / (1 - Dei)
        Qlim = Qi * ((-np.log(1 - Def)) / Dn)
        tlim = (((Qi / Qlim) - 1) / Dn) * 12  # output in months
    else:
        Type = 'hyp'
        Dn = (1 / b) * (((1 - Dei) ** -b) - 1)
        Qlim = Qi * ((-np.log(1 - Def)) / Dn) ** (1 / b)
        tlim = ((((Qi / Qlim) ** b) - 1) / (b * Dn)) * 12  # output in months

    # Generate volumes
    if Type == 'hyp':
        Dn_t = Dn / (1 + b * Dn * (t / 12))
        De_t = 1 - (1 / ((Dn_t * b) + 1)) ** (1 / b)
        if De_t > Def:
            q = Qi * (1 + b * Dn * (t / 12)) ** (-1 / b)
            Np = ((Qi ** b) / (Dn * (1 - b))) * ((Qi ** (1 - b)) - (q ** (1 - b))) * 365
        else:
            q = Qlim * np.exp(-(-np.log(1 - Def)) * ((t - tlim) / 12))
            Np = ((Qlim - q) / (-np.log(1 - Def)) * 365) + (((Qi ** b) / (Dn * (1 - b))) * ((Qi ** (1 - b)) - (Qlim ** (1 - b))) * 365)  # noqa
            De_t = Def
    elif Type == 'har':
        Dn_t = Dn / (1 + Dn * (t / 12))
        De_t = 1 - (1 / (Dn_t + 1))
        if De_t > Def:
            q = Qi / (1 + b * Dn * (t / 12))
            Np = (Qi / Dn) * np.log(Qi / q) * 365
        else:
            q = Qlim * np.exp(-(-np.log(1 - Def)) * ((t - tlim) / 12))
            Np = ((Qlim - q) / (-np.log(1 - Def)) * 365) + ((Qi / Dn) * np.log(Qi / Qlim) * 365)
            De_t = Def
    else:
        q = Qi * np.exp(-(-np.log(1 - Dei)) * (t / 12))
        Np = (Qi - q) / (-np.log(1 - Dei)) * 365
        De_t = Dei

    return UID, phase, t + prior_t, q, De_t, Np + prior_cum


# Vectorize the arps_decline function to allow it to work with numpy arrays
varps_decline = np.vectorize(arps_decline, otypes=[float, float, float, float, float, float])


# Smooth hyperbolic→exp tail in PyTensor (months in, rate out)
def arps_q_pt(t_mo, Qi, Dei, Def, b, exp_tol=1e-10):
    eps = 1e-8
    b_ = pt.maximum(b, eps)

    # time in days
    t_d = t_mo * (365.0 / 12.0)

    # Effective annual -> nominal instantaneous (per day)
    Di0_day = ((pt.power(1.0 - Dei, -b_) - 1.0) / b_) / 365.0
    Dmin_day = -pt.log(1.0 - Def) / 365.0

    # Pure exponential path when Dei ~= Def
    is_exp = pt.lt(pt.abs(Dei - Def), exp_tol)
    q_exp_pure = Qi * pt.exp(-Dmin_day * t_d)

    # Transition time (clamped for stability)
    Di0_day_c = pt.clip(Di0_day, 1e-12, np.inf)
    Dmin_day_c = pt.clip(Dmin_day, 1e-12, np.inf)
    tx = pt.maximum((1.0 / Dmin_day_c - 1.0 / Di0_day_c) / b_, 0.0)

    # Piecewise hyperbolic -> exponential tail
    q_hyp = Qi * pt.power(1.0 + b_ * Di0_day_c * pt.minimum(t_d, tx), -1.0 / b_)
    qx = Qi * pt.power(1.0 + b_ * Di0_day_c * tx, -1.0 / b_)
    q_exp_tail = qx * pt.exp(-Dmin_day_c * (t_d - tx))
    q_piecewise = pt.where(t_d <= tx, q_hyp, q_exp_tail)

    # Choose pure exp when Dei ~= Def, else piecewise
    return pt.switch(is_exp, q_exp_pure, q_piecewise)


# numpy twin of arps_q_pt
def arps_q_np(t_mo, Qi, Dei, Def, b):
    if np.isclose(Dei, Def):
        D_day = -np.log(1.0 - Dei) / 365.0
        t_d = np.asarray(t_mo, float) * (365.0 / 12.0)
        return Qi * np.exp(-D_day * t_d)
    eps = 1e-8
    b_ = np.maximum(b, eps)

    Di0_day = ((np.power(1.0 - Dei, -b_) - 1.0) / b_) / 365.0
    Dmin_day = -np.log(1.0 - Def) / 365.0

    t_d = np.asarray(t_mo, float) * (365.0 / 12.0)
    tx = np.maximum((1.0 / Dmin_day - 1.0 / Di0_day) / b_, 0.0)

    q_hyp = Qi * np.power(1.0 + b_ * Di0_day * np.minimum(t_d, tx), -1.0 / b_)
    qx = Qi * np.power(1.0 + b_ * Di0_day * tx, -1.0 / b_)
    q_exp = qx * np.exp(-Dmin_day * (t_d - tx))
    return np.where(t_d <= tx, q_hyp, q_exp)


# Fast warm-start helper for deterministic fits (tiny coarse grid)
def _coarse_grid_best(t, q, Qi_guess, Dei_low, Dei_guess, Dei_high, b_low, b_guess, b_high, Def):
    """
    Very small 3x3 grid around guesses to get a robust seed for bounded curve_fit.
    Returns [Qi0, Dei0, b0].
    """
    Qi_grid = np.array([Qi_guess * 0.85, Qi_guess, Qi_guess * 1.15], float)
    Dei_grid = np.array([Dei_low, Dei_guess, Dei_high], float)
    b_grid = np.array([max(0.05, b_low), b_guess, b_high], float)

    def q_model(tt, Qi, Dei, b):
        return arps_q_np(tt, Qi, Dei, Def, b)

    best = None
    for Qi in Qi_grid:
        for Dei in Dei_grid:
            for b in b_grid:
                pred = q_model(t, Qi, Dei, b)
                sse = float(np.sum((q - pred)**2))
                if (best is None) or (sse < best[0]):
                    best = (sse, Qi, Dei, b)
    _, Qi0, Dei0, b0 = best
    return [Qi0, Dei0, b0]


# Function to recalculate Arps decline parameters based on a future date
def arps_roll_forward(BaseDate, StartDate, UID, phase, Qi, Dei, Def, b):
    t = max(MonthDiff(BaseDate, StartDate), 0)
    return varps_decline(UID, phase, Qi, Dei, Def, b, t)


# Calculate Dei from Qi and Qf based on exponential decline equation
def exp_Dei(Qi, Qf, duration):
    '''
    Args:
    - Qi is the initial production rate typically in bbl/day or Mcf/day
    - Qf is the final production rate typically in bbl/day or Mcf/day
    - duration is the time interval in months over which you are trying to calculate the exponential decline rate
    Returns:
    - Dei: The initial effective annual decline rate
    '''
    Dei = 1 - np.exp(-np.log(Qi / Qf) / (duration / 12))
    return Dei


# Function to manage multiple segments
def arps_segments(UID, phase, Q1, Q2, Q3, Dei, Def, b, Qabn, t1, t2, duration, prior_cum=0, prior_t=0):  # noqa
    '''
    Args:
    - UID is a unique identifier for the well such as API, must be a number
    - phase is 1 = oil, 2 = gas, or 3 = water
    - Q1 is the initial production rate typically in bbl/day or Mcf/day
    - Q2 is the production rate at the end of the first segment
    - Q3 is the production rate at the end of the second segment
    - Dei is the initial effective annual decline rate
    - Def is the final effective annual decline rate at which point the decline becomes exponential
    - b is the b-factor used in hyperbolic or harmonic decline equations
    - Qabn is the minimum production rate to be included in the forecast
    - t1 is the duration of the first segment in months
    - t2 is the duration of the second segment in months
    - duration is the total duration of the forecast in months
    - prior_cum is the cumulative amount produced before the start of the decline calcuations
    - prior_t is an integer representing the final month from a previous decline segment

    Segment 1 is the initial incline period and uses Arps exponential equation
    Segment 2 is the period between the incline and decline periods and uses Arps exponential equation
    Segment 3 is the decline period

    Returns:
    - A numpy array containing the following values:
        - UID: Unique identifier for the well such as API
        - phase: 1 = oil, 2 = gas, or 3 = water
        - t: Time in months
        - q: Production rate
        - De_t: Effective annual decline rate
        - Np: Cumulative production
        - Monthly volume: Monthly production
    '''
    # Adjust duration if needed
    duration = duration - prior_t

    # Determine valid segment count
    if t1 > 0 and t2 > 0:
        segment_ct = 3
        if Q2 == Q3:
            Q2 = Q2 * 1.0001
    elif t1 > 0:
        segment_ct = 2
    elif Q3 == 0 or np.isnan(Q3):
        segment_ct = 0
    else:
        segment_ct = 1

    # 3 segment logic
    if segment_ct == 3:
        t_seg1 = np.arange(0, t1 + 1, 1)
        t_seg2 = np.arange(1, t2 + 1, 1)
        t_seg3 = np.arange(1, duration - t1 - t2 + 1, 1)
        Dei1 = exp_Dei(Q1, Q2, t1)
        Dei2 = exp_Dei(Q2, Q3, t2)
        seg1 = varps_decline(UID, phase, Q1, Dei1, Dei1, 1.0, t_seg1, prior_cum, prior_t)
        seg1_arr = np.array(seg1)
        prior_cum1 = np.max(seg1_arr[5])
        seg2 = varps_decline(UID, phase, Q2, Dei2, Dei2, 1.0, t_seg2, prior_cum1, t1)
        seg2_arr = np.array(seg2)
        prior_cum2 = np.max(seg2_arr[5])
        seg3 = varps_decline(UID, phase, Q3, Dei, Def, b, t_seg3, prior_cum2, t1 + t2)
        seg3_arr = np.array(seg3)
        # Filter out months where production is less than Qabn
        if Qabn > 0:
            Qabn_filter = seg3_arr[3] >= Qabn
            seg3_arr = seg3_arr[:, Qabn_filter]
        out_nparr = np.column_stack((seg1_arr, seg2_arr, seg3_arr))
    elif segment_ct == 2:
        t_seg1 = np.arange(0, t1 + 1, 1)
        t_seg3 = np.arange(1, duration - t1 + 1, 1)
        Dei1 = exp_Dei(Q1, Q3, t1)
        seg1 = varps_decline(UID, phase, Q1, Dei1, Dei1, 1.0, t_seg1, prior_cum, prior_t)
        seg1_arr = np.array(seg1)
        prior_cum1 = np.max(seg1_arr[5])
        seg3 = varps_decline(UID, phase, Q3, Dei, Def, b, t_seg3, prior_cum1, t1)
        seg3_arr = np.array(seg3)
        # Filter out months where production is less than Qabn
        if Qabn > 0:
            Qabn_filter = seg3_arr[3] >= Qabn
            seg3_arr = seg3_arr[:, Qabn_filter]
        out_nparr = np.column_stack((seg1_arr, seg3_arr))
    elif segment_ct == 1:
        t_seg3 = np.arange(0, duration + 1, 1)
        out_nparr = varps_decline(UID, phase, Q3, Dei, Def, b, t_seg3, prior_cum, prior_t)
        out_nparr = np.array(out_nparr)
        # Filter out months where production is less than Qabn
        if Qabn > 0:
            Qabn_filter = out_nparr[3] >= Qabn
            out_nparr = out_nparr[:, Qabn_filter]
    else:
        t = np.arange(0, duration + 1, 1)
        UID_arr = np.full_like(t, UID, dtype=float)
        phase_arr = np.full_like(t, phase, dtype=float)
        q = np.zeros_like(t, dtype=float)
        De = np.zeros_like(t, dtype=float)
        Np = np.zeros_like(t, dtype=float)
        out_nparr = np.vstack([UID_arr, phase_arr, t, q, De, Np])

    # Add monthly volumes to array
    Cum_i = out_nparr[5][:-1]
    Cum_f = out_nparr[5][1:]
    cum = Cum_f - Cum_i
    cum[cum < 0] = 0
    cum = np.insert(cum, 0, 0)
    out_nparr = np.vstack((out_nparr, cum))[:, 1:]

    return out_nparr


# Detect change points in production data
def detect_changepoints(df, id_col, prod_col, date_col, pen):
    '''
    Detects change points in time-series production data using the ruptures library.
    Args:
    - df (DataFrame): Pandas DataFrame containing the production data.
    - id_col (str): Column name identifying each unique property or well.
    - prod_col (str): Column name containing the production data values.
    - date_col (str): Column name containing the date information.
    - pen (float): Penalty value influencing the sensitivity of change point detection.
                   Higher values lead to fewer detected change points.
    Returns:
    - DataFrame: Modified DataFrame with an additional column 'segment' indicating the
                 segment number for each data point.
    Example Use:
    This function should be applied to partitions of the DataFrame. For example:
    - prod_df_oil = prod_df[prod_df['Measure'] == 'OIL'].groupby(['WellID', 'FitGroup']).apply(detect_changepoints, 'WellID', 'Value', 'Date', 10)
    - prod_df_gas = prod_df[prod_df['Measure'] == 'GAS'].groupby(['WellID', 'FitGroup']).apply(detect_changepoints, 'WellID', 'Value', 'Date', 10)
    - prod_df_wtr = prod_df[prod_df['Measure'] == 'WATER'].groupby(['WellID', 'FitGroup']).apply(detect_changepoints, 'WellID', 'Value', 'Date', 10)

    Note: The function sorts the data by the date column and assigns segment numbers based on detected change points
          Segments are determined by significant changes in the production data trend.
    '''  # noqa
    # Create a new column for the segment numbers
    df = df.copy()
    df['segment'] = 0

    # Get the list of unique properties
    properties = df[id_col].unique()

    for property in properties:
        # Filter the dataframe for the current property
        property_df = df[df[id_col] == property]

        # Sort by date
        property_df = property_df.sort_values(date_col)

        # Extract the prod_col values as a numpy array
        signal = property_df[prod_col].values

        # Perform the change point detection
        try:
            algo = rpt.Pelt(model="rbf", min_size=6).fit(signal)
            result = algo.predict(pen=pen)
        except rpt.exceptions.BadSegmentationParameters:
            # Handle exception
            result = []

        # Assign segment numbers
        segment_number = 1
        ends = set(result)
        for i, idx in enumerate(property_df.index):
            if (i + 1) in ends:
                segment_number += 1
            df.loc[idx, 'segment'] = segment_number

    return df


# Function to apply the bourdet derivative production data
def bourdet_outliers(y, x, L, xlog, ylog, z_threshold=2, min_array_size=6):
    '''
    Applies the bourdet derivative function from petbox-dca and removes outliers based on the z-score of the derivative.
    Args:
        y: numpy.NDFloat
        An array of y values to compute the derivative for.

        x: numpy.NDFloat
            An array of x values.

        L: float = 0.0
            Smoothing factor in units of log-cycle fractions. A value of zero returns the
            point-by-point first-order difference derivative.

        xlog: bool = True
            Calculate the derivative with respect to the log of x, i.e. ``dy / d[ln x]``.

        ylog: bool = False
            Calculate the derivative with respect to the log of y, i.e. ``d[ln y] / dx``.

        z_threshold: float = 2.0
            The z-score threshold for removing outliers

        min_array_size: int = 6
            The minimum number of points needed to apply the bourdet derivative
    Returns:
        y: numpy.NDFloat
            An array of y values with outliers removed.

        x: numpy.NDFloat
            An array of x values with outliers removed.
    '''
    if len(y) < min_array_size:
        return y, x
    else:
        # Calculate the bourdet derivative
        forward_derivative = dca.bourdet(y, x, L, xlog, ylog)
        reverse_derivative = dca.bourdet(y[::-1], x, L, xlog, ylog)[::-1]

        # Calculate the z-score of the bourdet derivatives
        forward_zscore = (forward_derivative - np.nanmean(forward_derivative)) / np.nanstd(forward_derivative)
        reverse_zscore = (reverse_derivative - np.nanmean(reverse_derivative)) / np.nanstd(reverse_derivative)

        # Identify the outliers
        outliers = (np.abs(forward_zscore) >= z_threshold) | (np.abs(reverse_zscore) >= z_threshold)

        # Exclude points where both forward and reverse derivatives are NaN
        valid_derivatives = ~(np.isnan(forward_derivative) & np.isnan(reverse_derivative))

        # Remove the outliers and where forward_derivative is NaN or reverse_derivative is NaN
        final_selection = ~outliers & valid_derivatives

        # Apply the final selection to keep valid points
        y = y[final_selection]
        x = x[final_selection]

        return y, x


# Function to derive b_factor from raw production data
def b_factor_diagnostics(df, rate_col, time_col, cadence='monthly', smoothing_factor=2, min_months=24, max_months=60):
    '''
    Function to calculate b-factor for Arps hyperbolic decline from production data
    Args:
    - df: DataFrame with production data (filtered to a single well and a single measure)
    - rate_col: Name of the column with production rate data
    - time_col: Name of the column with time data (in months or days not dates)
    - cadence: Frequency of the time data (monthly or daily)
    - smoothing_factor: Number of iterations of production data smoothing using a 3-month rolling average
    - min_months: Minimum number of months to consider for the calculation
    - max_months: Maximum number of months to consider for the calculation
    Returns:
    - Dictionary with the following keys
        - df: DataFrame with the filtered data
        - b_avg: Average b-factor
        - b_low: Low end of the 90% confidence interval for b-factor
        - b_high: High end of the 90% confidence interval for b-factor
        - summary: Summary of the best model
        - best_r2: Best R^2 value
        - best_max_time: Best number of months or days to consider
    '''
    best_r2 = -np.inf

    # Set default min_time based on cadence
    if cadence == 'monthly':
        min_time = min_months
        max_time = max_months
    else:
        min_time = round(min_months * 365.25 / 12, 0)
        max_time = round(max_months * 365.25 / 12, 0)

    best_max_time = min_time  # Generally assume that 1 year of monthly data is needed
    results = {}

    # Filter df to exclude rows where rate_col is 0 or negative and any values before the maximum rate
    df = df[df[rate_col] > 0].reset_index(drop=True)

    # Check if df is empty after filtering
    if df.empty:
        return None  # Return None early if DataFrame is empty

    is_max_value = df[rate_col] == df[rate_col].max()
    last_max_index = is_max_value[::-1].idxmax()
    df = df.loc[last_max_index:].reset_index(drop=True)

    # Check if the length of df is less than min_months
    if len(df) < min_months:
        return None  # Return None early if there are not enough data points

    # Apply smoothing by calculating a 3-month rolling average smoothing_factor times
    if smoothing_factor > 0:
        for i in range(smoothing_factor):
            df[rate_col] = df[rate_col].rolling(window=3, min_periods=1).mean()

    for time_limit in range(min_time, max_time + 1):  # Assuming time_col values are integers
        temp_df = df[df[time_col] <= time_limit].copy()
        temp_df['nominal_decline'] = (
            np.log(temp_df[rate_col].shift(1) / temp_df[rate_col]) / (temp_df[time_col] - temp_df[time_col].shift(1))
        )
        temp_df['b_integral'] = (1 - temp_df['nominal_decline']) / (temp_df['nominal_decline'])

        # Filter b_integral such that it results in a reasonable range of b_factors (0.001 as nearly exponential and 3.0 as an upper limit for b-factor)  # noqa
        temp_df = temp_df[
            (temp_df['b_integral'] > 0.001 * temp_df[time_col]) &
            (temp_df['b_integral'] < 3.0 * temp_df[time_col])
        ]

        # Fit model and check R^2
        if not temp_df.empty:
            X = sm.add_constant(temp_df[time_col])
            model = sm.OLS(temp_df['b_integral'], X).fit()

            if model.rsquared > best_r2:
                best_r2 = model.rsquared
                best_max_time = time_limit

                # Store the best model's details
                results['df'] = temp_df.reset_index(drop=True)
                results['b_avg'] = model.params.iloc[1]
                results['b_low'] = model.conf_int(alpha=0.10)[1:].values[0][0]
                results['b_high'] = model.conf_int(alpha=0.10)[1:].values[0][1]
                results['summary'] = model.summary()
                results['best_r2'] = best_r2
                results['best_max_time'] = best_max_time

    if results:
        return results
    else:
        return None  # In case all iterations result in an empty DataFrame or no improvement in R^2


# Function to calculate goodness of fit metrics between actual and predicted production data
def calc_goodness_of_fit(q_act, q_pred):
    q_act = np.asarray(q_act, float)
    q_pred = np.asarray(q_pred, float)
    if q_act.size < 2 or np.allclose(q_act, q_act[0]) or np.allclose(q_pred, q_pred[0]):
        r_squared = 0.0
    else:
        r, _ = scipy.stats.pearsonr(q_act, q_pred)
        r_squared = float(r * r)
    rmse = float(np.sqrt(np.mean((q_act - q_pred)**2)))
    mae = float(np.mean(np.abs(q_act - q_pred)))
    return r_squared, rmse, mae


# Functions to build reusable PyMC models
_MODEL_CACHE: Dict[Tuple, Tuple[pm.Model, dict]] = {}  # Reusable PyMC model cache


def _model_key(optimize_names, include_deterministics: bool, use_fixed_nu: bool):
    prior_sig = "b:triangular;Dei:gapBeta22"
    return (tuple(sorted(optimize_names)), bool(include_deterministics), bool(use_fixed_nu), prior_sig)


# --- PyMC data helper for PyMC3/4/5 compatibility ---
def _md(name, value):
    """Return a mutable-ish data container: pm.MutableData if available, else pm.Data."""
    if hasattr(pm, "MutableData"):
        return pm.MutableData(name, value)
    return pm.Data(name, value)


def _build_pymc_model(optimize_names, include_deterministics=False, use_fixed_nu=True):
    """
    Build a reusable PyMC model with MutableData for everything that changes per well:
    data, bounds, warm-start centers, kappas, fixed params, and v.
    Returns (model, handles) where 'handles' stores the MutableData tensors.
    """
    with pm.Model() as model:
        # Per-well data (variable length)
        t_mo = _md("t_mo", np.zeros(1, dtype=np.float64))
        q_obs = _md("q_obs", np.zeros(1, dtype=np.float64))

        # Per-parameter mutable bounds / centers / kappa
        handles = {}
        for name in ["Qi", "Dei", "Def", "b"]:
            handles[f"lo_{name}"] = _md(f"lo_{name}", 0.0)
            handles[f"hi_{name}"] = _md(f"hi_{name}", 1.0)
            handles[f"center_{name}"] = _md(f"center_{name}", 0.5)
            handles[f"kappa_{name}"] = _md(f"kappa_{name}", 8.0)

        # helper: bounded Beta prior mapped to [lo,hi]
        def _bounded_beta(name):
            lo = handles[f"lo_{name}"]
            hi = handles[f"hi_{name}"]
            c = handles[f"center_{name}"]
            kap = handles[f"kappa_{name}"]
            width = pm.math.clip(hi - lo, 1e-12, np.inf)
            u0 = pm.math.clip((c - lo) / width, 1e-3, 1 - 1e-3)
            alpha = u0 * kap
            beta = (1.0 - u0) * kap
            u = pm.Beta(f"{name}_u", alpha=alpha, beta=beta)
            return pm.Deterministic(name, lo + width * u)

        # Triangular prior for b-factor with mode at the guess
        def _bounded_triangular(name):
            lo = handles[f"lo_{name}"]
            hi = handles[f"hi_{name}"]
            c = handles[f"center_{name}"]
            width = pm.math.clip(hi - lo, 1e-12, np.inf)
            u_mode = pm.math.clip((c - lo) / width, 1e-6, 1 - 1e-6)
            u = pm.Triangular(f"{name}_u", lower=0.0, upper=1.0, c=u_mode)
            return pm.Deterministic(name, lo + width * u)

        RV = {}
        for name in ["Qi", "Def", "Dei", "b"]:
            if name not in optimize_names:
                handles[f"fixed_{name}"] = _md(f"fixed_{name}", 0.0)
                RV[name] = handles[f"fixed_{name}"]
                continue
            if name == "b":
                RV[name] = _bounded_triangular("b")
            elif name == "Dei":
                # Enforce Dei >= Def by modeling a gap fraction
                hi = handles["hi_Dei"]
                lo_eff = pm.math.maximum(handles["lo_Dei"], RV["Def"])
                width = pm.math.clip(hi - lo_eff, 1e-12, np.inf)
                gap_u = pm.Beta("Dei_gap_u", alpha=2.0, beta=2.0)
                RV["Dei"] = pm.Deterministic("Dei", lo_eff + width * gap_u)
            else:
                RV[name] = _bounded_beta(name)

        # Model mean: your PyTensor ARPS rate function
        q_model = arps_q_pt(t_mo, RV["Qi"], RV["Dei"], RV["Def"], RV["b"])
        log_q_mu = pm.math.log(q_model + 1e-9)
        if include_deterministics:
            pm.Deterministic("log_q_model", log_q_mu)
        sigma = pm.HalfNormal("sigma", 0.5)
        if use_fixed_nu:
            handles["nu_fixed"] = _md("nu_fixed", 4.0)
            pm.StudentT("Y_obs", nu=handles["nu_fixed"], mu=log_q_mu, sigma=sigma,
                        observed=pm.math.log(q_obs + 1e-9))
        else:
            nu = pm.Exponential("nu_raw", 1 / 5) + 2.0
            pm.StudentT("Y_obs", nu=nu, mu=log_q_mu, sigma=sigma,
                        observed=pm.math.log(q_obs + 1e-9))

    return model, handles


# Function to perform curve fitting with dynamic parameter optimization
def perform_curve_fit(  # noqa
        t_act,
        q_act,
        initial_guess,
        bounds,
        config,
        method='curve_fit',
        trials=1000,
        use_advi=False,
        *,
        include_deterministics=False,
        log_likelihood=False,
        return_trace=True
    ):
    """
    Fit ARPS parameters to (t_act, q_act) using one of:
    - 'curve_fit' (SciPy bounded least squares),
    - 'monte_carlo' (PyMC NUTS or ADVI with bounded priors; likelihood on log-rates),
    - 'differential_evolution' (SciPy DE).

    Notes
    -----
    • All methods warm-start from SciPy `curve_fit` when possible; if that fails,
    they fall back to the bounds-clipped `initial_guess`.
    • Returned point estimates are clipped to the provided bounds.
    • 'monte_carlo' uses a bounded reparameterization (logistic over [low, high])
    so the posterior stays inside bounds.
    • For Monte Carlo methods, PyMC models are cached and reused across wells
    to avoid rebuilding the model graph for each fit.
    • When `use_advi=True`, the model uses fixed v (`nu_fixed`) via
    `pm.MutableData` for faster ADVI convergence; otherwise v is learned.

    Parameters
    ----------
    t_act : array-like
        Time values (months). Will be cast to float64.
    q_act : array-like
        Rates. Non-positive values are clipped to a small positive number.
    initial_guess : list
        Initial guess for the parameters in the order given by `config["optimize"]`.
    bounds : tuple
        Either (low, high) for a single parameter or ((lows...), (highs...))
        aligned with `config["optimize"]`.
    config : dict
        {'optimize': [param names], 'fixed': {param: value}}
    method : {'curve_fit','monte_carlo','differential_evolution'}
    trials : int
        Iterations / draws depending on method.
    use_advi : bool
        If True and method == 'monte_carlo', use ADVI for fitting.
        Will use fixed v to improve speed/stability.
    include_deterministics : bool
        If True, records extra deterministics to the trace.
    log_likelihood : bool
        If True, save log_likelihood in the trace.
    return_trace : bool
        If True, return the trace object from the model fitting.

    Returns
    -------
    tuple
        If method == 'monte_carlo': (params, success, trace)
        Else: (params, success)
        where `params` aligns with `config["optimize"]`.
    """
    t_act = np.asarray(t_act, dtype=np.float64)
    q_act = np.clip(np.asarray(q_act, dtype=np.float64), 1e-9, None)

    # Validation shared by all methods
    valid_names = {"Qi", "Dei", "Def", "b"}
    unknown = [n for n in config["optimize"] if n not in valid_names]
    if unknown:
        raise ValueError(f"Unknown parameter(s) in optimize: {unknown}")

    def bounds_to_dict(opt_names, bounds):
        if len(bounds) == 2 and np.ndim(bounds[0]) == 0:
            lo, hi = float(bounds[0]), float(bounds[1])
            return {opt_names[0]: (lo, hi)}
        lows, highs = bounds
        return {n: (float(lo), float(hi)) for n, lo, hi in zip(opt_names, lows, highs)}

    # Build bounds mapping for optimized params
    param_bounds = bounds_to_dict(config["optimize"], bounds)

    missing = [n for n in config["optimize"] if n not in param_bounds]
    if missing:
        raise ValueError(f"Missing bounds for: {missing}")

    overlap = set(config["optimize"]).intersection(config["fixed"])
    if overlap:
        raise ValueError(f"Parameters cannot be both optimized and fixed: {sorted(overlap)}")

    def _normalize_bounds(bounds_in, expected_len):
        """
        Always return a list of (lo_i, hi_i) pairs of expected length.
        Accepts either (lo, hi) for a single param or (lows, highs) sequences.
        """
        if isinstance(bounds_in[0], numbers.Real):
            out = [(float(bounds_in[0]), float(bounds_in[1]))]
        else:
            lows, highs = bounds_in
            lows = np.asarray(lows, float).ravel()
            highs = np.asarray(highs, float).ravel()
            if lows.shape != highs.shape:
                raise ValueError(f"bounds lows/highs shape mismatch: {lows.shape} vs {highs.shape}")
            out = [(float(lo), float(hi)) for lo, hi in zip(lows, highs)]
        if len(out) != expected_len:
            raise ValueError(f"bounds length {len(out)} != expected {expected_len}")
        return out

    opt_names = list(config["optimize"])
    b_pairs_norm = _normalize_bounds(bounds, expected_len=len(opt_names))
    lo_vec = np.array([p[0] for p in b_pairs_norm], dtype=float)
    hi_vec = np.array([p[1] for p in b_pairs_norm], dtype=float)

    def arps_fit(t, Qi, Dei, b, Def):
        return arps_q_np(t, Qi, Dei, Def, b)

    # arps_q_pt signature: arps_q_pt(t_mo, Qi, Dei, Def, b) -> q(t)
    def arps_fit_mc(t, Qi, Dei, b, Def):
        return arps_q_pt(t, Qi, Dei, Def, b)

    # Dynamically construct the model function based on which parameters are being optimized
    def model_func(t, *params, method=method):
        param_values = dict(zip(config["optimize"], params))
        for fixed_param, fixed_value in config["fixed"].items():
            param_values[fixed_param] = fixed_value
        if method == 'monte_carlo':
            return arps_fit_mc(t, param_values["Qi"], param_values["Dei"], param_values["b"], param_values["Def"])
        else:
            return arps_fit(t, param_values["Qi"], param_values["Dei"], param_values["b"], param_values["Def"])

    def convert_bounds(bounds):
        if isinstance(bounds[0], numbers.Real):
            return [(float(bounds[0]), float(bounds[1]))]
        elif isinstance(bounds[0], (tuple, list, np.ndarray)):
            return list(zip(*bounds))
        else:
            raise ValueError("Invalid bounds format")

    if method == 'monte_carlo':
        # First use curve_fit to get the initial parameter estimates
        # Hygiene before warm-start (avoid zeros/NaNs):
        def wrapped_model_func_np(t, *params):
            p = dict(zip(config["optimize"], params))
            p.update(config["fixed"])
            return arps_q_np(t, p["Qi"], p["Dei"], p["Def"], p["b"])

        # Try a coarse-grid warm start before bounded curve_fit
        try:
            opt_names = config["optimize"]
            fixed = config["fixed"]
            lo = lo_vec
            hi = hi_vec
            tmp = dict(zip(opt_names, initial_guess))
            Qi_guess = tmp.get("Qi", fixed.get("Qi"))
            Dei_guess = tmp.get("Dei", fixed.get("Dei"))
            b_guess = tmp.get("b", fixed.get("b"))
            Def_val = tmp.get("Def", fixed.get("Def"))

            # Fast lookup for parameter indices in opt_names
            idx = {n: i for i, n in enumerate(opt_names)}

            # Dei bounds for coarse grid (or ±20% around guess if not optimized)
            if "Dei" in idx:
                i = idx["Dei"]
                Dei_low, Dei_high = lo[i], hi[i]
            else:
                Dei_low, Dei_high = Dei_guess * 0.8, Dei_guess * 1.2

            # If Def is fixed, ensure Dei_low >= Def for the coarse grid
            if "Def" not in idx and Def_val is not None:
                Dei_low = max(Dei_low, float(Def_val))

            # b bounds for coarse grid (or around guess if not optimized)
            if "b" in idx:
                i = idx["b"]
                b_low, b_high = lo[i], hi[i]
            else:
                b_low = max(0.3, b_guess * 0.8)
                b_high = min(2.0, b_guess * 1.2)

            # Run tiny coarse grid only if all seeds exist
            if all(v is not None for v in (Qi_guess, Dei_guess, b_guess, Def_val)):
                g0 = _coarse_grid_best(
                    t_act,
                    q_act,
                    Qi_guess,
                    Dei_low,
                    Dei_guess,
                    Dei_high,
                    b_low,
                    b_guess,
                    b_high,
                    Def_val
                )

                # Map coarse-grid outputs back to the order in opt_names, only if complete
                val_map = {"Qi": g0[0], "Dei": g0[1], "b": g0[2], "Def": Def_val}
                ig_new = []
                ok_seed = True
                for n in opt_names:
                    v = val_map.get(n, None)
                    if v is None:
                        ok_seed = False
                        break
                    ig_new.append(float(v))
                if ok_seed:
                    initial_guess = ig_new

        except Exception:
            # Swallow coarse-grid issues; downstream curve_fit handles warm start fallback
            pass

        # Then bounded curve_fit for a MAP-style warm start
        curve_ok = True
        try:
            # sanitize IG strictly inside bounds
            ig_arr = np.asarray(initial_guess, float)
            if ig_arr.size != len(opt_names):
                raise ValueError(f"initial_guess len {ig_arr.size} != {len(opt_names)}")
            eps = 1e-9 * np.maximum(1.0, np.abs(hi_vec - lo_vec))
            ig_arr = np.clip(ig_arr, lo_vec + eps, hi_vec - eps)
            popt, _ = curve_fit(
                wrapped_model_func_np,
                t_act,
                q_act,
                p0=ig_arr,
                bounds=(lo_vec, hi_vec),
                maxfev=max(trials, 4000),
            )
        except (RuntimeError, ValueError):
            curve_ok = False
            popt = np.clip(np.asarray(initial_guess, float), lo_vec, hi_vec)

        # If the warm-start curve_fit succeeded and produced Dei == Def, skip MCMC and return that fit.
        if curve_ok:
            fit_map = dict(zip(config['optimize'], np.asarray(popt, dtype=float)))
            dei_val = fit_map.get("Dei", config["fixed"].get("Dei"))
            def_val = fit_map.get("Def", config["fixed"].get("Def"))
            if (dei_val is not None) and (def_val is not None):
                if np.isclose(dei_val, def_val, rtol=1e-6, atol=1e-9):
                    popt_clipped = np.clip(np.asarray(popt, float), lo_vec, hi_vec)
                    return popt_clipped, True, None

        # Reuse or build a cached model
        use_fixed_nu = bool(use_advi)
        mkey = _model_key(config["optimize"], include_deterministics, use_fixed_nu)
        if mkey not in _MODEL_CACHE:
            _MODEL_CACHE[mkey] = _build_pymc_model(
                config["optimize"],
                include_deterministics,
                use_fixed_nu
            )
        model, handles = _MODEL_CACHE[mkey]

        # Centers from curve_fit warm start (+ fallbacks)
        centers = dict(zip(config['optimize'], popt))
        centers.update(config['fixed'])
        pri = config.get("priors", {})
        ig_map = dict(zip(config['optimize'], initial_guess))

        def _mid(lo, hi):
            return 0.5 * (float(lo) + float(hi))

        # b center: only adjust if b is being optimized; otherwise keep the fixed value
        if "b" in config["optimize"]:
            if "b" in param_bounds:
                b_lo, b_hi = param_bounds["b"]
                b_mid = _mid(b_lo, b_hi)
            else:
                b_mid = ig_map.get("b", 1.0)
            centers["b"] = float(pri.get("b_mu", ig_map.get("b", b_mid)))

        # Dei center: prefer YAML guess in initial_guess; else midpoint of Dei bounds
        if "Dei" not in centers and "Dei" in param_bounds:
            d_lo, d_hi = param_bounds["Dei"]
            d_mid = _mid(d_lo, d_hi)
            centers["Dei"] = float(ig_map.get("Dei", d_mid))

        # Def center is already set if fixed. If Def were optimized, fall back to its bounds midpoint.
        if "Def" not in centers and "Def" in param_bounds:
            f_lo, f_hi = param_bounds["Def"]
            centers["Def"] = _mid(f_lo, f_hi)

        # Qi fallback remains data-driven
        centers.setdefault("Qi", max(np.nanmax(q_act), 1e-3))

        with model:
            pm.set_data({
                "t_mo": np.asarray(t_act, np.float64).copy(),
                "q_obs": np.asarray(q_act, np.float64).copy()
            })
            kappa_map = {"Qi": 6.0}
            if "Def" in config["optimize"]:
                kappa_map["Def"] = 8.0
            # Optional user overrides for b prior
            if "b" in config["optimize"] and "b_mu" in pri:
                centers["b"] = float(pri["b_mu"])
            for name in ["Qi", "Dei", "Def", "b"]:
                lo_i, hi_i = param_bounds.get(name, (centers[name], centers[name]))
                pm.set_data({f"lo_{name}": float(lo_i), f"hi_{name}": float(hi_i)})
                pm.set_data({f"center_{name}": float(centers[name])})
                if name in kappa_map:
                    pm.set_data({f"kappa_{name}": float(kappa_map[name])})
            for name in [n for n in ["Qi", "Dei", "Def", "b"] if n not in config["optimize"]]:
                pm.set_data({f"fixed_{name}": float(centers[name])})

            if use_fixed_nu:
                pm.set_data({"nu_fixed": 4.0})

            if use_advi:
                approx = pm.fit(
                    method="advi",
                    n=min(trials, 300),
                    callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=5e-4)],
                    obj_optimizer=pm.adagrad_window(learning_rate=1e-2),
                )
                trace = None
                try:
                    # small number of draws is enough to get posterior means
                    draws = 64 if not return_trace else min(trials, 400)
                    idata = approx.sample(draws=draws)

                    post = idata.posterior
                    optimized_params = np.array([float(post[p].values.mean())
                                                 for p in config["optimize"]])
                    lo_hi = np.array([param_bounds[name] for name in config["optimize"]], dtype=float)
                    lo, hi = lo_hi[:, 0], lo_hi[:, 1]
                    eps = 1e-6 * np.maximum(1.0, np.abs(hi - lo))
                    optimized_params = np.clip(optimized_params, lo + eps, hi - eps)

                    if return_trace:
                        trace = idata
                    else:
                        del idata
                finally:
                    del approx
                    gc.collect()

                return optimized_params, True, trace
            else:
                try:
                    start = pm.initial_point()
                except Exception:
                    start = None
                use_jax = return_trace and getattr(pm, "sampling_jax", None) is not None
                if use_jax:
                    trace = pm.sampling_jax.sample_blackjax_nuts(
                        draws=min(trials, 600),
                        tune=max(trials // 4, 200),
                        chains=2,
                        chain_method="vectorized",
                        target_accept=0.9,
                        random_seed=42,
                        progressbar=False,
                        initvals=start
                    )
                else:
                    trace = pm.sample(
                        draws=min(trials, 600),
                        tune=max(trials // 4, 150),
                        chains=1,
                        cores=1,
                        target_accept=0.9,
                        progressbar=False,
                        initvals=start,
                        return_inferencedata=return_trace,
                        compute_convergence_checks=False,
                        discard_tuned_samples=True,
                        keep_untransformed=False,
                        idata_kwargs={"log_likelihood": log_likelihood} if return_trace else None
                    )

        # Posterior means: InferenceData or MultiTrace.
        if return_trace and hasattr(trace, "posterior"):
            optimized_params = np.array([trace.posterior[p].values.mean()
                                         for p in config["optimize"]])
            trace_to_return = trace
        else:
            # MultiTrace path: compute means, then drop the trace to save RAM.
            def _mean_mt(name):
                try:
                    return float(np.asarray(trace.get_values(name), dtype=float).mean())
                except Exception:
                    return np.nan
            optimized_params = np.array([_mean_mt(p) for p in config["optimize"]])
            trace_to_return = None
            del trace
            gc.collect()

        lo_hi = np.array([param_bounds[name] for name in config['optimize']], dtype=float)
        lo, hi = lo_hi[:, 0], lo_hi[:, 1]
        eps = 1e-6 * np.maximum(1.0, np.abs(hi - lo))
        optimized_params = np.clip(optimized_params, lo + eps, hi - eps)
        return optimized_params, True, trace_to_return

    elif method == 'differential_evolution':
        bounds_de = [(lo, hi) for lo, hi in b_pairs_norm]

        def de_model_func(params):
            model_values = model_func(t_act, *params, method='differential_evolution')
            return np.sum((q_act - model_values) ** 2)  # Sum of squared errors

        result = differential_evolution(
            de_model_func,
            bounds_de,
            maxiter=trials,
            updating='deferred'
        )
        x = np.asarray(result.x, float)
        lo = np.array([b[0] for b in bounds_de], float)
        hi = np.array([b[1] for b in bounds_de], float)
        return np.clip(x, lo, hi), True

    else:
        try:
            wrapped_model_func = partial(model_func, method='curve_fit')
            try:
                opt_names = config["optimize"]
                fixed = config["fixed"]
                lo = lo_vec
                hi = hi_vec
                tmp = dict(zip(opt_names, initial_guess))
                Qi_guess = tmp.get("Qi", fixed.get("Qi"))
                Dei_guess = tmp.get("Dei", fixed.get("Dei"))
                b_guess = tmp.get("b", fixed.get("b"))
                Def_val = tmp.get("Def", fixed.get("Def"))

                # Fast index map for optimized params
                idx = {n: i for i, n in enumerate(opt_names)}

                # Dei bounds for coarse grid (or ±20% around guess if not optimized)
                if "Dei" in idx:
                    i = idx["Dei"]
                    Dei_low, Dei_high = lo[i], hi[i]
                else:
                    Dei_low = Dei_guess * 0.8
                    Dei_high = Dei_guess * 1.2

                # b bounds for coarse grid (or around guess if not optimized)
                if "b" in idx:
                    i = idx["b"]
                    b_low, b_high = lo[i], hi[i]
                else:
                    b_low = max(0.3, b_guess * 0.8)
                    b_high = min(2.0, b_guess * 1.2)

                if all(v is not None for v in (Qi_guess, Dei_guess, b_guess, Def_val)):
                    g0 = _coarse_grid_best(
                        t_act,
                        q_act,
                        Qi_guess,
                        Dei_low,
                        Dei_guess,
                        Dei_high,
                        b_low,
                        b_guess,
                        b_high,
                        Def_val,
                    )

                    # Map coarse-grid outputs back to opt_names order
                    val_map = {"Qi": g0[0], "Dei": g0[1], "b": g0[2], "Def": Def_val}
                    ig_new = []
                    ok_seed = True
                    for name in opt_names:
                        v = val_map.get(name, None)
                        if v is None:
                            ok_seed = False
                            break
                        ig_new.append(float(v))
                    if ok_seed:
                        initial_guess = ig_new

            except Exception:
                # Swallow coarse-grid issues; curve_fit/de handles warm-start fallback.
                pass

            # sanitize IG strictly inside bounds
            ig_arr = np.asarray(initial_guess, float)
            if ig_arr.size != len(opt_names):
                raise ValueError(f"initial_guess len {ig_arr.size} != {len(opt_names)}")
            eps = 1e-9 * np.maximum(1.0, np.abs(hi - lo))
            ig_arr = np.clip(ig_arr, lo + eps, hi - eps)

            def wrapped_model_func_safe(t, *params):
                y = wrapped_model_func(t, *params)
                return np.nan_to_num(y, nan=1e12, posinf=1e12, neginf=0.0)
            popt, _ = curve_fit(
                wrapped_model_func_safe,
                t_act,
                q_act,
                p0=ig_arr,
                bounds=(lo, hi),
                maxfev=max(trials, 4000)
            )
            popt = np.clip(np.asarray(popt, float), lo, hi)
            return popt, True
        except (RuntimeError, ValueError) as e:
            # Differential evolution fallback for tough landscapes
            try:
                de_bounds = [(p[0], p[1]) for p in b_pairs_norm]

                def sse_obj(params):
                    return float(np.sum((q_act - wrapped_model_func(t_act, *params))**2))
                result = differential_evolution(
                    sse_obj,
                    de_bounds,
                    maxiter=max(trials // 100, 100),
                    tol=1e-4,
                    updating='deferred'
                )
                x = np.asarray(result.x, float)
                lo = np.array([p[0] for p in de_bounds], dtype=float)
                hi = np.array([p[1] for p in de_bounds], dtype=float)
                return np.clip(x, lo, hi), True
            except Exception:
                print(f"Curve fitting failed: {e}")
                return np.full(len(config["optimize"]), np.nan), False
