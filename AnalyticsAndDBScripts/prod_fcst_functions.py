import numpy as np
import ruptures as rpt
from petbox import dca
import statsmodels.api as sm
import scipy.stats
import pytensor.tensor as pt
import pymc as pm
from scipy.optimize import curve_fit, differential_evolution
from functools import partial

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
    if Dei == Def:
        Type = 'exp'
    elif Dei > Def and b == 1:
        Type = 'har'
        Dn = Dei / (1 - Dei)
        Qlim = Qi * ((-np.log(1 - Def)) / Dn)
        tlim = (((Qi / Qlim) - 1) / Dn) * 12 # output in months
    else:
        Type = 'hyp'
        Dn = (1 / b) * (((1 - Dei) ** -b) - 1)
        Qlim = Qi * ((-np.log(1 - Def)) / Dn) ** (1 / b)
        tlim = ((((Qi / Qlim) ** b) - 1) / ( b * Dn)) * 12 # output in months
    
    # Generate volumes
    if Type == 'hyp':
        Dn_t = Dn / (1 + b * Dn * (t / 12))
        De_t = 1 - (1 / ((Dn_t * b) + 1)) ** (1 / b)
        if De_t > Def:
            q = Qi * (1 + b * Dn * (t / 12)) ** (-1/b)
            Np = ((Qi ** b) / (Dn * (1 - b))) * ((Qi ** (1 - b)) - (q ** (1 - b))) * 365
        else:
            q = Qlim * np.exp(-(-np.log(1 - Def)) * ((t - tlim) / 12))
            Np = ((Qlim - q) / (-np.log(1 - Def)) * 365) + (((Qi ** b) / 
                    (Dn * (1 - b))) * ((Qi ** (1 - b)) - (Qlim ** (1 - b))) * 365)
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


# Version of function to use with pymc for auto-fitting
def arps_decline_mc(UID, phase, Qi, Dei, Def, b, t, prior_cum=0, prior_t=0):
    '''
    Args:
    - UID is a unique identifier for the well such as API, must be a number
    - phase is 1 = oil, 2 = gas, or 3 = water
    - Qi is the initial production rate typically in bbl/day or Mcf/day
    - Dei is the initial effective annual decline rate
    - Def is the final effective annual decline rate at which point the decline becomes exponential
    - b is the b-factor used in hyperbolic or harmonic decline equations
    - t is the time as a month integer
    - prior_cum is the cumulative amount produced before the start of the decline calculations
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
    Type_exp = pt.eq(Dei, Def)
    Type_har = pt.and_(pt.gt(Dei, Def), pt.eq(b, 1))

    # Exponential decline
    q_exp = Qi * pt.exp(-(-pt.log(1 - Dei)) * (t / 12))
    Np_exp = (Qi - q_exp) / (-pt.log(1 - Dei)) * 365
    De_t_exp = Dei

    # Harmonic decline
    Dn_har = Dei / (1 - Dei)
    Qlim_har = Qi * ((-pt.log(1 - Def)) / Dn_har)
    tlim_har = (((Qi / Qlim_har) - 1) / Dn_har) * 12
    Dn_t_har = Dn_har / (1 + Dn_har * (t / 12))
    De_t_har = 1 - (1 / (Dn_t_har + 1))
    q_har = pt.switch(De_t_har > Def, Qi / (1 + Dn_har * (t / 12)), Qlim_har * pt.exp(-(-pt.log(1 - Def)) * ((t - tlim_har) / 12)))
    Np_har = pt.switch(De_t_har > Def, (Qi / Dn_har) * pt.log(Qi / q_har) * 365, ((Qlim_har - q_har) / (-pt.log(1 - Def)) * 365) + ((Qi / Dn_har) * pt.log(Qi / Qlim_har) * 365))
    De_t_har = pt.switch(De_t_har > Def, De_t_har, Def)

    # Hyperbolic decline
    Dn_hyp = (1 / b) * (((1 - Dei) ** -b) - 1)
    Qlim_hyp = Qi * ((-pt.log(1 - Def)) / Dn_hyp) ** (1 / b)
    tlim_hyp = ((((Qi / Qlim_hyp) ** b) - 1) / (b * Dn_hyp)) * 12
    Dn_t_hyp = Dn_hyp / (1 + b * Dn_hyp * (t / 12))
    De_t_hyp = 1 - (1 / ((Dn_t_hyp * b) + 1)) ** (1 / b)
    q_hyp = pt.switch(De_t_hyp > Def, Qi * (1 + b * Dn_hyp * (t / 12)) ** (-1 / b), Qlim_hyp * pt.exp(-(-pt.log(1 - Def)) * ((t - tlim_hyp) / 12)))
    Np_hyp = pt.switch(De_t_hyp > Def, ((Qi ** b) / (Dn_hyp * (1 - b))) * ((Qi ** (1 - b)) - (q_hyp ** (1 - b))) * 365, ((Qlim_hyp - q_hyp) / (-pt.log(1 - Def)) * 365) + (((Qi ** b) / (Dn_hyp * (1 - b))) * ((Qi ** (1 - b)) - (Qlim_hyp ** (1 - b))) * 365))
    De_t_hyp = pt.switch(De_t_hyp > Def, De_t_hyp, Def)

    # Select the correct type of decline
    q = pt.switch(Type_exp, q_exp, pt.switch(Type_har, q_har, q_hyp))
    Np = pt.switch(Type_exp, Np_exp, pt.switch(Type_har, Np_har, Np_hyp))
    De_t = pt.switch(Type_exp, De_t_exp, pt.switch(Type_har, De_t_har, De_t_hyp))

    return UID, phase, t + prior_t, q, De_t, Np + prior_cum


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
def arps_segments(UID, phase, Q1, Q2, Q3, Dei, Def, b, Qabn, t1, t2, duration, prior_cum=0, prior_t=0):
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
        prior_cum2= np.max(seg2_arr[5])
        seg3 = varps_decline(UID, phase, Q3, Dei, Def, b, t_seg3, prior_cum2, t1 + t2)
        seg3_arr = np.array(seg3)
        # Filter out months where production is less than Qabn
        if Qabn > 0:
            Qabn_filter = seg3_arr[3] >= Qabn
            seg3_arr = seg3_arr[:,Qabn_filter]
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
            seg3_arr = seg3_arr[:,Qabn_filter]
        out_nparr = np.column_stack((seg1_arr, seg3_arr))
    elif segment_ct == 1:
        t_seg3 = np.arange(0, duration + 1, 1)
        out_nparr = varps_decline(UID, phase, Q3, Dei, Def, b, t_seg3, prior_cum, prior_t)
        out_nparr = np.array(out_nparr)
        # Filter out months where production is less than Qabn
        if Qabn > 0:
            Qabn_filter = out_nparr[3] >= Qabn
            out_nparr = out_nparr[:,Qabn_filter]
    else:
        t_nan = np.arange(0, duration + 1, 1)
        UID_nan = np.full((1, duration + 1), UID)
        phase_nan = np.full((1, duration + 1), phase)
        val_nan = np.full((3, duration + 1), 0)
        out_nparr = np.vstack((t_nan, val_nan))
    
    # Add monthly volumes to array
    Cum_i = out_nparr[5][:-1]
    Cum_f = out_nparr[5][1:]
    cum = Cum_f - Cum_i
    cum[cum < 0] = 0
    cum = np.insert(cum, 0, 0)
    out_nparr = np.vstack((out_nparr, cum))[:,1:]
    
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
    - pen (float): Penalty value influencing the sensitivity of change point detection. Higher values lead to fewer detected change points.
    Returns:
    - DataFrame: Modified DataFrame with an additional column 'segment' indicating the segment number for each data point.
    Example Use:
    This function should be applied to partitions of the DataFrame. For example:
    - prod_df_oil = prod_df[prod_df['Measure'] == 'OIL'].groupby(['WellID', 'FitGroup']).apply(detect_changepoints, 'WellID', 'Value', 'Date', 10)
    - prod_df_gas = prod_df[prod_df['Measure'] == 'GAS'].groupby(['WellID', 'FitGroup']).apply(detect_changepoints, 'WellID', 'Value', 'Date', 10)
    - prod_df_wtr = prod_df[prod_df['Measure'] == 'WATER'].groupby(['WellID', 'FitGroup']).apply(detect_changepoints, 'WellID', 'Value', 'Date', 10)

    Note: The function sorts the data by the date column and assigns segment numbers based on detected change points. Segments are determined by significant changes in the production data trend.
    '''
    # Create a new column for the segment numbers
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
        for i in range(len(property_df)):
            if i in result:
                segment_number += 1
            df.loc[property_df.index[i], 'segment'] = segment_number

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
    
    best_max_time = min_time # Generally assume that 1 year of monthly data is needed
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
        temp_df['nominal_decline'] = np.log(temp_df[rate_col].shift(1) / temp_df[rate_col]) / (temp_df[time_col] - temp_df[time_col].shift(1))
        temp_df['b_integral'] = (1 - temp_df['nominal_decline']) / (temp_df['nominal_decline'])

        # Filter b_integral such that it results in a reasonable range of b_factors (0.001 as nearly exponential and 3.0 as an upper limit for b-factor)
        temp_df = temp_df[(temp_df['b_integral'] > 0.001 * temp_df[time_col]) & (temp_df['b_integral'] < 3.0 * temp_df[time_col])]

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
    '''
    Args:
    - q_act: Actual production data
    - q_pred: Predicted production data
    Returns:
    - r_squared: R-squared value
    - rmse: Root mean squared error
    - mae: Mean absolute error
    '''
    r_squared = scipy.stats.pearsonr(q_act, q_pred)[0] ** 2
    rmse = np.sqrt(np.mean((q_act - q_pred) ** 2))
    mae = np.mean(np.abs(q_act - q_pred))
    return r_squared, rmse, mae

def perform_curve_fit(t_act, q_act, initial_guess, bounds, config, method='curve_fit', trials=1000):
    """
    Perform curve fitting with dynamic parameter optimization.
    Args:
        t_act (array-like): Actual time data.
        q_act (array-like): Actual production data.
        initial_guess (list): Initial guesses for the parameters being optimized.
        bounds (tuple): Bounds for the parameters being optimized.
        config (dict): Configuration specifying which parameters to optimize and which are fixed. 
                       Includes 'optimize' for parameters to optimize, and 'fixed' for fixed parameters.
        method (str): Method to use for fitting. Options are 'curve_fit', 'monte_carlo', and 'differential_evolution'.
                      Defaults to 'curve_fit'.
        trials (int): Number of trials to run. Defaults to 1000.
    Returns:
        tuple: Optimized parameters and a success flag (True if fitting succeeded, False otherwise).
    """

    def arps_fit(t, Qi, Dei, b, Def):
        return varps_decline(1, 1, Qi, Dei, Def, b, t, 0, 0)[3]
    
    def arps_fit_mc(t, Qi, Dei, b, Def):
        return arps_decline_mc(1, 1, Qi, Dei, Def, b, t, 0, 0)[3]

    def get_prior(name, initial, lower, upper):
        return pm.Triangular(name, lower=lower, upper=upper, c=initial)

    # Dynamically construct the model function based on which parameters are being optimized
    def model_func(t, *params, method=method):
        param_values = {param_name: value for param_name, value in zip(config["optimize"], params)}
        for fixed_param, fixed_value in config["fixed"].items():
            param_values[fixed_param] = fixed_value
        if method == 'monte_carlo':
            return arps_fit_mc(t, param_values["Qi"], param_values["Dei"], param_values["b"], param_values["Def"])
        else:
            return arps_fit(t, param_values["Qi"], param_values["Dei"], param_values["b"], param_values["Def"])
    
    def convert_bounds(bounds):
        if isinstance(bounds[0], float):
            # Single pair of bounds
            return [(bounds[0], bounds[1])]
        elif isinstance(bounds[0], tuple) or isinstance(bounds[0], list):
            # Tuple or list of bounds
            return list(zip(*bounds))
        else:
            raise ValueError("Invalid bounds format")

    if method == 'monte_carlo':
        # First use curve_fit to get the initial parameter estimates
        wrapped_model_func = partial(model_func, method='curve_fit')
        popt, _ = curve_fit(wrapped_model_func, t_act, q_act, p0=initial_guess, bounds=bounds)
        
        with pm.Model() as model:
            bounds = convert_bounds(bounds)
            bounds_mc = [(bound[0], bound[1]) for bound in bounds]
            priors = {}
            for param_name, (initial, (lower, upper)) in zip(config['optimize'], zip(popt, bounds_mc)):
                priors[param_name] = get_prior(param_name, initial, lower, upper)
            
            for fixed_param, fixed_value in config['fixed'].items():
                priors[fixed_param] = pm.Data(fixed_param, fixed_value)
            
            q_model = model_func(t_act, *priors.values(), method='monte_carlo')
            
            log_q_act = pm.Data('log_q_act', np.log(q_act))
            log_q_model = pm.Deterministic('log_q_model', pm.math.log(q_model))
            
            Y_obs = pm.Normal('Y_obs', mu=log_q_model, sigma=0.1, observed=log_q_act)
            
            trace = pm.sample(draws=trials, tune=trials//4, cores=4, nuts_sampler="blackjax", target_accept=0.95, return_inferencedata=True, progressbar=False, nuts_sampler_kwargs={'chain_method':'parallel'})
        
        optimized_params = np.array([trace.posterior[param].mean().item() for param in config['optimize']])
        return optimized_params

    elif method == 'differential_evolution':
        bounds = convert_bounds(bounds)
        bounds_de = [(bound[0], bound[1]) for bound in bounds]
        def de_model_func(params):
            model_values = model_func(t_act, *params, method='differential_evolution')
            return np.sum((q_act - model_values) ** 2)  # Sum of squared errors
        
        result = differential_evolution(de_model_func, bounds_de, maxiter=trials)
        return result.x

    else:
        try:
            wrapped_model_func = partial(model_func, method='curve_fit')
            popt, _ = curve_fit(wrapped_model_func, t_act, q_act, p0=initial_guess, bounds=bounds, maxfev=trials)
            return popt
        except RuntimeError as e:
            print(f"Curve fitting failed: {e}")
            return None
