import numpy as np
import pandas as pd
import statsmodels.api as sm

def b_factor_diagnostics(df, rate_col, time_col, data_type, cadence='monthly', smoothing_factor=2, min_months=24, max_months=60):
    '''
    Function to calculate b-factor for Arps hyperbolic decline from production data
    Args:
    - df: DataFrame with production data (filtered to a single well and a single measure)
    - rate_col: Name of the column with production rate data
    - time_col: Name of the column with time data (in months or days not dates)
    - data_type: String value "ProductionMaterialized" or "TypeCurve"
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

    # Apply smoothing by calculating a 3-month rolling average 2 times if DataType != 'Type Curve'
    if smoothing_factor > 0 and data_type != 'Type Curve':
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

def apply_b_factor_diagnostics(grouped_df, rate_col, time_col, cadence='monthly', smoothing_factor=2, min_months=24, max_months=60):
    '''
    Function to apply b_factor_diagnostics to a DataFrame grouped by DataType and Measure
    Args:
    - grouped_df: GroupBy object of DataFrame grouped by DataType and Measure
    - rate_col: Name of the column with production rate data
    - time_col: Name of the column with time data (in months or days not dates)
    - cadence: Frequency of the time data (monthly or daily)
    - min_months: Minimum number of months to consider for the calculation
    - max_months: Maximum number of months to consider for the calculation
    Returns:
    - DataFrame with the results of b_factor_diagnostics applied to each group
    Example:
    grouped_df = prod_df.groupby(['DataType', 'Measure'])
    results = apply_b_factor_diagnostics(grouped_df, 'MonthlyVolume', 'ProdMonth')
    '''
    results = []
    for (data_type, measure, scenario), group in grouped_df:
        temp_results = b_factor_diagnostics(group, rate_col, time_col, data_type, cadence, smoothing_factor, min_months, max_months)
        if temp_results:  # Ensure there are results before appending
            temp_results['DataType'] = data_type
            temp_results['Measure'] = measure
            temp_results['ScenarioName'] = scenario
            results.append(temp_results)
    
    if results:  # Ensure there are any results to convert to DataFrame
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no results
        
# prod_df, smoothing_factor, min_months, max_months are document properties defined in Spotfire and are set as input parameters
prod_df = prod_df[prod_df['DataType'] != 'monthly_forecast_volumes']

pivot_df = prod_df.pivot_table(index=['DataType', 'Measure', 'ProdMonth', 'ScenarioName'], values='MonthlyVolume', aggfunc='mean').reset_index()

grouped_df = pivot_df.groupby(['DataType', 'Measure', 'ScenarioName'])
results = apply_b_factor_diagnostics(grouped_df, 'MonthlyVolume', 'ProdMonth', cadence='monthly', smoothing_factor=smoothing_factor, min_months=min_months, max_months=max_months)

# Create combined dataframe for all resulting dataframes
combined_df = pd.concat(results['df'].tolist(), ignore_index=True)

# Create table output for b_factor estimates
b_factor_df = results[['DataType', 'Measure', 'ScenarioName', 'b_avg', 'b_low', 'b_high', 'best_r2']]
