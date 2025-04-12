import numpy as np
import pandas as pd

def MonthDiff(BaseDate, StartDate):
    '''
    Args:
    - BaseDate is a date from which the time difference is calculated
    - StartDate is a date for which the time difference is calculated
    Returns:
    - An integer representing the number of months between the two dates
    '''
    BaseDate = pd.to_datetime(BaseDate).to_numpy().astype('datetime64[M]')
    StartDate = pd.to_datetime(StartDate).to_numpy().astype('datetime64[M]')
    return int(((StartDate - (BaseDate - np.timedelta64(1, 'M'))) / np.timedelta64(1, 'M')) - 1)

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
varps_decline = np.vectorize(arps_decline)

# Function to recalculate Arps decline parameters based on a future date
def arps_roll_forward(BaseDate, StartDate, UID, phase, Qi, Dei, Def, b):
    t = max(MonthDiff(BaseDate, StartDate), 0)
    return varps_decline(UID, phase, Qi, Dei, Def, b, np.array([t]))

# Function to create a pivoted dataframe containing EUR values for each measure
def pivot_and_process(df):
    # Pivot the DataFrame
    pivoted_df = df.pivot_table(
        index=['WellID', 'DataSource'], 
        columns='Measure', 
        values='EUR', 
        aggfunc='mean'
    )

    # Rename the columns
    pivoted_df.columns = [f'EUR for {col}' for col in pivoted_df.columns]

    # Check and add missing columns for OIL, GAS, WATER
    for measure in ['OIL', 'GAS', 'WATER']:
        column_name = f'EUR for {measure}'
        if column_name not in pivoted_df.columns:
            pivoted_df[column_name] = 0.0

    return pivoted_df.reset_index()

# df is an input parameter from Spotfire and is the dbo.vw_FORECAST view from Analytics database
# Roll forward Qi and Dei
df[['Qi_rf', 'Dei_rf']] = df.apply(lambda row: arps_roll_forward(row['StartDate'], row['LastProdDate'], row['WellID'], row['PHASE_INT'], row['Q3'], row['Dei'], row['Def'], row['b_factor'])[3:5], axis=1, result_type='expand')

# Calculate EUR and add it to the DataFrame
df['EUR'] = df.apply(lambda row: varps_decline(row['WellID'], row['PHASE_INT'], row['Qi_rf'], row['Dei_rf'], row['Def'], row['b_factor'], 360, row['CumulativeProduction'], MonthDiff(row['FirstProdDate'], row['LastProdDate']))[5], axis=1)

# Ensure that EUR column contains scalar values, not arrays
df['EUR'] = df['EUR'].apply(lambda x: x if np.isscalar(x) else x[0])

# Pivot the dataframe using the pivot_and_process function
pivoted_df = pivot_and_process(df)
