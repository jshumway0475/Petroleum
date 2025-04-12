import numpy as np
import pandas as pd

def fcstVolume(fcst_df):
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

    # Fill null values with 0 in fcst_df
    fill_cols = ['StartMonth', 'Q1', 'Q2', 'Q3', 'Qabn', 'Dei', 'b_factor', 'Def', 't1', 't2']
    fcst_df[fill_cols] = fcst_df[fill_cols].fillna(0.0)

    # Function to apply arps_segments function to each row of a dataframe
    def apply_arps(row, duration):
        '''
        Apply arps_segments function to each row of a dataframe.
        :param row: A row from a dataframe
        '''
        # Dictionary for mapping PHASE_INT to a measure
        reverse_phase_dict = {1: 'OIL', 2: 'GAS', 3: 'WATER'}
        
        # Ensure StartMonth < duration
        if (row['StartMonth'] >= duration) | (row['Q3'] <= row['Qabn']):
            # Create a DataFrame with a single row of default values
            data = {
                'WellID': [row['WellID']],
                'Measure': [reverse_phase_dict.get(row['PHASE_INT'], 'UNKNOWN')],
                'ProdMonth': [row['StartMonth']],
                'ProductionRate': [None],
                'De': [None],
                'CumulativeProduction': [row['StartCumulative']],
                'MonthlyVolume': [None],
                'ForecastID': [row['ForecastID']],
                'StartDate': [row['StartDate']],
                'StartMonth': [row['StartMonth']]
            }
            df = pd.DataFrame(data)
        else:
            # Otherwise, apply arps_segments function
            arr = arps_segments(
                row['WellID'], 
                row['PHASE_INT'],
                row['Q1'], 
                row['Q2'], 
                row['Q3'], 
                row['Dei'], 
                row['Def'], 
                round(row['b_factor'], 4), 
                row['Qabn'],
                row['t1'],
                row['t2'],
                duration,
                row['StartCumulative'],
                row['StartMonth']
            )
            df = pd.DataFrame(np.stack(arr).T, columns=['WellID', 'Measure', 'ProdMonth', 'ProductionRate', 'De', 'CumulativeProduction', 'MonthlyVolume'])
            df = df.dropna(subset=['ProdMonth'])
            df['Measure'] = df['Measure'].map(reverse_phase_dict)
        df['ForecastID'] = row['ForecastID']
        df['StartDate'] = row['StartDate']
        df['StartMonth'] = row['StartMonth']
        df[['WellID', 'ProdMonth', 'StartMonth']] = df[['WellID', 'ProdMonth', 'StartMonth']].astype('int64')
        return df
        
    # Apply arps_segments function to each row of the dataframe
    monthly_df = pd.concat([apply_arps(row, 360) for _, row in fcst_df.iterrows()], ignore_index=True)

    # Modify monthly_df to add a column to help calculate a Date column in SQL and resort columns
    monthly_df['AdjustedMonth'] = monthly_df['ProdMonth'] - monthly_df['StartMonth']

    # Create a date column from AdjustedMonth and StartDate
    monthly_df['Date'] = monthly_df.apply(lambda x: x['StartDate'] + pd.DateOffset(months=x['AdjustedMonth']), axis=1)

    # Ensure that the Date is the end of the month
    monthly_df['Date'] = monthly_df['Date'] + pd.offsets.MonthEnd(0)

    col_order = ['ForecastID', 'WellID', 'Measure', 'Date', 'ProdMonth', 'ProductionRate', 'De', 'CumulativeProduction', 'MonthlyVolume']
    monthly_df = monthly_df[col_order]
    
    return monthly_df

# fcst_df is an input parameter from Spotfire and is the dbo.vw_FORECAST view from Analytics database    
monthly_df = fcstVolume(fcst_df)
