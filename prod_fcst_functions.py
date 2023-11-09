import numpy as np


def arps_decline(UID, phase, Qi, Dei, Def, b, t, prior_cum=0, prior_t=0):
    # UID is a unique identifier for the well such as API, must be a number
    # phase is 1 = oil, 2 = gas, or 3 = water
    # Qi is the initial production rate typically in bbl/day or Mcf/day
    # Dei is the initial effective annual decline rate
    # Def is the final effective annual decline rate at which point the decline becomes exponential
    # b is the b-factor used in hyperbolic or harmonic decline equations
    # t is the time as a month integer
    # prior_cum is the cumulative amount produced before the start of the decline calcuations
    # prior_t is an integer representing the final month from a previous decline segment

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


# Calculate Dei from Qi and Qf based on exponential decline equation
def exp_Dei(Qi, Qf, duration):
    # Qi is the initial production rate typically in bbl/day or Mcf/day
    # Qf is the final production rate typically in bbl/day or Mcf/day
    # duration is the time interval in months over which you are trying to calculate the exponential decline rate
    Dei = 1 - np.exp(-np.log(Qi / Qf) / (duration / 12))
    return Dei


# Function to manage multiple segments
def arps_segments(UID, phase, Q1, Q2, Q3, Dei, Def, b, Qabn, t1, t2, duration):
    '''
    Segment 1 is the initial incline period and uses Arps exponential equation
    Segment 2 is the period between the incline and decline periods and uses Arps exponential equation
    Segment 3 is the decline period
    '''
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
        seg1 = varps_decline(UID, phase, Q1, Dei1, Dei1, 1.0, t_seg1, 0, 0)
        prior_cum1 = np.max(seg1[5])
        seg2 = varps_decline(UID, phase, Q2, Dei2, Dei2, 1.0, t_seg2, prior_cum1, t1)
        prior_cum2= np.max(seg2[5])
        seg3 = varps_decline(UID, phase, Q3, Dei, Def, b, t_seg3, prior_cum2, t1 + t2)
        # Filter out months where production is less than Qabn
        if Qabn > 0:
            Qabn_filter = seg3[3] >= Qabn
            seg3 = seg3[:,Qabn_filter]
        out_nparr = np.column_stack((seg1, seg2, seg3))
    elif segment_ct == 2:
        t_seg1 = np.arange(0, t1 + 1, 1)
        t_seg3 = np.arange(1, duration - t1 + 1, 1)
        Dei1 = exp_Dei(Q1, Q3, t1)
        seg1 = varps_decline(UID, phase, Q1, Dei1, Dei1, 1.0, t_seg1, 0, 0)
        prior_cum1 = np.max(seg1[5])
        seg3 = varps_decline(UID, phase, Q3, Dei, Def, b, t_seg3, prior_cum1, t1)
        out_nparr = np.column_stack((seg1, seg3))
    elif segment_ct == 1:
        t_seg3 = np.arange(0, duration + 1, 1)
        out_nparr = varps_decline(UID, phase, Q3, Dei, Def, b, t_seg3, 0, 0)
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
