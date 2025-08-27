import numpy as np
import pandas as pd
import datetime
import scipy.optimize as optimize

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
def arps_segments(UID, phase, Q1, Q2, Q3, Dei, Def, b, t1, t2, duration): 
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
        out_nparr = np.vstack((UID_nan, phase_nan, t_nan, val_nan))
    
    # Add monthly volumes to array
    Cum_i = out_nparr[5][:-1]
    Cum_f = out_nparr[5][1:]
    cum = Cum_f - Cum_i
    cum[cum < 0] = 0
    cum = np.insert(cum, 0, 0)
    out_nparr = np.vstack((out_nparr, cum))[:,1:]
    
    return out_nparr
   
# function to calculate monthly cash flow output, no economic truncation applied to output
def monthly_cf(
        index, uid, wi, nri, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, pri_gas, paj_oil, 
        paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, opc_gas, opc_wtr, capex, volumes
    ):
    '''
    Args:
        - index (int): index of the well in the volumes array
        - uid (int): unique id of the well
        - wi (float): working interest of the well as a decimal
        - nri (float): net revenue interest of the well as a decimal
        - weight (float): weight of the well as a decimal, can be used for risking or fractional scheduling
        - prod_wt (float): production weight of the well as a decimal, can be used for risking production
        - inv_wt (float): investment weight of the well as a decimal, can be used for risking investment
        - shrink (float): gas shrinkage of the well as a decimal, fraction of gas remaining after processing or losses
        - btu (float): btu content of the gas as a decimal in units of mmbtu/mcf
        - ngl_yield (float): ngl yield of the well as a decimal, fraction of gas that is ngl, units of bbl/mmcf
        - pri_oil (float): price of oil in units of $/bbl
        - pri_gas (float): price of gas in units of $/mmbtu
        - paj_oil (float): price adjustment of oil in units of $/bbl
        - paj_gas (float): price adjustment of gas in units of $/mmbtu
        - paj_ngl (float): price adjustment of ngl as a fraction of the oil price
        - stx_oil (float): severance tax of oil as a decimal
        - stx_gas (float): severance tax of gas as a decimal
        - stx_ngl (float): severance tax of ngl as a decimal
        - adval (float): ad valorem tax as a decimal
        - opc_fix (float): fixed operating cost in units of $/month
        - opc_oil (float): operating cost of oil in units of $/bbl
        - opc_gas (float): operating cost of gas in units of $/mcf
        - opc_wtr (float): operating cost of water in units of $/bbl
        - capex (float): capital expenditure
        - volumes (list of arrays or nested arrays): array of monthly volumes for all wells in the project
    Returns:
        - out_full (array): array of monthly cash flow for the well
            - UID (int): unique id of the well
            - Month (int): month of the cash flow
            - Gross Oil Volume (float): gross oil volume in units of bbl
            - Gross Gas Volume (float): gross gas volume in units of mcf
            - Gross Water Volume (float): gross water volume in units of bbl
            - Oil Sales (float): net oil volumes (nri) in units of bbl
            - Gas Sales (float): net dry gas volumes (nri) in units of mcf
            - NGL Sales (float): net ngl volumes (nri) in units of bbl
            - Oil Revenue (float): net oil revenue (nri) in units of $
            - Gas Revenue (float): gas revenue (nri) in units of $
            - NGL Revenue (float): ngl revenue (nri) in units of $
            - Total Revenue (float): total revenue (nri) in units of $
            - Total Tax (float): net severance and ad valorem taxes (wi) in units of $
            - Operating Expense (float): net operating expense (wi) in units of $
            - Operating Cash Flow (float): net operating cash flow (wi), revenue - tax - opex,  in units of $
            - Cumulative Operating Cash Flow (float): cumulative net operating cash flow (wi) in units of $
            - Net Cash Flow (float): net cash flow (wi), opcf - capex, in units of $
            - Cumulative Net Cash Flow (float): cumulative net cash flow (wi) in units of $
    '''
    # Calculate cash flow components
    mos = np.extract(volumes[index][0] == uid, volumes[index][1])
    gr_oil = np.extract(volumes[index][0] == uid, volumes[index][2])
    gr_gas = np.extract(volumes[index][0] == uid, volumes[index][3])
    gr_wtr = np.extract(volumes[index][0] == uid, volumes[index][4])
    oil_sales = gr_oil * nri * weight * prod_wt
    gas_sales = gr_gas * nri * shrink * weight * prod_wt
    ngl_sales = (ngl_yield/1000) * gr_gas * nri * weight * prod_wt
    oil_rev = np.round(oil_sales * (pri_oil + paj_oil), 2)
    gas_rev = np.round(gas_sales * btu * (pri_gas + paj_gas), 2)
    ngl_rev = np.round(ngl_sales * (pri_oil * paj_ngl), 2)
    total_rev = oil_rev + gas_rev + ngl_rev
    oil_tax = oil_rev * stx_oil
    gas_tax = gas_rev * stx_gas
    ngl_tax = ngl_rev * stx_ngl
    adval_tax = total_rev * adval
    total_tax = oil_tax + gas_tax + ngl_tax + adval_tax
    FirstProd = np.min(np.nonzero(gr_oil + gr_gas))
    opex = np.where(mos >= FirstProd, (wi * weight) * ((gr_oil * prod_wt * opc_oil) + (gr_gas * prod_wt * shrink * opc_gas) + (gr_wtr * prod_wt * opc_wtr) + opc_fix), 0)
    op_cf = total_rev - total_tax - opex
    net_cf = np.where(mos == FirstProd, op_cf - (capex * weight * inv_wt * wi), op_cf)
    cum_opcf = np.cumsum(op_cf)
    cum_ncf = np.cumsum(net_cf)

    # create output array
    out_full = np.vstack(
        (volumes[index][:, volumes[index][0] == uid], oil_sales, gas_sales, ngl_sales, oil_rev, gas_rev, ngl_rev, total_rev, total_tax, opex, op_cf, cum_opcf, net_cf, cum_ncf)
    )
    return out_full

# function to calculate monthly cash flow output with economic truncation
def econ_cf(
        index, uid, wi, nri, roy, eloss, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, pri_gas, paj_oil, paj_gas,
        paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, opc_gas, opc_wtr, capex, aban, volumes
    ):
    '''
    Args:
        - index (int): index of the well in the volumes array
        - uid (int): unique id of the well
        - wi (float): working interest of the well as a decimal
        - nri (float): net revenue interest of the well as a decimal
        - roy (float): lease royalty as a decimal
        - eloss (int): economic loss of the well as an integer, number of months to continue producing after economic limit
        - weight (float): weight of the well as a decimal, can be used for risking or fractional scheduling
        - prod_wt (float): production weight of the well as a decimal, can be used for risking production
        - inv_wt (float): investment weight of the well as a decimal, can be used for risking investment
        - shrink (float): gas shrinkage of the well as a decimal, fraction of gas remaining after processing or losses
        - btu (float): btu content of the gas as a decimal in units of mmbtu/mcf
        - ngl_yield (float): ngl yield of the well as a decimal, fraction of gas that is ngl, units of bbl/mmcf
        - pri_oil (float): price of oil in units of $/bbl
        - pri_gas (float): price of gas in units of $/mmbtu
        - paj_oil (float): price adjustment of oil in units of $/bbl
        - paj_gas (float): price adjustment of gas in units of $/mmbtu
        - paj_ngl (float): price adjustment of ngl as a fraction of the oil price
        - stx_oil (float): severance tax of oil as a decimal
        - stx_gas (float): severance tax of gas as a decimal
        - stx_ngl (float): severance tax of ngl as a decimal
        - adval (float): ad valorem tax as a decimal
        - opc_fix (float): fixed operating cost in units of $/month
        - opc_oil (float): operating cost of oil in units of $/bbl
        - opc_gas (float): operating cost of gas in units of $/mcf
        - opc_wtr (float): operating cost of water in units of $/bbl
        - capex (float): capital expenditure
        - aban (float): abandonment cost, applied at the end of life
        - volumes (array): array of monthly volumes for all wells in the project
    Returns:
        - monthly_econ (array): array of monthly cash flow for the well with economic truncation
            - UID (int): unique id of the well
            - Month (int): month of the cash flow
            - Gross Oil Volume (float): gross oil volume in units of bbl
            - Gross Gas Volume (float): gross gas volume in units of mcf
            - Gross Water Volume (float): gross water volume in units of bbl
            - Oil Sales (float): net oil volumes (nri) in units of bbl
            - Gas Sales (float): net dry gas volumes (nri) in units of mcf
            - NGL Sales (float): net ngl volumes (nri) in units of bbl
            - Oil Revenue (float): net oil revenue (nri) in units of $
            - Gas Revenue (float): gas revenue (nri) in units of $
            - NGL Revenue (float): ngl revenue (nri) in units of $
            - Total Revenue (float): total revenue (nri) in units of $
            - Total Tax (float): net severance and ad valorem taxes (wi) in units of $
            - Operating Expense (float): net operating expense (wi) in units of $
            - Operating Cash Flow (float): net operating cash flow (wi), revenue - tax - opex,  in units of $
            - Cumulative Operating Cash Flow (float): cumulative net operating cash flow (wi) in units of $
            - Net Cash Flow (float): net cash flow (wi), opcf - capex, in units of $
            - Cumulative Net Cash Flow (float): cumulative net cash flow (wi) in units of $
    '''
    if wi == 0 or nri / wi > (1 - roy):
        life_cf = monthly_cf(
            index, uid, 1, 1 - roy, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, 
            pri_gas, paj_oil, paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, 
            opc_oil, opc_gas, opc_wtr, capex, volumes
        )
        try:
            life = np.where(life_cf[15] == np.max(life_cf[15]))[0][0]
        except:
            life = 1
    else:
        life_cf = monthly_cf(
            index, uid, wi, nri, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, 
            pri_gas, paj_oil, paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, 
            opc_oil, opc_gas, opc_wtr, capex, volumes
        )
        try:
            life = np.where(life_cf[15] == np.max(life_cf[15]))[0][0]
        except:
            life = 1
    
    result = monthly_cf(
        index, uid, wi, nri, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, 
        pri_gas, paj_oil, paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, 
        opc_oil, opc_gas, opc_wtr, capex, volumes
    )
    
    # add abandonment costs to end of life
    aban_month = min(life + eloss, volumes[0].shape[1]) - 1

    net_cf = np.where(result[1] == aban_month, result[16] - (aban * weight * inv_wt * wi), result[16])
    cum_ncf = np.cumsum(net_cf)
    
    # replace net cf array
    result = np.vstack((result[0:16], net_cf, cum_ncf))
    
    # zero out values past life
    monthly_econ = np.where(result[1] <= aban_month, result[:], np.vstack((result[0:2], result[2:] * 0)))
    return monthly_econ

# function to calculate npv, assumes monthly cash flow and annualized discount rate
def npv(disc_rate, ncf, n):
    '''
    Mid-period NPV for 0-based month indices.
    Args:
        - disc_rate (float): annualized discount rate as a decimal
        - ncf (array): array of monthly cash flow
        - n (array): array of month integers
    Returns:
        - pv (float): present value of the cash flow
    '''
    ncf = np.asarray(ncf, dtype=float)
    n = np.asarray(n, dtype=float)
    mask = ~np.isnan(ncf)
    exponents = (n[mask] + 0.5) / 12.0
    return np.sum(ncf[mask] / ((1.0 + disc_rate) ** exponents))

'''
q1, q2, q3, dei, def, b, t1, t2, duration, wi, nri, roy, eloss, weight, prod_wt, inv_wt, shrink, btu, 
ngl_yield, pri_oil, pri_gas, paj_oil, paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix,
opc_oil, opc_gas, opc_wtr, capex, aban, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, eloss,
duration, pri_disc, equiv_ratio are all document properties configured in Spotfire and are set as input parameters
'''
# Generate volume arrays from arps parameters
oil_arr = arps_segments(1, 1, q1_oil, q2_oil, q3_oil, dei_oil, def_oil, b_oil, t1_oil, t2_oil, duration)
gas_arr = arps_segments(1, 2, q1_gas, q2_gas, q3_gas, dei_gas, def_gas, b_gas, t1_gas, t2_gas, duration)
wtr_arr = arps_segments(1, 3, q1_wtr, q2_wtr, q3_wtr, dei_wtr, def_wtr, b_wtr, t1_wtr, t2_wtr, duration)

# Combine volume arrays
def volarray():
    uid = oil_arr[0]
    
    # create well specific volume arrays
    month = np.round(oil_arr[2] - 1, 0)
    oil_vol = np.round(oil_arr[6], 4)
    gas_vol = np.round(gas_arr[6], 4)
    wtr_vol = np.round(wtr_arr[6], 4)
    nvol_np = np.vstack((uid, month, oil_vol, gas_vol, wtr_vol))
    nvol_np = nvol_np[:, np.where(nvol_np[1] >= 0)[0]]
    return nvol_np
    
vol_arr = (volarray())

# Calculate monthly cash flow and add to a pandas dataframe
econ_ncf = econ_cf(
        0, 1, wi, nri, roy, eloss, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, pri_gas, 
        paj_oil, paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, opc_gas, opc_wtr, 
        capex, aban, [vol_arr]
    )

# Extract some useful arrays
month = econ_ncf[1]
cum_ocf = econ_ncf[15]
ncf = econ_ncf[16]
cum_ncf = econ_ncf[17]

# Create a dataframe
columns = [
    'UID', 'Month', 'Grs Oil', 'Grs Gas', 'Grs Wtr', 
    'Net Oil', 'Net Gas', 'Net NGL', 'Oil Revenue', 
    'Gas Revenue', 'NGL Revenue', 'Total Revenue', 
    'Total Tax', 'OPEX', 'EBITDA', 'Cumulative EBITDA', 
    'Net Cashflow', 'Cumulative Net CF'
]

ncf_df = pd.DataFrame(econ_ncf.T, columns=columns)

# Add a date column and reorder
today = datetime.date.today()
BaseDate = pd.Timestamp(today.year, today.month, today.day)

# Increment the month, adjust the year if necessary
if BaseDate.month == 12:
    BaseDate = pd.Timestamp(BaseDate.year + 1, 1, 1)
else:
    BaseDate = pd.Timestamp(BaseDate.year, BaseDate.month + 1, 1)

columns.insert(1, 'Date')
ncf_df['Date'] = BaseDate + ncf_df['Month'].apply(lambda m: pd.DateOffset(months=m))
ncf_df = ncf_df[columns].drop(columns=['UID'])

# Apply weight to gross volumes
ncf_df['Grs Oil'] = ncf_df['Grs Oil'] * weight * prod_wt
ncf_df['Grs Gas'] = ncf_df['Grs Gas'] * weight * prod_wt
ncf_df['Grs Wtr'] = ncf_df['Grs Wtr'] * weight * prod_wt

# Calculate economic metrics
npv_list = []

# NPV values at all discount rates
disc_rate = [0.05, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
pv_calc = lambda i: npv(i, ncf, month.astype(int))
pv = list(map(pv_calc, disc_rate))
npv_list.append(pv)

# calculate irr
if wi == 0 or capex == 0:
    irr = max(max(disc_rate), 1)
else:
    f = lambda x: npv(x, ncf, month.astype(int))
    r = optimize.root(f, [0])
    irr = max(r.x[0], 0)

# calculate life
try:
    life_calc = np.clip(np.where(cum_ocf == np.max(cum_ocf))[0][0] + eloss, 0, duration)
except:
    life_calc = 1 + eloss
life = np.round(life_calc / 12, 2)

# Filter ncf_df to remove trailing 0's
ncf_df = ncf_df[(ncf_df['Month'] <= life_calc) & (ncf_df['Total Revenue'] != 0.0)]

# calculate payout
ncf_cum = cum_ncf[:life_calc]
month_arr = month[:life_calc]
try:
    payout_interp = np.interp(0, ncf_cum, month_arr)
    payout_calc = payout_interp / 12
except:
    payout_calc = life_calc / 12
payout = np.round(payout_calc, 2)

# calculate ROI and DROI
net_capex = capex * weight * inv_wt * wi
if net_capex == 0:
    roi_calc = 0
    droi_calc = 0
else:
    roi_calc = np.round(((net_capex + np.sum(ncf)) / net_capex), 2)
    droi_calc = np.round(((net_capex + pv[3]) / net_capex), 2)

# Calculate oil and gas EUR
gr_oil_eur = sum(econ_ncf[5])/nri
gr_gas_eur = sum(econ_ncf[6])/nri/shrink

# Return NPV calcs to a dataframe
npv_df = pd.DataFrame(npv_list, columns = disc_rate)
npv_df = npv_df.T.reset_index().rename(columns={'index': 'Disc Rate', 0: 'NPV'})

# Calculate breakeven oil price
def econ_be(pri, paj_gas, disc, equiv_ratio):    
    econ_be_cf = econ_cf(
        0, 1, wi, nri, roy, eloss, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri, pri / equiv_ratio, 
        paj_oil, paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, opc_gas, opc_wtr,
        capex, aban, [vol_arr]
    )
    pv_beo = npv(disc, econ_be_cf[16], econ_be_cf[1].astype(int))
    return pv_beo

oil_breakeven = round(optimize.root(lambda p: econ_be(p, paj_gas, pri_disc, equiv_ratio), 50.0).x[0], 2)
