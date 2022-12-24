import numpy as np
import pandas as pd
import xlwings as xw
import scipy.optimize as optimize
from scipy.interpolate import interp1d

def arps_decline(UID, phase, Qi, Dei, Def, b, t, prior_cum, prior_t):
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


# Calculate Dei from Qi and Qf based on exponential decline equation
def exp_Dei(Qi, Qf, duration):
    # Qi is the initial production rate typically in bbl/day or Mcf/day
    # Qf is the final production rate typically in bbl/day or Mcf/day
    # duration is the time interval in months over which you are trying to calculate the exponential decline rate
    Dei = 1 - np.exp(-np.log(Qi / Qf) / (duration / 12))
    return Dei


# Function to manage multiple segments
# Segment 1 is the initial incline period and uses Arps exponential equation
# Segment 2 is the period between the incline and decline periods and uses Arps exponential equation
# Segment 3 is the decline period
def arps_segments(UID, phase, Q1, Q2, Q3, Dei, Def, b, t1, t2, duration):
    # Vectorize the arps_decline function to allow it to work with numpy arrays
    varps_decline = np.vectorize(arps_decline)
    
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


# Create MonthDiff function to calculate time difference in months
def MonthDiff(BaseDate, StartDate):
    BaseDate = np.datetime64(BaseDate, 'M')
    StartDate = np.datetime64(StartDate, 'M')
    MonthDiff = int(((StartDate - (BaseDate - np.timedelta64(1, 'M'))) / np.timedelta64(1, 'M')) - 1)
    return MonthDiff


# function to calculate monthly cash flow output, no economic truncation applied to output
def monthly_cf(index, uid, wi, nri, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, pri_gas, paj_oil, 
               paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, opc_gas, capex, volumes):
    # Calculate cash flow components
    mos = np.extract(volumes[index][0] == uid, volumes[index][1])
    gr_oil = np.extract(volumes[index][0] == uid, volumes[index][2])
    gr_gas = np.extract(volumes[index][0] == uid, volumes[index][3])
    oil_sales = gr_oil * nri * weight * prod_wt
    gas_sales = gr_gas * nri * shrink * weight * prod_wt
    ngl_sales = ngl_yield * gr_gas * nri * weight * prod_wt
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
    opex = np.where(mos >= FirstProd, (wi * weight * inv_wt) * ((gr_oil * prod_wt * opc_oil) + (gr_gas * prod_wt * shrink * opc_gas) + opc_fix), 0)
    op_cf = total_rev - total_tax - opex
    net_cf = np.where(mos == FirstProd, op_cf - (capex * weight * inv_wt * wi), op_cf)
    cum_opcf = np.cumsum(op_cf)
    cum_ncf = np.cumsum(net_cf)

    # create output array
    out_full = np.vstack((volumes[index][:, volumes[index][0] == uid], oil_sales, gas_sales, ngl_sales, oil_rev, gas_rev, 
                          ngl_rev, total_rev, total_tax, opex, op_cf, cum_opcf, net_cf, cum_ncf))
    return out_full


# function to calculate monthly cash flow output with economic truncation
def econ_cf(index, uid, wi, nri, roy, eloss, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, pri_gas, paj_oil, paj_gas,
            paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, opc_gas, capex, aban, volumes):
    if wi == 0 or nri / wi > (1 - roy):
        life_cf = monthly_cf(index, uid, 1, 1 - roy, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, 
                             pri_gas, paj_oil, paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, 
                             opc_gas, capex, volumes)
        try:
            life = np.where(life_cf[14] == np.max(life_cf[14]))[0][0]
        except:
            life = 1
    else:
        life_cf = monthly_cf(index, uid, wi, nri, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, 
                             pri_gas, paj_oil, paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, 
                             opc_gas, capex, volumes)
        try:
            life = np.where(life_cf[14] == np.max(life_cf[14]))[0][0]
        except:
            life = 1
    
    result = monthly_cf(index, uid, wi, nri, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, pri_gas, paj_oil, 
                        paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, opc_gas, capex, volumes)
    
    # add abandonment costs to end of life
    net_cf = np.where(result[1] == life + eloss, result[15] - (aban * weight * inv_wt * wi), result[15])
    cum_ncf = np.cumsum(net_cf)
    
    # replace net cf array
    result = np.vstack((result[0:15], net_cf, cum_ncf))
    
    # zero out values past life
    monthly_econ = np.where(result[1] <= life + eloss, result[:], np.vstack((result[0:2], result[2:] * 0)))
    return monthly_econ


# function to calculate npv, assumes monthly cash flow and annualized discount rate
def npv(disc_rate, ncf, n):
    # disc_rate is an annualized discount rate
    # ncf is the monthly cashflow array
    # n is the array of month integers
    rate = disc_rate / 12 # Corrects effective discount rate from annual to monthly for discounting
    factor = 1 / ((1 + rate) ** -0.5) # Adjusts discounting to mid-period similar to Aries output
    pv = np.sum(ncf / ((1 + rate) ** n)) * factor
    return pv

def calc_dcf():
    # Import data for decline parameters
    wb = xw.Book.caller()
    prop_list = (
        wb 
        .sheets('Property Editor')
        .range('A1')
        .options(pd.DataFrame,index=False, expand='table')
        .value
    )
    prop_list = prop_list.query('INCLUDE == 1')
    prop_list = prop_list.fillna(0)
    prop_list['Start_Date'] = pd.to_datetime(prop_list['Start_Date'])
    prop_list.reset_index(drop = True, inplace = True)
    R = prop_list.index # Rows
    
    # Loop through DataFrame and output monthly oil volumes
    duration = int(wb.sheets('Input Settings').range('B3').value)

    oil = lambda w: arps_segments(prop_list.loc[w, 'UID'], 1, prop_list.loc[w, 'OIL_IP'], prop_list.loc[w, 'OIL_IP2'], 
                                  prop_list.loc[w, 'OIL_IP3'], prop_list.loc[w, 'OIL_DI'] / 100, prop_list.loc[w, 'OIL_DEF'] / 100, 
                                  prop_list.loc[w, 'OIL_B'], prop_list.loc[w, 'OIL_SEG1_TIME'], prop_list.loc[w, 'OIL_SEG2_TIME'], 
                                  duration)
    v_oil = np.vectorize(oil, otypes = [object])
    oil_nparr = v_oil(R)

    # Loop through DataFrame and output monthly gas volumes

    gas = lambda w: arps_segments(prop_list.loc[w, 'UID'], 1, prop_list.loc[w, 'GAS_IP'], prop_list.loc[w, 'GAS_IP2'], 
                                  prop_list.loc[w, 'GAS_IP3'], prop_list.loc[w, 'GAS_DI'] / 100, prop_list.loc[w, 'GAS_DEF'] / 100, 
                                  prop_list.loc[w, 'GAS_B'], prop_list.loc[w, 'GAS_SEG1_TIME'], prop_list.loc[w, 'GAS_SEG2_TIME'], 
                                  duration)
    v_gas = np.vectorize(gas, otypes = [object])
    gas_nparr = v_gas(R)
    
    # Import price files
    BaseDate = np.datetime64(wb.sheets('Input Settings').range('B2').value)
    MaxDate = pd.to_datetime(prop_list['Start_Date']).max()
    MaxDate_np = MaxDate.to_datetime64()
    add_months = int(round((MaxDate - BaseDate) / np.timedelta64(1, 'M'), 0)) + 1
    str_periods = duration + add_months
    dates = pd.DataFrame(pd.date_range(BaseDate, periods = str_periods, freq = 'MS'), columns = ['Date'])
    strip_price = wb.sheets('Pricing Editor').range('A2').options(pd.DataFrame,index=False, expand='table').value
    strip_price['Date'] = pd.to_datetime(strip_price['Date'])

    # Create diff array
    gasdiff_pd = wb.sheets('Pricing Editor').range('F2').options(pd.DataFrame,index=False, expand='table').value
    gasdiff_pd['Date'] = pd.to_datetime(gasdiff_pd['Date'])

    # Create numpy arrays for cash flow calcs
    strip_price = pd.merge(dates, strip_price, how = 'left', on = 'Date')
    strip_price = pd.merge(strip_price, gasdiff_pd, how = 'left', on = 'Date')
    strip_price.fillna(method = 'ffill', inplace = True)
    strip_price['PAJ/GAS'].fillna(method = 'bfill', inplace = True)
    oil_price_full = np.transpose(strip_price[['Oil Price']].to_numpy())[0]
    gas_price_full = np.transpose(strip_price[['Gas Price']].to_numpy())[0]
    gasdiff_full = np.transpose(strip_price[['PAJ/GAS']].to_numpy())[0]
    
    # Generate price and volume arrays with proper indexing
    def oil_price(x):
        # Create shift integer for arrays
        StartDate = prop_list.loc[x, 'Start_Date']
        DateDiff = MonthDiff(BaseDate, StartDate)
        
        # Shift price arrays
        uid = np.full((duration + DateDiff,), oil_nparr[x][0][0])
        oil_pri = oil_price_full[:duration + DateDiff]
        oil_price = np.vstack((uid, oil_pri))
        return oil_price

    voil_price = np.vectorize(oil_price, otypes = [object])                         
    oilprice = voil_price(R)

    def gas_price(x):
        # Create shift integer for arrays
        StartDate = prop_list.loc[x, 'Start_Date']
        DateDiff = MonthDiff(BaseDate, StartDate)
        
        # Shift price arrays
        uid = np.full((duration + DateDiff,), gas_nparr[x][0][0])
        gas_pri = gas_price_full[:duration + DateDiff]
        gas_price = np.vstack((uid, gas_pri))
        return gas_price

    vgas_price = np.vectorize(gas_price, otypes = [object])                         
    gasprice = vgas_price(R)

    def gas_diff(x):
        # Create shift integer for arrays
        StartDate = prop_list.loc[x, 'Start_Date']
        DateDiff = MonthDiff(BaseDate, StartDate)
        
        # Shift price arrays
        uid = np.full((duration + DateDiff,), gas_nparr[x][0][0])
        pepl = gasdiff_full[:duration + DateDiff]
        gas_diff = np.vstack((uid, pepl))
        return gas_diff

    vgas_diff = np.vectorize(gas_diff, otypes = [object])                         
    gasdiff = vgas_diff(R)

    def volarray(x):
        uid = oil_nparr[x][0]
        StartDate = prop_list.loc[x, 'Start_Date']
        DateDiff = MonthDiff(BaseDate, StartDate)
        
        # create well specific volume arrays
        month = np.round(oil_nparr[x][2] + DateDiff - 1, 0)
        oil_vol = np.round(oil_nparr[x][6], 4)
        gas_vol = np.round(gas_nparr[x][6], 4)
        nvol_np = np.vstack((uid, month, oil_vol, gas_vol))
        prop = np.full((DateDiff,), uid[0])
        delay = np.arange(DateDiff)
        oil_zeros = np.zeros((DateDiff),)
        gas_zeros = oil_zeros
        shift = np.vstack((prop, delay, oil_zeros, gas_zeros))
        vol_arr = np.column_stack((shift, nvol_np)) 
        return vol_arr

    vvolarray = np.vectorize(volarray, otypes = [object])
    vol_np = vvolarray(R)
    
    # Generate full monthly cash flow arrays  
    # define constant input parameters
    eloss = int(wb.sheets('Input Settings').range('B5').value)
    weight = wb.sheets('Input Settings').range('B6').value
    prod_wt = wb.sheets('Input Settings').range('B7').value
    inv_wt = wb.sheets('Input Settings').range('B8').value

    # Create function for slicing the volume array and calculating the monthly cash flow
    def econ_ncf_iter(r):    
        econ_ncf_iter = econ_cf(
            index = r,
            uid = prop_list.loc[r, 'UID'],
            wi = prop_list.loc[r, 'WI'],
            nri = prop_list.loc[r, 'NRI'],
            roy = prop_list.loc[r, 'Royalty'],
            eloss = eloss,
            weight = weight,
            prod_wt = prod_wt,
            inv_wt = inv_wt,
            shrink = np.round(prop_list.loc[r, 'SHRINK'] / 100, 6),
            btu = np.round(prop_list.loc[r, 'BTU'] / 1000, 6),
            ngl_yield = np.round(prop_list.loc[r, 'NGL/GAS'], 6),
            pri_oil = np.extract(oilprice[r][0] == prop_list.loc[r, 'UID'], oilprice[r][1]),
            pri_gas = np.extract(gasprice[r][0] == prop_list.loc[r, 'UID'], gasprice[r][1]),
            paj_oil = prop_list.loc[r, 'PAJ/OIL'],
            paj_gas = np.extract(gasdiff[r][0] == prop_list.loc[r, 'UID'], gasdiff[r][1]),
            paj_ngl = prop_list.loc[r, 'PAJ/NGL'],
            stx_oil = prop_list.loc[r, 'STX/OIL'],
            stx_gas = prop_list.loc[r, 'STX/GAS'],
            stx_ngl = prop_list.loc[r, 'STX/NGL'],
            adval = prop_list.loc[r, 'ADVAL'],
            opc_fix = np.round(prop_list.loc[r, 'OPC/T'], 2),
            opc_oil = np.round(prop_list.loc[r, 'OPC/OIL'], 2),
            opc_gas = np.round(prop_list.loc[r, 'OPC/GAS'], 2),
            capex = np.round(prop_list.loc[r, 'CAPITAL'] * 1000, 2),
            aban = np.round(prop_list.loc[r, 'ABAN'] * 1000, 2),
            volumes = vol_np
        )
        return econ_ncf_iter

    # generate net cash flow array
    econ_ncf = lambda r: econ_ncf_iter(r)
    vecon_ncf = np.vectorize(econ_ncf_iter, otypes = [object])
    ncf_arr_packed = vecon_ncf(R)
    ncf_pd_dflist = []
    columns = ['UID', 'Month', 'Grs Oil', 'Grs Gas', 'Net Oil', 'Net Gas', 'Net NGL', 'Oil Revenue', 'Gas Revenue', 
               'NGL Revenue', 'Total Revenue', 'Total Tax', 'OPEX', 'Operating Income', 'Cumulative Op CF', 'Net Cashflow',
               'Cumulative Net CF']

    for r in R:
        ncf_pd_dflist.append(pd.DataFrame(np.transpose(ncf_arr_packed[r])))
    ncf_pd = pd.concat(ncf_pd_dflist)
    ncf_pd.columns = columns

    # Add a date column and reorder
    columns.insert(2, 'Date')
    ncf_pd['Date'] = pd.Timestamp(BaseDate) + ncf_pd['Month'].apply(lambda m: pd.DateOffset(months=m))
    ncf_pd = ncf_pd[columns]
    
    # Truncate pandas dataframe
    monthly_out = int(wb.sheets('Input Settings').range('B4').value)
    ncf_pd_trunc = ncf_pd.query(f'Month < {monthly_out}')
    ncf_pd_trunc = ncf_pd_trunc.drop(columns = ['Month'])
    wb.sheets('Monthly Output').range('A1').expand().clear_contents()
    wb.sheets('Monthly Output').range('A1').options(index=False, header=True).value = ncf_pd_trunc

    # loop though wells in array and create oneline output of economic metrics
    oneline_cat = prop_list.iloc[:, :9]
    disc_rate = [0.05, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    propnum = []
    ror = []
    life = []
    payout = []
    roi = []
    droi = []
    result_nparr = np.empty([6, 0])
    npv_list = []

    for r in R:
        propID = prop_list.loc[r, 'UID']
        wi = prop_list.loc[r, 'WI']
        capex = prop_list.loc[r, 'CAPITAL']
        ncf_r = np.extract(ncf_arr_packed[r][0] == propID, ncf_arr_packed[r][15])
        month_r = np.extract(ncf_arr_packed[r][0] == propID, ncf_arr_packed[r][1])
        cum_ocf_r = np.extract(ncf_arr_packed[r][0] == propID, ncf_arr_packed[r][14])
        cum_ncf_r = np.extract(ncf_arr_packed[r][0] == propID, ncf_arr_packed[r][14])
        
        # calculate npv at all discount rates
        pv_calc = lambda i: npv(i, ncf_r, month_r.astype(int))
        pv = list(map(pv_calc, disc_rate))
        npv_list.append(pv)
        
        # calculate irr
        if wi == 0 or capex == 0:
            irr = max(max(disc_rate), 1)
        else:
            f = lambda x: npv(x, ncf_r, month_r)
            r = optimize.root(f, [0])
            irr = min(r.x[0], 1)
        ror.append(irr)
        
        # calculate life
        try:
            life_calc = np.where(cum_ocf_r == np.max(cum_ocf_r))[0][0] + eloss
        except:
            life_calc = 1 + eloss
        life.append(np.round(life_calc / 12, 2))
            
        # calculate payout
        ncf_cum = cum_ncf_r[:life_calc]
        month_arr = month_r[:life_calc]
        try:
            payout_interp = interp1d(ncf_cum, month_arr, kind = 'cubic')
            payout_calc = payout_interp(0) / 12
        except:
            payout_calc = life_calc / 12
        payout.append(np.round(payout_calc, 2))
            
        # calculate ROI and DROI
        net_capex = capex * weight * inv_wt * wi
        if net_capex == 0:
            roi_calc = 0
            droi_calc = 0
        else:
            roi_calc = np.round(((net_capex + np.sum(ncf_r)) / net_capex), 2)
            droi_calc = np.round(((net_capex + pv[3]) / net_capex), 2)
        roi.append(roi_calc)
        droi.append(droi_calc)
        
        propnum.append(propID)
        oneline_nparr = np.array((propnum, ror, life, payout, roi, droi))

    result_nparr = np.column_stack((result_nparr, oneline_nparr))
    npv_pd = pd.DataFrame(npv_list, columns = disc_rate)

    # Create oneline output export to csv
    result_pd = pd.DataFrame(np.transpose(result_nparr), columns = ['UID', 'IRR', 'Life', 'Payout', 'ROI', 'DROI'])
    result_pd = pd.merge(result_pd, npv_pd, how = 'inner', left_index = True, right_index = True)

    # Calculate breakevens and add to pandas output
    equiv_ratio = wb.sheets('Input Settings').range('E3').value
    disc1 = wb.sheets('Input Settings').range('E4').value
    disc2 = wb.sheets('Input Settings').range('E5').value
    pri = 50 # breakeven price guess
    pajgas = wb.sheets('Input Settings').range('E6').value
    run_breakevens = int(wb.sheets('Input Settings').range('E2').value)

    if run_breakevens == 0:
        wb.sheets('Oneline Output').range('A1').expand().clear_contents()
        wb.sheets('Oneline Output').range('A1').options(index=False, header=True).value = result_pd
        pass

    else:
        # Create function for slicing the volume array and calculating the monthly cash flow
        def econ_be(pri, paj_gas, disc, equiv_ratio, r):    
            econ_be = econ_cf(
                index = r,
                uid = prop_list.loc[r, 'UID'],
                wi = prop_list.loc[r, 'WI'],
                nri = prop_list.loc[r, 'NRI'],
                roy = prop_list.loc[r, 'Royalty'],
                eloss = eloss, 
                weight = weight,
                prod_wt = prod_wt,
                inv_wt = inv_wt,
                shrink = np.round(prop_list.loc[r, 'SHRINK'] / 100, 6),
                btu = np.round(prop_list.loc[r, 'BTU'] / 1000, 6),
                ngl_yield = np.round(prop_list.loc[r, 'NGL/GAS'], 6),
                pri_oil = pri,
                pri_gas = pri / equiv_ratio,
                paj_oil = prop_list.loc[r, 'PAJ/OIL'],
                paj_gas = pajgas,
                paj_ngl = prop_list.loc[r, 'PAJ/NGL'],
                stx_oil = prop_list.loc[r, 'STX/OIL'],
                stx_gas = prop_list.loc[r, 'STX/GAS'],
                stx_ngl = prop_list.loc[r, 'STX/NGL'],
                adval = prop_list.loc[r, 'ADVAL'],
                opc_fix = np.round(prop_list.loc[r, 'OPC/T'], 2),
                opc_oil = np.round(prop_list.loc[r, 'OPC/OIL'], 2),
                opc_gas = np.round(prop_list.loc[r, 'OPC/GAS'], 2),
                capex = np.round(prop_list.loc[r, 'CAPITAL'] * 1000, 2),
                aban = np.round(prop_list.loc[r, 'ABAN'] * 1000, 2),
                volumes = vol_np
            )
            pv_beo = npv(disc, econ_be[15], econ_be[1])
            return pv_beo

        # calculate breakeven prices
        def econ_be_iter1(r):
            pv_beo1 = lambda p: econ_be(p, pajgas, disc1, equiv_ratio, r)
            r_beo1 = optimize.root(pv_beo1, pri)
            return round(r_beo1.x[0], 2)

        vpv_beo1 = np.vectorize(econ_be_iter1)
        beo1 = vpv_beo1(R)

        def econ_be_iter2(r):
            pv_beo2 = lambda p: econ_be(p, pajgas, disc2, equiv_ratio, r)
            r_beo2 = optimize.root(pv_beo2, pri)
            return round(r_beo2.x[0], 2)

        vpv_beo2 = np.vectorize(econ_be_iter2)
        beo2 = vpv_beo2(R)

        # Add results to oneline output
        result_pd['Oil_BE1'] = beo1
        result_pd['Oil_BE2'] = beo2
        wb.sheets('Oneline Output').range('A1').expand().clear_contents()
        wb.sheets('Oneline Output').range('A1').options(index=False, header=True).value = result_pd
