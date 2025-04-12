import numpy as np


# Create MonthDiff function to calculate time difference in months
def month_diff(BaseDate, StartDate):
    BaseDate = np.datetime64(BaseDate, 'M')
    StartDate = np.datetime64(StartDate, 'M')
    MonthDiff = int(((StartDate - (BaseDate - np.timedelta64(1, 'M'))) / np.timedelta64(1, 'M')) - 1)
    return MonthDiff


# function to calculate monthly cash flow output, no economic truncation applied to output
def monthly_cf(
        index, uid, wi, nri, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, pri_gas, paj_oil, 
        paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, opc_gas, capex, volumes
    ):
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
    out_full = np.vstack(
        (volumes[index][:, volumes[index][0] == uid], 
         oil_sales, gas_sales, ngl_sales, oil_rev, gas_rev, ngl_rev, total_rev, total_tax, opex, op_cf, cum_opcf, net_cf, cum_ncf)
    )
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
