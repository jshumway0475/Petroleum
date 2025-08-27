import numpy as np

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
        - ngl_yield (float): ngl yield of the well as a decimal, fraction of gas that is ngl, units of bbl/mcf
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
    opex = np.where(mos >= FirstProd, (wi * weight * inv_wt) * ((gr_oil * prod_wt * opc_oil) + (gr_gas * prod_wt * shrink * opc_gas) + (gr_wtr * prod_wt * opc_wtr) + opc_fix), 0)
    op_cf = total_rev - total_tax - opex
    net_cf = np.where(mos == FirstProd, op_cf - (capex * weight * inv_wt * wi), op_cf)
    cum_opcf = np.cumsum(op_cf)
    cum_ncf = np.cumsum(net_cf)

    # create output array
    out_full = np.vstack((volumes[index][:, volumes[index][0] == uid], oil_sales, gas_sales, ngl_sales, oil_rev, gas_rev, 
                          ngl_rev, total_rev, total_tax, opex, op_cf, cum_opcf, net_cf, cum_ncf))
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
        - ngl_yield (float): ngl yield of the well as a decimal, fraction of gas that is ngl, units of bbl/mcf
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
        life_cf = monthly_cf(index, uid, 1, 1 - roy, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, 
                             pri_gas, paj_oil, paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, 
                             opc_gas, opc_wtr, capex, volumes)
        try:
            life = np.where(life_cf[15] == np.max(life_cf[15]))[0][0]
        except:
            life = 1
    else:
        life_cf = monthly_cf(index, uid, wi, nri, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, 
                             pri_gas, paj_oil, paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, 
                             opc_gas, opc_wtr, capex, volumes)
        try:
            life = np.where(life_cf[15] == np.max(life_cf[15]))[0][0]
        except:
            life = 1
    
    result = monthly_cf(index, uid, wi, nri, weight, prod_wt, inv_wt, shrink, btu, ngl_yield, pri_oil, pri_gas, paj_oil, 
                        paj_gas, paj_ngl, stx_oil, stx_gas, stx_ngl, adval, opc_fix, opc_oil, opc_gas, opc_wtr, capex, volumes)
    
    # add abandonment costs to end of life
    aban_month = min(life + eloss, volumes.shape[1]) - 1
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
