import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve

'''
Collection of published empirical equations that can be used to calculate important crude oil and
natural gas fluid properties such as viscosity, compressibility, and formation volume factor. These
properties are important for reservoir engineering calculations such as material balance, well test
analysis, and production forecasting.
'''

# Helper function to convert input to float, returns default value if conversion fails
def _to_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        v = float(x)
        if np.isnan(v):
            return float(default)
        return v
    except Exception:
        return float(default)

# Calculation of pseudo critical Temperature for Natural Gases, units in deg R (McCain Petroleum Fluids, pp. 120, 512)
def Tpc(sg, co2=0.0, h2s=0.0):
    '''
    Args:
        sg: specific gravity of the gas, air = 1
        co2: mole fraction of CO2 in the gas
        h2s: mole fraction of H2S in the gas

    Returns:
        Tpc: pseudo critical temperature in deg R
    '''
    a_fac = (co2 + h2s)
    tc_adj = 120 * ((a_fac ** 0.9) - (a_fac ** 1.6)) + 15 * ((h2s ** 0.5) * (h2s ** 4))
    Tpc1 = 169.2 + 349.5 * sg - 74.0 * (sg ** 2)
    Tpc = Tpc1 - tc_adj
    return Tpc

# Calculation of pseudo critical Pressure for Natural Gases, units in psia (McCain Petroleum Fluids, pp. 120, 512)
def Ppc(sg, co2=0.0, h2s=0.0):
    '''
    Args:
        sg: specific gravity of the gas, air = 1
        co2: mole fraction of CO2 in the gas
        h2s: mole fraction of H2S in the gas

    Returns:
        Ppc: pseudo critical pressure in psia
    '''
    a_fac = (co2 + h2s)
    tc_adj = 120 * ((a_fac ** 0.9) - (a_fac ** 1.6)) + 15 * ((h2s ** 0.5) * (h2s ** 4))
    Tpc1 = 169.2 + 349.5 * sg - 74.0 * (sg ** 2)
    Ppc1 = 756.8 - 131.0 * sg - 3.6 * (sg ** 2)
    Ppc = (Ppc1 * Tpc(sg, co2, h2s)) / (Tpc1 + h2s * (1 - h2s) * tc_adj)
    return Ppc

# Calculation of z-factor, unitless (McCain Petroleum Fluids, pp. 111, 510)
def z_factor(p, t, sg, co2=0.0, h2s=0.0, tolerance=1e-5, max_iterations=1000):
    '''
    Args:
        p: pressure in psia
        t: temperature in deg F
        sg: specific gravity of the gas, air = 1
        co2: mole fraction of CO2 in the gas
        h2s: mole fraction of H2S in the gas
        tolerance: tolerance for convergence
        max_iterations: maximum number of iterations

    Returns:
        z_factor: compressibility factor, unitless
    '''
    # Constants for the calculations
    a1 = 0.3265
    a2 = -1.07
    a3 = -0.5339
    a4 = 0.01569
    a5 = -0.05165
    a6 = 0.5475
    a7 = -0.7361
    a8 = 0.1844
    a9 = 0.1056
    a10 = 0.6134
    a11 = 0.7210

    # Calculate the pseudoreduced temperature and pressure
    Tpr = (t + 460) / Tpc(sg, co2, h2s)
    Ppr = p / Ppc(sg, co2, h2s)

    # Initialize Rpr using the given formula
    Rpr = 0.27 * (Ppr / Tpr)

    # Initialize variables for the while loop
    relative_change = float('inf')
    iteration = 0

    while relative_change > tolerance and iteration < max_iterations:
        z1 = 1 + ((a1 + (a2 / Tpr) + (a3 / Tpr**3) + (a4 / Tpr**4) + (a5 / Tpr**5)) * Rpr)
        z2 = (a6 + (a7 / Tpr) + (a8 / Tpr**2)) * Rpr**2
        z3 = -a9 * ((a7 / Tpr) + (a8 / Tpr**2)) * Rpr**5
        z4 = a10 * (1 + a11 * Rpr**2) * (Rpr**2 / Tpr**3) * np.exp(-a11 * Rpr**2)
        z_factor = z1 + z2 + z3 + z4

        # Update Rpr and calculate the relative change
        Rpr_old = Rpr
        Rpr = 0.27 * (Ppr / (z_factor * Tpr))
        relative_change = abs((Rpr - Rpr_old) / Rpr)

        iteration += 1

    return z_factor

# Calculation of compressibility of natural gas, units of 1/psia (McCain Petroleum Fluids, pp. 176, 512 - 513)
def gas_compressibility(p, t, sg, co2=0.0, h2s=0.0):
    '''
    Args:
        p: pressure in psia
        t: temperature in deg F
        sg: specific gravity of the gas, air = 1
        co2: mole fraction of CO2 in the gas
        h2s: mole fraction of H2S in the gas

    Returns:
        cg: compressibility of the gas in 1/psia
    '''
    a1 = 0.3265
    a2 = -1.07
    a3 = -0.5339
    a4 = 0.01569
    a5 = -0.05165
    a6 = 0.5475
    a7 = -0.7361
    a8 = 0.1844
    a9 = 0.1056
    a10 = 0.6134
    a11 = 0.7210
    Tpr = (t + 460) / Tpc(sg, co2, h2s)
    Ppr = p / Ppc(sg, co2, h2s)
    z = z_factor(p, t, sg, co2, h2s)
    Rpr = 0.27 * (Ppr / (z *Tpr))
    dcpr1 = a1 + (a2 / Tpr) + (a3 / (Tpr ** 3)) + (a4 / (Tpr ** 4)) + (a5 / (Tpr ** 5))
    dcpr2 = 2 * Rpr * (a6 + (a7 / Tpr) + (a8 / (Tpr ** 2))) - 5 * (Rpr ** 4) * a9 * ((a7 / Tpr) + (a8 / (Tpr ** 2)))
    dcpr3 = ((2 * a10 * Rpr) / (Tpr ** 3)) * (1 + (a11 * (Rpr ** 2)) - ((a11 ** 2) * (Rpr ** 4))) * np.exp(-a11 * (Rpr ** 2))
    dcpr = dcpr1 + dcpr2 + dcpr3
    cpr = (1 / Ppr) - (0.27 / ((z ** 2) * Tpr)) * (dcpr / (1 + (Rpr / z) * dcpr))
    cg = cpr / Ppc(sg, co2, h2s)
    return cg

# Calculation of viscosity of natural gas, units of cp (McCain Petroleum Fluids, pp. 105, 514 - 515)
def gas_viscosity(p, t, sg, co2=0.0, h2s=0.0):
    '''
    Args:
        p: pressure in psia
        t: temperature in deg F
        sg: specific gravity of the gas, air = 1
        co2: mole fraction of CO2 in the gas
        h2s: mole fraction of H2S in the gas

    Returns:
        mu_g: viscosity of the gas in cp
    '''
    z = z_factor(p, t, sg, co2, h2s)
    mw = 28.97 * sg
    R = 10.732 # units of (psia * cu-ft) / (lb-mole * deg R)
    rhog = ((p * mw) / (z * R * (t + 460))) / (62.427962) # units of gm/cc
    mu_afac = ((9.379 + 0.01607 * mw) * ((t + 460) ** 1.5)) / (209.2 + 19.26 * mw + (t + 460))
    mu_bfac = 3.448 + (986.4 / (t + 460)) + 0.01009 * mw
    mu_cfac = 2.447 - 0.2224 * mu_bfac
    mu_g = (mu_afac / 10000) * np.exp(mu_bfac * (rhog ** mu_cfac))
    return mu_g

# Calculation of gas formation volume factor, units of cu-ft - scf (Craft & Hawkins Applied Petroleum Reservoir Engineering, pg. 23)
def gas_formation_volume_factor(p, t, sg, co2=0.0, h2s=0.0):
    '''
    Args:
        p: pressure in psia
        t: temperature in deg F
        sg: specific gravity of the gas, air = 1
        co2: mole fraction of CO2 in the gas
        h2s: mole fraction of H2S in the gas

    Returns:
        bg: gas formation volume factor in cu-ft/scf
    '''
    z = z_factor(p, t, sg, co2, h2s)
    bg = 0.02829 * (z * (t + 460)) / p
    return bg

# Calculation of gas pseudo pressure (Petroleum Engineering Handbook Va, pg 757)
def mp_gas(p, t, sg, co2=0.0, h2s=0.0):
    '''
    Args:
        p: pressure in psia
        t: temperature in deg F
        sg: specific gravity of the gas, air = 1
        co2: mole fraction of CO2 in the gas
        h2s: mole fraction of H2S in the gas

    Returns:
        mp: pseudo pressure of the gas in psia
    '''
    def integrand(p):
        mu_g = gas_viscosity(p, t, sg, co2, h2s)
        z = z_factor(p, t, sg, co2, h2s)
        integrand = p / (mu_g * z)
        return integrand
    mp = quad(integrand, 0, p)[0]
    return mp

# Calculation of bubble point pressure for black oil using Velarde, Blasingame, and McCain method (PETSOC-97-93, June 1997)
def bubble_point_pressure(t, rsb, api, sg, p_atm=14.7):
    '''
    Args:
        rsb: Gas-oil ratio at bubble point in scf/stb
        api: stock tank oil gravity in API
        sg: gas specific gravity, air = 1
        t: Reservoir temperature in deg F
        p_atm: atmospheric pressure in psia

    Returns:
        pb: bubble point pressure in psia
    '''
    x = (0.013098 * t ** 0.282372) - (8.2e-6 * api ** 2.176124)
    pb = 1091.47 * ((rsb ** 0.081465) * (sg ** -0.161488) * (10 ** x) - 0.740152) ** 5.354891
    return pb - p_atm

# Calculation of solution gas-oil ratio using Velarde, Blasingame, and McCain method (PETSOC-97-93, June 1997)
def gas_oil_ratio(p, t, rsb, api, sg, p_atm=14.7):
    '''
    Args:
        p: reservoir pressure in psig
        rsb: Gas-oil ratio at bubble point in scf/stb
        t: Reservoir temperature in deg F
        api: stock tank oil gravity in API
        sg: gas specific gravity, air = 1
        p_atm: atmospheric pressure in psia

    Returns:
        rs: gas-oil ratio at reservoir pressure in scf/stb
    '''
    # Calculate bubble point pressure and reduced pressure
    pb = bubble_point_pressure(t, rsb, api, sg, p_atm)
    pr = p / pb

    # Define coefficients for the correlation
    a0, a1, a2, a3, a4 = 9.7e-7, 1.672608, 0.92987, 0.247235, 1.056052
    b0, b1, b2, b3, b4 = 0.022339, -1.00475, 0.337711, 0.132795, 0.302065
    c0, c1, c2, c3, c4 = 0.725167, -1.48548, -0.164741, -0.09133, 0.047094

    # Functions from regression used to determine reduced gas-oil ratio
    a = a0 * (sg ** a1) * (api ** a2) * (t ** a3) * (pb ** a4)
    b = b0 * (sg ** b1) * (api ** b2) * (t ** b3) * (pb ** b4)
    c = c0 * (sg ** c1) * (api ** c2) * (t ** c3) * (pb ** c4)
    
    # Calculate reduced gas-oil ratio
    rsr = a * (pr ** b) + (1 - a) * (pr ** c)

    # Calculate and return gas-oil ratio
    return rsb * rsr

# Calculation of oil formation volume factor using modified Standing correlation (PETSOC-97-93, June 1997)
def oil_formation_volume_factor(p, t, rsb, api, sg, p_sep=100.0, t_sep=100.0, p_atm=14.7):
    '''
    Args:
        p: reservoir pressure in psig
        rsb: Gas-oil ratio at bubble point in scf/stb
        t: Reservoir temperature in deg F
        api: stock tank oil gravity in API
        sg: gas specific gravity, air = 1
        p_sep: separator pressure in psig
        t_sep: separator temperature in deg F
        p_atm: atmospheric pressure in psia

    Returns:
        bo: oil formation volume factor in bbl/stb
    '''
    # Calculate bubble point pressure
    pb = bubble_point_pressure(t, rsb, api, sg, p_atm)

    # Calculate oil formation volume factor using modified Standing correlation at bubble point
    bob = 1.023761 + 0.000122 * ((rsb ** 0.413179) * (sg ** 0.210293) * (api ** 0.127123) + (0.019073 * t)) ** 2.465976

    # Determine if fluid is saturated or undersaturated
    if p > pb:
        # Calculate oil compressibility (Vazquez, Beggs April 1978)
        sg_100 = sg * (1 + 5.912e-5 * api * t_sep * np.log((p_sep + p_atm) / (100 + p_atm)))
        co = (-1433 + (5 * rsb) + (17.2 * t) - (1180 * sg_100) + (12.61 * api)) / (100000 * (p + p_atm))
        bo = bob * np.exp(co * (p - pb))
    else:
        bo = bob

    return bo

# Calculate oil viscosity using Beggs and Robinson correlation
def oil_viscosity(p, t, rsb, api, sg, p_atm=14.7):
    '''
    Args:
        api: stock tank oil gravity in API
        p: pressure in psig
        t: temperature in deg F
        rsb: Gas-oil ratio at bubble point in scf/stb
        p_atm: atmospheric pressure in psia

    Returns:
        mu_o: oil viscosity in cp
    '''
    # Calculate bubble point pressure
    pb = bubble_point_pressure(t, rsb, api, sg, p_atm)

    # Calculate dead oil viscosity (uod)
    c = 3.0324 - 0.02023 * api
    b = 10 ** c
    a = b * t ** -1.163
    uod = 10 ** a - 1

    # Calculate live oil viscosity at bubble point pressure (uobo)
    a = 10.715 * (rsb + 100) ** -0.515
    b = 5.44 * (rsb + 150) ** -0.338
    uobo = a * uod ** b

    # Calculate oil viscosity
    if p < pb:
        return uobo
    else:
        # Calculate the coefficient `a` above bubble point
        a = 2.6 * (p + p_atm) ** 1.187 * np.exp(-0.0000898 * (p + p_atm) - 11.513)
        return uobo * (p / pb) ** a

def water_compressibility(p, t, rsw, salt):
    """
    Calculate water compressibility (Cw) with gas/water ratio correction and salinity adjustment using NumPy.
    Args:
        t (float): Temperature (degrees Fahrenheit)
        p (float): Pressure (psia)
        rsw (float): Gas/Water ratio (SCF/STBW)
        salt (float): Salt concentration (weight %)
    Returns:
        float: Water compressibility (1/psi)
    """
    a = 3.8546 - 0.000134 * p
    b = -0.01052 + 0.000000477 * p
    c = 0.000039267 - 0.00000000088 * p
    Cw_value = (a + b * t + c * t**2) / 1000000
    Cw_value *= (1 + 0.0089 * rsw)  # Dissolved gas correction
    Cw_value *= ((-0.052 + 0.00027 * t - 0.00000114 * t**2 + 0.000000001121 * t**3) * salt**0.7 + 1)
    return Cw_value

def water_formation_volume_factor(p, t, salt):
    """
    Calculate gas-saturated water formation volume factor (Bw) using NumPy.
    Args:
        t (float): Temperature (degrees Fahrenheit)
        p (float): Pressure (psia)
        salt (float): Salt concentration (weight %)
    Returns:
        float: Water formation volume factor
    """
    a = 0.9911 + 0.0000635 * t + 0.00000085 * t**2
    b = -0.000001093 - 0.000000003497 * t + 0.00000000000457 * t**2
    c = -0.00000000005 + 6.429e-13 * t - 1.43e-15 * t**2
    Bw_value = a + b * p + c * p**2
    Bw_value *= ((0.000000051 * p + (0.00000547 - 0.000000000195 * p) * (t - 60) + (-0.0000000323 + 0.00000000000085 * p) * (t - 60)**2) * salt + 1)
    return Bw_value

def water_viscosity(p, t, salt):
    """
    Calculate water viscosity (Uw) with salt concentration adjustment using NumPy.
    Args:
        t (float): Temperature (degrees Fahrenheit)
        p (float): Pressure (psia)
        salt (float): Salt concentration (weight %)
    Returns:
        float: Water viscosity (cp)
    """
    tc = 5 / 9 * (t - 32)  # Convert Fahrenheit to Celsius
    tk = tc + 273.15  # Convert Celsius to Kelvin
    coeff = (0.65 - 0.01 * tc)

    # Sum calculation for psat
    sum_exp = (
        -7.419242 * coeff**0
        - 0.29721 * coeff**1
        - 0.1155286 * coeff**2
        - 0.008685635 * coeff**3
        + 0.001094098 * coeff**4
        + 0.00439993 * coeff**5
        + 0.002520658 * coeff**6
        + 0.0005218684 * coeff**7
    )
    psat = 22088 * np.exp((374.136 - tc) * sum_exp / tk)

    Uw_value = 0.02414 * 10**(247.8 / (tk - 140)) * (1 + (p / 14.504 - psat) * 0.0000010467 * (tk - 305))
    Uw_value *= (1 - 0.00187 * np.sqrt(salt) + 0.000218 * salt**2.5 + (np.sqrt(t) - 0.0135 * t) * (0.00276 * salt - 0.000344 * salt**1.5))
    return Uw_value

def water_gas_solubility(p, t, salt):
    """
    Calculate gas solubility in water (RSwat) with salt concentration adjustment using NumPy.
    Args:
        t (float): Temperature (degrees Fahrenheit)
        p (float): Pressure (psia)
        salt (float): Salt concentration (weight %)
    Returns:
        float: Gas solubility in water (SCF/STBW)
    """
    a = 2.12 + 0.00345 * t - 0.0000359 * t**2
    b = 0.0107 - 0.0000526 * t + 0.000000148 * t**2
    c = -0.000000875 + 0.0000000039 * t - 0.0000000000102 * t**2
    RSwat_value = a + b * p + c * p**2
    RSwat_value *= (1 - (0.0753 - 0.000173 * t) * salt)
    return RSwat_value

# Estimate free gas saturation in a black oil reservoir
def free_gas_saturation(p, t, gor, api, sg, wtr_sat, co2=0.0, h2s=0.0, sro=0.3, srg=0.0, gas_end_point=0.8, oil_end_point=0.7, gas_exponent=3.0, oil_exponent=3.0, p_atm=14.7):
    '''
    Args:
        p: pressure in psig
        t: temperature in deg F
        gor: producing gas-oil ratio in scf/stb
        api: stock tank oil gravity in API
        sg: gas specific gravity, air = 1
        wtr_sat: water saturation in fraction
        co2: mole fraction of CO2 in the gas
        h2s: mole fraction of H2S in the gas
        sro: residual oil saturation in fraction
        srg: residual gas saturation in fraction
        gas_end_point: maximum gas relative permeability in fraction
        oil_end_point: maximum oil relative permeability in fraction
        gas_exponent: gas Corey exponent
        oil_exponent: oil Corey exponent
        p_atm: atmospheric pressure in psia

    Returns:
        gas_sat: free gas saturation in fraction
        oil_sat: oil saturation in fraction
        rsb: solution gas-oil ratio in scf/stb
        pb: bubble point pressure in psig
        bo: oil formation volume factor in bbl/stb
        bg: gas formation volume factor in cu-ft/scf
        mu_o: oil viscosity in cp
        mu_g: gas viscosity in cp
    '''
    # function to calculate relative perm from Corey model
    def rel_perm(sw, so, sro, srg, gas_end_point, oil_end_point, gas_exponent, oil_exponent):
        kro = oil_end_point * ((so - sro) / (1 - sro - sw)) ** oil_exponent
        krg = gas_end_point * (((1 - sw - so) - srg) / (1 - sro - sw - srg)) ** gas_exponent
        return kro, krg
    
    # Calculate bubble point pressure and correct rsb if needed
    pb = bubble_point_pressure(t, gor, api, sg, p_atm)
    rsb = gor
    if p < pb:
        def eqn_rsb(rsb):
            return bubble_point_pressure(t, rsb, api, sg, p_atm) - p
        
        rsb = fsolve(eqn_rsb, rsb)[0]
        pb = p

    # Calculate the oil formation volume factor
    bo = oil_formation_volume_factor(p, t, rsb, api, sg, p_atm=p_atm)

    # Calculate the gas formation volume factor
    bg = gas_formation_volume_factor(p, t, sg, co2, h2s)

    # Calculate the oil viscosity
    mu_o = oil_viscosity(p, t, rsb, api, sg, p_atm=p_atm)

    # Calculate the gas viscosity
    mu_g = gas_viscosity(p, t, sg, co2, h2s)

    # Perform fractional flow calcs
    qo = bo
    qg = ((gor - rsb) * bg) / 5.615
    qt = qo + qg
    fg1 = qg / qt

    oil_sat = 1.0 - wtr_sat  # Calculate starting point for oil saturation
    if rsb < gor:
        def eqn_oil_sat(oil_sat):
            kro, krg = rel_perm(wtr_sat, oil_sat, sro, srg, gas_end_point, oil_end_point, gas_exponent, oil_exponent)
            fg2 = 1.0 / (1.0 + (kro / krg) * (mu_g / mu_o))
            return fg1 - fg2
        
        oil_sat = fsolve(eqn_oil_sat, oil_sat)[0]
        gas_sat = 1.0 - wtr_sat - oil_sat
    else:
        gas_sat = 0.0

    return gas_sat, oil_sat, rsb, pb, bo, bg, mu_o, mu_g

# Gas Processing Calculations
_STD_PSI = 14.696

# Helper function to for by component calculations
def gas_processing(grs_vol, shrink_init, mol_pct, ideal_liquid_content, ideal_btu, pct_recovery, base_psi=14.65):
    '''
    Args:
        grs_vol: produced wet gas volume in mcf
        shrink_init: initial shrinkage factor before plant inlet
        mol_pct: molar fraction of the gas
        ideal_liquid_content: ideal liquid content in scf per gallon
        ideal_btu: ideal Btu content in Btu/scf
        pct_recovery: percentage fixed or actual ngl recovery as a decimal
        base_psi: base pressure in psia as base for the calculations
    Returns (all measued by component):
        recovered_gals: NGL gallons recovered by the plant
        wet_btu: Btu content of gas at the plant inlet in Btu/scf
        extracted_btu: Btu content extracted from the gas gas at the plant outlet in Btu/scf
        reduced_mol_pct: reduced molar fraction of the gas
    '''
    # Sanitize and clamp inputs
    grs_vol = _to_float(grs_vol, 0.0)
    shrink_init = np.clip(_to_float(shrink_init, 0.0), 0.0, 0.999999)
    mol_pct = max(_to_float(mol_pct, 0.0), 0.0)
    ideal_liquid_content = max(_to_float(ideal_liquid_content, 0.0), 0.0)
    ideal_btu = max(_to_float(ideal_btu, 0.0), 0.0)
    pct_recovery = np.clip(_to_float(pct_recovery, 0.0), 0.0, 1.0)
    base_psi = _to_float(base_psi, _STD_PSI)

    if base_psi <= 0 or not np.isfinite(base_psi):
        base_psi = _STD_PSI

    # Perform calculations
    processed_vol = max(0.0, grs_vol * 1000.0 * (1.0 - shrink_init))
    wet_btu = mol_pct * ideal_btu * base_psi / _STD_PSI
    if ideal_liquid_content > 0:
        denom = ideal_liquid_content * (_STD_PSI / base_psi)
        recovered_gals = (mol_pct / denom) * pct_recovery * processed_vol
    else:
        recovered_gals = 0.0
    extracted_btu = wet_btu * pct_recovery
    reduced_mol_pct = mol_pct * (1 - pct_recovery)

    return recovered_gals, wet_btu, extracted_btu, reduced_mol_pct

# Function to batch results for all components
def gas_processing_batch(grs_vol, shrink_init, mol_pct, ideal_liquid_content, ideal_btu, pct_recovery, pct_pop=None, base_psi=14.65):
    '''
    Args:
        grs_vol: produced wet gas volume in mcf
        shrink_init: initial shrinkage factor before plant inlet
        mol_pct: dictionary of molar fractions of the gas for each component
        ideal_liquid_content: dictionary of ideal liquid content in scf per gallon for each component
        ideal_btu: dictionary of ideal Btu content in Btu/scf for each component
        pct_recovery: dictionary of ngl recovery as a decimal for each component
        pct_pop: dictionary of percentage of proceeds the producer keeps as a decimal. Typically in lieu of a processing fee if less than 1.
        base_psi: base pressure in psia as base for the calculations

    Returns a dictionary of results for each component:
        recovered_gals: NGL gallons recovered by the plant
        settlement_gals: NGL gallons settled by the plant
        wet_btu: Btu content of gas at the plant inlet in Btu/scf
        extracted_btu: Btu content extracted from the inlet gas from processing in Btu/scf
        residue_btu: Btu content of the gas at the plant outlet in Btu/scf
    '''
    # Initialize dictionary to store results
    results = {}

    # Loop through each component and calculate results
    for component, mol in mol_pct.items():
        liq_cont = ideal_liquid_content.get(component, 0.0)
        btu_cont = ideal_btu.get(component, 0.0)
        pct_rec = pct_recovery.get(component, 0.0)
        pop_frac = (pct_pop or {}).get(component, 1.0)

        recovered_gals, wet_btu, extracted_btu, reduced_mol_pct = gas_processing(
            grs_vol, 
            shrink_init, 
            mol, 
            liq_cont, 
            btu_cont, 
            pct_rec, 
            base_psi
        )

        # Store results in the dictionary
        results[component] = {
            'recovered_gals': recovered_gals,
            'settlement_gals': recovered_gals * pop_frac,
            'wet_btu': wet_btu,
            'extracted_btu': extracted_btu,
            'reduced_mol_pct': reduced_mol_pct,
            'btu_cont': btu_cont
        }
    
    # Sum results for reduced_mol_pct for all components
    total_reduced_mol_pct = sum(v['reduced_mol_pct'] for v in results.values())
    den = total_reduced_mol_pct if total_reduced_mol_pct > 0 else 1.0

    # Calculate residue btu for each component and append to results
    for comp, val in results.items():
        val['residue_btu'] = val['reduced_mol_pct'] / den * val['btu_cont'] * (base_psi / _STD_PSI)
        val.pop('reduced_mol_pct', None)
        val.pop('btu_cont', None)

    return results
