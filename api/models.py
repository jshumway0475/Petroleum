from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any, Optional
from datetime import date


class MonthDiffInput(BaseModel):
    BaseDate: date = Field(..., description="The base date from which the time difference is calculated")
    StartDate: date = Field(..., description="The start date for which the time difference is calculated")

class MonthDiffOutput(BaseModel):
    months_difference: int = Field(..., description="The number of months between the two dates")

class ArpsSegmentsInput(BaseModel):
    UID: int = Field(..., description="Unique identifier of the well")
    phase: int = Field(..., description="phase is 1 = oil, 2 = gas, or 3 = water")
    Q1: float = Field(..., description="Q1 is the initial production rate typically in bbl/day or Mcf/day")
    Q2: float = Field(..., description="Q2 is the production rate at the end of the first segment")
    Q3: float = Field(..., description="Q3 is the production rate at the end of the second segment")
    Dei: float = Field(..., description="Dei is the initial effective annual decline rate")
    Def: float = Field(..., description="Def is the final effective annual decline rate at which point the decline becomes exponential")
    b: float = Field(..., description="b is the b-factor used in hyperbolic or harmonic decline equations")
    Qabn: float = Field(..., description="Qabn is the minimum production rate to be included in the forecast")
    t1: float = Field(..., description="t1 is the duration of the first segment in months")
    t2: float = Field(..., description="t2 is the duration of the second segment in months")
    duration: int = Field(..., description="duration is the total duration of the forecast in months")
    prior_cum: Optional[float] = Field(0, description="prior_cum is the cumulative amount produced before the start of the decline calcuations")
    prior_t: Optional[int] = Field(0, description="prior_t is an integer representing the final month from a previous decline segment")

class ArpsSegmentOutputRow(BaseModel):
    UID: int = Field(..., description="Unique identifier of the well")
    phase: int = Field(..., description="phase is 1 = oil, 2 = gas, or 3 = water")
    t: int = Field(..., description="t is the month number")
    q: float = Field(..., description="q is the production rate for the month")
    De_t: float = Field(..., description="De_t is the effective annual decline rate for the month")
    Np: float = Field(..., description="Np is the cumulative production for the month")
    Monthly_volume: float = Field(..., description="Monthly_volume is the volume produced for the month")

class ArpsSegmentsOutput(BaseModel):
    results: List[ArpsSegmentOutputRow] = Field(..., description="List of Arps segment output rows")

class EconCFInput(BaseModel):
    index: int = Field(..., description="Index of the well in the volumes array")
    uid: int = Field(..., description="Unique identifier of the well")
    wi: float = Field(..., description="Working interest of the well as a decimal")
    nri: float = Field(..., description="Net revenue interest of the well as a decimal")
    roy: float = Field(..., description="Lease royalty as a decimal")
    eloss: int = Field(..., description="Number of months to continue producing after economic limit")
    weight: float = Field(..., description="Weight of the well for risking or fractional scheduling")
    prod_wt: float = Field(..., description="Production weight for risking production")
    inv_wt: float = Field(..., description="Investment weight for risking investment")
    shrink: float = Field(..., description="Gas shrinkage as a decimal")
    btu: float = Field(..., description="BTU content of the gas as mmbtu/mcf")
    ngl_yield: float = Field(..., description="NGL yield as a decimal, fraction of gas that is NGL")
    pri_oil: float = Field(..., description="Price of oil in $/bbl")
    pri_gas: float = Field(..., description="Price of gas in $/mmbtu")
    paj_oil: float = Field(..., description="Price adjustment for oil in $/bbl")
    paj_gas: float = Field(..., description="Price adjustment for gas in $/mmbtu")
    paj_ngl: float = Field(..., description="Price adjustment for NGL as a fraction of oil price")
    stx_oil: float = Field(..., description="Severance tax for oil as a decimal")
    stx_gas: float = Field(..., description="Severance tax for gas as a decimal")
    stx_ngl: float = Field(..., description="Severance tax for NGL as a decimal")
    adval: float = Field(..., description="Ad valorem tax as a decimal")
    opc_fix: float = Field(..., description="Fixed operating cost in $/month")
    opc_oil: float = Field(..., description="Operating cost for oil in $/bbl")
    opc_gas: float = Field(..., description="Operating cost for gas in $/mcf")
    opc_wtr: float = Field(..., description="Operating cost for water in $/bbl")
    capex: float = Field(..., description="Capital expenditure")
    aban: float = Field(..., description="Abandonment cost, applied at the end of life")
    volumes: List[List[float]] = Field(..., description="Array of monthly volumes for all wells in the project")

class EconCFOutputRow(BaseModel):
    UID: int = Field(..., description="Unique identifier of the well")
    Month: int = Field(..., description="Month number")
    Gross_Oil_Volume: float = Field(..., description="Gross oil volume in bbl")
    Gross_Gas_Volume: float = Field(..., description="Gross gas volume in mcf")
    Gross_Water_Volume: float = Field(..., description="Gross water volume in bbl")
    Oil_Sales: float = Field(..., description="Net NRI oil volumes in bbl")
    Gas_Sales: float = Field(..., description="Net NRI gas volumes in mcf")
    NGL_Sales: float = Field(..., description="Net NRI NGL volumes in bbl")
    Oil_Revenue: float = Field(..., description="Net Oil revenue in $")
    Gas_Revenue: float = Field(..., description="Net Gas revenue in $")
    NGL_Revenue: float = Field(..., description="Net NGL revenue in $")
    Total_Revenue: float = Field(..., description="Total net revenue in $")
    Total_Tax: float = Field(..., description="Total tax in $")
    Operating_Expense: float = Field(..., description="Operating expense in $")
    Operating_Cash_Flow: float = Field(..., description="Operating cash flow in $")
    Cumulative_Operating_Cash_Flow: float = Field(..., description="Cumulative operating cash flow in $")
    Net_Cash_Flow: float = Field(..., description="Net cash flow in $")
    Cumulative_Net_Cash_Flow: float = Field(..., description="Cumulative net cash flow in $")

class EconCFOutput(BaseModel):
    results: List[EconCFOutputRow] = Field(..., description="List of economic cash flow output rows")

class NPVInput(BaseModel):
    disc_rate: float = Field(..., description="Annualized discount rate as a decimal")
    ncf: List[float] = Field(..., description="Array of monthly cash flow values")
    n: List[int] = Field(..., description="Array of month integers corresponding to each cash flow")

class NPVOutput(BaseModel):
    pv: float = Field(..., description="Present value of the cash flow")

class ChangePointDetectionInput(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="List of production data records.")
    id_col: str = Field(..., description="Column name identifying each unique property or well.")
    prod_col: str = Field(..., description="Column name containing the production data values.")
    date_col: str = Field(..., description="Column name containing the date information.")
    pen: float = Field(..., description="Penalty value for change point detection sensitivity.")

class ChangePointDetectionOutput(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Modified list of production data records with segmentation.")

class BourdetOutliersInput(BaseModel):
    y: List[float] = Field(..., description="Array of y values for derivative computation.")
    x: List[float] = Field(..., description="Array of x values.")
    L: float = Field(default=0.0, description="Smoothing factor in log-cycle fractions.")
    xlog: bool = Field(default=True, description="Calculate derivative with respect to the log of x.")
    ylog: bool = Field(default=False, description="Calculate derivative with respect to the log of y.")
    z_threshold: float = Field(default=2.0, description="Z-score threshold for removing outliers.")
    min_array_size: int = Field(default=6, description="Minimum number of points needed for calculation.")

class BourdetOutliersOutput(BaseModel):
    y_filtered: List[float] = Field(..., description="Filtered y values after outlier removal.")
    x_filtered: List[float] = Field(..., description="Filtered x values after outlier removal.")

class BFactorDiagnosticsInput(BaseModel):
    data: List[Dict[str, float]] = Field(..., description="List of records, each representing a row from a DataFrame.")
    rate_col: str = Field(..., description="Column name for production rate data.")
    time_col: str = Field(..., description="Column name for time data.")
    cadence: Optional[str] = Field(default='monthly', description="Frequency of the time data ('monthly' or 'daily').")
    smoothing_factor: Optional[int] = Field(default=2, description="Number of iterations for data smoothing.")
    min_months: Optional[int] = Field(default=24, description="Minimum number of months to consider for the calculation.")
    max_months: Optional[int] = Field(default=60, description="Maximum number of months to consider for the calculation.")

class BFactorDiagnosticsOutput(BaseModel):
    df: Optional[List[Dict[str, float]]]
    b_avg: Optional[float]
    b_low: Optional[float]
    b_high: Optional[float]
    summary: Optional[str]
    best_r2: Optional[float]
    best_max_time: Optional[int]

class GoodnessOfFitInput(BaseModel):
    q_act: List[float] = Field(..., description="Actual production data as an array of values.")
    q_pred: List[float] = Field(..., description="Predicted production data as an array of values.")

class GoodnessOfFitOutput(BaseModel):
    r_squared: float = Field(..., description="R-squared value indicating the goodness of fit.")
    rmse: float = Field(..., description="Root mean squared error between actual and predicted data.")
    mae: float = Field(..., description="Mean absolute error between actual and predicted data.")

class PerformCurveFitInput(BaseModel):
    t_act: List[float] = Field(..., description="Actual time data as an array of values.")
    q_act: List[float] = Field(..., description="Actual production data as an array of values.")
    initial_guess: List[float] = Field(..., description="Initial guess for the curve fit parameters.")
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = Field(..., description="Bounds for the parameters being optimized.")
    config: Dict[str, Dict[str, float]] = Field(..., description="Configuration specifying which parameters to optimize and which are fixed.")

class PerformCurveFitOutput(BaseModel):
    optimized_params: Optional[List[float]] = Field(..., description="Optimized parameters.")
    success: bool = Field(..., description="True if fitting succeeded, False otherwise.")


class OptimizeBufferInput(BaseModel):
    df: List[Dict[str, Any]] = Field(..., description="List of well survey data records.")
    geo_col: str = Field(..., description="The name of the geometry column.")
    sfc_lat_col: str = Field(..., description="The name of the surface latitude column.")
    sfc_long_col: str = Field(..., description="The name of the surface longitude column.")
    epsg: int = Field(4326, description="The EPSG code of the coordinate system.")
    start_buffer: float = Field(500.0, description="The starting radial buffer distance around the sfc_loc in feet.")
    max_buffer: float = Field(1500.0, description="The maximum radial buffer distance around the sfc_loc in feet.")
    max_iter: int = Field(20, description="The maximum number of iterations on the radial buffer size to perform.")
    buffer_distance_ft: float = Field(5280, description="The buffer distance around the LateralLine in feet.")
    rec_conformity_threshold: float = Field(0.5, description="The minimum rectangle conformity threshold to for optimization of the surface location buffer.")

class OptimizeBufferOutput(BaseModel):
    df: List[Dict[str, Any]] = Field(..., description="Modified list of well survey data records with additional columns.")

class PrepDFDistanceInput(BaseModel):
    df: List[Dict[str, Any]] = Field(..., description="List of well survey data records.")
    well_id_col: str = Field(..., description="The name of the well ID column.")

class PrepDFDistanceOutput(BaseModel):
    df: List[Dict[str, Any]] = Field(..., description="Modified list of well survey data records with additional columns.")

class CalculateDistanceInput(BaseModel):
    row: Dict[str, Any] = Field(..., description="Row from a DataFrame resulting from the prep_df_distance function.")
    min_distance_ft: float = Field(100.0, description="The minimum distance in feet between the two lines to be considered in calculations.")

class CalculateDistanceOutput(BaseModel):
    min_distance_ft: Optional[float] = Field(None, description="The minimum distance in feet between the two lines.")
    median_distance_ft: Optional[float] = Field(None, description="The median distance in feet between the two lines.")
    max_distance_ft: Optional[float] = Field(None, description="The maximum distance in feet between the two lines.")
    avg_distance_ft: Optional[float] = Field(None, description="The average distance in feet between the two lines.")
    intersection_fraction: Optional[float] = Field(None, description="The fraction of the neighboring lateral that intersects with the buffer of the reference lateral.")
    relative_position: Optional[float] = Field(None, description="The relative position of the neighboring lateral with respect to the reference lateral.")

class ParentChildProcessingInput(BaseModel):
    spacing_df: List[Dict[str, Any]] = Field(..., description="DataFrame containing well spacing information.")
    well_df: List[Dict[str, Any]] = Field(..., description="DataFrame containing well information.")
    co_completed_threshold: int = Field(..., description="Threshold in days to determine if wells are co-completed.")
    id_col: str = Field(..., description="Name of the column representing the well ID.")
    position_col: str = Field(..., description="Name of the column representing the relative position.")
    date_col: str = Field(..., description="Name of the column representing the date.")
    distance_col: str = Field(..., description="Name of the column representing the average distance.")
    neighbor_date_col: str = Field(..., description="Name of the column representing the neighbor's first production date.")
    scenario_name: str = Field(..., description="Name of the scenario.")

class ParentChildProcessingOutput(BaseModel):
    closest_wells: List[Dict[str, Any]] = Field(..., description="Processed DataFrame with well relationships and distances.")

class CalcVerticalDistanceInput(BaseModel):
    gdf: List[Dict[str, Any]] = Field(..., description="GeoDataFrame containing well geometries and information.")
    buffer_radius: float = Field(..., description="Radius to buffer each geometry by, in feet.")
    id_col: str = Field(..., description="Name of the column containing the well IDs.")
    geo_col: str = Field(..., description="Name of the column containing the geometries.")
    date_col: str = Field(..., description="Name of the column containing the first production date.")
    source_epsg: int = Field(4326, description="EPSG code of the geometries.")

class CalcVerticalDistanceOutput(BaseModel):
    exploded_df: List[Dict[str, Any]] = Field(..., description="Processed GeoDataFrame with intersecting WellIDs and distances.")
