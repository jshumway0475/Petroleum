from fastapi import FastAPI, HTTPException
from api.models import MonthDiffInput, MonthDiffOutput, ArpsSegmentsInput, ArpsSegmentsOutput, ArpsSegmentOutputRow
from api.models import ChangePointDetectionInput, ChangePointDetectionOutput, BourdetOutliersInput, BourdetOutliersOutput
from api.models import BFactorDiagnosticsInput, BFactorDiagnosticsOutput, GoodnessOfFitInput, GoodnessOfFitOutput
from api.models import PerformCurveFitInput, PerformCurveFitOutput
from api.models import OptimizeBufferInput, OptimizeBufferOutput, PrepDFDistanceInput, PrepDFDistanceOutput
from api.models import CalculateDistanceInput, CalculateDistanceOutput, ParentChildProcessingInput, ParentChildProcessingOutput
from api.models import CalcVerticalDistanceInput, CalcVerticalDistanceOutput
from api.models import EconCFInput, EconCFOutput, NPVInput, NPVOutput
from pydantic import BaseModel
from pydantic.tools import parse_obj_as
from AnalyticsAndDBScripts import dcf_functions as dcf
from AnalyticsAndDBScripts import geo_functions as geo
from AnalyticsAndDBScripts import prod_fcst_functions as fcst
from AnalyticsAndDBScripts import sql_connect as sql
from AnalyticsAndDBScripts import sql_schemas as schema
from AnalyticsAndDBScripts import well_spacing as ws
import numpy as np
import pandas as pd
import scipy.stats
import geopandas as gpd

app = FastAPI()

@app.post("/calculate_month_diff", response_model=MonthDiffOutput)
def calculate_month_diff(input_data: MonthDiffInput):
    try:
        # Convert dates from input data to required format if necessary
        months_diff = fcst.MonthDiff(input_data.BaseDate, input_data.StartDate)
        return MonthDiffOutput(months_difference=months_diff)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating month difference: {str(e)}")

@app.post("/arps_segments", response_model=ArpsSegmentsOutput)
def calculate_arps_segments(input_data: ArpsSegmentsInput):
    try:
        # Assuming the arps_segments function and all necessary imports are defined correctly
        output_data = fcst.arps_segments(
            UID=input_data.UID, phase=input_data.phase, Q1=input_data.Q1, Q2=input_data.Q2,
            Q3=input_data.Q3, Dei=input_data.Dei, Def=input_data.Def, b=input_data.b,
            Qabn=input_data.Qabn, t1=input_data.t1, t2=input_data.t2, duration=input_data.duration,
            prior_cum=input_data.prior_cum, prior_t=input_data.prior_t
        )
        # Process the numpy array to create a list of ArpsSegmentOutputRow instances
        results = [ArpsSegmentOutputRow(
            UID=row[0], phase=int(row[1]), t=int(row[2]), q=row[3],
            De_t=row[4], Np=row[5], Monthly_volume=row[6]
        ) for row in output_data]
        return ArpsSegmentsOutput(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ARPS segments: {str(e)}")

@app.post("/econ_cf", response_model=EconCFOutput)
def calculate_econ_cf(input_data: EconCFInput):
    try:
        results = dcf.econ_cf(**input_data.dict())
        # Process results into a suitable format for the response
        return EconCFOutput(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing economic cash flow: {str(e)}")

@app.post("/calculate_npv", response_model=NPVOutput)
def calculate_npv(input_data: NPVInput):
    try:
        # Call the npv function to calculate present value
        pv = dcf.npv(input_data.disc_rate, np.array(input_data.ncf), np.array(input_data.n))
        return NPVOutput(pv=pv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_changepoints", response_model=ChangePointDetectionOutput)
def detect_changepoints_endpoint(input_data: ChangePointDetectionInput):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(input_data.data)
        
        # Call the function
        result_df = fcst.detect_changepoints(
            df=df,
            id_col=input_data.id_col,
            prod_col=input_data.prod_col,
            date_col=input_data.date_col,
            pen=input_data.pen
        )
        
        # Convert DataFrame back to list of dicts
        result_data = result_df.to_dict('records')
        
        return ChangePointDetectionOutput(data=result_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in change point detection: {str(e)}")

@app.post("/bourdet_outliers", response_model=BourdetOutliersOutput)
def apply_bourdet_outliers(input_data: BourdetOutliersInput):
    try:
        # Convert input lists to numpy arrays
        y_array = np.array(input_data.y, dtype=np.float)
        x_array = np.array(input_data.x, dtype=np.float)

        # Call the bourdet_outliers function
        y_filtered, x_filtered = fcst.bourdet_outliers(
            y=y_array, 
            x=x_array, 
            L=input_data.L, 
            xlog=input_data.xlog, 
            ylog=input_data.ylog, 
            z_threshold=input_data.z_threshold, 
            min_array_size=input_data.min_array_size
        )
        
        # Return the filtered data
        return BourdetOutliersOutput(y_filtered=y_filtered.tolist(), x_filtered=x_filtered.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in Bourdet Outliers calculation: {str(e)}")

@app.post("/b_factor_diagnostics", response_model=BFactorDiagnosticsOutput)
def calculate_b_factor_diagnostics(input_data: BFactorDiagnosticsInput):
    try:
        # Convert list of dicts to DataFrame
        df = pd.DataFrame(input_data.data)

        # Call the b_factor_diagnostics function
        results = fcst.b_factor_diagnostics(
            df=df,
            rate_col=input_data.rate_col,
            time_col=input_data.time_col,
            cadence=input_data.cadence,
            smoothing_factor=input_data.smoothing_factor,
            min_months=input_data.min_months,
            max_months=input_data.max_months
        )
        
        # If results are None, return an appropriate response
        if results is None:
            return BFactorDiagnosticsOutput()

        # Prepare and return the output
        return BFactorDiagnosticsOutput(
            df=results['df'].to_dict('records') if 'df' in results else None,
            b_avg=results.get('b_avg'),
            b_low=results.get('b_low'),
            b_high=results.get('b_high'),
            summary=str(results['summary']) if 'summary' in results else None,
            best_r2=results.get('best_r2'),
            best_max_time=results.get('best_max_time')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in B-Factor diagnostics: {str(e)}")

@app.post("/calc_goodness_of_fit", response_model=GoodnessOfFitOutput)
def calculate_goodness_of_fit(input_data: GoodnessOfFitInput):
    try:
        # Ensure data is in numpy arrays for calculation
        q_act = np.array(input_data.q_act)
        q_pred = np.array(input_data.q_pred)

        # Calculate the goodness of fit metrics
        r_squared, rmse, mae = fcst.calc_goodness_of_fit(q_act, q_pred)

        return GoodnessOfFitOutput(r_squared=r_squared, rmse=rmse, mae=mae)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating goodness of fit metrics: {str(e)}")

@app.post("/perform_curve_fit", response_model=PerformCurveFitOutput)
def perform_curve_fit_endpoint(input_data: PerformCurveFitInput):
    try:
        optimized_params, success = fcst.perform_curve_fit(
            input_data.t_act, 
            input_data.q_act, 
            input_data.initial_guess, 
            input_data.bounds, 
            input_data.config
        )
        return PerformCurveFitOutput(optimized_params=optimized_params, success=success)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing curve fit: {str(e)}")

@app.post("/optimize_buffer")
def optimize_buffer_endpoint(input: OptimizeBufferInput):
    df = pd.DataFrame(input.df)
    output_df = ws.optimize_buffer(df, input.geo_col, input.sfc_lat_col, input.sfc_long_col, input.epsg, input.start_buffer, input.max_buffer, input.max_iter, input.buffer_distance_ft, input.rec_conformity_threshold)
    return OptimizeBufferOutput(df=output_df.to_dict('records'))

@app.post("/prep_df_distance")
def prep_df_distance_endpoint(input: PrepDFDistanceInput):
    df = pd.DataFrame(input.df)
    output_df = ws.prep_df_distance(df, input.well_id_col)
    return PrepDFDistanceOutput(df=output_df.to_dict('records'))

@app.post("/calculate_distance")
def calculate_distance_endpoint(input: CalculateDistanceInput):
    output = ws.calculate_distance(input.row, input.min_distance_ft)
    return CalculateDistanceOutput(
        min_distance_ft=output[0],
        median_distance_ft=output[1],
        max_distance_ft=output[2],
        avg_distance_ft=output[3],
        intersection_fraction=output[4],
        relative_position=output[5]
    )

@app.post("/parent_child_processing")
def parent_child_processing_endpoint(input: ParentChildProcessingInput):
    spacing_df = pd.DataFrame(input.spacing_df)
    well_df = pd.DataFrame(input.well_df)
    closest_wells = ws.parent_child_processing(spacing_df, well_df, input.co_completed_threshold, input.id_col, input.position_col, input.date_col, input.distance_col, input.neighbor_date_col, input.scenario_name)
    return ParentChildProcessingOutput(closest_wells=closest_wells.to_dict("records"))

@app.post("/calc_vertical_distance")
def calc_vertical_distance_endpoint(input: CalcVerticalDistanceInput):
    gdf = gpd.GeoDataFrame(input.gdf)
    exploded_df = ws.calc_vertical_distance(gdf, input.buffer_radius, input.id_col, input.geo_col, input.date_col, input.source_epsg)
    return CalcVerticalDistanceOutput(exploded_df=exploded_df.to_dict("records"))
