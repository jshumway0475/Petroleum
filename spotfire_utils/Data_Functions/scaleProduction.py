import pandas as pd
import numpy as np

def scaleProd(df, scale_to_LL, scale_to_FI, scale_to_dist, val_col, targetLL, targetFI, targetDist, factorLL, exponentPL, exponentDist, max_distance):
    scaled_col_name = f'scaled_{val_col}'

    # Separating out the series for efficiency
    val_data = df[val_col]
    lateral_length = df['LateralLength_FT'].fillna(targetLL)
    fluid_intensity = df['Fluid_Intensity'].fillna(targetFI)
    distance_left = np.minimum(df['ClosestHzDistance_Left_Filled'], max_distance)
    distance_right = np.minimum(df['ClosestHzDistance_Right_Filled'], max_distance)

    # Pre-computing scaling factors
    lateral_scale_factor = (targetLL / lateral_length) ** factorLL if scale_to_LL else 1
    fi_scale_factor = (targetFI / fluid_intensity) ** exponentPL if scale_to_FI else 1
    dist_left_scale_factor = ((targetDist / distance_left) ** (exponentDist / 2)) if scale_to_dist else 1
    dist_right_scale_factor = ((targetDist / distance_right) ** (exponentDist / 2)) if scale_to_dist else 1

    # Scale Data using computed factors
    scaled_data = val_data * lateral_scale_factor * fi_scale_factor * dist_left_scale_factor * dist_right_scale_factor

    # Assigning the result back to the DataFrame
    df[scaled_col_name] = scaled_data
    return df[scaled_col_name]

'''
df is dbo.ProductionMaterialized from Analytics database
scale_to_LL, scale_to_FI, scale_to_dist, targetLL, targetFI, targetDist, factorLL, exponentPL, exponentDist, and max_distance are document properties in Spotfire
'''
# Calculate new columns
df['scaled_monthly_volume'] = scaleProd(df, scale_to_LL, scale_to_FI, scale_to_dist, 'MonthlyVolume', targetLL, targetFI, targetDist, factorLL, exponentPL, exponentDist, max_distance)
df['scaled_cumulative_production'] = scaleProd(df, scale_to_LL, scale_to_FI, scale_to_dist, 'CumulativeProduction', targetLL, targetFI, targetDist, factorLL, exponentPL, exponentDist, max_distance)

# Pivot df to aggregate the maximum scaled_cumulative_production by WellID pivoted on measure
pivot_df = df.pivot_table(index=['WellID', 'ScenarioName'], columns='Measure', values='scaled_cumulative_production', aggfunc='max')
pivot_df.reset_index(inplace=True)

# Assign new columns to variables for output
scaled_monthly_volume = df['scaled_monthly_volume']
scaled_cumulative_production = df['scaled_cumulative_production']
scaled_eur_df = pivot_df
