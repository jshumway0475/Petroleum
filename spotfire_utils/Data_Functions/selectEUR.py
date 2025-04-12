import pandas as pd
# Function to select EUR column for visualizations
def selectEUR(df, measure):
    eur_col_dict = {
        'OIL': 'EUR for OIL',
        'GAS': 'EUR for GAS',
        'WATER': 'EUR for WATER'
    }

    # Use the get method to handle missing keys in the dictionary
    column_name = eur_col_dict.get(measure)

    # Check if the column exists in the DataFrame
    if column_name in df.columns:
        return df[column_name]
    else:
        # Create a new DataFrame with a column named after the missing column, filled with 0.0
        return pd.Series(0.0, index=df.index, name=column_name)

# df is dbo.vw_WELL_HEADER from Analytics database
# measure is a document property in Spotfire
df['SelectedEUR'] = selectEUR(df, measure)
SelectedEUR = df['SelectedEUR']
