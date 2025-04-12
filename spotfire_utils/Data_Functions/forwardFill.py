import pandas as pd

def dataframe_ffill(df, sort_columns, ascending_order, group, fill_cols, fill_values):
    # Create index for original sorting
    df['original_order'] = range(df.shape[0])
    
    # Sort dataframe for forward fill operation
    df.sort_values(by=sort_columns, ascending=ascending_order, inplace=True)
    
    # Perform forward fill and fill in null values
    for col in fill_cols:
        filled_col_name = f'{col}_Filled'
        df[filled_col_name] = df.groupby(group)[col].ffill()
        if fill_values.get(col):
            df[filled_col_name].fillna(value=fill_values[col], inplace=True)
    
    # Sort dataframe back to original order
    df.sort_values(by='original_order', ascending=True, inplace=True)
    
    # Drop 'original_order' column
    df.drop(columns=['original_order'], inplace=True)

# Specify the fill values for each column
fill_values = {
    'Relationship_Time': 'S',
    'ClosestHzDistance': 5280,
    'ClosestHzDistance_Left': 5280,
    'ClosestHzDistance_Right': 5280
}

# Call the function and pass the required parameters
dataframe_ffill(
    df, 
    ['WellID', 'Measure', 'Date'], 
    [True, True, True], 
    ['WellID', 'Measure'],
    ['Relationship_Time', 'ClosestHzDistance', 'ClosestHzDistance_Left', 'ClosestHzDistance_Right'],
    fill_values
)

# Accessing columns
relationship = df['Relationship_Time_Filled']
closestHzDist = df['ClosestHzDistance_Filled']
closestHzDistLeft = df['ClosestHzDistance_Left_Filled']
closestHzDistRight = df['ClosestHzDistance_Right_Filled']
