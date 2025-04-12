if LoadLinkedData:
    raise Exception('Data tables not done refreshing')
else:
    # Access the data manager
    dataManager = Document.Data

    # Variables to hold references to the specific data functions
    calcEURFunction = None
    forecastVolumeFunction = None

    # Iterate over all data functions in the document
    for function in dataManager.DataFunctions:
        # Find the 'calcEUR' data function
        if function.Name == 'calcEUR':
            calcEURFunction = function
        # Find the 'ForecastVolume' data function
        elif function.Name == 'ForecastVolume':
            forecastVolumeFunction = function

    # Check if the 'calcEUR' data function was found and execute it
    if calcEURFunction is not None:
        calcEURFunction.Execute()
    else:
        # Optional: Handle the case where the data function is not found
        print("Data function 'calcEUR' not found.")

    # Check if the 'ForecastVolume' data function was found and execute it
    if forecastVolumeFunction is not None:
        forecastVolumeFunction.Execute()
    else:
        # Optional: Handle the case where the data function is not found
        print("Data function 'ForecastVolume' not found.")
