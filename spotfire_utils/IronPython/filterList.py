import re
from Spotfire.Dxp.Application.Filters import ListBoxFilter

# Function to process input values
def process_value(val):
    # Remove any special characters and keep only digits
    val = re.sub(r'\D', '', val)

    # Truncate to 10 digits if longer
    val = val[:10] if len(val) > 10 else val

    # Add a leading zero if there are 9 digits
    val = '0' + val if len(val) == 9 else val

    return val

# Access the data table
dataTable = Document.Data.Tables["vw_WELL_HEADER"]

# Assuming you're using the default filtering scheme
filteringScheme = Document.FilteringSchemes[0]

# Access the ListBoxFilter for the 'API_UWI_Unformatted' column
column = dataTable.Columns["API_UWI_Unformatted"]
listBoxFilter = filteringScheme[dataTable].Item[column].As[ListBoxFilter]()

# Retrieve the raw values from the document property
raw_values = Document.Properties["api.list"]
# Process each value with the defined function
valuesToFilter = [process_value(value) for value in raw_values.split('\n') if value.strip()]
print("Processed Values to Filter: ", valuesToFilter)

# If the list is not empty, apply the filter
if valuesToFilter:
    # Attempt to set the selection
    try:
        listBoxFilter.IncludeAllValues = False
        listBoxFilter.SetSelection(valuesToFilter)
        print("Filter applied with values: ", valuesToFilter)
    except Exception as e:
        print("Error applying filter: ", str(e))
else:
    # If the list is empty, include all values (no filter)
    listBoxFilter.IncludeAllValues = True
    print("Filter cleared - no values provided")
