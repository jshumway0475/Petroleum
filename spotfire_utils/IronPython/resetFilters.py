from Spotfire.Dxp.Application.Filters import *

# Reset all filters
for table in Document.Data.Tables:
    for filteringScheme in Document.FilteringSchemes:
        filteringScheme[table].ResetAllFilters()

# Clear the api.list document property
Document.Properties["api.list"] = ""
