# Set document property to True
LoadLinkedData = True

if LoadLinkedData:
    # Start data table refresh
    Document.Data.Tables["ProductionMaterialized"].RefreshOnDemandData()
    Document.Data.Tables["vw_FORECAST"].RefreshOnDemandData()
    Document.Data.Tables["PARENT_CHILD"].RefreshOnDemandData()
    
    LoadLinkedData = False
