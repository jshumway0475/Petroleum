-- Creating nonclustered indexes on dbo.PRODUCTION
CREATE NONCLUSTERED INDEX IX_PROD_CVRG
ON		dbo.PRODUCTION(WellID, Date, Measure, Cadence, DataSource)
INCLUDE (Value, Units, ProducingDays, Comment, API_UWI_Unformatted, DateCreated);

CREATE NONCLUSTERED INDEX [IX_WELL_HEADER_WellID_DataSource] 
ON	[dbo].[WELL_HEADER] ([WellID], [DataSource]);

CREATE NONCLUSTERED INDEX IX_Prod_Monthly_OVG
ON		dbo.PRODUCTION (WellID, Date)
INCLUDE	(Measure, Value)
WHERE	Cadence = 'MONTHLY'
AND		SourceRank = 1
AND		Measure IN ('OIL','GAS','WATER');

-- Creating nonclustered indexes on dbo.WELL_HEADER
CREATE NONCLUSTERED INDEX [IX_WELL_HEADER_DateCreated] 
ON [dbo].[WELL_HEADER] ([DateCreated]);



