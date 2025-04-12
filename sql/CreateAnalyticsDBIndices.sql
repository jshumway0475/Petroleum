-- Creating nonclustered indexes on dbo.PRODUCTION
CREATE NONCLUSTERED INDEX [idx_production_main] 
ON [dbo].[PRODUCTION] ([WellID], [Cadence], [DataSource], [Date]) 
INCLUDE ([Measure], [Value]);

CREATE NONCLUSTERED INDEX [idx_production_measure_filtered] 
ON [dbo].[PRODUCTION] ([Measure], [WellID], [Cadence], [DataSource], [Date]) 
INCLUDE ([Value]);

CREATE NONCLUSTERED INDEX [idx_WellID] 
ON [dbo].[PRODUCTION] ([WellID]);

CREATE NONCLUSTERED INDEX [idx_WellID_DataSource_Cadence] 
ON [dbo].[PRODUCTION] ([WellID], [DataSource], [Cadence]);

CREATE NONCLUSTERED INDEX [IX_PRODUCTION_WellID_Measure_DataSource_Date] 
ON [dbo].[PRODUCTION] (WellID, Measure, DataSource, Date);

CREATE NONCLUSTERED INDEX [idx_Prod_Cadence_SourceRank_WellID_Date]
ON [Axia_Analytics].[dbo].[PRODUCTION] (Cadence, SourceRank, WellID, Date)
INCLUDE (DataSource);

-- Creating nonclustered indexes on dbo.WELL_HEADER
CREATE NONCLUSTERED INDEX [IX_WELL_HEADER_DateCreated] 
ON [dbo].[WELL_HEADER] ([DateCreated]);

CREATE NONCLUSTERED INDEX [IX_WELL_HEADER_WellID_DataSource] 
ON [dbo].[WELL_HEADER] ([WellID], [DataSource]);
