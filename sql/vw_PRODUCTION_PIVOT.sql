USE [Analytics]
GO

SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE OR ALTER VIEW [dbo].[vw_PRODUCTION_PIVOT] AS
SELECT
		WellID,
		Date,
		Cadence,
		DataSource,
		[OIL], 
		[GAS], 
		[WATER],
		ROW_NUMBER() OVER (PARTITION BY WellID, Cadence ORDER BY Date) AS DateRankAsc,
		ROW_NUMBER() OVER (PARTITION BY WellID, Cadence ORDER BY Date DESC) AS DateRankDesc,
		SUM(OIL) OVER (PARTITION BY WellID, Cadence ORDER BY Date) AS CumulativeOil,
		SUM(GAS) OVER (PARTITION BY WellID, Cadence ORDER BY Date) AS CumulativeGas,
		SUM(WATER) OVER (PARTITION BY WellID, Cadence ORDER BY Date) AS CumulativeWater
FROM  (
		SELECT	WellID, Date, DataSource, Cadence, Measure, Value
		FROM	dbo.PRODUCTION
		WHERE	Measure IN ('OIL', 'GAS', 'WATER')
		AND		SourceRank = 1
		AND		Cadence = 'MONTHLY'
) AS SourceData
PIVOT (
		SUM(Value)
		FOR Measure IN ([OIL], [GAS], [WATER])
) AS PivotTable;
GO
