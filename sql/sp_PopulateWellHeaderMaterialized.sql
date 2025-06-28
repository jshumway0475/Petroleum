USE [Analytics]
GO

/****** Object:  StoredProcedure [dbo].[sp_PopulateWellHeaderMaterialized]    Script Date: 2/26/2024 1:25:36 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE OR ALTER PROCEDURE [dbo].[sp_PopulateWellHeaderMaterialized] AS
BEGIN
    SET NOCOUNT ON;

    -- Assuming it's safe to truncate; otherwise, consider a different approach
    TRUNCATE TABLE dbo.WellHeaderMaterialized;

    -- Insert data into the materialized table
	WITH PROD_PIVOT_CTE AS (
		SELECT
			WellID,
			Date,
			[OIL], 
			[GAS], 
			[WATER],
			SUM(CASE WHEN COALESCE(OIL, 0.0) > 0.0 AND COALESCE(GAS, 0.0) > 0.0 THEN 1 ELSE 0 END) OVER (PARTITION BY WellID ORDER BY Date) AS MonthRankAsc_OG,
			SUM(CASE WHEN COALESCE(OIL, 0.0) > 0.0 AND COALESCE(WATER, 0.0) > 0.0 THEN 1 ELSE 0 END) OVER (PARTITION BY WellID ORDER BY Date DESC) AS MonthRankDesc_OW,
			ROW_NUMBER() OVER (PARTITION BY WellID ORDER BY Date) AS DateRankAsc
		FROM  (
				SELECT	WellID, Date, Measure, Value
				FROM	dbo.PRODUCTION
				WHERE	Measure IN ('OIL', 'GAS', 'WATER')
				AND		SourceRank = 1
				AND		Cadence = 'MONTHLY'
				AND		Value > 0.0
		) AS SourceData
		PIVOT (
				SUM(Value)
				FOR Measure IN ([OIL], [GAS], [WATER])
		) AS P
	),
	METRICS_CTE AS (
		SELECT		WellID, 
					SUM(CASE WHEN MonthRankAsc_OG <= 3 THEN COALESCE(GAS, 0.0) END) * 1000.0 / NULLIF(SUM(CASE WHEN MonthRankAsc_OG <= 3 THEN COALESCE(OIL, 0.0) END), 0.0) AS GOR,
					SUM(CASE WHEN MonthRankDesc_OW <= 12 AND DateRankAsc > 3 THEN COALESCE(WATER, 0.0) END) / NULLIF(SUM(CASE WHEN MonthRankDesc_OW <= 12 AND DateRankAsc > 3 THEN (COALESCE(OIL, 0.0) + COALESCE(WATER, 0.0)) END), 0.0) AS WTR_CUT,
					MAX(DateRankAsc) AS TotalProdMonths,
					MAX(CASE WHEN DateRankAsc >= 12 THEN 0.0 ELSE OIL END) AS OilMaxMonth,
					MAX(CASE WHEN DateRankAsc >= 12 THEN 0.0 ELSE GAS END) AS GasMaxMonth,
					MAX(CASE WHEN DateRankAsc >= 12 THEN 0.0 ELSE WATER END) AS WaterMaxMonth
		FROM		PROD_PIVOT_CTE
		GROUP BY	WellID
	)
    INSERT INTO dbo.WellHeaderMaterialized (
        WellID, CurrentCompletionID, GOR, WTR_CUT, TotalProdMonths, PrimaryPhase, OilMaxMonth,
		GasMaxMonth, WaterMaxMonth
    )
    SELECT		W.WellID, W.CurrentCompletionID, M.GOR, M.WTR_CUT, M.TotalProdMonths, 
				CASE WHEN M.GOR < 3200 THEN 'OIL' ELSE 'GAS' END AS PrimaryPhase,
				M.OilMaxMonth, M.GasMaxMonth, M.WaterMaxMonth
    FROM		dbo.WELL_HEADER W
    LEFT JOIN	METRICS_CTE M
	ON			W.WellID = M.WellID;
END;
GO
