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
    INSERT INTO dbo.WellHeaderMaterialized (
        WellID, CurrentCompletionID, GOR, WTR_CUT, TotalProdMonths, PrimaryPhase, OilMaxMonth,
		GasMaxMonth, WaterMaxMonth
    )
    SELECT		W.WellID, W.CurrentCompletionID, FP.GOR, FP.WTR_CUT, PP.TotalProdMonths, 
				CASE WHEN FP.GOR < 3200 THEN 'OIL' ELSE 'GAS' END AS PrimaryPhase,
				PP.OilMaxMonth, PP.GasMaxMonth, PP.WaterMaxMonth
    FROM		dbo.WELL_HEADER W
    LEFT JOIN	(SELECT	WellID, GOR, WTR_CUT 
				 FROM	dbo.vw_FLUID_PROPERTIES 
				 WHERE	DataSource = 'ENVERUS') FP 
	ON			W.WellID = FP.WellID
    LEFT JOIN	(SELECT		WellID, MAX(DateRankDesc) AS TotalProdMonths, MAX(OIL) AS OilMaxMonth,
							MAX(GAS) AS GasMaxMonth, MAX(WATER) AS WaterMaxMonth
				 FROM		dbo.vw_PRODUCTION_PIVOT 
				 WHERE		DataSource = 'ENVERUS' 
				 AND		Cadence = 'MONTHLY' 
				 GROUP BY	WellID) PP 
	ON			W.WellID = PP.WellID;
END;
GO
