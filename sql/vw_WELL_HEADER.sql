USE [Analytics]
GO

/****** Object:  View [dbo].[vw_WELL_HEADER]    Script Date: 10/8/2024 10:35:12 AM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE OR ALTER     VIEW [dbo].[vw_WELL_HEADER] AS
WITH PARENT_CHILD_CTE AS (
	SELECT	WellID, Relationship, ClosestHzDistance, ClosestHzDistance_Left, ClosestHzDistance_Right,
			ScenarioName, ROW_NUMBER() OVER (PARTITION BY WellID, ScenarioName ORDER BY Date DESC) AS DateRank
	FROM	dbo.PARENT_CHILD
), COMPLETION_CTE AS (
	SELECT		CH.WellID, CH.CompletionID, CH.Interval, CH.SubInterval, CH.Formation, CH.TargetLithology, CH.FracJobType, CH.FracStages, CH.Proppant_LBS, 
				CH.TotalFluidPumped_BBL, CH.UpperPerf_FT, CH.LowerPerf_FT, CH.Isopach_FT, CH.WaterSaturation_PCT, CH.EffectivePorosity_PCT, CH.ReservoirPressure_PSI, 
				CH.Bottom_Hole_Temp_DEGF, CH.OilGravity_API, CH.GasGravity_SG, CH.DataSource AS CompletionDataSource, X.IHS_Status,
				DENSE_RANK() OVER (PARTITION BY CH.CompletionID ORDER BY CASE WHEN CH.DataSource = 'DATASOURCE1' THEN 1 WHEN CH.DataSource = 'DATASOURCE2' THEN 2 ELSE 3 END) AS DataSourceRank
	FROM		dbo.COMPLETION_HEADER CH
	LEFT JOIN	(SELECT DISTINCT CompletionID, IHS_Status FROM dbo.vw_ID_XREF) X
	ON			CH.CompletionID = X.CompletionID
)
SELECT		W.*, C.IHS_Status, PC.Relationship, PC.ClosestHzDistance, PC.ClosestHzDistance_Left, PC.ClosestHzDistance_Right, COALESCE(PC.ScenarioName, 'NONE') AS ScenarioName, 
			UPPER(C.Interval) AS Interval, UPPER(C.SubInterval) AS SubInterval, C.Formation, C.TargetLithology, C.FracJobType, C.FracStages, C.Proppant_LBS, C.TotalFluidPumped_BBL, 
			C.Proppant_LBS / NULLIF(W.LateralLength_FT, 0) AS Prop_Intensity, C.TotalFluidPumped_BBL / NULLIF(W.LateralLength_FT, 0) AS Fluid_Intensity,
			CASE 
				WHEN COALESCE(W.LateralLength_FT, 0) < 4500 THEN 'SHORT' 
				WHEN W.LateralLength_FT BETWEEN 4500 AND 5499 THEN '5K' 
				WHEN W.LateralLength_FT BETWEEN 5500 AND 8499 THEN '7.5K' 
				WHEN W.LateralLength_FT BETWEEN 8500 AND 11000 THEN '10K' 
				ELSE 'LONG' 
			END AS LL_BIN,
			C.UpperPerf_FT, C.LowerPerf_FT, C.Isopach_FT, C.WaterSaturation_PCT, C.EffectivePorosity_PCT, 
			C.ReservoirPressure_PSI, C.Bottom_Hole_Temp_DEGF, C.OilGravity_API, C.GasGravity_SG, W2.GOR, 
			W2.WTR_CUT, W2.WTR_YIELD, W2.TotalProdMonths, W2.OilMaxMonth, W2.GasMaxMonth, W2.WaterMaxMonth, W2.PrimaryPhase, 
			C.CompletionDataSource
FROM		dbo.WELL_HEADER W
INNER JOIN	dbo.WellHeaderMaterialized W2
ON			((W.WellID = W2.WellID) AND (W.CurrentCompletionID = W2.CurrentCompletionID))
INNER JOIN 	COMPLETION_CTE C ON ((W.CurrentCompletionID = C.CompletionID) AND (W.WellID = C.WellID) AND C.DataSourceRank = 1)
LEFT JOIN 	(SELECT * FROM PARENT_CHILD_CTE WHERE DateRank = 1) PC ON W.WellID = PC.WellID
WHERE		W.WellStatus IN ('COMPLETED', 'DRILLED', 'DRILLING', 'DUC', 'PRODUCING', 'INACTIVE PRODUCER')
GO


