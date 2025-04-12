USE [Analytics]
GO

SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

-- Stored procedure used to query data from Enverus database and load into Analytics.dbo.COMPLETION_HEADER
CREATE OR ALTER PROCEDURE [dbo].[sp_InsertFromEnverusToCompletionHeader]
AS
BEGIN
    SET NOCOUNT ON;
	-- Create a temporary table to store data from the Enverus database query
	IF OBJECT_ID('tempdb..#TempEnverusCompData') IS NOT NULL DROP TABLE #TempEnverusCompData;
	CREATE TABLE #TempEnverusCompData (
		CompletionID				BIGINT NOT NULL, 
		WellID						BIGINT NOT NULL,
		API_UWI_12_Unformatted		VARCHAR(32), 
		API_UWI_14_Unformatted		VARCHAR(32),
		Interval					VARCHAR(256),
		Formation					VARCHAR(256),  
		TargetLithology				VARCHAR(256),
		UpperPerf_FT				INT,
		LowerPerf_FT				INT, 
		PerfInterval_FT				INT, 
		TopOfZone_FT				FLOAT,
		BottomOfZone_FT				FLOAT,
		DistFromBaseZone_FT			FLOAT, 
		DistFromTopZone_FT			FLOAT,
		PermitApprovedDate			DATETIME,
		CompletionDate				DATETIME,  
		FirstProdDate				DATETIME,
		CompletionDesign			VARCHAR(64),
		FracJobType					VARCHAR(32),  
		ProppantType				VARCHAR(32),  
		WellServiceProvider			VARCHAR(256),
		Proppant_LBS				FLOAT,
		TotalFluidPumped_BBL		FLOAT,
		FracStages					INT,
		TotalClusters				INT,
		AvgTreatmentPressure_PSI	INT, 
		AvgTreatmentRate_BBLPerMin	FLOAT,
		OilTestRate_BBLPerDAY		FLOAT,
		GasTestRate_MCFPerDAY		FLOAT,
		WaterTestRate_BBLPerDAY		FLOAT,
		TestFCP_PSI					FLOAT,
		TestFTP_PSI					FLOAT,
		TestChokeSize_64IN			INT, 
		ReservoirPressure_PSI		FLOAT,
		Bottom_Hole_Temp_DEGF		FLOAT,
		Isopach_FT					FLOAT,
		EffectivePorosity_PCT		FLOAT,  
		WaterSaturation_PCT			FLOAT,
		OilGravity_API				FLOAT,
		GasGravity_SG				FLOAT,     
		DataSource					VARCHAR(64) NOT NULL,
		DateCreated					DATETIME DEFAULT GETDATE(),
		UpdatedDate					DATETIME,
		SubInterval					VARCHAR(256),
		IHS_WellId					BIGINT
	);

	INSERT INTO #TempEnverusCompData (
		[CompletionID],
		[WellID],
		[API_UWI_12_Unformatted],
		[API_UWI_14_Unformatted],
		[Interval],
		[Formation],
		[TargetLithology],
		[UpperPerf_FT],
		[LowerPerf_FT],
		[PerfInterval_FT],
		[TopOfZone_FT],
		[BottomOfZone_FT],
		[DistFromBaseZone_FT],
		[DistFromTopZone_FT],
		[PermitApprovedDate],
		[CompletionDate],
		[FirstProdDate],
		[CompletionDesign],
		[FracJobType],
		[ProppantType],
		[WellServiceProvider],
		[Proppant_LBS],
		[TotalFluidPumped_BBL],
		[FracStages],
		[TotalClusters],
		[AvgTreatmentPressure_PSI],
		[AvgTreatmentRate_BBLPerMin],
		[OilTestRate_BBLPerDAY],
		[GasTestRate_MCFPerDAY],
		[WaterTestRate_BBLPerDAY],
		[TestFTP_PSI],
		[TestFCP_PSI],
		[TestChokeSize_64IN],
		[ReservoirPressure_PSI],
		[Bottom_Hole_Temp_DEGF],
		[Isopach_FT],
		[EffectivePorosity_PCT],
		[WaterSaturation_PCT],
		[OilGravity_API],
		[GasGravity_SG],
		[DataSource],
		[DateCreated],
		[UpdatedDate],
		[SubInterval],
		[IHS_WellId]
	)
	SELECT      E.CompletionID, E.WellID, E.UWI12 AS API_UWI_12_Unformatted, E.UWI14 AS API_UWI_14_Unformatted, COALESCE(O.Interval, E.Interval) AS Interval, 
				COALESCE(O.Formation, E.Formation) AS Formation, O.TargetLithology, E.UpperPerf_FT, E.LowerPerf_FT, E.PerfInterval_FT, E.TopOfZone_FT, 
				E.BottomOfZone_FT, ABS(COALESCE(O.TVD_FT, E.TVD_FT) - E.BottomOfZone_FT) AS DistFromBaseZone_FT, ABS(COALESCE(O.TVD_FT, E.TVD_FT) - E.TopOfZone_FT) AS DistFromTopZone_FT, 
				COALESCE(O.PermitApprovedDate, E.PermitApprovedDate) AS PermitApprovedDate, E.CompletionDate, COALESCE(O.FirstProdDate, E.FirstProdDate) AS FirstProdDate, 
				E.CompletionDesign, COALESCE(O.FracJobType, E.FracJobType) AS FracJobType, E.ProppantType, E.WellServiceProvider, COALESCE(O.Proppant_LBS, E.Proppant_LBS) AS Proppant_LBS, 
				COALESCE(O.TotalFluidPumped_BBL, E.TotalFluidPumped_BBL) AS TotalFluidPumped_BBL, COALESCE(O.FracStages, E.FracStages) AS FracStages, O.TotalClusters, 
				O.AvgTreatmentPressure_PSI, O.AvgTreatmentRate_BBLPerMin, E.OilTestRate_BBLPerDAY, E.GasTestRate_MCFPerDAY, E.WaterTestRate_BBLPerDAY, O.TestFTP_PSI, 
				O.TestFCP_PSI, E.ChokeSize_64IN AS TestChokeSize_64IN, O.ReservoirPressure_PSI, COALESCE(O.Bottom_Hole_Temp_DEGF, E.BottomHoleTemp_DEGF) AS Bottom_Hole_Temp_DEGF, 
				COALESCE(O.Isopach_FT, E.Isopach_FT) AS Isopach_FT, COALESCE(O.EffectivePorosity_PCT, E.EffectivePorosity_PCT) AS EffectivePorosity_PCT, 
				COALESCE(O.WaterSaturation_PCT, E.WaterSaturation_PCT) AS WaterSaturation_PCT, COALESCE(O.OilGravity_API, E.OilGravity_API) AS OilGravity_API, 
				COALESCE(O.GasGravity_SG, E.GasGravity_SG) AS GasGravity_SG, COALESCE(O.DataSource, 'ENVERUS') AS DataSource, GETDATE() AS DateCreated, E.UpdatedDate, O.SubInterval,
				X._WellId
	FROM        Enverus.dbo.Core_Well E
	INNER JOIN	(SELECT WellID FROM Analytics.dbo.WELL_HEADER) W
	ON			E.WellID = W.WellID
	LEFT JOIN	Analytics.dbo.WELL_OVERRIDE O
	ON			E.WellID = O.WellID
	LEFT JOIN	(SELECT DISTINCT CompletionID, _WellId FROM Analytics.dbo.vw_ID_XREF) X
	ON			E.CompletionID = X.CompletionID;
        
	-- Check if there is any data in the temporary table before attempting the merge operation
	IF EXISTS (SELECT 1 FROM #TempEnverusCompData)
	BEGIN
		-- Perform merge operation to update cases as needed and append new cases
		MERGE INTO [dbo].[COMPLETION_HEADER] AS Target
		USING #TempEnverusCompData AS Source
		ON (Target.[CompletionID] = Source.[CompletionID] AND Target.[WellID] = Source.[WellID] AND Target.[DataSource] = Source.[DataSource])
        
		-- When records are matched, update the records if the source has a newer UpdateDate
		WHEN MATCHED THEN 
			UPDATE SET 
				Target.[API_UWI_12_Unformatted] = Source.[API_UWI_12_Unformatted],
				Target.[API_UWI_14_Unformatted] = Source.[API_UWI_14_Unformatted],
				Target.[Interval] = Source.[Interval],
				Target.[Formation] = Source.[Formation],
				Target.[TargetLithology] = Source.[TargetLithology],
				Target.[UpperPerf_FT] = Source.[UpperPerf_FT],
				Target.[LowerPerf_FT] = Source.[LowerPerf_FT],
				Target.[PerfInterval_FT] = Source.[PerfInterval_FT],
				Target.[TopOfZone_FT] = Source.[TopOfZone_FT],
				Target.[BottomOfZone_FT] = Source.[BottomOfZone_FT],
				Target.[DistFromBaseZone_FT] = Source.[DistFromBaseZone_FT],
				Target.[DistFromTopZone_FT] = Source.[DistFromTopZone_FT],
				Target.[PermitApprovedDate] = Source.[PermitApprovedDate],
				Target.[CompletionDate] = Source.[CompletionDate],
				Target.[FirstProdDate] = Source.[FirstProdDate],
				Target.[CompletionDesign] = Source.[CompletionDesign],
				Target.[FracJobType] = Source.[FracJobType],
				Target.[ProppantType] = Source.[ProppantType],
				Target.[WellServiceProvider] = Source.[WellServiceProvider],
				Target.[Proppant_LBS] = Source.[Proppant_LBS],
				Target.[TotalFluidPumped_BBL] = Source.[TotalFluidPumped_BBL],
				Target.[FracStages] = Source.[FracStages],
				Target.[TotalClusters] = Source.[TotalClusters],
				Target.[AvgTreatmentPressure_PSI] = Source.[AvgTreatmentPressure_PSI],
				Target.[AvgTreatmentRate_BBLPerMin] = Source.[AvgTreatmentRate_BBLPerMin],
				Target.[OilTestRate_BBLPerDAY] = Source.[OilTestRate_BBLPerDAY],
				Target.[GasTestRate_MCFPerDAY] = Source.[GasTestRate_MCFPerDAY],
				Target.[WaterTestRate_BBLPerDAY] = Source.[WaterTestRate_BBLPerDAY],
				Target.[TestFTP_PSI] = Source.[TestFTP_PSI],
				Target.[TestFCP_PSI] = Source.[TestFCP_PSI],
				Target.[TestChokeSize_64IN] = Source.[TestChokeSize_64IN],
				Target.[ReservoirPressure_PSI] = Source.[ReservoirPressure_PSI],
				Target.[Bottom_Hole_Temp_DEGF] = Source.[Bottom_Hole_Temp_DEGF],
				Target.[Isopach_FT] = Source.[Isopach_FT],
				Target.[EffectivePorosity_PCT] = Source.[EffectivePorosity_PCT],
				Target.[WaterSaturation_PCT] = Source.[WaterSaturation_PCT],
				Target.[OilGravity_API] = Source.[OilGravity_API],
				Target.[GasGravity_SG] = Source.[GasGravity_SG],
				Target.[DateCreated] = Source.[UpdatedDate],
				Target.[SubInterval] = Source.[SubInterval],
				Target.[IHS_WellId] = Source.[IHS_WellId]
        
		-- When no records are matched, insert the incoming records from the source table
		WHEN NOT MATCHED BY TARGET THEN
			INSERT (
				[WellID],
				[CompletionID],
				[API_UWI_12_Unformatted],
				[API_UWI_14_Unformatted],
				[Interval],
				[Formation],
				[TargetLithology],
				[UpperPerf_FT],
				[LowerPerf_FT],
				[PerfInterval_FT],
				[TopOfZone_FT],
				[BottomOfZone_FT],
				[DistFromBaseZone_FT],
				[DistFromTopZone_FT],
				[PermitApprovedDate],
				[CompletionDate],
				[FirstProdDate],
				[CompletionDesign],
				[FracJobType],
				[ProppantType],
				[WellServiceProvider],
				[Proppant_LBS],
				[TotalFluidPumped_BBL],
				[FracStages],
				[TotalClusters],
				[AvgTreatmentPressure_PSI],
				[AvgTreatmentRate_BBLPerMin],
				[OilTestRate_BBLPerDAY],
				[GasTestRate_MCFPerDAY],
				[WaterTestRate_BBLPerDAY],
				[TestFTP_PSI],
				[TestFCP_PSI],
				[TestChokeSize_64IN],
				[ReservoirPressure_PSI],
				[Bottom_Hole_Temp_DEGF],
				[Isopach_FT],
				[EffectivePorosity_PCT],
				[WaterSaturation_PCT],
				[OilGravity_API],
				[GasGravity_SG],
				[DataSource],
				[DateCreated],
				[SubInterval],
				[IHS_WellId]
			)
			VALUES (
				Source.[WellID],
				Source.[CompletionID],
				Source.[API_UWI_12_Unformatted],
				Source.[API_UWI_14_Unformatted],
				Source.[Interval],
				Source.[Formation],
				Source.[TargetLithology],
				Source.[UpperPerf_FT],
				Source.[LowerPerf_FT],
				Source.[PerfInterval_FT],
				Source.[TopOfZone_FT],
				Source.[BottomOfZone_FT],
				Source.[DistFromBaseZone_FT],
				Source.[DistFromTopZone_FT],
				Source.[PermitApprovedDate],
				Source.[CompletionDate],
				Source.[FirstProdDate],
				Source.[CompletionDesign],
				Source.[FracJobType],
				Source.[ProppantType],
				Source.[WellServiceProvider],
				Source.[Proppant_LBS],
				Source.[TotalFluidPumped_BBL],
				Source.[FracStages],
				Source.[TotalClusters],
				Source.[AvgTreatmentPressure_PSI],
				Source.[AvgTreatmentRate_BBLPerMin],
				Source.[OilTestRate_BBLPerDAY],
				Source.[GasTestRate_MCFPerDAY],
				Source.[WaterTestRate_BBLPerDAY],
				Source.[TestFTP_PSI],
				Source.[TestFCP_PSI],
				Source.[TestChokeSize_64IN],
				Source.[ReservoirPressure_PSI],
				Source.[Bottom_Hole_Temp_DEGF],
				Source.[Isopach_FT],
				Source.[EffectivePorosity_PCT],
				Source.[WaterSaturation_PCT],
				Source.[OilGravity_API],
				Source.[GasGravity_SG],
				Source.[DataSource],
				Source.[UpdatedDate],
				Source.[SubInterval],
				Source.[IHS_WellId]
			);
        
		-- Output the number of rows affected by the merge operation
		PRINT CAST(@@ROWCOUNT AS NVARCHAR(50)) + ' rows merged into COMPLETION_HEADER from Enverus source data';
	END
	ELSE
	BEGIN
		-- If no data exists in the temp table, you might choose to log this or take other appropriate actions
        PRINT 'No data found. Skipping merge operation.';
	END
END;
GO
