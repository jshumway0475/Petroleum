USE [Analytics]
GO

SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

/*******************************************************************************************
Example Execution:
EXEC [dbo].[sp_InsertFromEnverusToWellHeader] 
    @PrimaryDataSource = 'DATASOURCE1';
********************************************************************************************/


-- Stored procedure used to query data from Enverus database and load into Analytics.dbo.WELL_HEADER
CREATE OR ALTER PROCEDURE [dbo].[sp_InsertFromEnverusToWellHeader]
	@PrimaryDataSource NVARCHAR(50)
AS
BEGIN
    SET NOCOUNT ON;
      
	-- Create a temporary table to store data from the Enverus database query
	IF OBJECT_ID('tempdb..#TempEnverusWellData') IS NOT NULL DROP TABLE #TempEnverusWellData;
	CREATE TABLE #TempEnverusWellData (
		WellID						BIGINT NOT NULL,
		CurrentCompletionID			BIGINT,
		API_UWI_Unformatted			VARCHAR(32),
		LeaseName					VARCHAR(256), 
		WellName					VARCHAR(256), 
		WellNumber					VARCHAR(256), 
		WellPadID					VARCHAR(128),
		Latitude					FLOAT, 
		Latitude_BH					FLOAT, 
		Longitude					FLOAT, 
		Longitude_BH				FLOAT,
		Country						VARCHAR(2), 
		StateProvince				VARCHAR(64),
		County						VARCHAR(32),
		Township					VARCHAR(32),
		Range						VARCHAR(32),
		Section						VARCHAR(32),
		Abstract					VARCHAR(16),
		Block						VARCHAR(64),
		Survey						VARCHAR(32),
		District					VARCHAR(256),
		Field						VARCHAR(256), 
		Basin						VARCHAR(64),
		Play						VARCHAR(128),
		SubPlay						VARCHAR(64),
		InitialOperator				VARCHAR(256),
		Operator					VARCHAR(256),
		Ticker						VARCHAR(128),
		GasGatherer					VARCHAR(256),
		OilGatherer					VARCHAR(256),   
		ProducingMethod				VARCHAR(256),
		WellStatus					VARCHAR(64), 
		WellType					VARCHAR(256),
		PermitApprovedDate			DATETIME,
		SpudDate					DATETIME,
		RigReleaseDate				DATETIME,
		FirstProdDate				DATETIME,
		MD_FT						FLOAT,
		TVD_FT						FLOAT,
		ElevationGL_FT				FLOAT,
		Trajectory					VARCHAR(64),
		LateralLength_FT			FLOAT,
		AzimuthFromGridNorth_DEG	FLOAT, 
		ToeUp						INT,
		Geometry					VARCHAR(MAX),
		SourceEPSG					INT,
		CasingSize_IN				VARCHAR(256),        
		TubingDepth_FT				FLOAT, 
		TubingSize_IN				VARCHAR(256), 
		DataSource					VARCHAR(64),
		DateCreated					DATETIME DEFAULT GETDATE(),
		UpdatedDate					DATETIME,
		FitGroup					VARCHAR(256),
		Comment						VARCHAR(MAX),
		FitMethod					VARCHAR(256)
	);

	DECLARE @currentDate DATETIME = GETDATE();
		
	-- CTE definitions used to query Enverus dataset
	WITH BaseCTE AS (
		SELECT		DISTINCT E.WellID, E.WellStatus AS BaseWellStatus
		FROM		Enverus.dbo.Core_Well E
		LEFT JOIN	Analytics.dbo.COMPLETION_HEADER C
		ON			E.WellID = C.WellID 
		AND			E.CompletionID = C.CompletionID
		LEFT JOIN	Analytics.dbo.WELL_OVERRIDE O
		ON			E.WellID = O.WellID
		WHERE		E.Country = 'US'
		AND			(E.WellStatus IS NOT NULL OR O.WellStatus IS NOT NULL)
		AND			(E.WellStatus NOT LIKE '%PERMIT%' OR O.WellStatus NOT LIKE '%PERMIT%')
		AND			E.WellStatus NOT IN ('UNREPORTED', 'SPUD DATE ONLY')
		AND			(E.UpdatedDate > C.DateCreated OR C.DateCreated IS NULL)
	),
	-- COMP_CTE: Enrich each base well with Core_Well details plus data from COMPLETION_HEADER and WELL_OVERRIDE.
	COMP_CTE AS (
		SELECT	B.WellID, CA.CompletionID, CA.API_UWI_Unformatted, CA.API_UWI_12_Unformatted, CA.API_UWI_14_Unformatted, CA.LeaseName, 
				CA.WellName, CA.WellPadID, CA.Latitude, CA.Latitude_BH, CA.Longitude, CA.Longitude_BH, CA.Country, CA.StateProvince, 
				CA.County, CA.Township, CA.[Range], CA.Section, CA.Abstract, CA.Block, CA.District, CA.Field, CA.Basin, CA.Play, CA.SubPlay,
				CA.InitialOperator, CA.Operator, CA.Ticker, CA.GasGatherer, CA.OilGatherer, CA.PermitApprovedDate, CA.ENVPermitApprovedDate, 
				CA.SpudDate, CA.RigReleaseDate, CA.FirstProdDate, CA.ENVSpudDate, CA.ENVRigReleaseDate, CA.ENVFirstProdDate, CA.MD_FT, 
				CA.TVD_FT, CA.ElevationGL_FT, CA.Trajectory, CA.LateralLength_FT, CA.AzimuthFromGridNorth_DEG, CA.ToeUp, CA.ProducingMethod, 
				CA.JoinedWellStatus, CA.WellType, CA.DataSource, CA.UpdatedDate, CA.ENVGeometry, CA.ENVEPSG, CA.Geometry, CA.EPSG
		FROM	BaseCTE B
		OUTER APPLY (
			SELECT TOP 1
				E.WellID, COALESCE(E.CompletionID, O.CurrentCompletionID) AS CompletionID, E.UWI AS API_UWI_Unformatted, E.UWI12 AS API_UWI_12_Unformatted,
				E.UWI14 AS API_UWI_14_Unformatted, E.LeaseName, COALESCE(O.WellName, E.WellName) AS WellName, E.WellPadID, E.Latitude, E.Latitude_BH, 
				E.Longitude, E.Longitude_BH, E.Country, E.StateProvince, E.County, E.Township, E.[Range], E.Section, E.Abstract, E.Block, E.District, 
				E.Field, COALESCE(O.Basin, E.Basin) AS Basin, COALESCE(O.Play, E.Play) AS Play, COALESCE(O.Subplay, E.SubPlay) AS SubPlay,
				COALESCE(O.InitialOperator, E.InitialOperator) AS InitialOperator, COALESCE(O.Operator, E.Operator) AS Operator, E.Ticker,
				COALESCE(O.GasGatherer, E.GasGatherer) AS GasGatherer, COALESCE(O.OilGatherer, E.OilGatherer) AS OilGatherer, O.PermitApprovedDate,
				(SELECT MIN(x.PermitApprovedDate) FROM Enverus.dbo.Core_Well x  WHERE x.WellID = E.WellID) AS ENVPermitApprovedDate,
				O.SpudDate, O.RigReleaseDate, O.FirstProdDate,
				(SELECT MIN(x.SpudDate) FROM Enverus.dbo.Core_Well x WHERE x.WellID = E.WellID) AS ENVSpudDate,
				(SELECT MIN(x.RigReleaseDate) FROM Enverus.dbo.Core_Well x WHERE x.WellID = E.WellID) AS ENVRigReleaseDate,
				(SELECT MIN(x.FirstProdDate) FROM Enverus.dbo.Core_Well x WHERE x.WellID = E.WellID) AS ENVFirstProdDate,
				COALESCE(O.MD_FT, E.MD_FT) AS MD_FT, COALESCE(O.TVD_FT, E.TVD_FT) AS TVD_FT, COALESCE(O.ElevationGL_FT, E.ElevationGL_FT) AS ElevationGL_FT,
				COALESCE(O.Trajectory, E.Trajectory) AS Trajectory, COALESCE(O.LateralLength_FT, E.LateralLength_FT) AS LateralLength_FT,
				E.AzimuthFromGridNorth_DEG, CASE WHEN E.ToeAngle_DEG > 90 THEN 1 ELSE 0 END AS ToeUp, 
				COALESCE(O.ProducingMethod, E.ProducingMethod) AS ProducingMethod, COALESCE(O.WellStatus, E.WellStatus) AS JoinedWellStatus,
				COALESCE(O.WellType, E.WellType) AS WellType, COALESCE(O.DataSource, 'ENVERUS') AS DataSource, E.UpdatedDate, C.DateCreated,
				E.LateralLine.STAsText() AS ENVGeometry, E.LateralLine.STSrid AS ENVEPSG, O.Geometry.STAsText() AS Geometry, O.Geometry.STSrid AS EPSG
			FROM 
				Enverus.dbo.Core_Well E
			LEFT JOIN 
				Analytics.dbo.COMPLETION_HEADER C
			ON	E.WellID = C.WellID AND E.CompletionID = C.CompletionID
			LEFT JOIN 
				Analytics.dbo.WELL_OVERRIDE O
			ON	(E.WellID = O.WellID AND (O.CurrentCompletionID IS NULL OR E.CompletionID = O.CurrentCompletionID))
			WHERE 
				E.WellID = B.WellID AND E.Country = 'US'
			ORDER BY 
				CASE WHEN O.DataSource = @PrimaryDataSource THEN 1 ELSE 2 END,
				COALESCE(E.CompletionNumber, 0) DESC
		) CA
	),
	-- CSG_CTE: Retrieve casing details (if available).
	CSG_CTE AS (
		SELECT	B.WellID, CSG.CasingSize_IN
		FROM	BaseCTE B
		OUTER APPLY (
			 SELECT TOP 1 
				 TRY_CAST(C.CasingSize_IN AS FLOAT) AS CasingSize_IN
			 FROM 
				Enverus.dbo.Casing C
			 INNER JOIN (
				 SELECT WellID, MIN(TRY_CAST(CasingSize_IN AS FLOAT)) AS MinSize
				 FROM Enverus.dbo.Casing
				 WHERE TRY_CAST(CasingSize_IN AS FLOAT) IS NOT NULL
				 GROUP BY WellID
			 ) M ON C.WellID = M.WellID
			 WHERE C.WellID = B.WellID AND TRY_CAST(C.CasingSize_IN AS FLOAT) = M.MinSize
			 ORDER BY C.SettingDepth_FT DESC
		) CSG
	),
	-- TBG_CTE: Retrieve tubing details (if available).
	TBG_CTE AS (
		SELECT	B.WellID, TA.TubingSize_IN, TA.TubingDepth_FT
		FROM	BaseCTE B
		OUTER APPLY (
			 SELECT TOP 1 
				 TRY_CAST(T.TubingSize_IN AS FLOAT) AS TubingSize_IN, T.TubingDepth_FT
			 FROM 
				Enverus.dbo.Tubing T
			 INNER JOIN (
				 SELECT WellID, MIN(TRY_CAST(TubingSize_IN AS FLOAT)) AS MinSize
				 FROM Enverus.dbo.Tubing
				 WHERE TRY_CAST(TubingSize_IN AS FLOAT) IS NOT NULL
				 GROUP BY WellID
			 ) M ON T.WellID = M.WellID
			 WHERE T.WellID = B.WellID AND TRY_CAST(T.TubingSize_IN AS FLOAT) = M.MinSize
			 ORDER BY T.TubingDepth_FT DESC
		) TA
	)

	INSERT INTO #TempEnverusWellData (
		[WellID],
		[CurrentCompletionID],
		[API_UWI_Unformatted],
		[LeaseName],
		[WellName],
		[WellNumber],
		[WellPadID],
		[Latitude],
		[Latitude_BH],
		[Longitude],
		[Longitude_BH],
		[Country],
		[StateProvince],
		[County],
		[Township],
		[Range],
		[Section],
		[Abstract],
		[Block],
		[Survey],
		[District],
		[Field],
		[Basin],
		[Play],
		[SubPlay],
		[InitialOperator],
		[Operator],
		[Ticker],
		[GasGatherer],
		[OilGatherer],
		[ProducingMethod],
		[WellStatus],
		[WellType],
		[PermitApprovedDate],
		[SpudDate],
		[RigReleaseDate],
		[FirstProdDate],
		[MD_FT],
		[TVD_FT],
		[ElevationGL_FT],
		[Trajectory],
		[LateralLength_FT],
		[AzimuthFromGridNorth_DEG],
		[ToeUp],
		[Geometry],
		[SourceEPSG],
		[CasingSize_IN],
		[TubingDepth_FT],
		[TubingSize_IN],
		[DataSource],
		[DateCreated],
		[UpdatedDate],
		[FitGroup],
		[Comment],
		[FitMethod]
	)
	SELECT		B.WellID, W.CompletionID AS CurrentCompletionID, W.API_UWI_Unformatted, W.LeaseName, W.WellName, NULL AS WellNumber,
				W.WellPadID, W.Latitude, W.Latitude_BH, W.Longitude, W.Longitude_BH, W.Country, W.StateProvince, W.County, W.Township, 
				W.[Range], W.Section, W.Abstract, W.Block, NULL AS Survey, W.District, W.Field, W.Basin, W.Play, W.SubPlay, W.InitialOperator, 
				W.Operator, W.Ticker, W.GasGatherer, W.OilGatherer, W.ProducingMethod, COALESCE(W.JoinedWellStatus, B.BaseWellStatus) AS WellStatus,
				W.WellType, COALESCE(W.PermitApprovedDate, W.ENVPermitApprovedDate) AS PermitApprovedDate, COALESCE(W.SpudDate, W.ENVSpudDate) AS SpudDate,
				COALESCE(W.RigReleaseDate, W.ENVRigReleaseDate) AS RigReleaseDate, COALESCE(W.FirstProdDate, W.ENVFirstProdDate) AS FirstProdDate,
				W.MD_FT, W.TVD_FT, W.ElevationGL_FT, W.Trajectory, W.LateralLength_FT, W.AzimuthFromGridNorth_DEG, W.ToeUp, 
				COALESCE(W.Geometry, W.ENVGeometry) AS Geometry, COALESCE(W.EPSG, W.ENVEPSG) AS SourceEPSG, CSG.CasingSize_IN, TBG.TubingDepth_FT,
				TBG.TubingSize_IN, W.DataSource, @currentDate AS DateCreated, W.UpdatedDate, NULL AS FitGroup, NULL AS Comment, NULL AS FitMethod
	FROM		BaseCTE B
	LEFT JOIN	COMP_CTE W ON B.WellID = W.WellID
	LEFT JOIN	CSG_CTE CSG ON B.WellID = CSG.WellID
	LEFT JOIN	TBG_CTE TBG ON B.WellID = TBG.WellID
	LEFT JOIN (
		SELECT WellID, DateCreated 
		FROM Analytics.dbo.WELL_HEADER
	) A ON B.WellID = A.WellID;
        
	-- Check if there is any data in the temporary table before attempting the merge operation
	IF EXISTS (SELECT 1 FROM #TempEnverusWellData)
	BEGIN
		-- Perform merge operation to update cases as needed and append new cases
		MERGE INTO [dbo].[WELL_HEADER] WITH (HOLDLOCK) AS Target
		USING #TempEnverusWellData AS Source
		ON Target.[WellID] = Source.[WellID]
		WHEN MATCHED THEN 
			UPDATE SET 
				Target.[CurrentCompletionID] = Source.[CurrentCompletionID],
				Target.[API_UWI_Unformatted] = Source.[API_UWI_Unformatted],
				Target.[LeaseName] = Source.[LeaseName],
				Target.[WellName] = Source.[WellName],
				Target.[WellNumber] = Source.[WellNumber],
				Target.[WellPadID] = Source.[WellPadID],
				Target.[Latitude] = Source.[Latitude],
				Target.[Latitude_BH] = Source.[Latitude_BH],
				Target.[Longitude] = Source.[Longitude],
				Target.[Longitude_BH] = Source.[Longitude_BH],
				Target.[Country] = Source.[Country],
				Target.[StateProvince] = Source.[StateProvince],
				Target.[County] = Source.[County],
				Target.[Township] = Source.[Township],
				Target.[Range] = Source.[Range],
				Target.[Section] = Source.[Section],
				Target.[Abstract] = Source.[Abstract],
				Target.[Block] = Source.[Block],
				Target.[Survey] = Source.[Survey],
				Target.[District] = Source.[District],
				Target.[Field] = Source.[Field],
				Target.[Basin] = Source.[Basin],
				Target.[Play] = Source.[Play],
				Target.[SubPlay] = Source.[SubPlay],
				Target.[InitialOperator] = Source.[InitialOperator],
				Target.[Operator] = Source.[Operator],
				Target.[Ticker] = Source.[Ticker],
				Target.[GasGatherer] = Source.[GasGatherer],
				Target.[OilGatherer] = Source.[OilGatherer],
				Target.[ProducingMethod] = Source.[ProducingMethod],
				Target.[WellStatus] = Source.[WellStatus],
				Target.[WellType] = Source.[WellType],
				Target.[PermitApprovedDate] = Source.[PermitApprovedDate],
				Target.[SpudDate] = Source.[SpudDate],
				Target.[RigReleaseDate] = Source.[RigReleaseDate],
				Target.[FirstProdDate] = Source.[FirstProdDate],
				Target.[MD_FT] = Source.[MD_FT],
				Target.[TVD_FT] = Source.[TVD_FT],
				Target.[ElevationGL_FT] = Source.[ElevationGL_FT],
				Target.[Trajectory] = Source.[Trajectory],
				Target.[LateralLength_FT] = Source.[LateralLength_FT],
				Target.[AzimuthFromGridNorth_DEG] = Source.[AzimuthFromGridNorth_DEG],
				Target.[ToeUp] = Source.[ToeUp],
				Target.[Geometry] = geometry::STGeomFromText(Source.[Geometry], 4326).MakeValid(),
				Target.[SourceEPSG] = Source.[SourceEPSG],
				Target.[CasingSize_IN] = Source.[CasingSize_IN],
				Target.[TubingDepth_FT] = Source.[TubingDepth_FT],
				Target.[TubingSize_IN] = Source.[TubingSize_IN],
				Target.[DataSource] = Source.[DataSource],
				Target.[DateCreated] = Source.[UpdatedDate]
		WHEN NOT MATCHED BY TARGET THEN
			INSERT (
				[WellID],
				[CurrentCompletionID],
				[API_UWI_Unformatted],
				[LeaseName],
				[WellName],
				[WellNumber],
				[WellPadID],
				[Latitude],
				[Latitude_BH],
				[Longitude],
				[Longitude_BH],
				[Country],
				[StateProvince],
				[County],
				[Township],
				[Range],
				[Section],
				[Abstract],
				[Block],
				[Survey],
				[District],
				[Field],
				[Basin],
				[Play],
				[SubPlay],
				[InitialOperator],
				[Operator],
				[Ticker],
				[GasGatherer],
				[OilGatherer],
				[ProducingMethod],
				[WellStatus],
				[WellType],
				[PermitApprovedDate],
				[SpudDate],
				[RigReleaseDate],
				[FirstProdDate],
				[MD_FT],
				[TVD_FT],
				[ElevationGL_FT],
				[Trajectory],
				[LateralLength_FT],
				[AzimuthFromGridNorth_DEG],
				[ToeUp],
				[Geometry],
				[SourceEPSG],
				[CasingSize_IN],
				[TubingDepth_FT],
				[TubingSize_IN],
				[DataSource],
				[DateCreated],
				[FitGroup],
				[Comment],
				[FitMethod]
			)
			VALUES (
				Source.[WellID],
				Source.[CurrentCompletionID],
				Source.[API_UWI_Unformatted],
				Source.[LeaseName],
				Source.[WellName],
				Source.[WellNumber],
				Source.[WellPadID],
				Source.[Latitude],
				Source.[Latitude_BH],
				Source.[Longitude],
				Source.[Longitude_BH],
				Source.[Country],
				Source.[StateProvince],
				Source.[County],
				Source.[Township],
				Source.[Range],
				Source.[Section],
				Source.[Abstract],
				Source.[Block],
				Source.[Survey],
				Source.[District],
				Source.[Field],
				Source.[Basin],
				Source.[Play],
				Source.[SubPlay],
				Source.[InitialOperator],
				Source.[Operator],
				Source.[Ticker],
				Source.[GasGatherer],
				Source.[OilGatherer],
				Source.[ProducingMethod],
				Source.[WellStatus],
				Source.[WellType],
				Source.[PermitApprovedDate],
				Source.[SpudDate],
				Source.[RigReleaseDate],
				Source.[FirstProdDate],
				Source.[MD_FT],
				Source.[TVD_FT],
				Source.[ElevationGL_FT],
				Source.[Trajectory],
				Source.[LateralLength_FT],
				Source.[AzimuthFromGridNorth_DEG],
				Source.[ToeUp],
				geometry::STGeomFromText(Source.[Geometry], 4326).MakeValid(),
				Source.[SourceEPSG],
				Source.[CasingSize_IN],
				Source.[TubingDepth_FT],
				Source.[TubingSize_IN],
				Source.[DataSource],
				Source.[UpdatedDate],
				Source.[FitGroup],
				Source.[Comment],
				Source.[FitMethod]
			);
        
		PRINT CAST(@@ROWCOUNT AS NVARCHAR(50)) + ' rows merged into WELL_HEADER from Enverus source data';
	END
	ELSE
	BEGIN
		-- If no data exists in the temp table, you might choose to log this or take other appropriate actions
        PRINT 'No data found. Skipping merge operation.';
	END
END;
GO
