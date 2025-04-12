USE [Analytics]
GO

/****** Object:  StoredProcedure [dbo].[sp_PopulateProductionMaterialized]    Script Date: 3/16/2024 2:37:43 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



CREATE OR ALTER     PROCEDURE [dbo].[sp_PopulateProductionMaterialized] AS

BEGIN
    SET NOCOUNT ON;
	SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

	DECLARE @BatchSize INT = 50000, @CurrentBatch INT = 1, @TotalBatches INT

	-- Drop Index on dbo.ProductionMaterialized
	DROP INDEX IF EXISTS IX_ProductionMaterialized_WellID ON dbo.ProductionMaterialized;

	-- Truncate dbo.ProductionMaterialized
	TRUNCATE TABLE dbo.ProductionMaterialized;

	-- Create temp table for the basic well data
	IF OBJECT_ID('tempdb..#WellData') IS NOT NULL DROP TABLE #WellData;
	CREATE TABLE #WellData (
		WellID                  BIGINT NOT NULL,
		Basin					VARCHAR(64),
		Trajectory				VARCHAR(64),
		LL_BIN					VARCHAR(10),
		LateralLength_FT		FLOAT,
		Prop_Intensity			FLOAT,
		Fluid_Intensity			FLOAT,
		DataSourceRank			INT,
		BatchNumber				INT
	);

	-- Populate #WellData, REPLACE #NAME# WITH YOUR QUALIFIER NAME
	INSERT INTO #WellData (WellID, Basin, Trajectory, LL_BIN, LateralLength_FT, Prop_Intensity, Fluid_Intensity, DataSourceRank, BatchNumber)
	SELECT		WellID, Basin, Trajectory, LL_BIN, LateralLength_FT, Prop_Intensity, Fluid_Intensity,
				ROW_NUMBER() OVER (PARTITION BY WellID ORDER BY CASE WHEN CompletionDataSource = '#NAME#' THEN 1 WHEN CompletionDataSource = 'ENVERUS' THEN 2 ELSE 3 END) AS DataSourceRank,
				CEILING(ROW_NUMBER() OVER (ORDER BY WellID) / CAST(@BatchSize AS FLOAT)) AS BatchNumber
	FROM		dbo.vw_WELL_HEADER
	WHERE		FirstProdDate IS NOT NULL

	-- Add indexes for #WellData
	CREATE NONCLUSTERED INDEX idx_WellData_BatchNumber ON #WellData (BatchNumber);
	CREATE NONCLUSTERED INDEX idx_WellData_WellID_BatchNumber ON #WellData (WellID, BatchNumber);

	-- Set value of @TotalBatches
	SELECT @TotalBatches = MAX(BatchNumber) FROM #WellData;

	-- Insert data into the materialized table
	WHILE @CurrentBatch <= @TotalBatches
	BEGIN

		PRINT 'Inserting batch ' + CAST(@CurrentBatch AS VARCHAR(10)) + ' of ' + CAST(@TotalBatches AS VARCHAR(10));

		INSERT INTO dbo.ProductionMaterialized WITH (TABLOCK) (
			SourceID, WellID, Basin, Trajectory, Measure, Date, ProdMonth, MonthlyVolume, CumulativeProduction, LateralLength_FT, Prop_Intensity,
			Fluid_Intensity, LL_BIN, DateCreated, Comment
		)
		SELECT		P.ProductionID AS SourceID, P.WellID, W.Basin, W.Trajectory, P.Measure, P.Date, 
					ROW_NUMBER() OVER (PARTITION BY P.WellID, P.Measure ORDER BY P.Date) AS ProdMonth, 
					P.Value AS MonthlyVolume, SUM(P.Value) OVER (PARTITION BY P.WellID, P.Measure ORDER BY P.Date) AS CumulativeProduction, 
					W.LateralLength_FT, W.Prop_Intensity, W.Fluid_Intensity, W.LL_BIN, P.DateCreated, P.Comment
		FROM		dbo.PRODUCTION P
		INNER JOIN	#WellData W
		ON			P.WellID = W.WellID
		WHERE		P.DataSource = 'ENVERUS'
		AND			P.Cadence = 'MONTHLY'
		AND			W.DataSourceRank = 1
		AND			W.BatchNumber = @CurrentBatch

		SET @CurrentBatch = @CurrentBatch + 1;
	END;

	-- Clean up the temporary table
	IF OBJECT_ID('tempdb..#WellData') IS NOT NULL 
		DROP TABLE #WellData;

	-- Recreate IX_ProductionMaterialized_WellID
	CREATE NONCLUSTERED INDEX IX_ProductionMaterialized_WellID ON dbo.ProductionMaterialized (WellID);

	-- Update statistics for performance
	UPDATE STATISTICS dbo.ProductionMaterialized;
END;
GO


