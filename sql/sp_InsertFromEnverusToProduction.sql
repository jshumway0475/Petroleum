USE [Analytics]
GO

/****** Object:  StoredProcedure [dbo].[sp_InsertFromEnverusToProduction]    Script Date: 2/26/2024 2:57:03 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- Stored procedure used to query data from Enverus database and load into Analytics.dbo.PRODUCTION
CREATE OR ALTER   PROCEDURE [dbo].[sp_InsertFromEnverusToProduction]
AS

BEGIN
    SET NOCOUNT ON;
	SET XACT_ABORT ON;

	-- Check and create Staging Table if it doesn't exist
    IF OBJECT_ID('dbo.Staging_EnverusProdData', 'U') IS NULL
    BEGIN
        CREATE TABLE dbo.Staging_EnverusProdData (
            WellID                  BIGINT NOT NULL,
            API_UWI_Unformatted     VARCHAR(32) NULL,
            Date                    DATETIME NOT NULL,
            ProducingDays           INT NULL,
            DataSource              VARCHAR(64) NULL,
            Cadence                 VARCHAR(64) NULL,
            Comment                 VARCHAR(MAX) NULL,
            Measure                 VARCHAR(64) NULL,
            Value                   FLOAT NULL, 
            Units                   VARCHAR(64) NULL,
            DateCreated             DATETIME NULL
        );

        -- Create Index on Staging Table
        CREATE NONCLUSTERED INDEX IX_Staging_WellID_Date ON dbo.Staging_EnverusProdData (WellID, Date);
    END
    ELSE
    BEGIN
        TRUNCATE TABLE dbo.Staging_EnverusProdData;
    END

	-- Declare variables for batch sizes and Update Date filtering
	DECLARE @BatchSize INT = 10000;
	DECLARE @UpdatedDateDiff INT = -30;

	-- Create a table variable to hold WellIDs
	DECLARE @WellIDs TABLE (WellID BIGINT PRIMARY KEY, API_UWI_Unformatted VARCHAR(32), BatchNumber INT);

	-- Populate the table variable with distinct WellIDs
	INSERT INTO @WellIDs (WellID, API_UWI_Unformatted, BatchNumber)
	SELECT	WellID, API_UWI_Unformatted, CEILING(ROW_NUMBER() OVER (ORDER BY WellID) / CAST(@BatchSize AS FLOAT)) AS BatchNumber
	FROM	Analytics.dbo.WELL_HEADER
	WHERE   FirstProdDate IS NOT NULL

    -- Initialize batch processing variables
    DECLARE @CurrentBatchNumber INT;
    DECLARE @TotalBatches INT;

	-- Calculate the total number of batches
	SELECT @TotalBatches = CEILING(CAST(COUNT(*) AS FLOAT) / @BatchSize)
	FROM @WellIDs;

	-- Initialize loop counter
    SET @CurrentBatchNumber = 1;

	-- Loop through each batch
	WHILE @CurrentBatchNumber <= @TotalBatches
	BEGIN
		PRINT 'Processing batch ' + CAST(@CurrentBatchNumber AS NVARCHAR(10)) + ' of ' + CAST(@TotalBatches AS NVARCHAR(10));

		;WITH ExistingProdData AS (
			SELECT		DISTINCT P.WellID, P.Date
			FROM		dbo.PRODUCTION P
			INNER JOIN	(SELECT WellID FROM @WellIDs WHERE BatchNumber = @CurrentBatchNumber) B
			ON			P.WellID = B.WellID
			WHERE		P.Cadence = 'MONTHLY'
			AND			P.DataSource = 'ENVERUS'
		),
		ProdData AS (
			SELECT		P.WellID, B.API_UWI_Unformatted, P.EOMonthDate AS Date, P.ProductionReportedMethod AS Comment, 
						MIN(P.UpdatedDate) AS UpdatedDate, SUM(CAST(ISNULL(P.ProducingDays, 0) AS INT)) AS ProducingDays, 
						SUM(ISNULL(P.GasProd_MCF, 0)) AS GAS, SUM(ISNULL(P.LiquidsProd_BBL, 0)) AS OIL, SUM(ISNULL(P.WaterProd_BBL, 0)) AS WATER, 
						'ENVERUS' AS DataSource, 'MONTHLY' AS Cadence
			FROM		Enverus.dbo.Production P
			INNER JOIN	(SELECT WellID, API_UWI_Unformatted FROM @WellIDs WHERE BatchNumber = @CurrentBatchNumber) B
			ON			P.WellID = B.WellID
			LEFT JOIN	ExistingProdData E 
			ON			P.WellID = E.WellID AND P.EOMonthDate = E.Date
			WHERE 
				(E.WellID IS NOT NULL AND P.UpdatedDate >= DATEADD(DAY, @UpdatedDateDiff, GETDATE()))
				OR (E.WellID IS NULL)
			GROUP BY P.WellID, B.API_UWI_Unformatted, P.EOMonthDate, P.ProductionReportedMethod
		),
		UnpivotedData AS (
			SELECT	WellID, API_UWI_Unformatted, Date, ProducingDays, DataSource, Cadence, Comment, UpdatedDate, Measure, Value
			FROM	ProdData
			UNPIVOT (
				Value FOR Measure IN (GAS, OIL, WATER)
			) AS Unpvt
		)

		-- Bulk insert into Staging Table
		INSERT INTO dbo.Staging_EnverusProdData WITH (TABLOCK) (
			WellID,
			API_UWI_Unformatted,
			Date,
			ProducingDays,
			DataSource,
			Cadence,
			Comment,
			Measure,
			Value,
			Units,
			DateCreated
		)
		SELECT 
			WellID, 
            API_UWI_Unformatted, 
            Date, 
            ProducingDays, 
            DataSource, 
            Cadence, 
            Comment, 
            Measure, 
            Value,
            CASE Measure
                WHEN 'GAS' THEN 'MCF'
                WHEN 'OIL' THEN 'BBL'
                WHEN 'WATER' THEN 'BBL'
            END AS Units, 
            UpdatedDate
        FROM 
			UnpivotedData;

			UPDATE T
			SET 
				T.API_UWI_Unformatted = S.API_UWI_Unformatted,
				T.Value = S.Value,
				T.Units = S.Units,
				T.ProducingDays = S.ProducingDays,
				T.Comment = S.Comment,
				T.DateCreated = S.DateCreated
			FROM 
				dbo.PRODUCTION T
			INNER JOIN 
				dbo.Staging_EnverusProdData S
				ON T.WellID = S.WellID
				AND T.Date = S.Date
				AND T.Measure = S.Measure
				AND T.Cadence = S.Cadence
				AND T.DataSource = S.DataSource
			WHERE 
				S.DateCreated > T.DateCreated;

			PRINT 'Update Complete for batch ' + CAST(@CurrentBatchNumber AS NVARCHAR(10));

			INSERT INTO dbo.PRODUCTION (
				WellID,
				API_UWI_Unformatted,
				Date,
				ProducingDays,
				DataSource,
				Cadence,
				Comment,
				Measure,
				Value,
				Units,
				DateCreated
			)
			SELECT 
				S.WellID,
				S.API_UWI_Unformatted,
				S.Date,
				S.ProducingDays,
				S.DataSource,
				S.Cadence,
				S.Comment,
				S.Measure,
				S.Value,
				S.Units,
				S.DateCreated
			FROM 
				dbo.Staging_EnverusProdData S
			LEFT JOIN 
				dbo.PRODUCTION T
				ON T.WellID = S.WellID
				AND T.Date = S.Date
				AND T.Measure = S.Measure
				AND T.Cadence = S.Cadence
				AND T.DataSource = S.DataSource
			WHERE 
				T.WellID IS NULL
			AND 
				S.WellID IN (SELECT WellID FROM @WellIDs WHERE BatchNumber = @CurrentBatchNumber);

			PRINT 'Data merged successfully into PRODUCTION table for batch ' + CAST(@CurrentBatchNumber AS NVARCHAR(10));

			TRUNCATE TABLE dbo.Staging_EnverusProdData;
			PRINT 'Staging table truncated after batch ' + CAST(@CurrentBatchNumber AS NVARCHAR(10));

            PRINT 'Batch ' + CAST(@CurrentBatchNumber AS NVARCHAR(10)) + ' processed successfully.';

		-- Increment the batch counter
        SET @CurrentBatchNumber = @CurrentBatchNumber + 1;
	END;
END;
GO
