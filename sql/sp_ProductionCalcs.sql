USE [Analytics]
GO

/****** Object:  StoredProcedure [dbo].[sp_ProductionCalcs]    Script Date: 12/18/2024 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

/*******************************************************************************************
Example Execution:
EXEC [dbo].[sp_ProductionCalcs] 
    @PrimaryDataSource = 'DATASOURCE1', 
    @SecondaryDataSource = 'DATASOURCE2';
********************************************************************************************/


CREATE OR ALTER PROCEDURE [dbo].[sp_ProductionCalcs] 
    @PrimaryDataSource NVARCHAR(50),
    @SecondaryDataSource NVARCHAR(50)
AS
BEGIN
    SET NOCOUNT ON;

    -- Use a Common Table Expression (CTE) to calculate ProdMonth and CumulativeProduction
    ;WITH RankedProduction AS (
        SELECT
            ProductionID,
            DENSE_RANK() OVER (PARTITION BY WellID, Measure, Cadence ORDER BY Date) AS DateRank,
			DENSE_RANK() OVER (
				PARTITION BY WellID, Measure, Cadence, Date 
				ORDER BY 
					CASE 
						WHEN DataSource = @PrimaryDataSource THEN 1 
						WHEN DataSource = @SecondaryDataSource THEN 2 
						ELSE 3 
					END
			) AS SourceRank 
        FROM
            dbo.PRODUCTION
        WHERE
            Measure IN ('OIL', 'GAS', 'WATER')
    )
    -- Update the dbo.PRODUCTION table by joining with the CTE
    UPDATE P
    SET 
        P.DateRank = RP.DateRank,
		P.SourceRank = RP.SourceRank
    FROM
        dbo.PRODUCTION P
    INNER JOIN
        RankedProduction RP ON P.ProductionID = RP.ProductionID;

END;
GO
