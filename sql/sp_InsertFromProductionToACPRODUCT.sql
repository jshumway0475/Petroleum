USE [Analytics_Aries]
GO

SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

-- Dropping the stored procedure if it already exists
IF OBJECT_ID('sp_UpdateInsertFromProductionToACPRODUCT', 'P') IS NOT NULL
    DROP PROCEDURE sp_UpdateInsertFromProductionToACPRODUCT;
GO

-- Creating the stored procedure
CREATE PROCEDURE [dbo].[sp_UpdateInsertFromProductionToACPRODUCT]
AS
BEGIN
    SET NOCOUNT ON;
        
    -- Temporary table to hold aggregated data
    SELECT
        CAST(s.WellID AS VARCHAR(128)) AS WellID,
		s.DataSource,
        s.Date,
        MAX(CASE WHEN s.Measure = 'OIL' AND s.Cadence = 'MONTHLY' THEN s.Value ELSE NULL END) AS OIL,
        MAX(CASE WHEN s.Measure = 'GAS' AND s.Cadence = 'MONTHLY' THEN s.Value ELSE NULL END) AS GAS,
        MAX(CASE WHEN s.Measure = 'WATER' AND s.Cadence = 'MONTHLY' THEN s.Value ELSE NULL END) AS WATER,
        MAX(s.ProducingDays) AS DAYSON
    INTO #AggregatedData
    FROM [Analytics].[dbo].[PRODUCTION] s
    WHERE s.Cadence = 'MONTHLY'
    GROUP BY s.WellID, s.Date, s.DataSource;

    -- Perform upsert using MERGE
    MERGE INTO [Analytics_Aries].[dbo].[AC_PRODUCT] AS Target
    USING (
        SELECT 
            ap.PROPNUM,
            ad.Date AS P_DATE,
            ad.OIL,
            ad.GAS,
            ad.WATER,
            ad.DAYSON,
			ad.DataSource
        FROM #AggregatedData ad
        INNER JOIN [Analytics_Aries].[dbo].[AC_PROPERTY] ap ON ad.WellID = ap.WELL_ID
    ) AS Source
    ON (Target.[PROPNUM] = Source.[PROPNUM] AND Target.[P_DATE] = Source.[P_DATE] AND Target.[DATA_SOURCE] = Source.[DataSource])
        
    WHEN MATCHED THEN 
        UPDATE SET 
            Target.[OIL] = Source.[OIL],
            Target.[GAS] = Source.[GAS],
            Target.[WATER] = Source.[WATER],
            Target.[DAYSON] = Source.[DAYSON]
        
    WHEN NOT MATCHED BY TARGET THEN
        INSERT ([PROPNUM], [P_DATE], [OIL], [GAS], [WATER], [DAYSON], [DATA_SOURCE])
        VALUES (
			Source.[PROPNUM], Source.[P_DATE], Source.[OIL], Source.[GAS], Source.[WATER], Source.[DAYSON], Source.[DataSource]
		);

    PRINT CAST(@@ROWCOUNT AS NVARCHAR(50)) + ' rows affected in AC_PRODUCT';
END;
GO
