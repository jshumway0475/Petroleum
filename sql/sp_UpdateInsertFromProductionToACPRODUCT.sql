USE [Analytics_Aries]
GO

/****** Object:  StoredProcedure [dbo].[sp_UpdateInsertFromProductionToACPRODUCT]    Script Date: 5/31/2024 12:40:47 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- Creating the stored procedure
CREATE OR ALTER PROCEDURE [dbo].[sp_UpdateInsertFromProductionToACPRODUCT]
AS
BEGIN
    SET NOCOUNT ON;
    SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

    BEGIN TRY
        BEGIN TRANSACTION;
        
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
        ON (Target.[PROPNUM] = Source.[PROPNUM] AND Target.[P_DATE] = Source.[P_DATE])
        
        WHEN MATCHED THEN 
            UPDATE SET 
                Target.[OIL] = Source.[OIL],
                Target.[GAS] = Source.[GAS],
                Target.[WATER] = Source.[WATER],
                Target.[DAYSON] = Source.[DAYSON],
				Target.[DATA_SOURCE] = Source.[DataSource]
        
        WHEN NOT MATCHED BY TARGET THEN
            INSERT ([PROPNUM], [P_DATE], [OIL], [GAS], [WATER], [DAYSON], [DATA_SOURCE])
            VALUES (Source.[PROPNUM], Source.[P_DATE], Source.[OIL], Source.[GAS], Source.[WATER], Source.[DAYSON], Source.[DataSource]);

        COMMIT TRANSACTION;
        PRINT CAST(@@ROWCOUNT AS NVARCHAR(50)) + ' rows affected in AC_PRODUCT';
        
    END TRY
    BEGIN CATCH
        ROLLBACK TRANSACTION;
        DECLARE @ErrorMessage NVARCHAR(4000), @ErrorSeverity INT, @ErrorState INT;
        SELECT @ErrorMessage = ERROR_MESSAGE(), @ErrorSeverity = ERROR_SEVERITY(), @ErrorState = ERROR_STATE();
        RAISERROR (@ErrorMessage, @ErrorSeverity, @ErrorState);
    END CATCH
END;
GO


