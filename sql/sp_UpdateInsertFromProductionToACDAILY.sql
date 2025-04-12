USE [Analytics_Aries]
GO

/****** Object:  StoredProcedure [dbo].[sp_UpdateInsertFromProductionToACDAILY]    Script Date: 5/31/2024 12:40:25 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- Creating the stored procedure
CREATE OR ALTER PROCEDURE [dbo].[sp_UpdateInsertFromProductionToACDAILY]
AS
BEGIN
    SET NOCOUNT ON;
    SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

    BEGIN TRY
        BEGIN TRANSACTION;
        
        -- Temporary table to hold aggregated data
        SELECT
            s.WellID,
			s.DataSource,
            s.Date,
            MAX(CASE WHEN s.Measure = 'OIL' AND s.Cadence = 'DAILY' THEN s.Value ELSE NULL END) AS OIL,
            MAX(CASE WHEN s.Measure = 'GAS' AND s.Cadence = 'DAILY' THEN s.Value ELSE NULL END) AS GAS,
            MAX(CASE WHEN s.Measure = 'WATER' AND s.Cadence = 'DAILY' THEN s.Value ELSE NULL END) AS WATER,
			MAX(CASE WHEN s.Measure = 'CP' AND s.Cadence = 'DAILY' THEN s.Value ELSE NULL END) AS CP,
			MAX(CASE WHEN s.Measure = 'TP' AND s.Cadence = 'DAILY' THEN s.Value ELSE NULL END) AS TP,
			MAX(CASE WHEN s.Measure = 'CHOKE' AND s.Cadence = 'DAILY' THEN s.Value ELSE NULL END) AS CHOKE
        INTO #AggregatedData
        FROM [Analytics].[dbo].[PRODUCTION] s
        WHERE s.Cadence = 'DAILY'
        GROUP BY s.WellID, s.Date, s.DataSource;

        -- Perform upsert using MERGE
        MERGE INTO [Analytics_Aries].[dbo].[AC_DAILY] AS Target
        USING (
            SELECT 
                ap.PROPNUM,
                ad.Date AS D_DATE,
                ad.OIL,
                ad.GAS,
                ad.WATER,
                ad.CP,
				ad.TP,
				ad.CHOKE,
				ad.DataSource
            FROM #AggregatedData ad
            INNER JOIN [Analytics_Aries].[dbo].[AC_PROPERTY] ap ON (ad.WellID = ap.WELL_ID and ad.DataSource = ap.DATA_SOURCE)
        ) AS Source
        ON (Target.[PROPNUM] = Source.[PROPNUM] AND Target.[D_DATE] = Source.[D_DATE] AND Target.[DATA_SOURCE] = Source.[DataSource])
        
        WHEN MATCHED THEN 
            UPDATE SET 
                Target.[OIL] = Source.[OIL],
                Target.[GAS] = Source.[GAS],
                Target.[WATER] = Source.[WATER],
                Target.[CP] = Source.[CP],
				Target.[TP] = Source.[TP],
				Target.[CHOKE] = Source.[CHOKE]
        
        WHEN NOT MATCHED BY TARGET THEN
            INSERT ([PROPNUM], [D_DATE], [OIL], [GAS], [WATER], [CP], [TP], [CHOKE], [DATA_SOURCE])
            VALUES (Source.[PROPNUM], Source.[D_DATE], Source.[OIL], Source.[GAS], Source.[WATER], Source.[CP], Source.[TP], Source.[CHOKE], Source.[DataSource]);

        COMMIT TRANSACTION;
        PRINT CAST(@@ROWCOUNT AS NVARCHAR(50)) + ' rows affected in AC_DAILY';
        
    END TRY
    BEGIN CATCH
        ROLLBACK TRANSACTION;
        DECLARE @ErrorMessage NVARCHAR(4000), @ErrorSeverity INT, @ErrorState INT;
        SELECT @ErrorMessage = ERROR_MESSAGE(), @ErrorSeverity = ERROR_SEVERITY(), @ErrorState = ERROR_STATE();
        RAISERROR (@ErrorMessage, @ErrorSeverity, @ErrorState);
    END CATCH
END;
GO


