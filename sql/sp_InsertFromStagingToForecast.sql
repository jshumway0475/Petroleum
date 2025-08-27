USE [Analytics]
GO

/****** Object:  StoredProcedure [dbo].[sp_InsertFromStagingToForecast]    Script Date: 2/21/2024 12:51:32 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- Creating the stored procedure
CREATE OR ALTER PROCEDURE [dbo].[sp_InsertFromStagingToForecast]
AS
BEGIN
    -- Preventing extra result sets from interfering with SELECT statements.
    SET NOCOUNT ON;

    -- We set the transaction isolation level to avoid dirty reads
    SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

    BEGIN TRY
        -- Starting a transaction
        BEGIN TRANSACTION;
        
        -- The actual insert operation with a MERGE statement to handle duplicates
        MERGE INTO [dbo].[FORECAST] AS Target
        USING FORECAST_STAGE AS Source
        ON (Target.[WellID] = Source.[WellID] AND Target.[Measure] = Source.[Measure] AND Target.[Analyst] = Source.[Analyst])
        
        -- When records are matched, update the records
        WHEN MATCHED THEN 
            UPDATE SET 
                Target.[Units] = Source.[Units],
                Target.[StartDate] = Source.[StartDate],
                Target.[Q1] = Source.[Q1],
				Target.[Q2] = Source.[Q2],
				Target.[Q3] = Source.[Q3],
				Target.[Qabn] = Source.[Qabn],
				Target.[Dei] = Source.[Dei],
				Target.[b_factor] = Source.[b_factor],
				Target.[Def] = Source.[Def],
                Target.[t1] = Source.[t1],
				Target.[t2] = Source.[t2],
                Target.[DateCreated] = Source.[DateCreated],
				Target.[TraceBlob]  = COALESCE(Source.[TraceBlob], Target.[TraceBlob])
        
        -- When no records are matched, insert the incoming records from the source table
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (
                [WellID],
                [Measure],
                [Units],
                [StartDate],
                [Q1],
                [Q2],
                [Q3],
                [Qabn],
				[Dei],
				[b_factor],
				[Def],
				[t1],
				[t2],
				[Analyst],
				[DateCreated],
				[TraceBlob]
            )
            VALUES (
                Source.[WellID],
                Source.[Measure],
                Source.[Units],
                Source.[StartDate],
                Source.[Q1],
                Source.[Q2],
                Source.[Q3],
                Source.[Qabn],
				Source.[Dei],
				Source.[b_factor],
				Source.[Def],
				Source.[t1],
				Source.[t2],
				Source.[Analyst],
				Source.[DateCreated],
				Source.[TraceBlob]
            );

        -- If the operation was successful, commit the transaction
        COMMIT TRANSACTION;
        
        -- Output the number of rows affected by the merge operation
        PRINT CAST(@@ROWCOUNT AS NVARCHAR(50)) + ' rows merged into FORECAST from FORECAST_STAGE';

		-- Clean-up: Drop the staging table if it's no longer needed
        IF OBJECT_ID('dbo.FORECAST_STAGE', 'U') IS NOT NULL
        BEGIN
            DROP TABLE dbo.FORECAST_STAGE;
            PRINT 'Staging table FORECAST_STAGE has been dropped.';
        END

    END TRY
    BEGIN CATCH
        -- If there is any error, rollback the transaction
        ROLLBACK TRANSACTION;

        -- Capture the error information and re-throw
        DECLARE @ErrorMessage NVARCHAR(4000);
        DECLARE @ErrorSeverity INT;
        DECLARE @ErrorState INT;

        SELECT 
            @ErrorMessage = ERROR_MESSAGE(),
            @ErrorSeverity = ERROR_SEVERITY(),
            @ErrorState = ERROR_STATE();

        -- Use RAISERROR inside the CATCH block to return error
        -- information about the original error that caused
        -- execution to jump to the CATCH block.
        RAISERROR (@ErrorMessage, -- Message text.
                   @ErrorSeverity, -- Severity.
                   @ErrorState -- State.
                   );
    END CATCH
END;
GO


