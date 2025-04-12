USE [Analytics]
GO

/****** Object:  StoredProcedure [dbo].[sp_InsertFromStagingToProduction]    Script Date: 2/21/2024 12:52:49 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



-- Creating the stored procedure
CREATE PROCEDURE [dbo].[sp_InsertFromStagingToProduction]
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
        MERGE INTO [dbo].[PRODUCTION] AS Target
        USING PRODUCTION_STAGE AS Source
        ON (
			Target.[WellID] = Source.[WellID] AND 
			Target.[Date] = Source.[Date] AND 
			Target.[Measure] = Source.[Measure] AND 
			Target.[Cadence] = Source.[Cadence] AND 
			Target.[DataSource] = Source.[DataSource]
		)
        
        -- When records are matched, update the records if the source has a newer DateCreated
        WHEN MATCHED AND Target.[DateCreated] < Source.[DateCreated] THEN 
            UPDATE SET 
				Target.[API_UWI_Unformatted] = Source.[API_UWI_Unformatted],
				Target.[Value] = Source.[Value],
				Target.[Units] = Source.[Units],
				Target.[ProducingDays] = Source.[ProducingDays],
				Target.[Comment] = Source.[Comment],
				Target.[DateCreated] = Source.[DateCreated]
        
        -- When no records are matched, insert the incoming records from the source table
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (
                [WellID],
				[API_UWI_Unformatted],
				[Date],
				[Measure],
				[Value],
				[Units],
				[Cadence],
				[ProducingDays],
				[DataSource],
				[Comment],
				[DateCreated]
            )
            VALUES (
                Source.[WellID],
				Source.[API_UWI_Unformatted],
				Source.[Date],
				Source.[Measure],
				Source.[Value],
				Source.[Units],
				Source.[Cadence],
				Source.[ProducingDays],
				Source.[DataSource],
				Source.[Comment],
				Source.[DateCreated]
            );

        -- If the operation was successful, commit the transaction
        COMMIT TRANSACTION;
        
        -- Output the number of rows affected by the merge operation
        PRINT CAST(@@ROWCOUNT AS NVARCHAR(50)) + ' rows merged into PRODUCTION from PRODUCTION_STAGE';

		-- Clean-up: Drop the staging table if it's no longer needed
        IF OBJECT_ID('dbo.PRODUCTION_STAGE', 'U') IS NOT NULL
        BEGIN
            DROP TABLE dbo.PRODUCTION_STAGE;
            PRINT 'Staging table PRODUCTION_STAGE has been dropped.';
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


