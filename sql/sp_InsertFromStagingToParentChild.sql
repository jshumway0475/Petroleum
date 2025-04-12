USE [Analytics]
GO

/****** Object:  StoredProcedure [dbo].[sp_InsertFromStagingToParentChild]    Script Date: 2/21/2024 12:52:25 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- Creating the stored procedure
CREATE OR ALTER PROCEDURE [dbo].[sp_InsertFromStagingToParentChild]
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
        MERGE INTO [dbo].[PARENT_CHILD] AS Target
        USING PARENT_CHILD_STAGE AS Source
        ON (Target.[WellID] = Source.[WellID] AND Target.[Date] = Source.[Date] AND Target.[ScenarioName] = Source.[ScenarioName])
        
        -- When records are matched, update the records
        WHEN MATCHED THEN 
            UPDATE SET 
                Target.[Relationship] = Source.[Relationship],
                Target.[ClosestHzDistance] = Source.[ClosestHzDistance],
                Target.[ClosestHzDistance_Left] = Source.[ClosestHzDistance_Left],
                Target.[ClosestHzDistance_Right] = Source.[ClosestHzDistance_Right],
                Target.[UpdateDate] = Source.[UpdateDate]
        
        -- When no records are matched, insert the incoming records from the source table
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (
                [WellID],
                [Date],
                [Relationship],
                [ClosestHzDistance],
                [ClosestHzDistance_Left],
                [ClosestHzDistance_Right],
                [ScenarioName],
                [UpdateDate]
            )
            VALUES (
                Source.[WellID],
                Source.[Date],
                Source.[Relationship],
                Source.[ClosestHzDistance],
                Source.[ClosestHzDistance_Left],
                Source.[ClosestHzDistance_Right],
                Source.[ScenarioName],
                Source.[UpdateDate]
            );

        -- If the operation was successful, commit the transaction
        COMMIT TRANSACTION;
        
        -- Output the number of rows affected by the merge operation
        PRINT CAST(@@ROWCOUNT AS NVARCHAR(50)) + ' rows merged into PARENT_CHILD from PARENT_CHILD_STAGE';

		-- Clean-up: Drop the staging table if it's no longer needed
        IF OBJECT_ID('dbo.PARENT_CHILD_STAGE', 'U') IS NOT NULL
        BEGIN
            DROP TABLE dbo.PARENT_CHILD_STAGE;
            PRINT 'Staging table PARENT_CHILD_STAGE has been dropped.';
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


