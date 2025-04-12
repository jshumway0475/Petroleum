USE [Analytics]
GO

/****** Object:  StoredProcedure [dbo].[sp_DeleteFromWellHeader]    Script Date: 5/22/2024 4:29:11 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE OR ALTER PROCEDURE [dbo].[sp_DeleteFromCompletionHeader]
AS
BEGIN
    SET NOCOUNT ON;

    -- Variables to handle batch processing
    DECLARE @BatchSize INT = 100000;
    DECLARE @RowsAffected INT;

    -- Loop until no rows are deleted in the batch
    WHILE 1 = 1
    BEGIN
        -- Begin transaction for each batch
        BEGIN TRANSACTION;

        -- Delete in batches using NOT EXISTS (anti-join)
        DELETE TOP (@BatchSize) FROM dbo.COMPLETION_HEADER
        WHERE NOT EXISTS (
            SELECT 1
            FROM Enverus.dbo.Core_Well e
            WHERE e.CompletionID = COMPLETION_HEADER.CompletionID
			AND e.UWI = LEFT(COMPLETION_HEADER.API_UWI_12_Unformatted, 10)
        );

        -- Check the number of rows affected by the DELETE statement
        SET @RowsAffected = @@ROWCOUNT;

        -- Commit the transaction for the current batch
        COMMIT TRANSACTION;

        -- Exit the loop if no rows were deleted
        IF @RowsAffected = 0
            BREAK;

        -- Checkpoint to reduce the log size
        CHECKPOINT;
    END;

    PRINT 'Deletion process completed successfully.';
END;
GO
