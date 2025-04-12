USE [Analytics];
GO

SET ANSI_NULLS ON;
GO

SET QUOTED_IDENTIFIER ON;
GO

-- Stored procedure to create indexes on the Enverus.dbo.Core_Well table
CREATE OR ALTER PROCEDURE [dbo].[sp_CreateEnverusCoreWellIndexes]
AS
BEGIN
    SET NOCOUNT ON;

    BEGIN TRY
        -- Attempt to create IDX_Core_Well_Country index
        BEGIN TRY
            EXEC('CREATE INDEX IDX_Core_Well_Country ON Enverus.dbo.Core_Well (Country)');
            PRINT 'Created index IDX_Core_Well_Country';
        END TRY
        BEGIN CATCH
            IF ERROR_NUMBER() = 1913 -- Index already exists
                PRINT 'Index IDX_Core_Well_Country already exists';
            ELSE
                THROW; -- Rethrow unexpected errors
        END CATCH;

        -- Attempt to create IDX_Core_Well_CompletionNumber index
        BEGIN TRY
            EXEC('CREATE INDEX IDX_Core_Well_CompletionNumber ON Enverus.dbo.Core_Well (CompletionNumber)');
            PRINT 'Created index IDX_Core_Well_CompletionNumber';
        END TRY
        BEGIN CATCH
            IF ERROR_NUMBER() = 1913 -- Index already exists
                PRINT 'Index IDX_Core_Well_CompletionNumber already exists';
            ELSE
                THROW; -- Rethrow unexpected errors
        END CATCH;

        -- Attempt to create IDX_Core_Well_WellStatus index
        BEGIN TRY
            EXEC('CREATE INDEX IDX_Core_Well_WellStatus ON Enverus.dbo.Core_Well (WellStatus)');
            PRINT 'Created index IDX_Core_Well_WellStatus';
        END TRY
        BEGIN CATCH
            IF ERROR_NUMBER() = 1913 -- Index already exists
                PRINT 'Index IDX_Core_Well_WellStatus already exists';
            ELSE
                THROW; -- Rethrow unexpected errors
        END CATCH;

        -- Attempt to create IDX_Core_Well_WellID_CompletionID_UpdatedDate index
        BEGIN TRY
            EXEC('CREATE INDEX IDX_Core_Well_WellID_CompletionID_UpdatedDate ON Enverus.dbo.Core_Well (WellID, CompletionID, UpdatedDate)');
            PRINT 'Created index IDX_Core_Well_WellID_CompletionID_UpdatedDate';
        END TRY
        BEGIN CATCH
            IF ERROR_NUMBER() = 1913 -- Index already exists
                PRINT 'Index IDX_Core_Well_WellID_CompletionID_UpdatedDate already exists';
            ELSE
                THROW; -- Rethrow unexpected errors
        END CATCH;
    END TRY
    BEGIN CATCH
        DECLARE @ErrorMessage NVARCHAR(4000), @ErrorSeverity INT, @ErrorState INT;
        SELECT @ErrorMessage = ERROR_MESSAGE(), @ErrorSeverity = ERROR_SEVERITY(), @ErrorState = ERROR_STATE();
        RAISERROR (@ErrorMessage, @ErrorSeverity, @ErrorState);
    END CATCH
END;
GO
