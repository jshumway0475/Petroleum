USE [Analytics]
GO

/****** Object:  StoredProcedure [dbo].[sp_BackfillMissingGeometry]    Script Date: 1/31/2025 1:26:28 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE OR ALTER PROCEDURE [dbo].[sp_BackfillMissingGeometry]
AS
BEGIN
    SET NOCOUNT ON;
    SET XACT_ABORT ON;

    BEGIN TRY
        BEGIN TRANSACTION;
        
        UPDATE		W
        SET			W.Geometry = MB.PathWGS84
        FROM		Analytics.dbo.WELL_HEADER W
        INNER JOIN	Enverus.mapping.WellBase MB
        ON			W.WellID = MB.WellID
        AND			W.CurrentCompletionID = MB.CompletionID
        WHERE		W.Geometry IS NULL
        AND			MB.PathWGS84 IS NOT NULL;

        COMMIT TRANSACTION;
        PRINT 'Update successful';
    END TRY
    BEGIN CATCH
        ROLLBACK TRANSACTION;
        PRINT 'Error: ' + ERROR_MESSAGE();
    END CATCH;
END;
GO


