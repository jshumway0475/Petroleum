USE [Analytics]
GO

/****** Object:  StoredProcedure [dbo].[sp_InsertFromStagingToWellSpacing]    Script Date: 2/21/2024 12:53:33 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- Creating the stored procedure
CREATE OR ALTER PROCEDURE [dbo].[sp_InsertFromStagingToWellSpacing]
AS
BEGIN
    -- Preventing extra result sets from interfering with SELECT statements.
    SET NOCOUNT ON;
        
	-- Delete rows in WELL_SPACING that have matching WellID in WELL_SPACING_STAGE
	DELETE WS
	FROM [dbo].[WELL_SPACING] WS
	INNER JOIN [dbo].[WELL_SPACING_STAGE] WSS
	ON WS.[WellID] = WSS.[WellID];
        
	INSERT INTO [dbo].[WELL_SPACING] (
		[WellID],
		[neighboring_WellID],
		[clipped_lateral_geometry],
		[lateral_geometry_buffer],
		[clipped_neighbor_lateral_geometry],
		[neighbor_lateral_geometry_buffer],
		[MinDistance],
		[MedianDistance],
		[MaxDistance],
		[AvgDistance],
		[neighbor_IntersectionFraction],
		[RelativePosition],
		[Projection],
		[UpdateDate]
	)
	SELECT
		[WellID],
		[neighboring_WellID],
		geometry::STGeomFromText([clipped_lateral_geometry], 4326),
		geometry::STGeomFromText([lateral_geometry_buffer], 4326),
		geometry::STGeomFromText([clipped_neighbor_lateral_geometry], 4326),
		geometry::STGeomFromText([neighbor_lateral_geometry_buffer], 4326),
		[MinDistance],
		[MedianDistance],
		[MaxDistance],
		[AvgDistance],
		[neighbor_IntersectionFraction],
		[RelativePosition],
		[Projection],
		[UpdateDate]
	FROM [dbo].[WELL_SPACING_STAGE];
        
	-- Output the number of rows affected by the merge operation
	PRINT CAST(@@ROWCOUNT AS NVARCHAR(50)) + ' rows merged into WELL_SPACING from WELL_SPACING_STAGE';

	-- Clean-up: Drop the staging table if it's no longer needed
	IF OBJECT_ID('dbo.WELL_SPACING_STAGE', 'U') IS NOT NULL
	BEGIN
		DROP TABLE dbo.WELL_SPACING_STAGE;
		PRINT 'Staging table WELL_SPACING_STAGE has been dropped.';
	END
END;
GO


