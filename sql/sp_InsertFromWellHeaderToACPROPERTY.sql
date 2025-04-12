USE [Analytics_Aries]
GO

/****** Object:  StoredProcedure [dbo].[sp_InsertFromWellHeaderToACPROPERTY]    Script Date: 5/31/2024 12:38:44 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- Creating the stored procedure
CREATE OR ALTER   PROCEDURE [dbo].[sp_InsertFromWellHeaderToACPROPERTY] AS
BEGIN
    -- Preventing extra result sets from interfering with SELECT statements.
    SET NOCOUNT ON;

    -- We set the transaction isolation level to avoid dirty reads
    SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

	DECLARE @DBSKEY VARCHAR(12);

    -- Retrieve the DBSKEY for '_NAME_'. Replace '_NAME_' with the preferred name
    SELECT @DBSKEY = DBSKEY FROM dbo.DBSLIST WHERE NAME = '_NAME_';

    BEGIN TRY
        -- Starting a transaction
        BEGIN TRANSACTION;

		-- CTE to create a unique row for each well (Replace _NAME_ with preferred name)
		;WITH WELL_CTE AS (
			SELECT	*, 
			ROW_NUMBER() OVER (
				PARTITION BY WellID, API_UWI_Unformatted
				ORDER BY 
					CASE 
						WHEN CompletionDataSource = '_NAME_' THEN 1 
						WHEN CompletionDataSource = 'ENVERUS' THEN 2 
						ELSE 3 
					END,
					CASE 
						WHEN ScenarioName = '_NAME_' THEN 1 
						WHEN ScenarioName = 'ENVERUS' THEN 2 
                        ELSE 3
					END
			) AS RowNum
			FROM	[Analytics].[dbo].[vw_WELL_HEADER]
			WHERE	WellStatus = 'PRODUCING'
		)
        
        -- The actual insert operation with a MERGE statement to handle duplicates
        MERGE INTO [Analytics_Aries].[dbo].[AC_PROPERTY] AS Target
        USING (SELECT * FROM WELL_CTE WHERE RowNum = 1) AS Source
        ON (Target.[WELL_ID] = CAST(Source.[WellID] AS VARCHAR(128)) AND Target.[API] = Source.[API_UWI_Unformatted])
        
        -- When records are matched, update the records
        WHEN MATCHED THEN 
            UPDATE SET 
				Target.[LEASE] = CAST(Source.[WellName] AS VARCHAR(64)),
				Target.[FIELD] = CAST(Source.[Field] AS VARCHAR(64)),
				Target.[OPERATOR] = CAST(Source.[Operator] AS VARCHAR(64)),
				Target.[COUNTY] = Source.[County],
				Target.[STATUS] = Source.[WellStatus],
				Target.[MAJOR] = Source.[PrimaryPhase],
				Target.[FIRST_PROD] = Source.[FirstProdDate],
				Target.[LIQ_GRAV] = Source.[OilGravity_API],
				Target.[GAS_GRAV] = Source.[GasGravity_SG],
				Target.[TEMP_BH] = Source.[Bottom_Hole_Temp_DEGF],
				Target.[UPR_PERF] = Source.[UpperPerf_FT],
				Target.[LWR_PERF] = Source.[LowerPerf_FT],
				Target.[STATE] = Source.[StateProvince],
				Target.[COMPLETION_ID] = CAST(Source.[CurrentCompletionID] AS VARCHAR(64)),
				Target.[BASIN] = Source.[Basin],
				Target.[PLAY] = Source.[Play],
				Target.[SUBPLAY] = Source.[SubPlay],
				Target.[INTERVAL] = CAST(Source.[Interval] AS VARCHAR(64)),
				Target.[TRAJECTORY] = Source.[Trajectory],
				Target.[LATITUDE] = Source.[Latitude],
				Target.[LONGITUDE] = Source.[Longitude],
				Target.[LATITUDE_BH] = Source.[Latitude_BH],
				Target.[LONGITUDE_BH] = Source.[Longitude_BH],
				Target.[TOWNSHIP] = Source.[Township],
				Target.[RANGE] = Source.[Range],
				Target.[SECTION] = Source.[Section],
				Target.[ABSTRACT] = Source.[Abstract],
				Target.[BLOCK] = Source.[Block],
				Target.[SURVEY] = Source.[Survey],
				Target.[TVD_FT] = Source.[TVD_FT],
				Target.[MD_FT] = Source.[MD_FT],
				Target.[LATERAL_LENGTH_FT] = Source.[LateralLength_FT],
				Target.[PERMIT_APPROVED_DATE] = Source.[PermitApprovedDate],
				Target.[SPUD_DATE] = Source.[SpudDate],
				Target.[RIG_RELEASE_DATE] = Source.[RigReleaseDate],
				Target.[DATA_SOURCE] = Source.[DataSource],
				Target.[DATE_CREATED] = Source.[DateCreated]
        
        -- When no records are matched, insert the incoming records from the source table
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (
				[DBSKEY],
				[PROPNUM],
				[LEASE],
				[FIELD],
				[OPERATOR],
				[COUNTY],
				[API],
				[STATUS],
				[MAJOR],
				[WELL_ID],
				[FIRST_PROD],
				[LIQ_GRAV],
				[GAS_GRAV],
				[TEMP_BH],
				[UPR_PERF],
				[LWR_PERF],
				[STATE],
				[COMPLETION_ID],
				[RSV_CAT],
				[BASIN],
				[PLAY],
				[SUBPLAY],
				[INTERVAL],
				[TRAJECTORY],
				[LATITUDE],
				[LONGITUDE],
				[LATITUDE_BH],
				[LONGITUDE_BH],
				[TOWNSHIP],
				[RANGE],
				[SECTION],
				[ABSTRACT],
				[BLOCK],
				[SURVEY],
				[TVD_FT],
				[MD_FT],
				[LATERAL_LENGTH_FT],
				[PERMIT_APPROVED_DATE],
				[SPUD_DATE],
				[RIG_RELEASE_DATE],
				[DATA_SOURCE],
				[DATE_CREATED]
            )
            VALUES (
				@DBSKEY,
				NEWID(),
                CAST(Source.[WellName] AS VARCHAR(64)),
				CAST(Source.[Field] AS VARCHAR(64)),
				CAST(Source.[Operator] AS VARCHAR(64)),
				Source.[County],
				Source.[API_UWI_Unformatted],
				Source.[WellStatus],
				Source.[PrimaryPhase],
				CAST(Source.[WellID] AS VARCHAR(128)),
				Source.[FirstProdDate],
				Source.[OilGravity_API],
				Source.[GasGravity_SG],
				Source.[Bottom_Hole_Temp_DEGF],
				Source.[UpperPerf_FT],
				Source.[LowerPerf_FT],
				Source.[StateProvince],
				CAST(Source.[CurrentCompletionID] AS VARCHAR(64)),
				'1PDP',
				Source.[Basin],
				Source.[Play],
				Source.[SubPlay],
				CAST(Source.[Interval] AS VARCHAR(64)),
				Source.[Trajectory],
				Source.[Latitude],
				Source.[Longitude],
				Source.[Latitude_BH],
				Source.[Longitude_BH],
				Source.[Township],
				Source.[Range],
				Source.[Section],
				Source.[Abstract],
				Source.[Block],
				Source.[Survey],
				Source.[TVD_FT],
				Source.[MD_FT],
				Source.[LateralLength_FT],
				Source.[PermitApprovedDate],
				Source.[SpudDate],
				Source.[RigReleaseDate],
				Source.[DataSource],
				Source.[DateCreated]
            );

        -- Update the SEQNUM for newly inserted rows with NULL SEQNUM
        DECLARE @CurrentSEQNUM INT;
        SELECT @CurrentSEQNUM = MAX(SEQNUM) FROM [Analytics_Aries].[dbo].[AC_PROPERTY];

        ;WITH CTE AS (
            SELECT 
                SEQNUM, 
                (@CurrentSEQNUM + ROW_NUMBER() OVER (ORDER BY (SELECT NULL))) AS NewSEQNUM
            FROM 
                [Analytics_Aries].[dbo].[AC_PROPERTY]
            WHERE 
                SEQNUM IS NULL
        )
        UPDATE CTE
        SET SEQNUM = NewSEQNUM;

        -- If the operation was successful, commit the transaction
        COMMIT TRANSACTION;
        
        -- Output the number of rows affected by the merge operation
        PRINT CAST(@@ROWCOUNT AS NVARCHAR(50)) + ' rows merged into AC_PROPERTY';

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


