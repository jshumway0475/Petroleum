USE [Analytics_Aries]
GO

/****** Object:  StoredProcedure [dbo].[sp_UpdateInsertFromForecastToACECONOMIC]    Script Date: 5/31/2024 12:39:45 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- Creating the stored procedure
CREATE OR ALTER PROCEDURE [dbo].[sp_UpdateInsertFromForecastToACECONOMIC]
AS
BEGIN
    SET NOCOUNT ON;
    SET TRANSACTION ISOLATION LEVEL READ COMMITTED;

    BEGIN TRY
        BEGIN TRANSACTION;
        
		-- First, delete all rows from SECTION 4 of the AC_ECONOMIC table
        DELETE FROM [Analytics_Aries].[dbo].[AC_ECONOMIC] WHERE SECTION = 4;

		;WITH START_DATE_CTE AS (
			SELECT		M.PROPNUM, FCST.WellID, M.MAJOR, FCST.Analyst AS QUALIFIER, FCST.Measure, FORMAT(FCST.StartDate, 'MM/yyyy') AS START,
						ROW_NUMBER() OVER (
							PARTITION BY FCST.WellID, FCST.Measure 
							ORDER BY
								CASE WHEN FCST.Analyst = '_NAME_' THEN 1
								WHEN FCST.Analyst = 'auto_fit_1' THEN 2
								WHEN FCST.Analyst = 'auto_fit_2' THEN 3
								ELSE 4 END) AS QUALIFIER_RANK,
						RANK() OVER (
							PARTITION BY M.PROPNUM
							ORDER BY 
									CASE WHEN M.MAJOR = 'OIL' THEN 
									CASE WHEN FCST.Measure = 'OIL' THEN 1 WHEN FCST.Measure = 'GAS' THEN 2 ELSE 3 END 
								ELSE 
									CASE WHEN FCST.Measure = 'GAS' THEN 1 WHEN FCST.Measure = 'OIL' THEN 2 ELSE 3 END 
								END
						) AS STREAM_ORDER, 0 AS EXP_ORDER
			FROM		[Analytics].[dbo].[FORECAST] FCST
			INNER JOIN 	[Analytics_Aries].[dbo].[AC_PROPERTY] M ON (FCST.WellID = M.WELL_ID)
			WHERE		FCST.StartDate IS NOT NULL
		),
		FCST_CTE AS (
			SELECT  WellID, Analyst AS QUALIFIER, Measure, DateCreated,
					CONCAT_WS(
						' ', 
						1, ':', COALESCE(Q1, 0), 
							CASE WHEN COALESCE(t2, 0) = 0 THEN COALESCE(Q3, 0) ELSE COALESCE(Q2, 0) END,
							CASE WHEN Measure = 'GAS' THEN 'M/D' ELSE 'B/D' END, COALESCE(t1, 0), 'MO', 'EXP', 'X', ';', 
						2, ':', COALESCE(Q2, 0), COALESCE(Q3, 0), CASE WHEN Measure = 'GAS' THEN 'M/D' ELSE 'B/D' END, 
							COALESCE(t2, 0), 'IMO', 'EXP', 'X', ';',
						3, ':', COALESCE(Q3, 0), 'X', CASE WHEN Measure = 'GAS' THEN 'M/D' ELSE 'B/D' END, 
							ROUND(COALESCE(Def, 0) * 100, 6), 'EXP', CONCAT('B/', ROUND(COALESCE(b_factor, 1.0), 4)), 
							ROUND(COALESCE(Dei, 0) * 100, 6), ';',
						4, ':', 'X', COALESCE(Qabn, 0), CASE WHEN Measure = 'GAS' THEN 'M/D' ELSE 'B/D' END, 'X', 'YRS', 'EXP', 
							ROUND(COALESCE(Def, 0) * 100, 6)
					) AS EXPRESSION
			FROM    [Analytics].[dbo].[FORECAST]
		),
		RANKED_CTE AS (
			SELECT  F.WellID, F.QUALIFIER, F.Measure, 
					CAST(LEFT(SS.value, CHARINDEX(':', SS.value) - 1) AS INT) AS EXP_ORDER,
					LTRIM(SUBSTRING(SS.value, CHARINDEX(':', SS.value) + 1, LEN(SS.value))) AS EXPRESSION
			FROM    FCST_CTE F
			CROSS 	APPLY STRING_SPLIT(F.EXPRESSION, ';') AS SS
		),
		JOINED_CTE AS (
			SELECT  	R.WellID, R.QUALIFIER, R.Measure, DENSE_RANK() OVER (PARTITION BY R.WellID, R.QUALIFIER ORDER BY R.EXP_ORDER) AS EXP_ORDER, 
						R.EXPRESSION, SD.PROPNUM, SD.MAJOR, SD.START, SD.STREAM_ORDER, SD.QUALIFIER_RANK
			FROM    	RANKED_CTE R
			INNER JOIN	START_DATE_CTE SD ON (R.WellID = SD.WellID AND R.QUALIFIER = SD.QUALIFIER AND R.Measure = SD.Measure)
			WHERE   	CHARINDEX('0 MO', R.EXPRESSION) = 0
			AND     	CHARINDEX('0 IMO', R.EXPRESSION) = 0
		),
		KEYWORDS_CTE AS (
			SELECT		PROPNUM, QUALIFIER, Measure, 'START' AS KEYWORD, START AS EXPRESSION, STREAM_ORDER, EXP_ORDER, QUALIFIER_RANK
			FROM		START_DATE_CTE
			UNION ALL
			SELECT		PROPNUM, QUALIFIER, Measure, Measure AS KEYWORD, EXPRESSION, STREAM_ORDER, EXP_ORDER, QUALIFIER_RANK
			FROM		JOINED_CTE
		),
		FCST_DATA_CTE AS (
			SELECT		PROPNUM, 4 AS SECTION,
						ROW_NUMBER() OVER (
							PARTITION BY	PROPNUM
							ORDER BY		QUALIFIER, STREAM_ORDER, EXP_ORDER
						) AS SEQUENCE,
						QUALIFIER, QUALIFIER_RANK, CASE WHEN EXP_ORDER <=1 THEN KEYWORD ELSE '"' END AS KEYWORD, EXPRESSION
			FROM		KEYWORDS_CTE
			WHERE		QUALIFIER_RANK < 2
		)
		SELECT	PROPNUM, SECTION, SEQUENCE, CASE WHEN QUALIFIER LIKE '%auto_fit%' THEN 'AUTO_FIT' ELSE '_NAME_' END AS QUALIFIER, 
				CASE WHEN KEYWORD = 'WATER' THEN 'WTR' ELSE KEYWORD END AS KEYWORD, EXPRESSION
		INTO	#ForecastData
		FROM	FCST_DATA_CTE;

		-- Then, insert the processed data into AC_ECONOMIC from the temp table
        INSERT INTO [Analytics_Aries].[dbo].[AC_ECONOMIC] ([PROPNUM], [SECTION], [SEQUENCE], [QUALIFIER], [KEYWORD], [EXPRESSION])
        SELECT [PROPNUM], [SECTION], [SEQUENCE], [QUALIFIER], [KEYWORD], [EXPRESSION]
        FROM #ForecastData;

        COMMIT TRANSACTION;
        PRINT CAST(@@ROWCOUNT AS NVARCHAR(50)) + ' rows affected in AC_ECONOMIC';
        
    END TRY
    BEGIN CATCH
        ROLLBACK TRANSACTION;
        DECLARE @ErrorMessage NVARCHAR(4000), @ErrorSeverity INT, @ErrorState INT;
        SELECT @ErrorMessage = ERROR_MESSAGE(), @ErrorSeverity = ERROR_SEVERITY(), @ErrorState = ERROR_STATE();
        RAISERROR (@ErrorMessage, @ErrorSeverity, @ErrorState);
    END CATCH
END;
GO


