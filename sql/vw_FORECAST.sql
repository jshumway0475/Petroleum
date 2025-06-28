USE [Analytics]
GO

/****** Object:  View [dbo].[vw_FORECAST]    Script Date: 2/21/2024 12:54:33 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE OR ALTER VIEW [dbo].[vw_FORECAST] AS
WITH ValidMeasures AS (
    SELECT DISTINCT WellID, Measure
    FROM dbo.PRODUCTION
    WHERE SourceRank = 1 AND Cadence = 'MONTHLY'
)
SELECT
    W.WellID,
    F.ForecastID,
    VM.Measure,
    PA.Units,
    F.StartDate,
    F.Q1, F.Q2, F.Q3, F.Qabn, F.Dei,
    F.b_factor, F.Def, F.t1, F.t2,
    F.Analyst,
    F.DateCreated,
    PA.FirstProdDate,
    PA.LastProdDate,
    PA.CumulativeProduction,
    COALESCE(PS.StartCumulative, 0) AS StartCumulative,
    CASE
        WHEN F.StartDate > PA.MaxDate THEN
            PA.MaxProdMonth + DATEDIFF(MONTH, PA.MaxDate, F.StartDate)
        ELSE DR.DateRank
    END AS StartMonth,
    CASE 
        WHEN VM.Measure = 'OIL' THEN 1
        WHEN VM.Measure = 'GAS' THEN 2
        ELSE 3
    END AS Phase_Int
FROM
    dbo.WELL_HEADER W
JOIN
    ValidMeasures VM ON VM.WellID = W.WellID
LEFT JOIN (
    SELECT *
    FROM dbo.FORECAST F1
    WHERE EXISTS (
        SELECT 1
        FROM (
            SELECT ForecastID,
                   ROW_NUMBER() OVER (
                       PARTITION BY WellID, Measure 
                       ORDER BY 
                           CASE 
                               WHEN Analyst = 'PRIMARY_ANALYST' THEN 1
                               WHEN Analyst = 'auto_fit_1' THEN 2
                               WHEN Analyst = 'auto_fit_2' THEN 3
                               WHEN Analyst = 'auto_fit_3' THEN 4
                               ELSE 5
                           END
                   ) AS RN
            FROM dbo.FORECAST
        ) Ranked
        WHERE Ranked.ForecastID = F1.ForecastID
          AND Ranked.RN = 1
          AND F1.StartDate IS NOT NULL
    )
) F ON F.WellID = W.WellID AND F.Measure = VM.Measure
OUTER APPLY (
    SELECT
        MIN(Date) AS FirstProdDate,
        MAX(Date) AS LastProdDate,
        SUM(Value) AS CumulativeProduction,
        MAX(Date) AS MaxDate,
        MAX(DateRank) AS MaxProdMonth,
        MIN(Units) AS Units
    FROM dbo.PRODUCTION P0
    WHERE P0.WellID = W.WellID
      AND P0.Measure = VM.Measure
      AND P0.SourceRank = 1
      AND P0.Cadence = 'MONTHLY'
) PA
OUTER APPLY (
    SELECT SUM(Value) AS StartCumulative
    FROM dbo.PRODUCTION P1
    WHERE P1.WellID = W.WellID
      AND P1.Measure = VM.Measure
      AND P1.SourceRank = 1
      AND P1.Cadence = 'MONTHLY'
      AND P1.Date <= F.StartDate
) PS
LEFT JOIN dbo.PRODUCTION DR
    ON DR.WellID = W.WellID
   AND DR.Measure = VM.Measure
   AND DR.SourceRank = 1
   AND DR.Cadence = 'MONTHLY'
   AND DR.Date = CASE 
                    WHEN F.StartDate > PA.MaxDate THEN PA.MaxDate 
                    ELSE F.StartDate 
                END;
GO