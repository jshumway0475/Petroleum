USE [Analytics]
GO

/****** Object:  View [dbo].[vw_FORECAST]    Script Date: 2/21/2024 12:54:33 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO




CREATE OR ALTER VIEW [dbo].[vw_FORECAST] AS
WITH PROD_CTE AS (
	SELECT		WellID, Measure, Units, SUM(Value) AS CumulativeProduction, MIN(Date) AS FirstProdDate, 
				MAX(Date) AS LastProdDate
	FROM		dbo.PRODUCTION
	WHERE		SourceRank = 1
	AND			Cadence = 'MONTHLY'
	GROUP BY	WellID, Measure, Units
), PROD_CTE2 AS (
	SELECT		F.ForecastID, SUM(P.Value) AS StartCumulative
	FROM		dbo.PRODUCTION P
	INNER JOIN  dbo.FORECAST F ON P.WellID = F.WellID AND P.Measure = F.Measure
	WHERE		P.SourceRank = 1
	AND			P.Cadence = 'MONTHLY'
	AND			P.Date <= F.StartDate
	GROUP BY	F.ForecastID
), MaxDateCTE AS (
    SELECT      WellID, Measure, MAX(Date) AS MaxDate, MAX(DateRank) AS MaxProdMonth
    FROM        dbo.PRODUCTION
	WHERE		Cadence = 'MONTHLY'
    GROUP BY    WellID, Measure
), PROD_CTE3 AS (
    SELECT      F.ForecastID,
				CASE 
                    WHEN F.StartDate > M.MaxDate THEN M.MaxProdMonth + DATEDIFF(MONTH, M.MaxDate, F.StartDate)
                    ELSE P2.DateRank 
                END AS StartMonth
    FROM        dbo.PRODUCTION P2
    INNER JOIN  dbo.FORECAST F ON P2.WellID = F.WellID AND P2.Measure = F.Measure
    INNER JOIN  MaxDateCTE M ON P2.WellID = M.WellID AND P2.Measure = M.Measure
    WHERE       P2.Date = CASE WHEN F.StartDate > M.MaxDate THEN M.MaxDate ELSE F.StartDate END
	AND			P2.SourceRank = 1
	AND			P2.Cadence = 'MONTHLY'
	GROUP BY    F.ForecastID, M.MaxProdMonth, F.StartDate, M.MaxDate, P2.DateRank
), ANALYST_RANKING AS (
    SELECT      ForecastID, Analyst,
                ROW_NUMBER() OVER (PARTITION BY WellID, Measure ORDER BY 
                    CASE
                        WHEN Analyst = 'PRIMARY_ANALYST' THEN 1
                        WHEN Analyst = 'auto_fit_1' THEN 2
                        WHEN Analyst = 'auto_fit_2' THEN 3
                        WHEN Analyst = 'auto_fit_3' THEN 4
                        ELSE 5
                    END) AS AnalystRank
    FROM        dbo.FORECAST
)
SELECT		F.ForecastID, P.WellID, P.Measure, P.Units, F.StartDate, F.Q1, F.Q2, F.Q3, F.Qabn, F.Dei, F.b_factor, F.Def, 
			F.t1, F.t2, F.Analyst, F.DateCreated, P.FirstProdDate, P.LastProdDate, P.CumulativeProduction, 
			P2.StartCumulative, P3.StartMonth, 
			CASE WHEN P.Measure = 'OIL' THEN 1 WHEN P.Measure = 'GAS' THEN 2 ELSE 3 END AS PHASE_INT
FROM		dbo.FORECAST F
FULL JOIN	PROD_CTE P
ON			F.WellID = P.WellID AND F.Measure = P.Measure
LEFT JOIN	ANALYST_RANKING A 
ON			F.ForecastID = A.ForecastID
LEFT JOIN	PROD_CTE2 P2
ON			F.ForecastID = P2.ForecastID
LEFT JOIN	PROD_CTE3 P3
ON			F.ForecastID = P3.ForecastID
WHERE       ((A.AnalystRank = 1 AND	F.StartDate IS NOT NULL) OR F.ForecastID IS NULL)
;
GO
