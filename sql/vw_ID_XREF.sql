USE [Analytics]
GO

/****** Object:  View [dbo].[vw_ID_XREF]    Script Date: 3/21/2025 1:01:00 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO




CREATE OR ALTER     VIEW [dbo].[vw_ID_XREF] AS

SELECT		W.UWI10 AS API10, W.API, W._Id AS _WellId, PW._HeaderId, E.WellID, E.CompletionID, 
			W.StatusCurrent AS IHS_Status, E.WellStatus AS ENV_Status
FROM		IHS.dbo.Wells W
INNER JOIN	IHS.dbo.ProdnWells PW
ON			W.API = PW.API
LEFT JOIN	(SELECT DISTINCT WellID, CompletionID, UWI14, WellStatus FROM Enverus.dbo.Core_Well WHERE NULLIF(UWI14, '') IS NOT NULL) E
ON			W.API = E.UWI14
WHERE		NULLIF(W.API, '') IS NOT NULL
GO


