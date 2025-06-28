USE [Analytics_Aries]
GO

/****** Object:  StoredProcedure [dbo].[sp_UpdateTestDataInACPRODUCT]    Script Date: 6/23/2025 3:27:46 PM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


-- Create the stored procedure
CREATE OR ALTER PROCEDURE [dbo].[sp_UpdateTestDataInACPRODUCT]
AS
BEGIN
    SET NOCOUNT ON;

    UPDATE P
    SET 
        P.TEST_OIL = X.TEST_OIL,
        P.TEST_GAS = X.TEST_GAS,
        P.TEST_WATER = X.TEST_WATER
    FROM Analytics_Aries.dbo.AC_PRODUCT P
    INNER JOIN (
        SELECT	
            M.PROPNUM, 
            T.Date,         
            MAX(CASE WHEN T.Measure = 'OIL' THEN T.TestValue END) AS TEST_OIL,
            MAX(CASE WHEN T.Measure = 'GAS' THEN T.TestValue END) AS TEST_GAS,
            MAX(CASE WHEN T.Measure = 'WATER' THEN T.TestValue END) AS TEST_WATER
        FROM Analytics.dbo.vw_PROD_TEST T
        INNER JOIN Analytics_Aries.dbo.AC_PROPERTY M
            ON CAST(T.WellID AS VARCHAR(128)) = M.WELL_ID
        WHERE T.Date IS NOT NULL
        GROUP BY M.PROPNUM, T.Date
    ) X
        ON P.PROPNUM = X.PROPNUM
        AND P.P_DATE = X.Date;

    PRINT CAST(@@ROWCOUNT AS NVARCHAR(50)) + ' rows updated in AC_PRODUCT with test data.';
END;
GO


