USE [Axia_Analytics_Aries]
GO

/****** Object:  StoredProcedure [dbo].[sp_CreateUpdateEconLines]    Script Date: 10/21/2024 10:32:52 AM ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



CREATE OR ALTER     PROCEDURE [dbo].[sp_CreateUpdateEconLines]
AS
BEGIN
	-- Include name of the pricing sidefile
	DECLARE @STRIP_FILENAME VARCHAR(80) = 'STR092024'
    
	-- Delete existing data from AC_ECONOMIC
    DELETE FROM dbo.AC_ECONOMIC
    WHERE (SECTION NOT IN (4, 7) AND QUALIFIER = 'AXIA') 
       OR (SECTION = 7 AND QUALIFIER IN ('OP', 'MIN'));

    -- Insert new data into AC_ECONOMIC
    INSERT INTO AC_ECONOMIC (PROPNUM, SECTION, SEQUENCE, QUALIFIER, KEYWORD, EXPRESSION)
    SELECT 
        P.PROPNUM, 
        ExampleData.SECTION, 
        ExampleData.SEQUENCE, 
        ExampleData.QUALIFIER, 
        ExampleData.KEYWORD, 
        ExampleData.EXPRESSION
    FROM 
        AC_PROPERTY P
    CROSS JOIN 
        (SELECT '2' AS SECTION, '10' AS SEQUENCE, 'AXIA' AS QUALIFIER, 'BTU' AS KEYWORD, '@ED.BTU' AS EXPRESSION
         UNION ALL
         SELECT '2', '20', 'AXIA', 'SHRINK', '@ED.SHRINK'
         UNION ALL
         SELECT '2', '30', 'AXIA', 'ELOSS', 'OPINC 0 NOH P 1'
         UNION ALL
         SELECT '2', '40', 'AXIA', 'OPNET', '@ED.LSE_ROY*-1+100 @ED.LSE_ROY*-1+100'
         UNION ALL
         SELECT '2', '50', 'AXIA', 'INVWT', '@ED.INV_WEIGHT'
         UNION ALL
         SELECT '2', '60', 'AXIA', 'XINVWT', '@ED.PROD_WEIGHT'
         UNION ALL
         SELECT '5', '10', 'AXIA', 'SIDEFILE', @STRIP_FILENAME
         UNION ALL
         SELECT '5', '20', 'AXIA', 'PAJ/OIL', '@ED.PAJ_OIL_VAL X $/B TO LIFE PC 0'
         UNION ALL
         SELECT '5', '30', 'AXIA', 'PAJ/GAS', '@ED.PAJ_GAS_VAL X $/M TO LIFE PC 0'
         UNION ALL
         SELECT '5', '40', 'AXIA', 'PAJ/NGL', '@ED.PAJ_NGL_FRAC X FRAC TO LIFE PC 0'
         UNION ALL
         SELECT '6', '10', 'AXIA', 'OPC/T', '@ED.OPC1_FIXED X $/M @ED.OPC1_DATE AD PC 0'
         UNION ALL
         SELECT '6', '20', 'AXIA', '"', '@ED.OPC2_FIXED X $/M TO LIFE PC 0'
         UNION ALL
         SELECT '6', '30', 'AXIA', 'OPC/OIL', '@ED.OPC_OIL X $/B TO LIFE PC 0'
         UNION ALL
         SELECT '6', '40', 'AXIA', 'OPC/GAS', '@ED.OPC_GAS X $/M TO LIFE PC 0'
         UNION ALL
         SELECT '6', '50', 'AXIA', 'OPC/WTR', '@ED.OPC_WTR X $/B TO LIFE PC 0'
         UNION ALL
         SELECT '6', '60', 'AXIA', 'STX/OIL', '@ED.STX_OIL X % TO LIFE PC 0'
         UNION ALL
         SELECT '6', '70', 'AXIA', 'STX/GAS', '@ED.STX_GAS X % TO LIFE PC 0'
         UNION ALL
         SELECT '6', '80', 'AXIA', 'STX/NGL', '@ED.STX_NGL X % TO LIFE PC 0'
         UNION ALL
         SELECT '6', '90', 'AXIA', 'ATX', '@ED.ATX X % TO LIFE PC 0'
         UNION ALL
         SELECT '6', '100', 'AXIA', 'ABAN', '@ED.CAPEX_ABAN X M$/M TO LIFE PC 0'
         UNION ALL
         SELECT '7', '10', 'OP', 'NET', '@ED.WI @ED.LSE_ROY*-1+100 @ED.LSE_ROY*-1+100 %'
         UNION ALL
         SELECT '7', '20', 'MIN', 'LSE', '@ED.WI @ED.LSE_ROY 1.0 0 %'
         UNION ALL
         SELECT '7', '30', 'MIN', 'OWN', '0.0 0.0 100.0 0 %'
         UNION ALL
         SELECT '8', '10', 'AXIA', 'CAPITAL', '@ED.CAPEX_DCE*0.3 @ED.CAPEX_DCE*0.7 G @M.FIRST_PROD AD PC 0'
         UNION ALL
         SELECT '8', '20', 'AXIA', 'CAPITAL', '@ED.CAPEX_AL*0.5 @ED.CAPEX_AL*0.5 G @ED.OPC1_DATE AD PC 0'
         UNION ALL
         SELECT '9', '10', 'AXIA', 'NGL/GAS', '@ED.NGL_YIELD X B/MM TO LIFE LIN TIME'
    ) AS ExampleData;
END;
GO


