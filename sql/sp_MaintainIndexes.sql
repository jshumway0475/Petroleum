USE [Analytics];
GO

SET ANSI_NULLS ON;
GO

SET QUOTED_IDENTIFIER ON;
GO

CREATE OR ALTER PROCEDURE [dbo].[sp_MaintainIndexes]
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @tableName NVARCHAR(128);
    DECLARE @indexName NVARCHAR(128);
    DECLARE @sql NVARCHAR(MAX);
    DECLARE @fragmentation FLOAT;

    -- Fragmentation thresholds
    DECLARE @lowFragmentationThreshold FLOAT = 10.0;
    DECLARE @highFragmentationThreshold FLOAT = 30.0;

    -- Cursor to iterate through each table
    DECLARE table_cursor CURSOR FOR
    SELECT 
        QUOTENAME(SCHEMA_NAME(t.schema_id)) + '.' + QUOTENAME(t.name) AS TableName
    FROM 
        sys.tables AS t
    WHERE 
        t.is_ms_shipped = 0;

    OPEN table_cursor;

    FETCH NEXT FROM table_cursor INTO @tableName;

    WHILE @@FETCH_STATUS = 0
    BEGIN
        -- Cursor to iterate through each index in the current table
        DECLARE index_cursor CURSOR FOR
        SELECT 
            QUOTENAME(i.name) AS IndexName
        FROM 
            sys.indexes AS i
        WHERE 
            i.object_id = OBJECT_ID(@tableName) 
            AND i.type_desc IN ('CLUSTERED', 'NONCLUSTERED') 
            AND i.is_primary_key = 0 
            AND i.is_unique_constraint = 0;

        OPEN index_cursor;

        FETCH NEXT FROM index_cursor INTO @indexName;

        WHILE @@FETCH_STATUS = 0
        BEGIN
            -- Get the fragmentation percentage
            SELECT @fragmentation = avg_fragmentation_in_percent
            FROM sys.dm_db_index_physical_stats(DB_ID(), OBJECT_ID(@tableName), NULL, NULL, 'SAMPLED')
            WHERE index_id = (SELECT index_id FROM sys.indexes WHERE object_id = OBJECT_ID(@tableName) AND name = @indexName);

            -- Reorganize or rebuild the index based on fragmentation level
            IF @fragmentation >= @lowFragmentationThreshold AND @fragmentation < @highFragmentationThreshold
            BEGIN
                -- Reorganize the index
                SET @sql = 'ALTER INDEX ' + @indexName + ' ON ' + @tableName + ' REORGANIZE';
                EXEC sp_executesql @sql;
            END
            ELSE IF @fragmentation >= @highFragmentationThreshold
            BEGIN
                -- Rebuild the index
                SET @sql = 'ALTER INDEX ' + @indexName + ' ON ' + @tableName + ' REBUILD WITH (ONLINE = ON)';
                EXEC sp_executesql @sql;
            END;

            FETCH NEXT FROM index_cursor INTO @indexName;
        END;

        CLOSE index_cursor;
        DEALLOCATE index_cursor;

        FETCH NEXT FROM table_cursor INTO @tableName;
    END;

    CLOSE table_cursor;
    DEALLOCATE table_cursor;

    SET NOCOUNT OFF;
END;
GO
