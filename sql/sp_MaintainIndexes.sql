CREATE OR ALTER PROCEDURE dbo.sp_MaintainIndexes
AS
BEGIN
  SET NOCOUNT ON;

  DECLARE
    @object_id       INT,
    @index_id        INT,
    @schema_name     SYSNAME,
    @table_name      SYSNAME,
    @index_name      SYSNAME,
    @frag            FLOAT,
    @sql             NVARCHAR(MAX),
    @engineEdition   INT;

  -- thresholds
  DECLARE
    @lowThreshold  FLOAT = 10.0,
    @highThreshold FLOAT = 30.0;

  -- figure out if we’re on an “Enterprise” SKU
  SET @engineEdition = CAST(SERVERPROPERTY('EngineEdition') AS INT);
  -- 3 = Enterprise, 6 = Enterprise Core

  DECLARE idx_cursor CURSOR LOCAL FAST_FORWARD FOR
    SELECT  
      t.object_id,
      i.index_id,
      s.name       AS schema_name,
      t.name       AS table_name,
      i.name       AS index_name
    FROM sys.indexes AS i
    JOIN sys.tables   AS t ON t.object_id = i.object_id
    JOIN sys.schemas  AS s ON s.schema_id = t.schema_id
    WHERE t.is_ms_shipped = 0
      AND i.type_desc     IN ('CLUSTERED','NONCLUSTERED')
      AND i.name IS NOT NULL;

  OPEN idx_cursor;
  FETCH NEXT FROM idx_cursor 
    INTO @object_id, @index_id, @schema_name, @table_name, @index_name;

  WHILE @@FETCH_STATUS = 0
  BEGIN
    -- get fragmentation for that index
    SELECT @frag = avg_fragmentation_in_percent
    FROM sys.dm_db_index_physical_stats(
      DB_ID(),
      @object_id,
      @index_id,
      NULL,
      'SAMPLED'
    );

    SET @sql = NULL;
    IF @frag >= @highThreshold
    BEGIN
      -- rebuild: online only on Enterprise SKUs
      IF @engineEdition IN (3,6)
        SET @sql = N'ALTER INDEX ' 
          + QUOTENAME(@index_name)
          + N' ON ' + QUOTENAME(@schema_name) + N'.' + QUOTENAME(@table_name)
          + N' REBUILD WITH (ONLINE = ON);';
      ELSE
        SET @sql = N'ALTER INDEX '
          + QUOTENAME(@index_name)
          + N' ON ' + QUOTENAME(@schema_name) + N'.' + QUOTENAME(@table_name)
          + N' REBUILD;';
    END
    ELSE IF @frag >= @lowThreshold
    BEGIN
      -- always-supported reorganize
      SET @sql = N'ALTER INDEX '
        + QUOTENAME(@index_name)
        + N' ON ' + QUOTENAME(@schema_name) + N'.' + QUOTENAME(@table_name)
        + N' REORGANIZE;';
    END

    IF @sql IS NOT NULL
      EXEC sp_executesql @sql;

    FETCH NEXT FROM idx_cursor 
      INTO @object_id, @index_id, @schema_name, @table_name, @index_name;
  END

  CLOSE idx_cursor;
  DEALLOCATE idx_cursor;

  -- refresh stats too
  EXEC sp_updatestats;
END;
GO
