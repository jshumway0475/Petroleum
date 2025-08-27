import json
from sqlalchemy import create_engine, text
import sqlalchemy as sa
from urllib.parse import quote_plus
from sqlalchemy.exc import SQLAlchemyError
import pyodbc
import pandas as pd

ODBC_DRIVER = "ODBC Driver 18 for SQL Server"

# Function to return a dictionary from a json file
def get_json_dict(json_file):
    with open(json_file, 'r') as file:
        json_dict = json.load(file)
    return json_dict

def _server_with_port(servername: str, port: str | int | None):
    """Return SERVER string acceptable to msodbcsql: 'host,port' if port is provided."""
    if port is None or str(port).strip() == "":
        return str(servername)
    return f"{servername},{port}"

def _common_security_kv(creds_dict: dict) -> str:
    """
    v18 defaults to Encrypt=yes. If you're using on-prem with self-signed certs,
    TrustServerCertificate=yes avoids certificate validation failures.
    Allow opt-out via creds_dict if you've installed a proper CA chain.
    """
    encrypt = creds_dict.get("encrypt", "yes")
    trust = creds_dict.get("trust_server_certificate", "yes")
    return f"Encrypt={encrypt};TrustServerCertificate={trust};"

# Function to query using pyodbc
def sql_query_pyodbc(query, creds_dict):
    """
    Query a SQL Server database using pyodbc.
    :query: SQL query to execute
    :creds_dict: {servername, db_name, username, password, [port], [encrypt], [trust_server_certificate]}
    :return: pandas DataFrame or None on error
    """
    conn = cursor = None
    try:
        server = _server_with_port(creds_dict['servername'], creds_dict.get('port'))
        security = _common_security_kv(creds_dict)
        conn_str = (
            f"DRIVER={{{ODBC_DRIVER}}};"
            f"SERVER={server};"
            f"DATABASE={creds_dict['db_name']};"
            f"UID={creds_dict['username']};PWD={creds_dict['password']};"
            f"{security}"
        )
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        columns = [c[0] for c in cursor.description]
        df = pd.DataFrame.from_records(results, columns=columns)
        return df

    except Exception as e:
        print(f"An error occurred while querying the database: {e}")
        try:
            # Helpful hint if driver lookup fails
            print("pyodbc.drivers():", pyodbc.drivers())
        except Exception:
            pass
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# Function to connect to SQL Server database
def sql_connect(
        username, 
        password, 
        db_name, 
        server_name, 
        port, 
        return_engine=True, 
        pool_size=10, 
        max_overflow=20, 
        pool_timeout=60, 
        pool_recycle=3600, 
        autocommit=False,
        creds_dict: dict | None = None
    ):
    '''
    Connect to a SQL Server database using SQLAlchemy.
    Args:
    - username (str): The username for the database connection.
    - password (str): The password for the database connection.
    - db_name (str): The name of the database to connect to.
    - server_name (str): The name of the server to connect to.
    - port (str): The port number to connect to.
    - return_engine (bool): If True, return the SQLAlchemy engine object. If False, return the connection_string.
    - pool_size (int): The number of connections to keep in the connection pool.
    - max_overflow (int): The number of connections to allow that can exceed the pool_size.
    - pool_timeout (int): The number of seconds to wait before giving up on getting a connection from the pool.
    - pool_recycle (int): The number of seconds after which a connection is automatically recycled.
    - autocommit (bool): If True, set the connection to autocommit mode (useful for stored procedures).
    - creds_dict (dict | None): Optional dictionary containing additional connection parameters like 'encrypt' and 'trust_server_certificate'.
    Returns:
    - engine (sqlalchemy.engine.base.Engine): The SQLAlchemy engine object.
    - connection_string (str): The connection string used to connect to the database.
    '''
    security = _common_security_kv(creds_dict or {})
    server = _server_with_port(server_name, port)

    # ODBC connection string (URL-encoded for SQLAlchemy's odbc_connect)
    conn_stmt = (
        f"DRIVER={{{ODBC_DRIVER}}};"
        f"SERVER={server};"
        f"DATABASE={db_name};"
        f"UID={username};PWD={password};"
        f"{security}"
    )
    encoded = quote_plus(conn_stmt)
    connection_string = f"mssql+pyodbc:///?odbc_connect={encoded}"

    engine_params = {
        'pool_size': pool_size,
        'max_overflow': max_overflow,
        'pool_timeout': pool_timeout,
        'pool_recycle': pool_recycle,
        'fast_executemany': True
    }

    engine = create_engine(
        connection_string,
        isolation_level="AUTOCOMMIT" if autocommit else "READ COMMITTED",
        **engine_params
    )
    return engine if return_engine else connection_string

# Helper function to ensure table exists
def _ensure_table_exists(engine, connection, table_schema):
    if not engine.dialect.has_table(connection, table_schema.name):
        table_schema.create(bind=engine, checkfirst=True)
        print(f"Created table {table_schema.name}")

# Function to load data into a SQL Server database
def load_data_to_sql(df, creds_dict, table_schema, lock=None):
    '''
    Load data into a SQL Server table.
    :df: DataFrame containing the data to load
    :creds_dict: Dictionary containing connection credentials
    :table_schema: SQLAlchemy Table object defining the schema of the table
        - creds_dict can include: encrypt, trust_server_certificate
    :table_name: The name of the database table
    :lock: A threading lock object to synchronize access to the database
    '''
    engine = sql_connect(
        username=creds_dict['username'], 
        password=creds_dict['password'], 
        db_name=creds_dict['db_name'], 
        server_name=creds_dict['servername'], 
        port=creds_dict['port'],
        creds_dict=creds_dict
    )

    # NaN -> None
    null_cols = df.columns[df.isna().any()]
    df[null_cols] = df[null_cols].astype(object).where(pd.notna(df[null_cols]), None)

    # Coerce LargeBinary columns to bytes/None
    binary_cols = [c.name for c in table_schema.columns if isinstance(c.type, sa.LargeBinary)]
    for col in binary_cols:
        if col in df.columns:
            def _to_bytes(v):
                if v is None:
                    return None
                if isinstance(v, (bytes, bytearray, memoryview)):
                    return bytes(v)
                return None
            df[col] = df[col].map(_to_bytes).astype(object)

    # Only bind columns that appear in the DataFrame
    cols = [c for c in table_schema.columns if c.name in df.columns]

    # Typed bindparams straight from the schema (no special-casing)
    values_map = {c.name: sa.bindparam(c.name, type_=c.type) for c in cols}
    stmt = sa.insert(table_schema).values(**values_map)

    rows = df.to_dict(orient='records')

    with engine.connect() as connection:
        with connection.begin():
            try:
                _ensure_table_exists(engine, connection, table_schema)
                if lock:
                    with lock:
                        connection.execute(stmt, rows)
                else:
                    connection.execute(stmt, rows)
                print(f"Data loaded into table {table_schema.name}")
            except SQLAlchemyError as e:
                print(f"SQLAlchemyError occurred: {e}")
                raise

def execute_stored_procedure(creds_dict, procedure_name, lock=None):
    '''
    Execute a stored procedure.
    :param creds_dict: Dictionary containing connection credentials
    :param procedure_name: The name of the stored procedure
    '''
    try:
        # Establish a database connection
        engine = sql_connect(
            username=creds_dict['username'], 
            password=creds_dict['password'], 
            db_name=creds_dict['db_name'], 
            server_name=creds_dict['servername'], 
            port=creds_dict['port'],
            autocommit=True,
            creds_dict=creds_dict
        )
        # Start a connection and transaction
        with engine.connect() as connection:
            if lock:
                with lock:
                    proc = text(f"EXEC {procedure_name}")
                    connection.execute(proc)
            else:
                proc = text(f"EXEC {procedure_name}")
                connection.execute(proc)
            print(f"Stored procedure {procedure_name} executed successfully.")
    except SQLAlchemyError as e:
        print(f"An error occurred while executing the stored procedure: {e}")
        connection.rollback()
        raise

# Function to generate an update query
def generate_update_query(df, dest_table_name, source_table_name, join_columns):
    '''
    Generates an SQL UPDATE query to update data from a source table to a destination table in MS SQL,
    based on the schema of a given pandas DataFrame.

    Args:
    df (pandas.DataFrame): A DataFrame representing the schema of the data that will be loaded into the source table.
    dest_table_name (str): The name of the destination table to update.
    source_table_name (str): The name of the source table where data is loaded.
    join_columns (list of str): The columns to join on, which must exist in both the DataFrame and the destination table.

    Example:
    generate_update_query(df, 'dbo.COMPLETION_HEADER', 'GEO_STAGE', ['WellID', 'DataSource'])

    Returns:
    str: An SQL UPDATE query string.
    '''
    # Validate join columns are in DataFrame
    if not all(col in df.columns for col in join_columns):
        raise ValueError("One or more join columns do not exist in the DataFrame.")

    source_columns = df.columns
    update_columns = [col for col in source_columns if col not in join_columns]
    set_clause = ', '.join(f"{dest_table_name}.{col} = temp.{col}" for col in update_columns)
    join_condition = ' AND '.join(f"{dest_table_name}.{col} = temp.{col}" for col in join_columns)

    query = f"""
    UPDATE {dest_table_name}
    SET {set_clause}
    FROM {dest_table_name}
    INNER JOIN {source_table_name} AS temp
    ON {join_condition};
    """
    return query
