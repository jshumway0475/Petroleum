import json
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from sqlalchemy.exc import SQLAlchemyError
import pyodbc
import pandas as pd

# Function to return a dictionary from a json file
def get_json_dict(json_file):
    with open(json_file, 'r') as file:
        json_dict = json.load(file)
    return json_dict

# Function to query using pyodbc
def sql_query_pyodbc(query, creds_dict):
    '''
    Query a SQL Server database using pyodbc.
    :param query: SQL query to execute
    :param creds_dict: Dictionary containing connection credentials
    :return: Query results as a pandas DataFrame
    '''
    try:
        # Form the connection string
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={creds_dict['servername']};DATABASE={creds_dict['db_name']};UID={creds_dict['username']};PWD={creds_dict['password']}"
        if 'port' in creds_dict:
            conn_str += f";PORT={creds_dict['port']}"

        # Establish a database connection
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()

        # Extract column names from cursor
        columns = [column[0] for column in cursor.description]

        # Create a DataFrame from the rows
        df = pd.DataFrame.from_records(results, columns=columns)

    except Exception as e:
        print(f"An error occurred while querying the database: {e}")
        return None

    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    return df

# Function to connect to SQL Server database
def sql_connect(username, password, db_name, server_name, port, return_engine=True, pool_size=10, max_overflow=20, pool_timeout=60, pool_recycle=3600, autocommit=False):
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
    Returns:
    - engine (sqlalchemy.engine.base.Engine): The SQLAlchemy engine object.
    - connection_string (str): The connection string used to connect to the database.
    '''
    # Construct the ODBC connection string
    conn_stmt = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server_name},{port};DATABASE={db_name};UID={username};PWD={password}'
    
    # URL-encode the ODBC connection string
    encoded_conn_stmt = quote_plus(conn_stmt)
    
    # Create the SQLAlchemy engine using the encoded connection string
    connection_string = f"mssql+pyodbc:///?odbc_connect={encoded_conn_stmt}"

    # Set up the connection parameters with autocommit if needed
    engine_params = {
        'pool_size': pool_size,
        'max_overflow': max_overflow,
        'pool_timeout': pool_timeout,
        'pool_recycle': pool_recycle
    }

    engine = create_engine(
        connection_string,
        isolation_level="AUTOCOMMIT" if autocommit else "READ COMMITTED",
        **engine_params
    )

    if return_engine:
        return engine
    else:
        return connection_string

# Helper function to ensure table exists
def _ensure_table_exists(engine, connection, table_schema):
    if not engine.dialect.has_table(connection, table_schema.name):
        table_schema.create(bind=engine, checkfirst=True)
        print(f"Created table {table_schema.name}")

# Function to load data into a SQL Server database
def load_data_to_sql(df, creds_dict, table_schema, lock=None):
    '''
    Load data into a SQL Server table.
    :param df: DataFrame containing the data to load
    :param creds_dict: Dictionary containing connection credentials
    :param table_name: The name of the database table
    :param lock: A threading lock object to synchronize access to the database
    '''
    engine = sql_connect(
        username=creds_dict['username'], 
        password=creds_dict['password'], 
        db_name=creds_dict['db_name'], 
        server_name=creds_dict['servername'], 
        port=creds_dict['port']
    )
    with engine.connect() as connection:
        with connection.begin():
            try:
                if lock:
                    with lock:
                        _ensure_table_exists(engine, connection, table_schema)
                    
                        # Load data into the table
                        df.to_sql(table_schema.name, con=connection, if_exists='append', index=False)
                        print(f"Data loaded into table {table_schema.name}")
                else:
                    _ensure_table_exists(engine, connection, table_schema)

                    # Load data into the table
                    df.to_sql(table_schema.name, con=connection, if_exists='append', index=False)
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
            autocommit=True
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
