# Schemas for data being loaded into SQL
from sqlalchemy import MetaData, Table, Column, BigInteger, String, Float, DateTime, Text, Integer
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

# Define schemas for staging tables
metadata = MetaData()

# Define the WELL_HEADER_STAGE table structure
well_header_stage = Table('WELL_HEADER_STAGE', metadata,
    Column('WellID', BigInteger),
    Column('CurrentCompletionID', BigInteger),
    Column('API_UWI_Unformatted', String(32)),
    Column('LeaseName', String(256)),
    Column('WellName', String(256)),
    Column('WellNumber', String(256)),
    Column('WellPadID', String(128)),
    Column('Latitude', Float),
    Column('Latitude_BH', Float),
    Column('Longitude', Float),
    Column('Longitude_BH', Float),
    Column('Country', String(2)),
    Column('StateProvince', String(64)),
    Column('County', String(32)),
    Column('Township', String(32)),
    Column('Range', String(32)),
    Column('Section', String(32)),
    Column('Abstract', String(16)),
    Column('Block', String(64)),
    Column('Survey', String(32)),
    Column('District', String(256)),
    Column('Field', String(256)),
    Column('Basin', String(64)),
    Column('Play', String(128)),
    Column('SubPlay', String(64)),
    Column('InitialOperator', String(256)),
    Column('Operator', String(256)),
    Column('Ticker', String(128)),
    Column('GasGatherer', String(256)),
    Column('OilGatherer', String(256)),
    Column('ProducingMethod', String(256)),
    Column('WellStatus', String(64)),
    Column('WellType', String(256)),
    Column('PermitApprovedDate', DateTime),
    Column('SpudDate', DateTime),
    Column('RigReleaseDate', DateTime),
    Column('FirstProdDate', DateTime),
    Column('MD_FT', Float),
    Column('TVD_FT', Float),
    Column('ElevationGL_FT', Float),
    Column('Trajectory', String(64)),
    Column('LateralLength_FT', Float),
    Column('AzimuthFromGridNorth_DEG', Float),
    Column('ToeUp', Integer),
    Column('Geometry', Text),
    Column('SourceEPSG', Float),
    Column('CasingSize_IN', String(256)),
    Column('TubingDepth_FT', Float),
    Column('TubingSize_IN', String(256)),
    Column('DataSource', String(64)),
    Column('DateCreated', DateTime, default="GETDATE()"),
    Column('FitGroup', String(256)),
    Column('Comment', Text)
)
# Define the COMPLETION_HEADER_STAGE table structure
completion_header_stage = Table('COMPLETION_HEADER_STAGE', metadata,
    Column('CompletionID', BigInteger),
    Column('WellID', BigInteger),
    Column('API_UWI_12_Unformatted', String(32)),
    Column('API_UWI_14_Unformatted', String(32)),
    Column('Interval', String(256)),
    Column('Formation', String(256)),
    Column('TargetLithology', String(256)),
    Column('UpperPerf_FT', Integer),
    Column('LowerPerf_FT', Integer),
    Column('PerfInterval_FT', Integer),
    Column('TopOfZone_FT', Float),
    Column('BottomOfZone_FT', Float),
    Column('DistFromBaseZone_FT', Float),
    Column('DistFromTopZone_FT', Float),
    Column('PermitApprovedDate', DateTime),
    Column('CompletionDate', DateTime),
    Column('FirstProdDate', DateTime),
    Column('CompletionDesign', String(64)),
    Column('FracJobType', String(32)),
    Column('ProppantType', String(32)),
    Column('WellServiceProvider', String(256)),
    Column('Proppant_LBS', Float),
    Column('TotalFluidPumped_BBL', Float),
    Column('FracStages', Integer),
    Column('TotalClusters', Integer),
    Column('AvgTreatmentPressure_PSI', Integer),
    Column('AvgTreatmentRate_BBLPerMin', Float),
    Column('OilTestRate_BBLPerDAY', Float),
    Column('GasTestRate_MCFPerDAY', Float),
    Column('WaterTestRate_BBLPerDAY', Float),
    Column('TestFCP_PSI', Float),
    Column('TestFTP_PSI', Float),
    Column('TestChokeSize_64IN', Integer),
    Column('ReservoirPressure_PSI', Float),
    Column('Bottom_Hole_Temp_DEGF', Float),
    Column('Isopach_FT', Float),
    Column('EffectivePorosity_PCT', Float),
    Column('WaterSaturation_PCT', Float),
    Column('OilGravity_API', Float),
    Column('GasGravity_SG', Float),
    Column('DataSource', String(64)),
    Column('DateCreated', DateTime, default="GETDATE()"),
)
# Define the PRODUCTION_STAGE table structure
production_stage = Table('PRODUCTION_STAGE', metadata,
    Column('WellID', BigInteger),
    Column('API_UWI_Unformatted', String(32)),
    Column('Date', DateTime),
    Column('ProducingDays', Integer),
    Column('Measure', String(64)),
    Column('Value', Float),
    Column('Units', String(64)),
    Column('Comment', Text),
    Column('DataSource', String(64)),
    Column('Cadence', String(64)),
    Column('DateCreated', DateTime, default="GETDATE()"),
)

# Define the WELL_SPACING_STAGE table structure
well_spacing_stage = Table('WELL_SPACING_STAGE', metadata,
    Column('WellID', BigInteger),
    Column('neighboring_WellID', BigInteger),
    Column('clipped_lateral_geometry', Text),
    Column('lateral_geometry_buffer', Text),
    Column('clipped_neighbor_lateral_geometry', Text),
    Column('neighbor_lateral_geometry_buffer', Text),
    Column('MinDistance', Float),
    Column('MedianDistance', Float),
    Column('MaxDistance', Float),
    Column('AvgDistance', Float),
    Column('neighbor_IntersectionFraction', Float),
    Column('RelativePosition', String(64)),
    Column('Projection', String(64)),
    Column('UpdateDate', DateTime, default="GETDATE()")
)

# Define the PARENT_CHILD_STAGE table structure
parent_child_stage = Table('PARENT_CHILD_STAGE', metadata,
    Column('WellID', BigInteger),
    Column('Date', DateTime),
    Column('Relationship', String(10)),
    Column('ClosestHzDistance', Float),
    Column('ClosestHzDistance_Left', Float),
    Column('ClosestHzDistance_Right', Float),
    Column('ScenarioName', String(64)),
    Column('UpdateDate', DateTime, default="GETDATE()")
)

# Define the FORECAST_STAGE table structure
forecast_stage = Table('FORECAST_STAGE', metadata,
    Column('WellID', BigInteger),
    Column('Measure', String(64)),
    Column('Units', String(64)),
    Column('StartDate', DateTime),
    Column('Q1', Float),
    Column('Q2', Float),
    Column('Q3', Float),
    Column('Qabn', Float),
    Column('Dei', Float),
    Column('b_factor', Float),
    Column('Def', Float),
    Column('t1', Integer),
    Column('t2', Integer),
    Column('Analyst', String(64)),
    Column('DateCreated', DateTime, default="GETDATE()")
)

# Define the FORECAST_VOLUME_STAGE table structure
forecast_volume_stage = Table('FORECAST_VOLUME_STAGE', metadata,
    Column('ForecastID', String(36)),
    Column('WellID', BigInteger),
    Column('Measure', String(64)),
    Column('StartDate', DateTime),
    Column('AdjustedMonth', Integer),
    Column('ProdMonth', Integer),
    Column('ProductionRate', Float),
    Column('De', Float),
    Column('CumulativeProduction', Float),
    Column('MonthlyVolume', Float)
)

# Function to create a table schema from a pandas dataframe
def dataframe_to_table_schema(df, table_name, type_overrides=None):
    '''
    Create SQLAlchemy Table schema from pandas DataFrame.
    :param df: DataFrame to infer schema
    :param table_name: Name of the table to create
    :param type_overrides: Optional dictionary to override column data types
    :return: SQLAlchemy Table object
    '''

    if not isinstance(df, pd.DataFrame):
        raise ValueError("The provided data is not a pandas DataFrame.")

    metadata = MetaData()
    columns = []
    for col_name, dtype in df.dtypes.items():
        if type_overrides and col_name in type_overrides:
            # Use the override if specified
            column_type = type_overrides[col_name]
        elif 'float' in str(dtype):
            column_type = Float()
        elif 'int' in str(dtype):
            column_type = Integer()
        elif 'object' in str(dtype):
            column_type = String(255)
        else:
            raise ValueError(f"Unhandled dtype: {dtype}")
        
        columns.append(Column(col_name, column_type))

    return Table(table_name, metadata, *columns)
