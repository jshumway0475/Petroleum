-- Create a database and schema for data pulled from Enverus.
-- Jacob Shumway 10-2023
DROP DATABASE IF EXISTS Analytics;

-- Create the database 'Analytics'
CREATE DATABASE Analytics;
GO


-- Switch to the 'Analytics' database
USE Analytics;
GO

-- Create table with well header data
CREATE TABLE dbo.WELL_HEADER (
	WellID						BIGINT NOT NULL PRIMARY KEY,
	CurrentCompletionID			BIGINT,
	API_UWI_Unformatted			VARCHAR(32),
	LeaseName					VARCHAR(256), 
	WellName					VARCHAR(256), 
	WellNumber					VARCHAR(256), 
	WellPadID					VARCHAR(128),
	Latitude					FLOAT, 
	Latitude_BH					FLOAT, 
	Longitude					FLOAT, 
	Longitude_BH				FLOAT,
	Country						VARCHAR(2), 
	StateProvince				VARCHAR(64),
	County						VARCHAR(32),
	Township					VARCHAR(32),
	Range						VARCHAR(32),
	Section						VARCHAR(32),
	Abstract					VARCHAR(16),
	Block						VARCHAR(64),
	Survey						VARCHAR(32),
	District					VARCHAR(256),
	Field						VARCHAR(256), 
	Basin						VARCHAR(64),
	Play						VARCHAR(128),
	SubPlay						VARCHAR(64),
	InitialOperator				VARCHAR(256),
	Operator					VARCHAR(256),
	Ticker						VARCHAR(128),
	GasGatherer					VARCHAR(256),
	OilGatherer					VARCHAR(256),   
	ProducingMethod				VARCHAR(256),
	WellStatus					VARCHAR(64), 
	WellType					VARCHAR(256),
	PermitApprovedDate			DATETIME,
	SpudDate					DATETIME,
	RigReleaseDate				DATETIME,
	FirstProdDate				DATETIME,
	MD_FT						FLOAT,
	TVD_FT						FLOAT,
	ElevationGL_FT				FLOAT,
	Trajectory					VARCHAR(64),
	LateralLength_FT			FLOAT,
	AzimuthFromGridNorth_DEG	FLOAT, 
	ToeUp						INT,
	Geometry					GEOMETRY,
	SourceEPSG					INT,
	CasingSize_IN				VARCHAR(256),        
	TubingDepth_FT				FLOAT, 
	TubingSize_IN				VARCHAR(256), 
	DataSource					VARCHAR(64),
	DateCreated					DATETIME,
	FitGroup					VARCHAR(256),
	Comment						VARCHAR(MAX),
	FitMethod					VARCHAR(256) CHECK (FitMethod IN ('curve_fit','monte_carlo','differential_evolution'))
);

-- Create table with completion header data
CREATE TABLE dbo.COMPLETION_HEADER (
	CompletionID				BIGINT NOT NULL, 
	WellID						BIGINT NOT NULL,
	API_UWI_12_Unformatted		VARCHAR(32), 
	API_UWI_14_Unformatted		VARCHAR(32),
	Interval					VARCHAR(256),
	SubInterval					VARCHAR(256),
	Formation					VARCHAR(256),  
	TargetLithology				VARCHAR(256),
	UpperPerf_FT				INT,
	LowerPerf_FT				INT, 
	PerfInterval_FT				INT, 
	TopOfZone_FT				FLOAT,
	BottomOfZone_FT				FLOAT,
	DistFromBaseZone_FT			FLOAT, 
	DistFromTopZone_FT			FLOAT,
	PermitApprovedDate			DATETIME,
	CompletionDate				DATETIME,  
	FirstProdDate				DATETIME,
	CompletionDesign			VARCHAR(64),
	FracJobType					VARCHAR(32),  
	ProppantType				VARCHAR(32),  
	WellServiceProvider			VARCHAR(256),
	Proppant_LBS				FLOAT,
	TotalFluidPumped_BBL		FLOAT,
	FracStages					INT,
	TotalClusters				INT,
	AvgTreatmentPressure_PSI	INT, 
	AvgTreatmentRate_BBLPerMin	FLOAT,
	OilTestRate_BBLPerDAY		FLOAT,
	GasTestRate_MCFPerDAY		FLOAT,
	WaterTestRate_BBLPerDAY		FLOAT,
	TestFCP_PSI					FLOAT,
	TestFTP_PSI					FLOAT,
	TestChokeSize_64IN			INT, 
	ReservoirPressure_PSI		FLOAT,
	Bottom_Hole_Temp_DEGF		FLOAT,
	Isopach_FT					FLOAT,
	EffectivePorosity_PCT		FLOAT,  
	WaterSaturation_PCT			FLOAT,
	OilGravity_API				FLOAT,
	GasGravity_SG				FLOAT,     
	DataSource					VARCHAR(64) NOT NULL,
	DateCreated					DATETIME DEFAULT GETDATE(),
	SubInterval					VARCHAR(256),
	PRIMARY KEY (CompletionID, WellID, DataSource),
	FOREIGN KEY (WellID) 
		REFERENCES dbo.WELL_HEADER(WellID)
		ON DELETE CASCADE
);

-- Create table with production volumes
CREATE TABLE dbo.PRODUCTION (  
	ProductionID				UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
	WellID						BIGINT NOT NULL,
	API_UWI_Unformatted			VARCHAR(32),  
	Date						DATETIME NOT NULL,
	Measure						VARCHAR(64) CHECK (Measure IN ('OIL', 'GAS', 'WATER', 'NGL', 'TP', 'CP', 'CHOKE')),
	Value						FLOAT, 
	Units						VARCHAR(64),
	Cadence						VARCHAR(64) CHECK (Cadence IN ('DAILY', 'MONTHLY')),
	ProducingDays				INT,
	DataSource					VARCHAR(64),
	Comment						VARCHAR(MAX),
	DateCreated					DATETIME DEFAULT GETDATE(),
	DateRank					INT NULL,
	SourceRank					INT NULL,
	FOREIGN KEY (WellID) 
		REFERENCES dbo.WELL_HEADER(WellID)
		ON DELETE CASCADE
);

-- Create table with well spacing
CREATE TABLE dbo.WELL_SPACING (
	WellSpacingID						UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
	WellID								BIGINT NOT NULL,
    neighboring_WellID					BIGINT NOT NULL,
    clipped_lateral_geometry			GEOMETRY,
    lateral_geometry_buffer				GEOMETRY,
    clipped_neighbor_lateral_geometry	GEOMETRY,
    neighbor_lateral_geometry_buffer	GEOMETRY,
    MinDistance							FLOAT,
    MedianDistance						FLOAT,
    MaxDistance							FLOAT,
    AvgDistance							FLOAT,
    neighbor_IntersectionFraction		FLOAT,
    RelativePosition					VARCHAR(5),
    Projection							VARCHAR(24),
    UpdateDate							DATETIME NOT NULL,
	FOREIGN KEY (WellID) 
		REFERENCES dbo.WELL_HEADER(WellID)
		ON DELETE CASCADE
);

-- Create table for parent-child relationships
CREATE TABLE dbo.PARENT_CHILD (
	ParentChildID				UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
	WellID						BIGINT NOT NULL,
    Date						DATETIME,
    Relationship				VARCHAR(10),
    ClosestHzDistance			FLOAT,
    ClosestHzDistance_Left		FLOAT,
    ClosestHzDistance_Right		FLOAT,
    ScenarioName				VARCHAR(64),
    UpdateDate					DATETIME,
	FOREIGN KEY (WellID) 
		REFERENCES dbo.WELL_HEADER(WellID)
		ON DELETE CASCADE
);

-- Create a table to store forecasts and type curves
CREATE TABLE dbo.FORECAST (
	ForecastID					UNIQUEIDENTIFIER PRIMARY KEY DEFAULT NEWID(),
	WellID						BIGINT,
	Measure						VARCHAR(64) CHECK (Measure IN ('OIL', 'GAS', 'WATER', 'OIL_CUT', 'WATER_CUT', 'GOR', 'OIL_YIELD')),
	Units						VARCHAR(64),
	StartDate					DATETIME,
	Q1							FLOAT, -- Initial rate of the first segment, typically an initial incline or cleanup period
	Q2							FLOAT, -- Initial rate of the second segment, also the final rate of the first segment
	Q3							FLOAT, -- Initial rate at start of decline period. Can also be the final rate of the second segment
	Qabn						FLOAT, -- Final abandonment rate
	Dei							FLOAT,
	b_factor					FLOAT,
	Def							FLOAT,
	t1							INT,
	t2							INT,
	Analyst						VARCHAR(64),
	DateCreated					DATETIME DEFAULT GETDATE(),
	TraceBlob					VARBINARY(MAX),
	FOREIGN KEY (WellID) 
		REFERENCES dbo.WELL_HEADER(WellID)
		ON DELETE CASCADE
);

-- Create a Materialized Table of well header, completion header, and aggregated production metrics for use in analytics
CREATE TABLE dbo.WellHeaderMaterialized (
	WellID						BIGINT PRIMARY KEY NOT NULL,
	CurrentCompletionID			BIGINT NOT NULL,
	GOR							FLOAT,
	WTR_CUT						FLOAT,
	TotalProdMonths				INT,
	PrimaryPhase				VARCHAR(10),
	Oil_Max_Month				FLOAT,
	Gas_Max_Month				FLOAT,
	Water_Max_Month				FLOAT,
	WTR_YIELD					FLOAT
	FOREIGN KEY (WellID) 
		REFERENCES dbo.WELL_HEADER(WellID)
		ON DELETE CASCADE
);

-- Create a Materialized Table of production data for use in analytics
CREATE TABLE dbo.ProductionMaterialized (
	SourceID					UNIQUEIDENTIFIER NOT NULL,
	WellID						BIGINT NOT NULL,
	Basin						VARCHAR(64),
	Trajectory					VARCHAR(64),
	Measure						VARCHAR(64),
	Date						DATETIME,
	ProdMonth					INT,
	MonthlyVolume				FLOAT,
	CumulativeProduction		FLOAT,
	LateralLength_FT			FLOAT,
	Prop_Intensity				FLOAT,
	Fluid_Intensity				FLOAT,
	LL_BIN						VARCHAR(50),
	Comment						VARCHAR(MAX),
	DateCreated					DATETIME
	PRIMARY KEY (SourceID),
	FOREIGN KEY (WellID) 
		REFERENCES dbo.WELL_HEADER(WellID)
		ON DELETE CASCADE
);

CREATE TABLE dbo.WELL_OVERRIDE (
    WellID						bigint,
    CurrentCompletionID			bigint,
	API10						varchar(32),
	DataSource					varchar(64),
    WellName					varchar(256),
    Basin						varchar(64),
    Play						varchar(128),
    SubPlay						varchar(64),
    InitialOperator				varchar(256),
    Operator					varchar(256),
	GasGatherer					varchar(256),
    OilGatherer					varchar(256),
    ProducingMethod				varchar(256),
    WellStatus					varchar(64),
    WellType					varchar(256),
    PermitApprovedDate			datetime,
    SpudDate					datetime,
    RigReleaseDate				datetime,
    FirstProdDate				datetime,
    MD_FT						float,
    TVD_FT						float,
    ElevationGL_FT				float,
    Trajectory					varchar(64),
    LateralLength_FT			float,
    Interval					varchar(256),
	SubInterval					varchar(256),
    Formation					varchar(256),
    TargetLithology				varchar(256),
    FracJobType					varchar(32),
    FracStages					int,
	TotalClusters				int, 
	AvgTreatmentPressure_PSI	float, 
	AvgTreatmentRate_BBLPerMin	float, 
	TestFTP_PSI					float, 
	TestFCP_PSI					float,
    Proppant_LBS				float,
    TotalFluidPumped_BBL		float,
    Prop_Intensity				float,
    Fluid_Intensity				float,
    Isopach_FT					float,
    WaterSaturation_PCT			float,
    EffectivePorosity_PCT		float,
    ReservoirPressure_PSI		float,
    Bottom_Hole_Temp_DEGF		float,
    OilGravity_API				float,
    GasGravity_SG				float,
	Geometry					geometry,
	TotalClusters				INT,
	AvgTreatmentPressure_PSI	INT, 
	AvgTreatmentRate_BBLPerMin	FLOAT,
	TestFCP_PSI					FLOAT,
	TestFTP_PSI					FLOAT,
	SubInterval					VARCHAR(256),
	TopOfZone_FT				FLOAT,
	BottomOfZone_FT				FLOAT,
	PRIMARY KEY (WellID, DataSource)
);

