# PlayInsight Suite: Oil and Gas Well Spacing, Forecasting, and Economic Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Introduction

PlayInsight Suite is a Python-based toolkit designed to accelerate oil and gas property evaluations by automating key technical workflows. It provides fast, scalable calculations for well spacing, parent-child relationship identification, production forecasting using Arps models, and discounted cash flow analysis. PlayInsight integrates seamlessly with analytics databases, enabling engineers, analysts, and asset teams to handle large datasets efficiently and make informed economic decisions across unconventional resource plays. The recommended usage involves integrating the application with a Data Warehouse containing Well and Production data. The workflow enabled by the application includes:

1. Calculation of 2D distances between horizontal wells
2. Identification of parent-child relationships based on well spacing calculations
3. Production forecasting using Arps hyperbolic, harmonic, and exponential models
4. Calculation of future cash flow streams based on production forecasts, including discounted cash flow analysis

## Repository Guide

This repository consists of the following folders/files:

### AnalysisAndDBScripts
Python files that contain the core logic associated with the PlayInsight Suite:
- `__init__.py`: Initializes the code within this folder
- `dcf_functions.py`: Functions related to future cash flow and discounted cash flow calculations
- `fluid_properties.py`: Functions used to calculate oil and gas fluid properties
- `geo_functions.py`: Functions used to derive values from geologic grids
- `prod_fcst_functions.py`: Logic related to forecasting oil and gas time-series production data
- `sql_connect.py`: Helper functions used to interact with MS SQL Server databases
- `sql_schemas.py`: MS SQL Server database table schemas recommended for use with the PlayInsight Suite
- `well_spacing.py`: Functions related to the calculation of well spacing and parent-child relationships between wells

### config
- `__init__.py`: Initializes the code within this folder
- `config_loader.py`: Enables the usage of the YAML file within this folder
- `analytics_config.yaml`: Contains user-defined parameters and defaults that are arguments to the functions in the `AnalysisAndDBScripts` folder

### data/grids
Folder with some example files that can be used with the config.

### play_assessment_tools
A collection of data transformations based on the functions in the `AnalysisAndDBScripts` folder, representing the complete workflow intended by the PlayInsight Suite.
- Python files:
  - `arps_autofit.py`: Complete workflow needed to forecast oil and gas production data
  - `parent_child_assignments.py`: Calculates parent-child relationships between horizontal wells
  - `well_spacing_calcs.py`: Calculates 2D distances between horizontal wells within a user-defined distance
  - `update_geology.py`: Samples geologic grids to extract values and associate those values with wells
- Jupyter Notebooks: A collection of notebooks used for testing and product development, not likely to be included in the final production of PlayInsight.

### spotfire_utils
A collection of CSS scripts, Python data functions, and IronPython scripts that can be used to create a GUI using Spotfire to interact with the PlayInsight Suite.

### sql
SQL scripts that serve as a working example of the Data Warehouse needed for the PlayInsight Suite. Includes:
- Scripts to create the Data Warehouse
- Stored procedures to populate the Data Warehouse (using Enverus data as an example)
- Views to enable the Spotfire UI

## Other Files
- `.gitignore`: Instructs Git on the maintenance of the repository
- `setup.py`: Manages import and versioning of the files in this repository
- `Dockerfile`: Instructions to create a Docker image containing the Python environment from which the PlayInsight Suite can be deployed
- `requirements.txt`: Used by the Dockerfile to import the necessary Python libraries to execute the code in the PlayInsight Suite
- `PlayInsight.pdf`: Presentation explaining the functionality of the PlayInsight Suite
- `README.md`: This file

## Usage
1. **Pull the Docker image:** Run the following command to create the Python environment with the necessary dependencies and import the files needed to use the PlayInsight Suite:
    ```bash
    docker pull jshumway0475/playinsight-image:latest
    ```
2. **Set up and populate the Data Warehouse:** Use the files in the `sql` folder as a starting point.
3. **Modify the `analytics_config.yaml`:** Update this file with the necessary permissions and user-defined defaults needed to execute the Python functions.
4. **Use the transforms:** Use the transforms in the `play_assessment_tools` folder to populate the Data Warehouse with spacing, parent-child relationships, and forecasts. Add these transforms to a schedule.
5. **Configure Spotfire or another BI product:** Visualize and interact with the data in the Analytics database populated by the PlayInsight Suite.

## License

This project is licensed under the [MIT License](LICENSE).

© 2025 Jay Engineering, LLC.  
By using this software, you agree to the terms of the license.

For questions or further information, please contact [jshumway0475@gmail.com](mailto:jshumway0475@gmail.com).
