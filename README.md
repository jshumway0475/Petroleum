# PlayInsight Suite: Oil and Gas Well Spacing, Forecasting, and Economic Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Introduction

PlayInsight Suite is a Python-based toolkit designed to accelerate oil and gas property evaluations by automating key technical workflows. It provides fast, scalable calculations for:

- Well spacing and 2D distance analysis
- Parent-child relationship identification
- Arps production forecasting (hyperbolic, harmonic, exponential)
- Discounted cash flow (DCF) economic evaluation

PlayInsight integrates seamlessly with analytics databases, enabling engineers, analysts, and asset teams to handle large datasets and drive decisions in unconventional resource plays.

---

## Repository Overview

This repository includes:

### `AnalysisAndDBScripts/`

Core modules implementing the PlayInsight workflows:

- `dcf_functions.py`: Future cash flow and DCF calculations
- `fluid_properties.py`: Oil and gas fluid property calculations
- `geo_functions.py`: Sampling from geologic grids
- `prod_fcst_functions.py`: Arps-based production forecasting
- `sql_connect.py`: Helpers for MS SQL Server interaction
- `sql_schemas.py`: Sample SQL schema for use with the suite
- `well_spacing.py`: 2D spacing and parent-child logic

### `config/`

Configuration system for customizing workflows:

- `analytics_config.yaml`: Stores all user-defined settings
- `config_loader.py`: Helper to load config in Python

### `data/grids/`

Example geologic grids to test grid sampling functions.

### `play_assessment_tools/`

End-to-end pipeline scripts for applying spacing, forecasting, and analysis:

- `arps_autofit.py`: Fit Arps models to production data
- `parent_child_assignments.py`: Determine parent-child well relationships
- `well_spacing_calcs.py`: Measure inter-well spacing
- `update_geology.py`: Associate wells with geologic data
- Jupyter notebooks used during prototyping and development

### `spotfire_utils/`

Scripts and helpers (CSS, IronPython, data functions) to build a GUI in TIBCO Spotfire.

### `sql/`

Example SQL scripts for building and populating a data warehouse compatible with PlayInsight.

- Table creation scripts
- Stored procedures
- SQL views to power analytics dashboards

### `devcontainer/` (Optional)

VS Code Remote - Containers configuration:

- `Dockerfile`: Python + dependencies environment
- `devcontainer.json`: VS Code configuration

Using this is **optional**, but highly recommended for contributors and power users.

---

## Other Files

- `requirements.txt`: Python libraries required
- `setup.py`: Installation and module packaging
- `.gitignore`: Files and folders excluded from version control
- `Dockerfile`: Standalone image build for manual Docker usage
- `PlayInsight.pdf`: Overview presentation of core workflows
- `README.md`: This file
- `USAGE.md`: Step-by-step instructions for setup and usage

---

## Getting Started

See [USAGE.md](USAGE.md) for setup instructions, including:

- Cloning the repo
- Using VS Code with Dev Containers
- Running analysis pipelines
- Connecting to SQL and Spotfire

---

## License

This project is licensed under the [MIT License](LICENSE).  
© 2025 Jay Engineering, LLC.  
By using this software, you agree to the terms of the license.

📧 Questions? Contact [jshumway0475@gmail.com](mailto:jshumway0475@gmail.com).
