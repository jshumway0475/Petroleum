import os
from setuptools import setup, find_packages

NAME = "AnalysisAndDBScripts"
VERSION = "0.96"

this_directory = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(this_directory, "README.md")
try:
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Utilities for geospatial analytics, forecasting, and database workflows."

install_requires = [
    "numpy>=1.23",
    "pandas>=1.5",
    "scipy>=1.10",
    "sqlalchemy>=2.0",
    "PyYAML>=6.0",
]

extras_ml = [
    "scikit-learn>=1.2",
    "statsmodels>=0.14",
    "numba>=0.57",
    "ruptures>=1.1",
    "loky>=3.4",
    "petbox-dca>=1.1",
]

extras_bayes = [
    "pymc>=5.10",
    "arviz>=0.16",
    "jax>=0.4.20",
    "jaxlib>=0.4.20",
    "blackjax>=1.0",
]

extras_geo = [
    "geopandas>=0.13",
    "shapely>=2.0",
    "pyproj>=3.6",
    "fiona>=1.9",
    "rasterio>=1.3",
    "dask_geopandas>=0.4",
]

extras_dev = [
    "dask[dataframe,distributed,diagnostics]>=2024.5.0",
    "ipykernel>=6.29",
]

extras = {
    "geo": extras_geo,
    "ml": extras_ml,
    "bayes": extras_bayes,
    "dev": extras_dev,
}

extras["all"] = sorted({pkg for group in extras.values() for pkg in group})

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=install_requires,
    extras_require=extras,
    include_package_data=True,
    zip_safe=False,
    author='Jacob Shumway',
    author_email='jshumway0475@gmail.com',
    description='Geospatial, database, production forecasting, mixed linear modeling, fluid property, material balance, and DCF utilities.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jshumway0475/Petroleum',
    license='MIT',
    license_files=['LICENSE'],
)
