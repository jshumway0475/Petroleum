import os
from setuptools import setup, find_packages

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
    
setup(
    name='AnalysisAndDBScripts',
    version='0.94',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'pandas',
        'ruptures',
        'sqlalchemy',
        'geopandas',
        'shapely',
        'pyproj',
        'scipy',
        'petbox-dca',
        'statsmodels',
        'PyYAML',
        'arviz',
        'pymc',
        'aesara',
        'jax',
        'blackjax',
        'numba',
        'dask[complete]',
        'dask_geopandas',
        'loky'
    ],
    author='Jacob Shumway',
    author_email='jshumway0475@gmail.com',
    description='A collection of geospatial, database, oil and gas production forecasting, fluid property, material balance, and discounted cash flow functions.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jshumway0475/Petroleum',
    license='MIT',
)
