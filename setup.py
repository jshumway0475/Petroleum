from setuptools import setup, find_packages

setup(
    name='AnalysisAndDBScripts',
    version='0.94',
    packages=find_packages(),
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
)
