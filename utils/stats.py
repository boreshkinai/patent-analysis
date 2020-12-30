import pandas as pd
import dask.dataframe as dask


def print_ddf_stats(ddf: dask):
    print("Dataset info")
    print("Number of rows:", len(ddf))
    print(ddf.head())
    print()
    print("Stats")
    print(ddf.describe().compute())
    print()
    print("Missing values")
    print(ddf.isnull().sum(axis=0).compute())

