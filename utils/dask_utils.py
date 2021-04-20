import dask.dataframe as dask
import pandas as pd
import numpy as np
from typing import List


def month_diff(a, b):
    return 12 * (a.dt.year - b.dt.year) + (a.dt.month - b.dt.month)


def year_diff(a, b):
    return a.dt.year - b.dt.year


def chunk(s):
    '''
    The function applied to the
    individual partition (map)
    '''    
    return s.apply(lambda x: list(set(x)))


def agg(s):
    '''
    The function whic will aggrgate 
    the result from all the partitions(reduce)
    '''
    s = s._selected_obj    
    return s.groupby(level=list(range(s.index.nlevels))).sum()


def finalize(s):
    '''
    The optional functional that will be 
    applied to the result of the agg_tu functions
    '''
    return s.apply(lambda x: len(set(x)))


tunique = dask.Aggregation('tunique', chunk, agg, finalize)


def compute_ltc(df: pd.DataFrame, year_range: List[int], last_date: pd.Timestamp) -> pd.DataFrame:
    """
    This computes the citation life time curve: the number of citations in a given number of years
    
    """
    df = df[['patent_date', 'citedby_patent_number', 'citedby_patent_date', 'citation_age']]
    df = df.reset_index().drop_duplicates()
    agg_fns = {}
    for year in year_range:
        citaton_column = f"CY{year}"
        df[citaton_column] = (df['citation_age'] < year)
        agg_fns[citaton_column] = 'sum'
    
    agg_fns.update({'patent_date': 'first', 'citedby_patent_date': 'first'})
    df = df.groupby('index').agg(agg_fns)
    
    for year in year_range:
        citaton_column = f"CY{year}"
        df.at[df.patent_date > last_date - pd.DateOffset(years=year), citaton_column] = np.NaN
        
    return df

