from keras.utils import get_file
from typing import List, Dict
import pathlib
import os
import pandas as pd
import numpy as np
import argparse
import sys
import csv

import dask.dataframe as dask
from dask.distributed import Client


DATA_TIMESTAMP = '20171226'
ROOT_URL = 'http://www.patentsview.org/data/' + DATA_TIMESTAMP + '/'

DATASETS = ['patent.tsv.zip', 
            'uspatentcitation.tsv.zip',
            ]
DATASETS_EXCLUDE_COLUMNS = {'patent.tsv.zip': ['abstract', 'title'],
                            'uspatentcitation.tsv.zip': [],
                            }

numeric = lambda x: pd.to_numeric(x, errors='coerce')

DATASETS_SPEC = {
    'patent': {
        'index': ['id'],
        'dtype': {'date': 'datetime64[ns]', 'num_claims': 'int16'},
        'usecols': ['id', 'type', 'number', 'country', 'date', 'title', 'kind', 'num_claims'],
        'parse_dates': ['date'],
        'parse_numeric': {'num_claims': np.int16},
        'drop_columns': [],
    },
    'uspatentcitation': {
        'index': ['patent_id'],
        'dtype': {},
        'usecols': [],
        'parse_dates': [],
        'parse_numeric': {},
        'drop_columns': [],
        
    },
#     'location': {
#         'index': ['id'],
#     },
}


def download_file_cached(file_name: str, drop_columns: List[str], local_storage: str, permanent_storage: str) -> None:
    pathlib.Path(local_storage).mkdir(parents=True, exist_ok=True)
    pathlib.Path("/tmp/patent-analysis/").mkdir(parents=True, exist_ok=True)
    if not os.path.exists(f"{permanent_storage}/{file_name}"):
        print(f"Downloading {file_name} locally")
        get_file(file_name, origin=os.path.join(ROOT_URL, file_name), cache_dir=local_storage, cache_subdir='')

        print(f"Creating pickled {file_name}")
        df = pd.read_csv(os.path.join(local_storage, file_name), sep='\t', error_bad_lines=False, dtype=str)
        df.drop(columns=drop_columns, inplace=True)
        df.to_pickle(path=f"{local_storage}/{file_name}.pickle")

        print(f"Transferring to permanent storage {file_name}")
        os.system(f"rsync -avr --progress {local_storage}* {permanent_storage}/")
        

def prepare_datasets_cached(dataset_name: str, 
                            local_storage: str, permanent_storage: str,
                            params: Dict) -> None:
    
    pathlib.Path(local_storage).mkdir(parents=True, exist_ok=True)
    pathlib.Path("/tmp/patent-analysis/").mkdir(parents=True, exist_ok=True)
    file_name_zip = f"{dataset_name}.tsv.zip"
    file_name_tsv = f"{dataset_name}.tsv"
    if not os.path.exists(os.path.join(permanent_storage, file_name_zip)):
        print(f"Downloading {file_name_zip} locally in {local_storage}")
        get_file(file_name_zip, origin=os.path.join(ROOT_URL, file_name_zip), 
                 cache_dir=local_storage, cache_subdir='')
        
        print(f"Unzipping {file_name_zip} locally in {local_storage}")
        cmd = f"unzip -u {os.path.join(local_storage, file_name_zip)} -d {local_storage}/"
        os.system(cmd)
        
        print("Create dask local cluster")
        ncpus = len(os.sched_getaffinity(0))
        client = Client(n_workers=ncpus, threads_per_worker=1, memory_limit='6GB', dashboard_address='6006')
    
        print(f"Creating parquete {dataset_name}.parquet")
        ddf = dask.read_csv(os.path.join(local_storage, file_name_tsv),
                            sep='\t', error_bad_lines=False, 
                            dtype=str, 
                            usecols=params['usecols'],
                            quoting=csv.QUOTE_NONNUMERIC,
#                             engine='python'
                            )
        
        for k in params['parse_dates']:
            ddf[k] = dask.to_datetime(ddf[k], errors='coerce')
        ddf = ddf.dropna(subset=params['parse_dates'])
            
        for k, v in params['parse_numeric'].items():
            ddf[k] = dask.to_numeric(ddf[k], errors='coerce')
            ddf = ddf.dropna(subset=[k])
            ddf[k] = ddf[k].astype(v)
            
        ddf = ddf.set_index(params['index'])
        ddf = ddf.drop(columns=params['drop_columns'])
            
        ddf.to_parquet(os.path.join(local_storage, f"{dataset_name}.parquet"), engine='pyarrow', schema="infer")

        print(f"Transferring to permanent storage {dataset_name}")
        os.system(f"rsync -hvrt --progress {local_storage}/* {permanent_storage}/")
        
        client.close()
        
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_storage', type=str,
        default='/tmp/patent-analysis/',
        help='Path to the raw data, permanent storage for the zip')
    parser.add_argument(
        '--permanent_storage', type=str,
        default='data/patents-view/',
        help='Path to the raw data, unzipped data stored locally in the job storage')

    args = parser.parse_args()
    for d, params in DATASETS_SPEC.items():
        prepare_datasets_cached(dataset_name=d, 
                                local_storage=args.local_storage, 
                                permanent_storage=args.permanent_storage,
                                params=params)
        
    os.system(f"rsync -hvrt --progress {args.permanent_storage}/*parquet {args.local_storage}/")
