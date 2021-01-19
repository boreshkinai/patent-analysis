from typing import List, Dict
import pathlib
import os
import pandas as pd
import numpy as np
import argparse
import sys
import csv

# Suppress TF warnings
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
from keras.utils import get_file

import dask.dataframe as dask
from dask.distributed import Client


DATA_TIMESTAMP = '20200929'
ROOT_URL = f"https://s3.amazonaws.com/data.patentsview.org/{DATA_TIMESTAMP}/download/"



DATASETS_SPEC = {
    'patent': {
        'index': ['id'],
        'usecols': ['id', 'type', 'number', 'country', 'date', 'title', 'kind', 'num_claims'],
        'parse_dates': ['date'],
        'parse_numeric': {'num_claims': np.int16},
        'drop_columns': [],
    },
    'uspatentcitation': {
        'index': ['patent_id'],
        'usecols': ['uuid', 'patent_id', 'citation_id', 'date', 'name', 'kind', 'country', 'category', 'sequence'],
        'parse_dates': ['date'],
        'parse_numeric': {'sequence': np.int16},
        'drop_columns': [],
    },
    'location': {
        'index': ['id'],
        'usecols': ['id', 'city', 'state', 'country', 'latitude', 'longitude', 'county'],
        'parse_dates': [],
        'parse_numeric': {'latitude': np.float32, 'longitude': np.float32},
        'drop_columns': [],
    },
    'assignee': {
        'index': ['id'],
        'usecols': ['id', 'type', 'name_first', 'name_last', 'organization'],
        'parse_dates': [],
        'parse_numeric': {},
        'drop_columns': [],
    },
    'patent_assignee': {
        'index': ['patent_id'],
        'usecols': ['patent_id', 'assignee_id', 'location_id'],
        'renamecols': {'location_id': 'location_id_assignee'},
        'parse_dates': [],
        'parse_numeric': {},
        'drop_columns': [],
    },
    'patent_inventor': {
        'index': ['patent_id'],
        'usecols': ['patent_id', 'inventor_id', 'location_id'],
        'renamecols': {'location_id': 'location_id_inventor'},
        'parse_dates': [],
        'parse_numeric': {},
        'drop_columns': [],
    },
    'cpc_current': {
        'index': ['patent_id'],
        'usecols': ['patent_id', 'section_id', 'subsection_id', 'group_id', 'subgroup_id', 'category', 'sequence'],
        'parse_dates': [],
        'parse_numeric': {},
        'drop_columns': [],
    },
    
}
        

def prepare_datasets_cached(dataset_name: str, 
                            local_storage: str, permanent_storage: str,
                            params: Dict) -> None:
    
    pathlib.Path(local_storage).mkdir(parents=True, exist_ok=True)
    pathlib.Path(permanent_storage).mkdir(parents=True, exist_ok=True)
    file_name_zip = f"{dataset_name}.tsv.zip"
    file_name_tsv = f"{dataset_name}.tsv"
    dir_name_parquet = f"{dataset_name}.parquet"
    print(f"Preparing dataset {dataset_name}")
    if not os.path.exists(os.path.join(permanent_storage, dir_name_parquet)):
        print(f"Downloading {file_name_zip} locally in {local_storage}")
        get_file(file_name_zip, origin=os.path.join(ROOT_URL, file_name_zip), 
                 cache_dir=local_storage, cache_subdir='')
        
        print(f"Unzipping {file_name_zip} locally in {local_storage}")
        cmd = f"unzip -u {os.path.join(local_storage, file_name_zip)} -d {local_storage}/"
        os.system(cmd)
        
        print("Create dask local cluster")
        ncpus = len(os.sched_getaffinity(0))
        client = Client(n_workers=ncpus, threads_per_worker=1, memory_limit='6GB', dashboard_address='6006')
        
        if len(params['usecols']) == 0:
            print("Empty usecols, no dataset produced.")
            df = pd.read_csv(os.path.join(local_storage, file_name_tsv), sep = '\t', error_bad_lines = False, 
                             dtype = str, engine = 'python', nrows = 10)
            print("Please specify columns to use in the dataset in usecols based on the dataset contents:")
            print()
            print(df.columns)
            return
    
        print(f"Creating parquet {dir_name_parquet}")
        ddf = dask.read_csv(os.path.join(local_storage, file_name_tsv),
                            sep='\t', error_bad_lines=False, 
                            dtype=str, 
                            usecols=params['usecols'],
                            engine='python'
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
        if 'renamecols' in params.keys():
            ddf = ddf.rename(columns=params['renamecols'])
        
        ddf.repartition(npartitions=ddf.npartitions)
            
        ddf.to_parquet(os.path.join(local_storage, f"{dir_name_parquet}"), engine='pyarrow', schema="infer")

        print(f"Transferring to permanent storage {dataset_name}")
        os.system(f"rsync -hvrt --progress {local_storage}/{dataset_name}* {permanent_storage}/")
        
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
