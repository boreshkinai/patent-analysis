from tensorflow.keras.utils import get_file
from typing import List
import pathlib
import os
import pandas as pd
import argparse


DATA_TIMESTAMP = '20171226'
ROOT_URL = 'http://www.patentsview.org/data/' + DATA_TIMESTAMP + '/'

DATASETS = ['patent.tsv.zip',
            ]
DATASETS_EXCLUDE_COLUMNS = {'patent.tsv.zip': ['abstract', 'title'],
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
    for d in DATASETS:
        download_file_cached(file_name=d,
                             drop_columns=DATASETS_EXCLUDE_COLUMNS[d],
                             permanent_storage=args.permanent_storage,
                             local_storage=args.local_storage)
