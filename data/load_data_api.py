import urllib.request
import numpy as np
import json
import math
import pathlib
import gzip
import os
import pandas as pd
from tqdm.auto import tqdm
from typing import List

START_YEAR = 1976
START_MONTH = 1
END_YEAR = 2020
END_MONTH = 12

NUM_PATENT_PER_PAGE = 100

BASE_URL = 'http://www.patentsview.org/api/patents/query?'

Q_PART = 'q={"_and":[{"_gte":{"patent_date":"2016-01-01"}},{"_lte":{"patent_date":"2016-02-01"}}]}'

F_PART = '&f=["patent_number","patent_title","patent_date","patent_kind","patent_type",\
"app_number","app_date","app_type",\
"inventor_id","inventor_first_name","inventor_last_name","inventor_location_id","inventor_city","inventor_state","inventor_country","inventor_latitude","inventor_longitude","inventor_sequence",\
"assignee_id","assignee_organization","assignee_first_name","assignee_last_name","assignee_location_id","assignee_city","assignee_state","assignee_country","assignee_latitude","assignee_longitude","assignee_sequence","assignee_type",\
"cited_patent_number","cited_patent_date","cited_patent_sequence",\
"citedby_patent_number","citedby_patent_date",\
"cpc_section_id","cpc_subsection_id","cpc_group_id","cpc_subgroup_id","cpc_category","cpc_sequence"]'

# SORT_PART = '&s=[{"patent_title":"asc"}]'
DEFAULT_FORMAT_PART = '&o={"page":1,"per_page":%d}' %(NUM_PATENT_PER_PAGE)

RESULTS_DIR = './datasets/api/'

# URL = BASE_URL + Q_PART + F_PART + FORMAT_PART


def get_period_start(year, month):
    start_month = month - 1
    if month == 1:
        start_month = 12
        start_year = year - 1
    else:
        start_year = year
    return start_year, start_month


def get_full_url(q_part, output_part):
    url = BASE_URL  + q_part + F_PART + output_part
    return url

def get_json_from_url(url):
    url_feed = urllib.request.urlopen(url=url)
    response_text = url_feed.read().decode('utf-8')
    url_feed.close()
    response_json = json.loads(response_text)
    return response_json, url_feed

def fetch_month_data(year, month):
    # Get the query part of the URL
    start_year, start_month = get_period_start(year, month)
    q_part = 'q={"_and":[{"_gte":{"patent_date":"%d-%d-01"}},{"_lt":{"patent_date":"%d-%d-01"}}]}' % (start_year, start_month, year, month)

    patents = []
    # Make first request to figure out the total number of patents
    url = get_full_url(q_part, DEFAULT_FORMAT_PART)
    response_json, url_feed = get_json_from_url(url=url)
    log_txt=''
    is_success = True
    if url_feed.status != 200:
        is_success = False
        log_txt = 'Url request in the year %d, month %d, page %d, failed with status %d, message and reason %s' %(year, month, 1, url_feed.status, url_feed.message, url_feed.reason)
        print(log_txt)
    else:
        total_patent_count = response_json['total_patent_count']
        if total_patent_count > 0:
            num_pages = math.floor(total_patent_count / NUM_PATENT_PER_PAGE) + 1
            for page in range(2, num_pages+2):
                if response_json['patents'] != None:
                    patents.extend(response_json['patents'])

                format_part = '&o={"page":%d,"per_page":%d}' %(page, NUM_PATENT_PER_PAGE)
                url = get_full_url(q_part, format_part)
                response_json, url_feed = get_json_from_url(url=url)
                if url_feed.status != 200:
                    is_success = False
                    log_txt = 'Url request in the year %d, month %d, page %d, failed with status %d, message and reason %s' % (
                    year, month, 1, url_feed.status, url_feed.message, url_feed.reason)
                    print(log_txt)
                    break

    return patents, is_success, log_txt


def get_month_file_name(logdir, year, month: str) -> str:
    return logdir + '%d-%d' %(year,month) + '.json'


def get_month_file_zipped_name(json_file_name: str) -> str:
    return json_file_name + '.gz'


def get_month_file_parquet_name(json_file_name: str) -> str:
    return ".".join(json_file_name.split(".")[:-1]) + '.parquet'


def save_month_data(json_file_name, patents, is_success, log_txt):
    log_file_name = json_file_name + '.log'
    gz_file = get_month_file_zipped_name(json_file_name)
    if is_success:
        with open(json_file_name, 'w') as fp:
            json.dump(patents, fp)

        if os.path.isfile(log_file_name):
            os.remove(log_file_name)
    else:
        with open(log_file_name, 'w') as fp:
            fp.write(log_txt)
            
            
def flatten_column(df: pd.DataFrame, column: str, index: str) -> pd.DataFrame:
    flat_list = []
    for row in tqdm(df.iterrows(), total=len(df)):
        if isinstance(row[1][column], list):
            x = pd.json_normalize(row[1][column])
            x.index = [row[1][index] for i in range(len(x))]
            flat_list.append(x)
        
    return pd.concat(flat_list)


def flatten_dataframe(df: pd.DataFrame, columns:List[str], index: str) -> pd.DataFrame:
    df_out = df.drop(columns=columns)
    df_out = df_out.set_index(index)
    for c in columns:
        df_c = flatten_column(df, column=c, index=index)
        df_out = df_out.join(df_c)
    return df_out


pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
for year in range(START_YEAR, END_YEAR+1):
    for month in range(START_MONTH, END_MONTH + 1):
        print('Processing year %d month %d' %(year, month))
        
        json_file_name = get_month_file_name(RESULTS_DIR, year, month)
        if not os.path.isfile(json_file_name):
            patents, is_success, log_txt = fetch_month_data(year, month)
            save_month_data(json_file_name, patents, is_success, log_txt)
            
        parquet_file_name = get_month_file_parquet_name(json_file_name)
        if not os.path.isfile(parquet_file_name):
            df = pd.read_json(json_file_name)
            if not df.empty:
                df_flat = flatten_dataframe(df, index="patent_number",
                                            columns=["inventors", "assignees", "cited_patents", 
                                                     "citedby_patents", "cpcs"])
                df_flat.to_parquet(parquet_file_name)
            
        


