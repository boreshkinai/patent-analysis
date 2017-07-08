import urllib.request
import numpy as np
import json
import math
import pathlib
import gzip
import os
import pandas as pd

START_YEAR = 1976
START_MONTH = 1
END_YEAR = 2017
END_MONTH = 12

NUM_PATENT_PER_PAGE = 100

BASE_URL = 'http://www.patentsview.org/api/patents/query?'
Q_PART = 'q={"_and":[{"_gte":{"patent_date":"2016-01-01"}},{"_lte":{"patent_date":"2016-02-01"}}]}'
F_PART = '&f=["patent_number","patent_title","patent_abstract","patent_date","patent_year","patent_kind","patent_type","patent_average_processing_time","patent_processing_time","app_number","app_date","app_type","govint_org_id","govint_org_name","govint_org_level_one","govint_org_level_two","govint_org_level_three","govint_contract_award_number","govint_raw_statement","inventor_id","inventor_first_name","inventor_last_name","rawinventor_first_name","rawinventor_last_name","inventor_first_seen_date","inventor_last_seen_date","inventor_location_id","inventor_city","inventor_state","inventor_country","inventor_latitude","inventor_longitude","inventor_lastknown_location_id","inventor_lastknown_city","inventor_lastknown_state","inventor_lastknown_country","inventor_lastknown_latitude","inventor_lastknown_longitude","inventor_sequence","assignee_city","assignee_country","assignee_first_name","assignee_first_seen_date","assignee_id","assignee_lastknown_city","assignee_lastknown_country","assignee_lastknown_latitude","assignee_lastknown_location_id","assignee_lastknown_longitude","assignee_lastknown_state","assignee_last_name","assignee_last_seen_date","assignee_latitude","assignee_location_id","assignee_longitude","assignee_organization","assignee_sequence","assignee_state","assignee_type","appcit_category","appcit_date","appcit_kind","appcit_app_number","appcit_sequence","cited_patent_number","cited_patent_title","cited_patent_date","cited_patent_kind","cited_patent_category","cited_patent_sequence","citedby_patent_number","citedby_patent_title","citedby_patent_date","citedby_patent_kind","citedby_patent_category","cpc_section_id","cpc_subsection_id","cpc_group_id","cpc_subgroup_id","cpc_category","cpc_subsection_title","cpc_group_title","cpc_subgroup_title","cpc_first_seen_date","cpc_last_seen_date","cpc_sequence","ipc_section","ipc_class","ipc_subclass","ipc_main_group","ipc_subgroup","ipc_first_seen_date","ipc_last_seen_date","ipc_sequence","ipc_symbol_position","ipc_action_date","ipc_classification_data_source","ipc_classification_value","ipc_version_indicator","nber_category_id","nber_subcategory_id","nber_category_title","nber_subcategory_title","nber_first_seen_date","nber_last_seen_date","uspc_mainclass_id","uspc_subclass_id","uspc_mainclass_title","uspc_subclass_title","uspc_first_seen_date","uspc_last_seen_date","uspc_sequence","wipo_field_id","wipo_sector_title","wipo_field_title","wipo_sequence"]'
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


def get_month_file_name(logdir, year, month):
    return logdir + '%d-%d' %(year,month) + '.json'

def get_month_file_zipped_name(json_file_name):
    return json_file_name + '.gz'


def save_month_data(json_file_name, patents, is_success, log_txt):
    log_file_name = json_file_name + '.log'
    gz_file = get_month_file_zipped_name(json_file_name)
    if is_success:
        with open(json_file_name, 'w') as fp:
            json.dump(patents, fp)
        with open(json_file_name, 'rb') as fp, gzip.open(gz_file, 'wb') as fzip:
            fzip.writelines(fp)
        os.remove(json_file_name)

        if os.path.isfile(log_file_name):
            os.remove(log_file_name)
    else:
        with open(log_file_name, 'w') as fp:
            fp.write(log_txt)


pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
for year in range(START_YEAR, END_YEAR+1):
    for month in range(START_MONTH, END_MONTH + 1):
        print('Processing year %d month %d' %(year, month))

        json_file_name = get_month_file_name(RESULTS_DIR, year, month)

        if not os.path.isfile( get_month_file_zipped_name(json_file_name) ):
            patents, is_success, log_txt = fetch_month_data(year, month)

            save_month_data(json_file_name, patents, is_success, log_txt)


