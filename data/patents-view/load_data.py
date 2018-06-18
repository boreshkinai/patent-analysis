from keras.utils import get_file

CACHE_DIR = '.'
DATA_TIMESTAMP='20171226'
ROOT_URL = 'http://www.patentsview.org/data/' + DATA_TIMESTAMP + '/'

get_file('application.tsv.tsv.zip', origin=ROOT_URL+'application.tsv.zip', cache_dir=CACHE_DIR)
get_file('assignee.tsv.zip', origin=ROOT_URL+'assignee.tsv.zip', cache_dir=CACHE_DIR)
get_file('brf_sum_text.tsv.zip', origin=ROOT_URL+'brf_sum_text.tsv.zip', cache_dir=CACHE_DIR)
get_file('claim.tsv.zip', origin=ROOT_URL+'claim.tsv.zip', cache_dir=CACHE_DIR)
get_file('cpc_current.tsv.zip', origin=ROOT_URL+'cpc_current.tsv.zip', cache_dir=CACHE_DIR)
get_file('cpc_group.tsv.zip', origin=ROOT_URL+'cpc_group.tsv.zip', cache_dir=CACHE_DIR)
get_file('cpc_subgroup.tsv.zip', origin=ROOT_URL+'cpc_subgroup.tsv.zip', cache_dir=CACHE_DIR)
get_file('cpc_subsection.tsv.zip', origin=ROOT_URL+'cpc_subsection.tsv.zip', cache_dir=CACHE_DIR)
get_file('draw_desc_text.tsv.zip', origin=ROOT_URL+'draw_desc_text.tsv.zip', cache_dir=CACHE_DIR)
get_file('figures.tsv.zip', origin=ROOT_URL+'figures.tsv.zip', cache_dir=CACHE_DIR)
get_file('inventor.tsv.zip', origin=ROOT_URL+'inventor.tsv.zip', cache_dir=CACHE_DIR)
get_file('ipcr.tsv.zip', origin=ROOT_URL+'ipcr.tsv.zip', cache_dir=CACHE_DIR)
get_file('lawyer.tsv.zip', origin=ROOT_URL+'lawyer.tsv.zip', cache_dir=CACHE_DIR)
# Disambiguated location data, including latitude and longitude
get_file('location.tsv.zip', origin=ROOT_URL+'location.tsv.zip', cache_dir=CACHE_DIR)
get_file('location_assignee.tsv.zip', origin=ROOT_URL+'location_assignee.tsv.zip', cache_dir=CACHE_DIR)
get_file('location_inventor.tsv.zip', origin=ROOT_URL+'location_inventor.tsv.zip', cache_dir=CACHE_DIR)
get_file('non_inventor_applicant.tsv.zip', origin=ROOT_URL+'non_inventor_applicant.tsv.zip', cache_dir=CACHE_DIR)
get_file('otherreference.tsv.zip', origin=ROOT_URL+'otherreference.tsv.zip', cache_dir=CACHE_DIR)
get_file('patent.tsv.zip', origin=ROOT_URL+'patent.tsv.zip', cache_dir=CACHE_DIR)
get_file('patent_contractawardnumber.tsv.zip', origin=ROOT_URL+'patent_contractawardnumber.tsv.zip', cache_dir=CACHE_DIR)
get_file('patent_inventor.tsv.zip', origin=ROOT_URL+'patent_inventor.tsv.zip', cache_dir=CACHE_DIR)
get_file('patent_lawyer.tsv.zip', origin=ROOT_URL+'patent_lawyer.tsv.zip', cache_dir=CACHE_DIR)
get_file('rel_app_text.tsv.zip', origin=ROOT_URL+'rel_app_text.tsv.zip', cache_dir=CACHE_DIR)
get_file('usapplicationcitation.tsv.zip', origin=ROOT_URL+'usapplicationcitation.tsv.zip', cache_dir=CACHE_DIR)
get_file('uspatentcitation.tsv.zip', origin=ROOT_URL+'uspatentcitation.tsv.zip', cache_dir=CACHE_DIR)
get_file('uspc.tsv.zip', origin=ROOT_URL+'uspc.tsv.zip', cache_dir=CACHE_DIR)
# WIPO technology codes coded as 0,1,...
get_file('wipo.tsv.zip', origin=ROOT_URL+'wipo.tsv.zip', cache_dir=CACHE_DIR)
# translation from WIPO technology codes to wipo technology names, e.g. IT
get_file('wipo_field.tsv.zip', origin=ROOT_URL+'wipo_field.tsv.zip', cache_dir=CACHE_DIR)
# get_file('', origin='', cache_dir=CACHE_DIR)
# get_file('', origin='', cache_dir=CACHE_DIR)
# get_file('', origin='', cache_dir=CACHE_DIR)
# get_file('', origin='', cache_dir=CACHE_DIR)
# get_file('', origin='', cache_dir=CACHE_DIR)

