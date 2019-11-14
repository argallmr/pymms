import pytest
import datetime
from . import mrmms_sdc_api as api

def test_init_attributes():
	start_date = datetime.datetime(2017, 11, 24, 0, 0, 0)
	end_date = datetime.datetime(2017, 11, 24, 23, 59, 59)
	sdc = api.MrMMS_SDC_API('mms1', 'fgm', 'l2', 'srvy',
	                        start_date=start_date,
	                        end_date=end_date)
	assert sdc.sc == 'mms1'
	assert sdc.instr == 'fgm'
	assert sdc.level == 'l2'
	assert sdc.mode == 'srvy'

def test_science_download_url():
	sdc = api.MrMMS_SDC_API('mms1', 'fgm', 'l2', 'srvy')
	assert sdc.url() == 'https://'