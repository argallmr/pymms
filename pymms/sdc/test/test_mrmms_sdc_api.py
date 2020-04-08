import pytest
import datetime
from pymms import mrmms_sdc_api as sdc

@pytest.fixture
def api():
    start_date = datetime.datetime(2017, 11, 24, 0, 0, 0)
    end_date = datetime.datetime(2017, 11, 24, 23, 59, 59)
    api = sdc.MrMMS_SDC_API('mms1', 'fgm', 'srvy', 'l2',
                            start_date=start_date,
                            end_date=end_date)
    return api


def test_init_attributes(api):
    """
    Check that the input values set the correct attributes.
    """
    assert api.sc == 'mms1'
    assert api.instr == 'fgm'
    assert api.level == 'l2'
    assert api.mode == 'srvy'
    assert api.start_date == datetime.datetime(2017, 11, 24, 0, 0, 0)
    assert api.end_date == datetime.datetime(2017, 11, 24, 23, 59, 59)

def test_science_download_url(api):
    """
    Check that the correct URL is created for downloading mms1_fgm_srvy_l2
    science data from the public side of the SDC.
    """
    assert api.url() == 'https://lasp.colorado.edu/' \
                        'mms/sdc/public/files/api/v1/download/science?' \
                        'sc_id=mms1&' \
                        'instrument_id=fgm&' \
                        'data_rate_mode=srvy&' \
                        'data_level=l2&' \
                        'start_date=2017-11-24&' \
                        'end_date=2017-11-25&'