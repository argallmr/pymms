import pytest
import os
import datetime
from pymms.sdc import mrmms_sdc_api as sdc

sc = 'mms1'
instr = 'fgm'
mode = 'srvy'
level = 'l2'
start_date = datetime.datetime(2017, 11, 24, 0, 0, 0)
end_date = datetime.datetime(2017, 11, 24, 23, 59, 59)
api = sdc.MrMMS_SDC_API(sc, instr, mode, level,
                        start_date=start_date, end_date=end_date)

# Sample file names
f_burst = '_'.join((sc, instr, 'brst', level,
                    start_date.strftime('%Y%m%d%H%M%S'),
                    'v1.0.0')
                   ) + '.cdf'
f_survey = '_'.join((sc, instr, 'srvy', level,
                     start_date.strftime('%Y%m%d'),
                     'v1.0.0')
                     ) + '.cdf'
f_selections = '_'.join(('gls_selections', 'mp-dl-unh',
                         start_date.strftime('%Y-%m-%d-%H-%M-%S'))
                         ) + '.sav'


def test_init_attributes():
    """
    Check that the input values set the correct attributes.
    """
    assert api.sc == 'mms1'
    assert api.instr == 'fgm'
    assert api.level == 'l2'
    assert api.mode == 'srvy'
    assert api.start_date == datetime.datetime(2017, 11, 24, 0, 0, 0)
    assert api.end_date == datetime.datetime(2017, 11, 24, 23, 59, 59)


def test_science_download_url():
    """
    Check that the correct URL is created for downloading mms1_fgm_srvy_l2
    science data from the public side of the SDC.
    """
    assert api.url() == ('https://lasp.colorado.edu/'
                         'mms/sdc/public/files/api/v1/download/science?'
                         'sc_id=mms1&'
                         'instrument_id=fgm&'
                         'data_rate_mode=srvy&'
                         'data_level=l2&'
                         'start_date=2017-11-24&'
                         'end_date=2017-11-25&'
                         )


def test_construct_file_names():

    # Test parameters
    burst_tstart = start_date.strftime('%Y%m%d%H%M%S')
    survey_tstart = start_date.strftime('%Y%m%d')
    science_version = '1.0.0'
    gls_datatype = 'gls_selections'
    gls_type = 'mp-dl-unh'
    gls_tstart = start_date.strftime('%Y-%m-%d-%H-%M-%S')

    # Create the file names
    f_test_burst = sdc.construct_file_names(sc, instr, 'brst', level,
                                            tstart=burst_tstart,
                                            version=science_version)[0]

    f_test_survey = sdc.construct_file_names(sc, instr, 'srvy', level,
                                             tstart=survey_tstart,
                                             version=science_version)[0]

    f_test_gls = sdc.construct_file_names(data_type=gls_datatype,
                                          gls_type=gls_type,
                                          tstart=gls_tstart)[0]

    # Test that the file names were created correctly
    assert f_test_burst == f_burst
    assert f_test_survey == f_survey
    assert f_test_gls == f_selections


def test_construct_paths():

    # Test parameters
    survey_tstart = start_date.strftime('%Y%m%d')
    gls_datatype = 'gls_selections'

    # Create the paths
    p_science = sdc.construct_path(sc, instr, mode, level,
                                   tstart=survey_tstart)[0]

    p_gls = sdc.construct_path(data_type=gls_datatype)[0]

    # Test
    assert p_science == os.path.sep.join((sc, instr, mode, level,
                                          '{0:04d}'.format(start_date.year),
                                          '{0:02d}'.format(start_date.month)))
    assert p_gls == os.path.sep.join(('sitl', gls_datatype))


def test_file_start_time():

    # Extract file start times from file names
    t_burst = sdc.file_start_time(f_burst)
    t_survey = sdc.file_start_time(f_survey)
    t_selections = sdc.file_start_time(f_selections)

    # Test
    assert t_burst == start_date
    assert t_survey == start_date
    assert t_selections == start_date


def test_filename2path():
    p_burst = sdc.filename2path(f_burst)
    p_survey = sdc.filename2path(f_survey)
    p_selections = sdc.filename2path(f_selections)

    assert p_burst == os.path.sep.join((sc, instr, 'brst', level,
                                        '{0:04d}'.format(start_date.year),
                                        '{0:02d}'.format(start_date.month),
                                        '{0:02d}'.format(start_date.day),
                                        f_burst)
                                       )
    assert p_survey == os.path.sep.join((sc, instr, 'srvy', level,
                                         '{0:04d}'.format(start_date.year),
                                         '{0:02d}'.format(start_date.month),
                                         f_survey)
                                        )
    assert p_selections == os.path.sep.join(('gls_selections', f_selections))

def test_filter_time():
    burst_files = ['mms1_fgm_brst_l2_20171124231933_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124020733_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124020503_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124020233_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124020003_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124015733_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124015503_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124015233_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124015003_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124014733_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124014503_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124014233_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124014003_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124013733_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124013503_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124013233_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124013003_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124012733_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124012503_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124012233_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124012003_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124011733_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124011503_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124011233_v5.113.0.cdf',
                   'mms1_fgm_brst_l2_20171124011003_v5.113.0.cdf']

    survey_files = ['mms/data/mms1/fgm/srvy/l2/2017/12/mms1_fgm_srvy_l2_20171204_v5.116.0.cdf',
                    'mms/data/mms1/fgm/srvy/l2/2017/12/mms1_fgm_srvy_l2_20171203_v5.114.0.cdf',
                    'mms/data/mms1/fgm/srvy/l2/2017/12/mms1_fgm_srvy_l2_20171202_v5.114.0.cdf',
                    'mms/data/mms1/fgm/srvy/l2/2017/12/mms1_fgm_srvy_l2_20171201_v5.115.0.cdf',
                    'mms/data/mms1/fgm/srvy/l2/2017/11/mms1_fgm_srvy_l2_20171130_v5.114.0.cdf',
                    'mms/data/mms1/fgm/srvy/l2/2017/11/mms1_fgm_srvy_l2_20171129_v5.114.0.cdf',
                    'mms/data/mms1/fgm/srvy/l2/2017/11/mms1_fgm_srvy_l2_20171128_v5.115.0.cdf',
                    'mms/data/mms1/fgm/srvy/l2/2017/11/mms1_fgm_srvy_l2_20171127_v5.113.0.cdf',
                    'mms/data/mms1/fgm/srvy/l2/2017/11/mms1_fgm_srvy_l2_20171126_v5.113.0.cdf',
                    'mms/data/mms1/fgm/srvy/l2/2017/11/mms1_fgm_srvy_l2_20171125_v5.114.0.cdf',
                    'mms/data/mms1/fgm/srvy/l2/2017/11/mms1_fgm_srvy_l2_20171124_v5.113.0.cdf']

    burst_filtered = sdc.filter_time(burst_files,
                                     datetime.datetime(2017, 11, 24, 1, 15, 0),
                                     datetime.datetime(2017, 11, 24, 1, 50, 0))

    survey_filtered = sdc.filter_time(survey_files,
                                      datetime.datetime(2017, 11, 25, 1, 15, 0),
                                      datetime.datetime(2017, 12, 2, 22, 8, 16))

    assert burst_filtered == burst_files[-2:-17:-1]
    assert survey_filtered == survey_files[-2:-10:-1]


def test_filter_version():
    files = ['mms1_fgm_brst_l2_20171124231933_v3.84.0.cdf',
             'mms1_fgm_brst_l2_20171124231933_v3.110.0.cdf',
             'mms1_fgm_brst_l2_20171124231933_v4.1.0.cdf',
             'mms1_fgm_brst_l2_20171124231933_v4.1.0.cdf',
             'mms1_fgm_brst_l2_20171124231933_v4.53.0.cdf',
             'mms1_fgm_brst_l2_20171124231933_v4.110.0.cdf',
             'mms1_fgm_brst_l2_20171124231933_v4.110.1.cdf',
             'mms1_fgm_brst_l2_20171124231933_v4.110.2.cdf',
             'mms1_fgm_brst_l2_20171124231933_v4.113.0.cdf',
             'mms1_fgm_brst_l2_20171124231933_v5.12.0.cdf',
             'mms1_fgm_brst_l2_20171124231933_v5.63.0.cdf',
             'mms1_fgm_brst_l2_20171124231933_v5.102.0.cdf',
             'mms1_fgm_brst_l2_20171124231933_v5.113.0.cdf',
             'mms1_fgm_brst_l2_20171124231933_v5.113.0.cdf',
             'mms1_fgm_brst_l2_20171124231933_v5.113.1.cdf',
             'mms1_fgm_brst_l2_20171124231933_v5.113.2.cdf']

    assert sdc.filter_version(files, latest=True)[0] == files[-1]
    assert sdc.filter_version(files, min_version='4.110.1') == files[6:]


def test_parse_file_name():
    f_parse_burst = sdc.parse_file_name(f_burst)
    f_parse_survey = sdc.parse_file_name(f_survey)
    f_parse_selections = sdc.parse_file_name(f_selections)

    assert f_parse_burst == (sc, instr, 'brst', level, '',
                             start_date.strftime('%Y%m%d%H%M%S'),
                             '1.0.0')
    assert f_parse_survey == (sc, instr, 'srvy', level, '',
                              start_date.strftime('%Y%m%d'),
                              '1.0.0')
    assert f_parse_selections == ('gls_selections', 'mp-dl-unh',
                                  start_date.strftime('%Y-%m-%d-%H-%M-%S'))


def test_parse_time():
    t_burst = '20171124231933'
    t_survey = '20171124'
    t_selections = '2017-11-24-23-19-33'

    assert sdc.parse_time(t_burst)[0] == ('2017', '11', '24',
                                          '23', '19', '33')
    assert sdc.parse_time(t_survey)[0] == ('2017', '11', '24',
                                           '00', '00', '00')
    assert sdc.parse_time(t_selections)[0] == ('2017', '11', '24',
                                               '23', '19', '33')
    assert (sdc.parse_time([t_burst, t_survey, t_selections])
            == [('2017', '11', '24', '23', '19', '33'),
                ('2017', '11', '24', '00', '00', '00'),
                ('2017', '11', '24', '23', '19', '33')]
            )


def test_sort_files():
    fgm_brst_files = ['mms/data/mms1/fgm/srvy/l2/2017/11/mms1_fgm_srvy_l2_20171128_v5.115.0.cdf',
                      'mms/data/mms1/fgm/srvy/l2/2017/11/mms1_fgm_srvy_l2_20171127_v5.113.0.cdf',
                      'mms/data/mms1/fgm/srvy/l2/2017/11/mms1_fgm_srvy_l2_20171126_v5.113.0.cdf',
                      'mms/data/mms1/fgm/srvy/l2/2017/11/mms1_fgm_srvy_l2_20171125_v5.114.0.cdf',
                      'mms/data/mms1/fgm/srvy/l2/2017/11/mms1_fgm_srvy_l2_20171124_v5.113.0.cdf']

    fgm_srvy_files = ['mms1_fgm_brst_l2_20171124231933_v5.113.0.cdf',
                      'mms1_fgm_brst_l2_20171124020733_v5.113.0.cdf',
                      'mms1_fgm_brst_l2_20171124020503_v5.113.0.cdf',
                      'mms1_fgm_brst_l2_20171124020233_v5.113.0.cdf',
                      'mms1_fgm_brst_l2_20171124020003_v5.113.0.cdf']

    edp_files = ['mms1_edp_brst_l2_dce_20171124014733_v3.0.0.cdf',
                 'mms1_edp_brst_l2_dce_20171124014503_v3.0.0.cdf',
                 'mms1_edp_brst_l2_dce_20171124014233_v3.0.0.cdf',
                 'mms1_edp_brst_l2_dce_20171124014003_v3.0.0.cdf',
                 'mms1_edp_brst_l2_dce_20171124013733_v3.0.0.cdf',
                 'mms1_edp_brst_l2_dce_20171124013503_v3.0.0.cdf',
                 'mms1_edp_brst_l2_dce_20171124013233_v3.0.0.cdf',
                 'mms2_edp_brst_l2_dce_20171124012733_v3.0.0.cdf',
                 'mms2_edp_brst_l2_dce_20171124012503_v3.0.0.cdf',
                 'mms2_edp_brst_l2_dce_20171124012233_v3.0.0.cdf',
                 'mms2_edp_brst_l2_dce_20171124012003_v3.0.0.cdf',
                 'mms2_edp_brst_l2_dce_20171124011733_v3.0.0.cdf',
                 'mms2_edp_fast_l2_dce_20171125_v3.0.0.cdf',
                 'mms2_edp_fast_l2_dce_20171124_v3.0.0.cdf']

    assert sdc.sort_files(fgm_brst_files)[0] == fgm_brst_files[::-1]
    assert sdc.sort_files(fgm_srvy_files)[0] == fgm_srvy_files[::-1]

    edp_sorted = sdc.sort_files(edp_files)
    for file_type in edp_sorted:
        ref_parts = file_type[0].split('_')
        try:
            tref = datetime.datetime.strptime(ref_parts[5], '%Y%m%d%H%M%S')
        except ValueError:
            tref = datetime.datetime.strptime(ref_parts[5], '%Y%m%d')


        for file in file_type:
            fparts = file.split('_')
            try:
                ftime = datetime.datetime.strptime(fparts[5], '%Y%m%d%H%M%S')
            except ValueError:
                ftime = datetime.datetime.strptime(fparts[5], '%Y%m%d')

            assert ((fparts[0:5] == ref_parts[0:5]) and
                    (tref <= ftime)
                    )
