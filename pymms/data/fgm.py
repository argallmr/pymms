from pymms.sdc import mrmms_sdc_api as api
from . import util
from metaarray import metaarray
import datetime as dt


def check_spacecraft(sc):
    if sc not in ('mms1', 'mms2', 'mms3', 'mms4'):
        raise ValueError('{} is not a recongized SC ID. '
                         'Must be ("mms1", "mms2", "mms3", "mms4")'
                         .format(sc))

def check_mode(mode, level='l2'):
    
    # Fast and slow mode data are combined into survey data
    if level == 'l2':
        if mode in ('fast', 'slow'):
            mode = 'srvy'
    
    modes = ('brst', 'srvy', 'fast', 'slow')
    if mode not in modes:
        raise ValueError('Mode "{0}" is not in {1}'.format(mode, modes))

    return mode


def check_instr(instr):
    instrs = ('fgm', 'afg', 'dfg')
    if instr not in instrs:
        raise ValueError('Instr "{0}" is not in {1}'.format(instr, instrs))


def check_level(level, instr='fgm'):
    if instr == 'fgm':
        levels = ('l2',)
    else:
        levels = ('l1a', 'l2pre')

    if level not in levels:
        raise ValueError('Level "{0}" is not in {1}'.format(level, levels))


def check_coords(coords, instr='fgm', level='l2'):
    coord_systems = ('dbcs', 'gse')
    if coords not in coord_systems:
        raise ValueError(('Coordinate systems "{0}" is not in {1}'
                          .format(coords, coord_systems)
                          )
                         )


def load_data_xr(sc, mode, start_date, end_date,
                 instr='fgm', level='l2', coords='gse'):
    """
    Load FPI distribution function data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    
    Returns
    -------
    dist : `metaarray.metaarray`
        Particle distribution function.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    check_level(level, instr=instr)
    check_coords(coords)
    
    # File and variable name parameters
    b_vname = '_'.join((sc, instr, 'b', coords, mode, level))
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, instr, mode, level,
                            start_date=start_date,
                            end_date=end_date)
    fgm_files = sdc.download_files()
    
    # Read the data from files
    fgm_ds = util.cdf_to_ds(fgm_files, b_vname)
    
    # Read into Pandas DataFrame
#    fgm_df = util.cdf_to_df(fgm_files, b_vname)
#    util.rename_df_cols(fgm_df, b_vname, ('Bx', 'By', 'Bz', '|B|'))
    
    return fgm_ds


def load_data(sc, mode, start_date, end_date,
              instr='fgm', level='l2', coords='gse'):
    """
    Load FPI distribution function data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    
    Returns
    -------
    dist : `metaarray.metaarray`
        Particle distribution function.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    check_level(level, instr=instr)
    check_coords(coords)
    
    # File and variable name parameters
    b_vname = '_'.join((sc, instr, 'b', coords, mode, level))
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, instr, mode, level,
                            start_date=start_date,
                            end_date=end_date)
    fgm_files = sdc.download_files()
    
    # Read the data from files
    fgm_df = util.cdf_to_df(fgm_files, b_vname)
    util.rename_df_cols(fgm_df, b_vname, ('Bx', 'By', 'Bz', '|B|'))
#    bfield = metaarray.from_cdflib(fgm_files, b_vname,
#                                   start_date=start_date,
#                                   end_date=end_date)
    
    return fgm_df


if __name__ == '__main__':
    pass
    