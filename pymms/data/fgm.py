from pymms.sdc import mrmms_sdc_api as api
from . import util
import datetime as dt
import xarray as xr


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


def load_data(sc, mode, start_date, end_date,
              instr='fgm', level='l2', coords='gse',
              pd=False):
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
    t_vname = 'Epoch'
    b_vname = '_'.join((sc, instr, 'b', coords, mode, level))
    b_labl_vname = '_'.join(('label', 'b', coords))
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, instr, mode, level,
                            start_date=start_date,
                            end_date=end_date)
    fgm_files = sdc.download_files()
    fgm_files = api.sort_files(fgm_files)[0]
    
    # Read the data from files
    if pd:
        fgm_data = util.cdf_to_df(fgm_files, b_vname)
        util.rename_df_cols(fgm_data, b_vname, ('Bx', 'By', 'Bz', '|B|'))
    else:
        # Concatenate data along the records (time) dimension, which
        # should be equivalent to the DEPEND_0 variable name of the
        # magnetic field variable.
        fgm_data = []
        for file in fgm_files:
            fgm_data.append(util.cdf_to_ds(file, b_vname))
        fgm_data = xr.concat(fgm_data, dim=fgm_data[0][b_vname].dims[0])
        
        fgm_data = fgm_data.rename({t_vname: 'time',
                                    b_vname: 'B',
                                    b_labl_vname: 'B_index'})
        fgm_data = fgm_data.assign_coords(B_index=['Bx', 'By', 'Bz', '|B|'])
        fgm_data = fgm_data.sel(time=slice(start_date, end_date))
        
    return fgm_data


if __name__ == '__main__':
    pass
    