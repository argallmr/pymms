from pymms.sdc import mrmms_sdc_api as api
from . import util
import xarray as xr


def check_spacecraft(sc):
    if sc not in ('mms1', 'mms2', 'mms3', 'mms4'):
        raise ValueError('{} is not a recongized SC ID. '
                         'Must be ("mms1", "mms2", "mms3", "mms4")'
                         .format(sc))

def check_mode(mode):
    modes = ('brst', 'fast', 'slow')
    if mode == 'srvy':
        mode = 'fast'
    if mode not in modes:
        raise ValueError('Mode "{0}" is not in {1}'.format(mode, modes))

    return mode


def check_instr(instr):
    instrs = ('edp', 'scpot')
    if instr not in instrs:
        raise ValueError('Instr "{0}" is not in {1}'.format(instr, instrs))


def check_level(level, instr='edp'):
    levels = ('l1a', 'l2pre')
    if level not in levels:
        raise ValueError('Level "{0}" is not in {1}'.format(level, levels))


def check_coords(coords, instr='edp', level='l2'):
    coord_systems = ('gse')
    if coords not in coord_systems:
        raise ValueError(('Coordinate systems "{0}" is not in {1}'
                          .format(coords, coord_systems)
                          )
                         )


def load_data(sc, mode, start_date, end_date,
              level='l2', coords='gse'):
    """
    Load EDP spacecraft potential data.
    
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
    scpot : `metaarray.metaarray`
        Spacecraft potential.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    
    # File and variable name parameters
    instr = 'edp'
    optdesc = 'dce'
    t_vname = '_'.join((sc, instr, 'epoch', mode, level))
    e_vname = '_'.join((sc, instr, optdesc, coords, mode, level))
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, instr, mode, level,
                            optdesc=optdesc,
                            start_date=start_date,
                            end_date=end_date)
    edp_files = sdc.download_files()
    edp_files = api.sort_files(edp_files)[0]
    
    # Concatenate data along the records (time) dimension, which
    # should be equivalent to the DEPEND_0 variable name of the
    # magnetic field variable.
    edp_data = []
    for file in edp_files:
        edp_data.append(util.cdf_to_ds(file, e_vname))
    edp_data = xr.concat(edp_data, dim=edp_data[0][e_vname].dims[0])
    edp_data = edp_data.rename({t_vname: 'time',
                                e_vname: 'E'})
    edp_data = edp_data.sel(time=slice(start_date, end_date))
    edp_data.attrs['files'] = edp_files

    return edp_data


def load_scpot(sc, mode, start_date, end_date,
               level='l2'):
    """
    Load EDP spacecraft potential data.
    
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
    scpot : `metaarray.metaarray`
        Spacecraft potential.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    
    # File and variable name parameters
    instr = 'edp'
    optdesc = 'scpot'
    t_vname = '_'.join((sc, instr, 'epoch', mode, level))
    scpot_vname = '_'.join((sc, instr, optdesc, mode, level))
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, instr, mode, level,
                            optdesc=optdesc,
                            start_date=start_date,
                            end_date=end_date)
    edp_files = sdc.download_files()
    edp_files = api.sort_files(edp_files)[0]
    
    # Concatenate data along the records (time) dimension, which
    # should be equivalent to the DEPEND_0 variable name of the
    # magnetic field variable.
    edp_data = []
    for file in edp_files:
        edp_data.append(util.cdf_to_ds(file, scpot_vname))
    edp_data = xr.concat(edp_data, dim=edp_data[0][scpot_vname].dims[0])
    edp_data = edp_data.rename({t_vname: 'time',
                                scpot_vname: 'Vsc'})
    edp_data = edp_data.sel(time=slice(start_date, end_date))

    return edp_data

if __name__ == '__main__':
    pass
    