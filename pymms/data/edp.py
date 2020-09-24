from pymms.sdc import mrmms_sdc_api as api
from . import util
from metaarray import metaarray


def check_spacecraft(sc):
    if sc not in ('mms1', 'mms2', 'mms3', 'mms4'):
        raise ValueError('{} is not a recongized SC ID. '
                         'Must be ("mms1", "mms2", "mms3", "mms4")'
                         .format(sc))

def check_mode(mode):
    modes = ('brst', 'fast', 'slow')
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
    epoch_vname = '_'.join((sc, instr, 'epoch', mode, level))
    scpot_vname = '_'.join((sc, instr, optdesc, mode, level))
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, instr, mode, level,
                            optdesc=optdesc,
                            start_date=start_date,
                            end_date=end_date)
    edp_files = sdc.download_files()
    
    # Read the data from files
    scpot_df = util.read_cdf_vars(edp_files, scpot_vname,
                                  epoch=epoch_vname)
#    scpot = metaarray.from_cdflib(edp_files, scpot_vname,
#                                  start_date=start_date,
#                                  end_date=end_date)
    scpot_df.rename(columns={scpot_vname: 'V_sc'}, inplace=True)

    return scpot_df

if __name__ == '__main__':
    pass
    