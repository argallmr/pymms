from pymms.data import util


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

def rename(data, sc, mode, level, optdesc):
    '''
    Rename standard variables names to something more memorable.
    
    Parameters
    ----------
    data : `xarray.Dataset`
        Data to be renamed
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    level : str
        Data quality level ('l1a', 'l2pre', 'l2')
    optdesc : str
        Optional descriptor. Options are: ('dce', 'scpot')
    
    Returns
    -------
    data : `xarray.Dataset`
        Dataset with variables renamed
    '''
    
    # Variable name parameters
    instr = 'edp'
    t_vname = '_'.join((sc, instr, 'epoch', mode, level))

    if optdesc == 'dce':
        e_dsl_vname = '_'.join((sc, instr, optdesc, 'dsl', mode, level))
        e_gse_vname = '_'.join((sc, instr, optdesc, 'gse', mode, level))
        new_names = {t_vname: 'time',
                     e_dsl_vname: 'E_DSL',
                     e_gse_vname: 'E_GSE'}
    
    elif optdesc == 'scpot':
        scpot_vname = '_'.join((sc, instr, optdesc, mode, level))
        new_names = {t_vname: 'time',
                     scpot_vname: 'Vsc'}
    else:
        raise ValueError('Optional descriptor {0} not in (dce, scpot).'
                         .format(optdesc))

    # Change names
    return data.rename(new_names)


def load_data(sc='mms1', mode='fast', level='l2', optdesc='dce',
              start_date=None, end_date=None, rename_vars=True,
              **kwargs):
    """
    Load EDP data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('slow', 'srvy', 'fast', 'brst').
    level : str
        Data quality level ('l1a', 'l2pre', 'l2')
    optdesc : str
        Optional descriptor ('dce', 'scpot', 'hmfe')
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    rename_vars : bool
        If true (default), rename the standard MMS variable names
        to something more memorable and easier to use.
    \*\*kwargs : dict
        Any keyword accepted by *pymms.data.util.load_data*
    
    Returns
    -------
    data : `xarray.Dataset`
        EDP data.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    
    # Load the data
    t_vname = '_'.join((sc, 'edp', 'epoch', mode, level))
    data = util.load_data(sc=sc, instr='edp', mode=mode, level=level,
                          optdesc=optdesc,
                          start_date=start_date, end_date=end_date,
                          record_dim=t_vname, **kwargs)
    
    # Trim time interval
    data = data.sel({t_vname: slice(start_date, end_date)})
    
    # Rename variables
    if rename_vars:
        data = rename(data, sc, mode, level, optdesc)
    
    # Add data descriptors to attributes
    data.attrs['sc'] = sc
    data.attrs['instr'] = 'fpi'
    data.attrs['mode'] = mode
    data.attrs['level'] = level
    data.attrs['optdesc'] = optdesc

    return data


def load_scpot(sc='mms1', mode='fast', level='l2',
              start_date=None, end_date=None, rename_vars=True,
              **kwargs):
    """
    Load spacecraft potential data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('slow', 'srvy', 'fast', 'brst').
    level : str
        Data quality level ('l1a', 'l2pre', 'l2')
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    rename_vars : bool
        If true (default), rename the standard MMS variable names
        to something more memorable and easier to use.
    \*\*kwargs : dict
        Any keyword accepted by *pymms.data.util.load_data*
    
    Returns
    -------
    data : `xarray.Dataset`
        Spacecraft potential data.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    optdesc = 'scpot'
    
    # Load the data
    t_vname = '_'.join((sc, 'edp', 'epoch', mode, level))
    data = util.load_data(sc=sc, instr='edp', mode=mode, level=level,
                          optdesc=optdesc,
                          start_date=start_date, end_date=end_date,
                          record_dim=t_vname, **kwargs)
    
    # Trim time interval
    t_vname = '_'.join((sc, 'edp', 'epoch', mode, level))
    data = data.sel({t_vname: slice(start_date, end_date)})
    
    if rename_vars:
        data = rename(data, sc, mode, level, optdesc)
    
    # Add data descriptors to attributes
    data.attrs['sc'] = sc
    data.attrs['instr'] = 'edp'
    data.attrs['mode'] = mode
    data.attrs['level'] = level
    data.attrs['optdesc'] = optdesc

    return data