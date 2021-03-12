from pymms.data import util


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

def rename(fgm_data, sc, instr, mode, level):
    '''
    Rename standard variables names to something more memorable.
    
    Parameters
    ----------
    ds : `xarray.Dataset`
        Data to be renamed
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    level : str
        Data quality level ('l1a', 'l2pre', 'l2')
    
    Returns
    -------
    data : `xarray.Dataset`
        Dataset with variables renamed
    '''
        
    t_vname = 'Epoch'
    tr_vname = 'Epoch_state'
    t_delta_vname = '_'.join((sc, instr, 'bdeltahalf', mode, level))
    tr_delta_vname = '_'.join((sc, instr, 'rdeltahalf', mode, level))
    
    r_gse_vname = '_'.join((sc, instr, 'r', 'gse', mode, level))
    r_gsm_vname = '_'.join((sc, instr, 'r', 'gsm', mode, level))
    
    r_gse_labl_vname = '_'.join(('label', 'r', 'gse'))
    r_gsm_labl_vname = '_'.join(('label', 'r', 'gsm'))
    repr_vname = '_'.join(('represent', 'vec', 'tot'))
    
    b_dmpa_vname = '_'.join((sc, instr, 'b', 'dmpa', mode, level))
    b_bcs_vname = '_'.join((sc, instr, 'b', 'bcs', mode, level))
    b_gse_vname = '_'.join((sc, instr, 'b', 'gse', mode, level))
    b_gsm_vname = '_'.join((sc, instr, 'b', 'gsm', mode, level))
    
    b_dmpa_lbl_vname = '_'.join(('label', 'b', 'dmpa'))
    b_bcs_lbl_vname = '_'.join(('label', 'b', 'bcs'))
    b_gse_lbl_vname = '_'.join(('label', 'b', 'gse'))
    b_gsm_lbl_vname = '_'.join(('label', 'b', 'gsm'))
    
    labels = [b_dmpa_lbl_vname, b_bcs_lbl_vname, b_gse_lbl_vname,
              b_gsm_lbl_vname, r_gse_labl_vname, r_gsm_labl_vname]
    
    # Rename variables
    names = {t_vname: 'time',
             tr_vname: 'time_r',
             t_delta_vname: 'time_delta',
             tr_delta_vname: 'time_r_delta',
             b_dmpa_vname: 'B_DMPA',
             b_bcs_vname: 'B_BCS',
             b_gse_vname: 'B_GSE',
             b_gsm_vname: 'B_GSM',
             r_gse_vname: 'r_GSE',
             r_gsm_vname: 'r_GSM'}

    names = {key:val for key, val in names.items() if key in fgm_data}
    fgm_data = fgm_data.rename(names)
    
    # Standardize labels
    labels = [label for label in labels if label in fgm_data]
    new_labels = {key: ('r_index' if key.startswith('label_r') else 'b_index')
                  for key in labels}
    fgm_data = (fgm_data.assign_coords({'b_index': ['x', 'y', 'z', 't'],
                                        'r_index': ['x', 'y', 'z']})
                        .drop(labels + [repr_vname,])
                        .rename(new_labels)
                )
    
    return fgm_data


def load_data(sc='mms1', instr='fgm', mode='srvy', level='l2',
              start_date=None, end_date=None, rename_vars=True,
              **kwargs):
    """
    Load FPI distribution function data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    instr : str
        Instrument ID: ('afg', 'dfg', 'fgm')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    level : str
        Data quality level ('l1a', 'l2pre', 'l2')
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    coords : str
        Data coordinate system ('gse', 'gsm', 'dmpa', 'omb')
    pd : bool
        If true, read data as a pandas dataframe
    rename_vars : bool
        If true (default), rename the standard MMS variable names
        to something more memorable and easier to use.
    \*\*kwargs : dict
        Keywords for `pymms.data.util.load_data`
    
    Returns
    -------
    dist : `metaarray.metaarray`
        Particle distribution function.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    check_level(level, instr=instr)
    
    # Load the data
    #   - R is concatenated along Epoch, but depends on Epoch_state
    data = util.load_data(sc=sc, instr=instr, mode=mode, level=level,
                          start_date=start_date, end_date=end_date, 
                          **kwargs)
    
    if rename_vars:
        data = rename(data, sc, instr, mode, level)
    
    # Add data descriptors to attributes
    data.attrs['sc'] = sc
    data.attrs['instr'] = instr
    data.attrs['mode'] = mode
    data.attrs['level'] = level
    
    return data
