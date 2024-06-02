from pymms.data import util

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
        Optional descriptor. Options are: ('8khz',)
    
    Returns
    -------
    data : `xarray.Dataset`
        Dataset with variables renamed
    '''
    b_gse_vname = '_'.join((sc, 'scm', 'acb', 'gse', optdesc, mode, level))
    b_gse_lbl_vname = '_'.join((sc, 'scm', 'acb', 'gse',
                                optdesc, mode, level, 'labl', '1'))
    b_gse_repr_vname = '_'.join((sc, 'scm', 'acb', 'gse',
                                 optdesc, mode, level,
                                 'representation', '1'))

    names = {'Epoch': 'time',
             'Epoch_delta': 'time_delta',
             b_gse_vname: 'B_GSE',
             b_gse_lbl_vname: 'b_index'}

    names = {key:val for key, val in names.items() if key in data}
    data = (data.assign_coords({'b_index': ['x', 'y', 'z']})
                .drop([b_gse_lbl_vname, b_gse_repr_vname], errors='ignore')
                .rename(names)
            )
    return data

def load_data(sc='mms1', mode='srvy', level='l2',
              start_date=None, end_date=None, rename_vars=True,
              **kwargs):
    """
    Load SCM data.
    
    CDF variable names are renamed to something easier to remember and
    use. Original CDF variable names are kept as an attribute "cdf_name"
    in each individual variable.
    
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
        SCM data.
    """
    if mode == 'srvy':
        optdesc = 'scsrvy'
    elif mode == 'slow':
        optdesc = 'scs'
    elif mode == 'fast':
        optdesc = 'scf'
    elif mode == 'brst':
        optdesc = 'scb'
    else:
        raise ValueError('SCM data rate mode {0} invalid. Try {1}'
                         .format(mode, ('slow', 'fast', 'srvy', 'brst')))
    
    # Load the data
    #   - R is concatenated along Epoch, but depends on Epoch_state
    data = util.load_data(sc=sc, instr='scm', mode=mode, level=level,
                          optdesc=optdesc,
                          start_date=start_date, end_date=end_date,
                          **kwargs)
    
    # Rename data variables to something simpler
    if rename_vars:
        data = rename(data, sc, mode, level, optdesc)
    
    return data
