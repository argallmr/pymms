from pymms.data import util

def load_data(sc='mms1', instr='epd-eis', mode='srvy', level='l2',
              optdesc='phxtof', start_date=None, end_date=None,
              rename_vars=True, **kwargs):
    """
    Load EPD-EIS data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    level : str
        Data quality level ('l1a', 'l2pre', 'l2')
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    optdesc : str
        Optional descriptor of the file name
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
        
    # Load the data
    #   - R is concatenated along Epoch, but depends on Epoch_state
    data = util.load_data(sc=sc, instr=instr, mode=mode, level=level,
                          optdesc=optdesc, start_date=start_date, end_date=end_date, 
                          **kwargs)
    
#    if rename_vars:
#        data = rename(data, sc, instr, mode, level, product=product)
    
    # Add data descriptors to attributes
    data.attrs['sc'] = sc
    data.attrs['instr'] = instr
    data.attrs['mode'] = mode
    data.attrs['level'] = level
    
    return data
