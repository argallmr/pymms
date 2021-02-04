from pymms.data import util

def load_data(sc='mms1', mode='srvy', level='l2', optdesc='efield',
              start_date=None, end_date=None, **kwargs):
    """
    Load EDI data.
    
    CDF variable names are renamed to something easier to remember and
    use. Original CDF variable names are kept as an attribute "cdf_name"
    in each individual variable.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('slow', 'srvy', 'fast', 'brst').
    optdesc : str
        Optional descriptor. Options are: {'efield' | 'amb' | 'amb-pm2' |
        'amb-alt-cc', 'amb-alt-oc', 'amb-alt-oob', 'amb-perp-c',
        'amb-perp-ob'}
    \*\*kwargs : dict
    	Any keyword accepted by *pymms.data.util.load_data*
    
    Returns
    -------
    dist : `xarray.Dataset`
        EDI data.
    """
    
    # Load the data
    data = util.load_data(sc=sc, instr='edi', mode=mode, level=level,
                          optdesc=optdesc, start_date=start_date, 
                          end_date=end_date, **kwargs)
    
   # Rename data variables to something simpler
    if optdesc == 'efield':
        v_dsl_vname = '_'.join((sc, 'edi', 'vdrift', 'dsl', mode, level))
        v_gse_vname = '_'.join((sc, 'edi', 'vdrift', 'gse', mode, level))
        v_gsm_vname = '_'.join((sc, 'edi', 'vdrift', 'gsm', mode, level))
        e_dsl_vname = '_'.join((sc, 'edi', 'e', 'dsl', mode, level))
        e_gse_vname = '_'.join((sc, 'edi', 'e', 'gse', mode, level))
        e_gsm_vname = '_'.join((sc, 'edi', 'e', 'gsm', mode, level))
        v_labl_vname = 'v_labls'
        e_labl_vname = 'e_labls'

        names = {v_dsl_vname: 'V_DSL',
                 v_gse_vname: 'V_GSE',
                 v_gsm_vname: 'V_GSM',
                 e_dsl_vname: 'E_DSL',
                 e_gse_vname: 'E_GSE',
                 e_gsm_vname: 'E_GSM',
                 v_labl_vname: 'V_index',
                 e_labl_vname: 'E_index'}

        names = {key:val for key, val in names.items() if key in data}
        data = data.rename(names)
    
    return data
