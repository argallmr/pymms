import xarray as xr
from pymms.data import util
from pymms.sdc import mrmms_sdc_api as api

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
        Data quality level ('l1a', 'l2')
    optdesc : str
        Optional descriptor. Options are: {'efield' | 'amb' | 'amb-pm2' |
        'amb-alt-cc', 'amb-alt-oc', 'amb-alt-oob', 'amb-perp-c',
        'amb-perp-ob'}
    
    Returns
    -------
    data : `xarray.Dataset`
        Dataset with variables renamed
    '''
    
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
    else:
        raise ValueError('Optional descriptor not recognized: "{0}"'
                         .format(optdesc))
    
    return data
    

def load_data(sc='mms1', mode='srvy', level='l2', optdesc='efield',
              start_date=None, end_date=None, rename_vars=True,
              **kwargs):
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
    level : str
        Data quality level ('l1a', 'l2pre', 'l2')
    optdesc : str
        Optional descriptor. Options are: {'efield' | 'amb' | 'amb-pm2' |
        'amb-alt-cc', 'amb-alt-oc', 'amb-alt-oob', 'amb-perp-c',
        'amb-perp-ob'}
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    rename_vars : bool
        If true (default), rename the standard MMS variable names
        to something more memorable and easier to use.
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
    
    # EDI generates empty efield files when in ambient mode. These files
    # have a version number of '0.0.0' and get read in as empty datasets,
    # which fail the concatenation in util.load_data. Remove the datasets
    # associated with empty files and concatenate the datasets with data
    if isinstance(data, list):
        # Remove empty datasets
        data = [ds 
                for ds in data
                if api.parse_file_name(ds.attrs['filename'])[-1] != '0.0.0']
        
        # Concatenate again
        data = xr.concat(data, dim='Epoch')
    
    # Rename data variables to something simpler
    if rename_vars:
        data = rename(data, sc, mode, level, optdesc)
    
    # Add data descriptors to attributes
    data.attrs['sc'] = sc
    data.attrs['instr'] = 'fpi'
    data.attrs['mode'] = mode
    data.attrs['level'] = level
    data.attrs['optdesc'] = optdesc
    
    return data
