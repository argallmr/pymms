import xarray as xr
from pymms.sdc import mrmms_sdc_api as api
from pymms.data.util import cdf_to_ds
#from pymms.data import util


# The variable pointed to by the DELTA_(PLUS|MINUS)_VAR variable attribute
# for the Epoch time tags is missing its DEPEND_0 variable attribute. THis
# causes a ValueError because the attributes cannot be concatenated without
# coordinates. Here, tweak util.load_data to give the missing attribute.
def util_load_data(sc='mms1', instr='fgm', mode='srvy', level='l2',
                   optdesc=None, start_date=None, end_date=None,
                   offline=False, record_dim='Epoch', team_site=False,
                   **kwargs):
    """
    Load FPI distribution function data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    instr : str
        Instrument ID
    mode : str
        Instrument mode: ('slow', 'fast', 'srvy', 'brst').
    optdesc : str
        Optional descriptor for dataset
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    offline : bool
        If True, search only for local files
    record_dim : str
        Name of the record varying dimension. This is the dimension
        along which the data from different files will be concatenated.
        If *None*, the name of the leading dimension of the first data
        variable will be used.
    team_site : bool
        If True, search the password-protected team site
    \*\*kwargs : dict
        Keywords passed to *cdf_to_ds*
    
    Returns
    -------
    data : `xarray.DataArray`
        The requested data.
    """
    if start_date is None:
        start_date = np.datetime64('2015-10-16T13:06:04')
    if end_date is None:
        end_date = np.datetime64('2015-10-16T13:07:20')
    
    site = 'public'
    if team_site:
        site = 'private'
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, instr, mode, level,
                            optdesc=optdesc,
                            start_date=start_date,
                            end_date=end_date,
                            offline=offline)
    
    # The data level parameter will automatically set the site keyword.
    # If the user specifies the site, set it after instantiation.
    sdc.site = site
    
    files = sdc.download_files()
    files = api.sort_files(files)[0]
    
    # Concatenate data along the records (time) dimension, which
    # should be equivalent to the DEPEND_0 variable name of the
    # density variable.
    data = []
    for file in files:
        data.append(cdf_to_ds(file, **kwargs))
        
        # Assign missing coordinates to DELTA_(PLUS|MINUS)_VAR
        #   - Start by removing circular reference
        #   - This means that the Epoch variable for the DELTA_(PLUS|MINUS)
        #     variables will not itself have DELTA_(PLUS|MINUS) variables
        epoch = data[-1]['Epoch']
#        dplus = epoch.attrs['DELTA_PLUS']
#        dminus = epoch.attrs['DELTA_MINUS']
        del epoch.attrs['DELTA_PLUS']
        del epoch.attrs['DELTA_MINUS']
        
        # DELTA_(PLUS|MINUS) have their CDF variable names as their dimension
        # name. We want the dimension name to be Epoch so that we can assign
        # coordinates to Epoch
#        delta_vname = '_'.join((sc, instr, 'epoch', 'delta', mode, level))
#        dplus = dplus.rename({delta_vname: 'Epoch'})
#        dminus = dminus.rename({delta_vname: 'Epoch'})
        
        # Assign coordiantess
#        dplus = dplus.assign_coords({'Epoch': epoch})
#        dminus = dplus.assign_coords({'Epoch': epoch})

#        epoch.attrs['DELTA_PLUS'] = dplus
#        epoch.attrs['DELTA_MINUS'] = dminus
    
    # Concatenate all datasets along the time dimension. If not given,
    # assume that the time dimension is the leading dimension of the data
    # variables.
    if record_dim is None:
        varnames = [name for name in data[0].data_vars]
        rec_vname = data[0].data_vars[varnames[0]].dims[0]
    else:
        rec_vname = record_dim
    
    data = xr.concat(data, dim=rec_vname)
    data = data.sel({rec_vname: slice(start_date, end_date)})
    
    # Keep information about the data
    data.attrs['sc'] = sc
    data.attrs['instr'] = instr
    data.attrs['mode'] = mode
    data.attrs['level'] = level
    data.attrs['optdesc'] = optdesc
    data.attrs['files'] = files
    
    return data


def load_data(sc='mms1', mode='brst', level='l3', optdesc='8khz',
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
    #   - R is concatenated along Epoch, but depends on Epoch_state
    data = util_load_data(sc=sc, instr='fsm', mode=mode, level=level,
                          optdesc=optdesc, start_date=start_date, 
                          end_date=end_date, team_site=True, **kwargs)
    
    # Rename data variables to something simpler
    if optdesc == '8khz':
        b_gse_vname = '_'.join((sc, 'fsm', 'b', 'gse', mode, level))
        b_mag_vname = '_'.join((sc, 'fsm', 'b', 'mag', mode, level))
        r_gse_vname = '_'.join((sc, 'fsm', 'r', 'gse', mode, level))
        b_labl_vname = '_'.join((sc, 'fsm', 'b', 'gse', 'labls', mode, level))
        r_labl_vname = 'label_r_gse'

        names = {'Epoch': 'time',
                 'Epoch_state': 'time_r',
                 b_gse_vname: 'B_GSE',
                 b_mag_vname: '|B|',
                 r_gse_vname: 'R_GSE',
                 b_labl_vname: 'b_index',
                 r_labl_vname: 'r_index'}

        names = {key:val for key, val in names.items() if key in data}
        data = data.rename(names)
    
    return data
