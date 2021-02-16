import xarray as xr
from pymms.data import util

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
    data = util.load_data(sc=sc, instr='fsm', mode=mode, level=level,
                          optdesc=optdesc, start_date=start_date, 
                          end_date=end_date, team_site=True, **kwargs)
    
    # The FSM CDFs do not have a DEPEND_0 attribute for the time delta
    # variable. Coordinates have to be assigned and its index reset
    if isinstance(data, list):
        t_delta_vname = '_'.join((sc, 'fsm', 'epoch', 'delta', mode, level))
        for idx, ds in enumerate(data):
            data[idx] = (ds.assign_coords({t_delta_vname: ds['Epoch']})
                           .reset_coords(t_delta_vname))
        
        # Add attributes about the data request
        data.attrs['sc'] = sc
        data.attrs['instr'] = 'fsm'
        data.attrs['mode'] = mode
        data.attrs['level'] = level
        data.attrs['optdesc'] = optdesc
    
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
