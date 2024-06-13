import xarray as xr

from pymms.data import util


def rename(data, sc, mode, level, optdesc, product):
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
    optdesc : str
        Optional descriptor. Options are: ('8khz',)
    
    Returns
    -------
    data : `xarray.Dataset`
        Dataset with variables renamed
    '''
    if product == 'b':
        t_delta_vname = '_'.join((sc, 'fsm', 'epoch', 'delta', mode, level))
        b_gse_vname = '_'.join((sc, 'fsm', 'b', 'gse', mode, level))
        b_mag_vname = '_'.join((sc, 'fsm', 'b', 'mag', mode, level))
        b_labl_vname = '_'.join((sc, 'fsm', 'b', 'gse', 'labls', mode, level))
        
        names = {'Epoch': 'time',
                 t_delta_vname: 'time_delta',
                 b_gse_vname: 'B_GSE',
                 b_mag_vname: '|B|',
                 b_labl_vname: 'b_index'}

        names = {key:val for key, val in names.items() if key in data}
        data = (data.assign_coords({'b_index': ['x', 'y', 'z']})
                    .drop([b_labl_vname,], errors='ignore')
                    .rename(names)
                )
        
    elif product == 'r':
        r_gse_vname = '_'.join((sc, 'fsm', 'r', 'gse', mode, level))
        r_labl_vname = 'label_r_gse'
        repr_vname = 'represent_vec_tot'
        
        names = {'Epoch_state': 'time',
                 r_gse_vname: 'R_GSE',
                 r_labl_vname: 'r_index'}

        names = {key:val for key, val in names.items() if key in data}
        data = (data.assign_coords({'r_index': ['x', 'y', 'z', '|r|']})
                    .drop([r_labl_vname, repr_vname], errors='ignore')
                    .rename(names)
                )
    else:
        raise ValueError('Invalid data product {0}. Choose from (b, r)'
                         .format(product))
    
    return data

def load_data(sc='mms1', mode='brst', level='l3', optdesc='8khz',
              start_date=None, end_date=None, rename_vars=True,
              coords='gse', product='b', **kwargs):
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
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    level : str
        Data quality level ('l1a', 'l2pre', 'l2')
    optdesc : str
        Optional descriptor. Options are: ('8khz',)
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    coords : str, list
        Data coordinate system ('gse', 'gsm')
    product : str
        Data product to be loaded ('b', 'r')
    rename_vars : bool
        If true (default), rename the standard MMS variable names
        to something more memorable and easier to use.
    \*\*kwargs : dict
        Any keyword accepted by *pymms.data.util.load_data*
    
    Returns
    -------
    dist : `xarray.Dataset`
        FSM data.
    """
    if isinstance(coords, str):
        coords = [coords]
    
    # Select either magnetic field or position data
    if product in ('b', 'b-field'):
        product = 'b'
    elif product in ('r', 'ephemeris', 'state'):
        product = 'r'
    else:
        raise ValueError('Invalid data product {0}. Choose from (b, r)'
                         .format(product))
    if product == 'b':
        coords += ['mag']
    varformat = '_'+product+'_(' + '|'.join(coords) + ')_'
    
    # Load the data
    #   - R is concatenated along Epoch, but depends on Epoch_state
    data = util.load_data(sc=sc, instr='fsm', mode=mode, level=level,
                          optdesc=optdesc, start_date=start_date, 
                          end_date=end_date, team_site=True,
                          varformat=varformat, **kwargs)
    
    # The FSM CDFs do not have a DEPEND_0 attribute for the time delta
    # variable. Coordinates have to be assigned and its index reset
    if isinstance(data, list):
        t_delta_vname = '_'.join((sc, 'fsm', 'epoch', 'delta', mode, level))
        for idx, ds in enumerate(data):
            #t_delta = (ds[t_delta_vname]
            #            .rename({t_delta_vname: 'Epoch'})
            #            .assign_coords({'Epoch': ds['Epoch']})
            #            )
            #data[idx] = ds.assign_coords({t_delta_vname: t_delta})
            data[idx] = ds.drop([t_delta_vname,])
        
        # Concatenate the dat
        data = xr.concat(data, dim='Epoch')
    
    # Rename data variables to something simpler
    if rename_vars:
        data = rename(data, sc, mode, level, optdesc, product=product)
        
    # Add attributes about the data request
    data.attrs['sc'] = sc
    data.attrs['instr'] = 'fsm'
    data.attrs['mode'] = mode
    data.attrs['level'] = level
    data.attrs['optdesc'] = optdesc
    
    return data
