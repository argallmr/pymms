import numpy as np
import xarray as xr
import warnings
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
    
    elif optdesc == 'amb-pm2':
        time = 'Epoch'
        flux1_0_vname = '_'.join((sc, 'edi', 'flux1', '0', mode, level))
        flux2_0_vname = '_'.join((sc, 'edi', 'flux2', '0', mode, level))
        flux3_0_vname = '_'.join((sc, 'edi', 'flux3', '0', mode, level))
        flux4_0_vname = '_'.join((sc, 'edi', 'flux4', '0', mode, level))
        flux1_180_vname = '_'.join((sc, 'edi', 'flux1', '180', mode, level))
        flux2_180_vname = '_'.join((sc, 'edi', 'flux2', '180', mode, level))
        flux3_180_vname = '_'.join((sc, 'edi', 'flux3', '180', mode, level))
        flux4_180_vname = '_'.join((sc, 'edi', 'flux4', '180', mode, level))
        traj1_0_dbcs_vname = '_'.join((sc, 'edi', 'traj1', 'dbcs', '0', mode, level))
        traj2_0_dbcs_vname = '_'.join((sc, 'edi', 'traj2', 'dbcs', '0', mode, level))
        traj3_0_dbcs_vname = '_'.join((sc, 'edi', 'traj3', 'dbcs', '0', mode, level))
        traj4_0_dbcs_vname = '_'.join((sc, 'edi', 'traj4', 'dbcs', '0', mode, level))
        traj1_180_dbcs_vname = '_'.join((sc, 'edi', 'traj1', 'dbcs', '180', mode, level))
        traj2_180_dbcs_vname = '_'.join((sc, 'edi', 'traj2', 'dbcs', '180', mode, level))
        traj3_180_dbcs_vname = '_'.join((sc, 'edi', 'traj3', 'dbcs', '180', mode, level))
        traj4_180_dbcs_vname = '_'.join((sc, 'edi', 'traj4', 'dbcs', '180', mode, level))
        traj1_0_gse_vname = '_'.join((sc, 'edi', 'traj1', 'gse', '0', mode, level))
        traj2_0_gse_vname = '_'.join((sc, 'edi', 'traj2', 'gse', '0', mode, level))
        traj3_0_gse_vname = '_'.join((sc, 'edi', 'traj3', 'gse', '0', mode, level))
        traj4_0_gse_vname = '_'.join((sc, 'edi', 'traj4', 'gse', '0', mode, level))
        traj1_180_gse_vname = '_'.join((sc, 'edi', 'traj1', 'gse', '180', mode, level))
        traj2_180_gse_vname = '_'.join((sc, 'edi', 'traj2', 'gse', '180', mode, level))
        traj3_180_gse_vname = '_'.join((sc, 'edi', 'traj3', 'gse', '180', mode, level))
        traj4_180_gse_vname = '_'.join((sc, 'edi', 'traj4', 'gse', '180', mode, level))
        
        traj1_0_dbcs_labl_vname = '_'.join((sc, 'edi', 'traj1', 'dbcs', '0', 'labl', mode, level))
        traj1_180_dbcs_labl_vname = '_'.join((sc, 'edi', 'traj1', 'dbcs', '180', 'labl', mode, level))
        traj1_0_gse_labl_vname = '_'.join((sc, 'edi', 'traj1', 'gse', '0', 'labl', mode, level))
        traj1_180_gse_labl_vname = '_'.join((sc, 'edi', 'traj1', 'gse', '180', 'labl', mode, level))

        vnames = {time: 'time',
                  flux1_0_vname: 'flux1_0',
                  flux2_0_vname: 'flux2_0',
                  flux3_0_vname: 'flux3_0',
                  flux4_0_vname: 'flux4_0',
                  flux1_180_vname: 'flux1_180',
                  flux2_180_vname: 'flux2_180',
                  flux3_180_vname: 'flux3_180',
                  flux4_180_vname: 'flux4_180',
                  traj1_0_dbcs_vname: 'traj1_0_dbcs',
                  traj2_0_dbcs_vname: 'traj2_0_dbcs',
                  traj3_0_dbcs_vname: 'traj3_0_dbcs',
                  traj4_0_dbcs_vname: 'traj4_0_dbcs',
                  traj1_180_dbcs_vname: 'traj1_180_dbcs',
                  traj2_180_dbcs_vname: 'traj2_180_dbcs',
                  traj3_180_dbcs_vname: 'traj3_180_dbcs',
                  traj4_180_dbcs_vname: 'traj4_180_dbcs',
                  traj1_0_gse_vname: 'traj1_0_gse',
                  traj2_0_gse_vname: 'traj2_0_gse',
                  traj3_0_gse_vname: 'traj3_0_gse',
                  traj4_0_gse_vname: 'traj4_0_gse',
                  traj1_180_gse_vname: 'traj1_180_gse',
                  traj2_180_gse_vname: 'traj2_180_gse',
                  traj3_180_gse_vname: 'traj3_180_gse',
                  traj4_180_gse_vname: 'traj4_180_gse'}
        
        labels = {traj1_0_dbcs_labl_vname: 'DBCS',
                  traj1_180_dbcs_labl_vname: 'DBCS',
                  traj1_0_gse_labl_vname: 'GSE',
                  traj1_180_gse_labl_vname: 'GSE'}
        
        coords = {'DBCS': ['phi', 'theta'],
                  'GSE': ['phi', 'theta']}
        
        names = {key:val for key, val in vnames.items() if key in data}
        labels = {key:val for key, val in labels.items() if key in data}
        data = data.drop(labels)
        names.update(labels)
        data = (data.rename(names)
                    .assign_coords(coords))
    
    elif optdesc == 'amb':
        pass
    
    else:
        raise ValueError('Optional descriptor not recognized: "{0}"'
                         .format(optdesc))
    
    return data


def prune_time_overlap(data, start_date, end_date):
    
    # Some of the EDI ambient files contain extra data. E.g. A file from 2016-07-07
    # has data from the first few seconds of 2016-07-08. The data from the two files
    # overlaps and causes slicing by time to fail.
    idx = np.argwhere(np.diff(data['Epoch']) < np.array(0, dtype='timedelta64[ns]'))
    if len(idx[0]) > 0:
        tf_keep = np.ones(len(data['Epoch']), dtype='bool')
        for i in idx[0]:
            new_idx = i - 1
            while data['Epoch'][new_idx] > data['Epoch'][i+1]:
                new_idx -= 1
            tf_keep[new_idx+1:i+1] = False

        data = data.sel(Epoch=tf_keep)
    
    return data.sel(indexers={'Epoch': slice(start_date, end_date)})


def load_efield(sc='mms1', mode='srvy', level='l2', optdesc='efield',
              start_date=None, end_date=None, rename_vars=True,
              **kwargs):
    """
    Load EDI electric field data.
    
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
                          optdesc='efield', start_date=start_date, 
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
    data.attrs['instr'] = 'edi'
    data.attrs['mode'] = mode
    data.attrs['level'] = level
    data.attrs['optdesc'] = optdesc
    
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
    
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        
        # Load the data
        data = util.load_data(sc=sc, instr='edi', mode=mode, level=level,
                              optdesc=optdesc, start_date=start_date, 
                              end_date=end_date, **kwargs)
        
        # Verify some things
        import pdb
        pdb.set_trace()
        if len(w) > 0:
            data = prune_time_overlap(data, start_date, end_date)
    
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
    data.attrs['instr'] = 'edi'
    data.attrs['mode'] = mode
    data.attrs['level'] = level
    data.attrs['optdesc'] = optdesc
    
    return data


def load_amb_l1a(sc='mms1', mode='fast', level='l1a', optdesc='amb',
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
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, 'edi', mode, level,
                            optdesc=optdesc,
                            start_date=start_date,
                            end_date=end_date,
                            **kwargs)
    
    files = sdc.download_files()
    try:
        files = api.sort_files(files)[0]
    except IndexError:
        raise IndexError('No files found: {0}'.format(sdc))
    
    # Read all of the data files. Skip empty files unless all files are empty
    data = []
    for file in files:
        data.append(util.cdf_to_ds(file, **kwargs))
    
    # Variables must be concatenated based on their DEPEND_0 variable
    rec_vnames = ['Epoch', 'epoch_angle', 'epoch_timetag']
    out_data = []
    for recname in rec_vnames:
        # Remove all data not associated with the current record name
        drop_vars = [varname
                     for varname in data[0]
                     if recname not in data[0][varname].coords]
        drop_coords = [coord
                       for coord in data[0].coords
                       if coord != recname]
        rec_data = [ds.drop(drop_vars + drop_coords) for ds in data]
        
        # Concatenate remaining variables together
        out = xr.concat(rec_data, dim=recname)
        
        # Select the desired time range
        out = out.sel(indexers={recname: slice(start_date, end_date)})
        
        # All datasets will be merged back together, so keep track of them
        out_data.append(out)
    
    # Combine the datasets back together
    data = xr.merge(out_data)
    
    # Rename data variables to something simpler
    if rename_vars:
        data = rename(data, sc, mode, level, optdesc)
    
    # Add data descriptors to attributes
    data.attrs['sc'] = sc
    data.attrs['instr'] = 'edi'
    data.attrs['mode'] = mode
    data.attrs['level'] = level
    data.attrs['optdesc'] = optdesc
    
    return data
