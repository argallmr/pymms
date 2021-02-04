from cdflib import cdfread, epochs
from pymms.sdc import mrmms_sdc_api as api
import pandas as pd
import xarray as xr
import numpy as np
import re

# Note used
import pathlib
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

# Downloader
from pymms import config
import pathlib
data_root = pathlib.Path(config['data_root'])

class Downloader():
    '''
    A class for downloading a single dataset.
    
    The following methods must be implemented by sub-classes:
    '''
    def load(self, starttime, endtime):
        pass
    
    def local_path(self, interval):
        '''
        Absolute path to a single file.
        
        Parameters
        ----------
        interval : tuple of datetime.datetime
            Start and end time associated with a single file
        
        Returns
        -------
        path : str
            Absolute file path
        '''
        local_path = self.local_dir(interval) / self.fname(interval)
        return data_root / local_path
    
    def local_file_exists(self, interval):
        '''
        Check if a local file exists.
        
        Parameters
        ----------
        interval : tuple of datetime.datetime
            Start and end time associated with a single file
        
        Returns
        -------
        exists : bool
            True if local file exists. False otherwise.
        '''
        return self.local_path(interval).exists()
    
    def intervals(self, starttime, endtime):
        '''
        Break the time interval down into a set of intervals associated
        with individual file names.
        
        Parameters
        ----------
        starttime, endtime : datetime.datetime
            Start and end times of the data interval
        
        Returns
        -------
        intervals : list of tuples
            Time intervals (starttime, endtime) associated with individual
            data files
        '''
        pass
    
    def fname(self, interval):
        '''
        Create the file name associated with a given interval.
        
        Parameters
        ----------
        interval : tuple of datetime.datetime
            Start and end time of the data interval
        
        Returns
        -------
        filename : str
            File name
        '''
        pass
    
    def local_dir(self, interval):
        '''
        Local directory for a given interval. This is relative to the
        PyMMS data directory.
        
        Parameters
        ----------
        interval : tuple of datetime.datetime
            Start and end time of the data interval
        
        Returns
        -------
        dir : pathlib.Path
            Local directory
        '''
        pass
    
    def download(self, interval):
        pass
    
    def load_local_file(self, interval):
        pass


def cdf_to_df(cdf_files, cdf_vars, epoch='Epoch'):
    '''
    Read variables from CDF files into a dataframe
    
    Parameters
    ----------
    cdf_files : str or list
        CDF files to be read
    cdf_vars : str or list
        Names of the variables to be read
    epoch : str
        Name of the time variable that serves as the data frame index
    
    Returns
    -------
    out : `pandas.DataFrame`
        The data. If a variable is 2D, "_#" is appended, where "#"
        increases from 0 to var.shape[1]-1.
    '''
    tepoch = epochs.CDFepoch()
    if isinstance(cdf_files, str):
        cdf_files = [cdf_files]
    if isinstance(cdf_vars, str):
        cdf_vars = [cdf_vars]
    if epoch not in cdf_vars:
        cdf_vars.append(epoch)
    
    out = []
    for file in cdf_files:
        file_df = pd.DataFrame()
        cdf = cdfread.CDF(file)
        
        for var_name in cdf_vars:
            # Read the variable data
            data = cdf.varget(var_name)
            if var_name == epoch:
                data = tepoch.to_datetime(data, to_np=True)
            
            # Store as column in data frame
            if data.ndim == 1:
                file_df[var_name] = data
                
            # 2D variables get "_#" appended to name for each column
            elif data.ndim == 2:
                for idx in range(data.shape[1]):
                    file_df['{0}_{1}'.format(var_name, idx)] = data[:,idx]
                    
            # 3D variables gets reshaped to 2D and treated as 2D
            # This includes variables like the pressure and temperature tensors
            elif data.ndim == 3:
                dims = data.shape
                data = data.reshape(dims[0], dims[1]*dims[2])
                for idx in range(data.shape[1]):
                    file_df['{0}_{1}'.format(var_name, idx)] = data[:,idx]
            else:
                print('cdf_var.ndims > 3. Skipping. {0}'.format(var_name))
                continue
        
        # Close the file
        cdf.close()
        
        # Set the epoch variable as the index
        file_df.set_index(epoch, inplace=True)
        out.append(file_df)
    
    # Concatenate all of the file data
    out = pd.concat(out)
    
    # Check that the index is unique
    # Some contiguous low-level data files have data overlap at the edges of the files (e.g., AFG)
    if not out.index.is_unique:
        out['index'] = out.index
        out.drop_duplicates(subset='index', inplace=True, keep='first')
        out.drop(columns='index', inplace=True)
    
    # File names are not always given in order, so sort the data
    out.sort_index(inplace=True)
    return out

def cdf_varnames(cdf, data_vars=True):
    '''
    Return the variable names in the CDF file.
    
    Parameters
    ----------
    cdf : `cdflib.cdfread.CDF`
        CDF file object
    data : bool
        Only variables with VAR_TYPE of "data" will be returned
    
    Returns
    -------
    varnames : list
        Variable names
    '''
    varnames = cdf.cdf_info()['zVariables']
    if data_vars:
        varnames = [varname
                    for varname in varnames
                    if cdf.attget('VAR_TYPE', entry=varname)['Data'] == 'data'
                    ]
    return varnames


def cdf_to_ds(filename, variables=None, varformat=None, data_vars=True):
    '''
    Read variables from CDF files into an XArray DataSet
    
    Parameters
    ----------
    filename : str
        Name of the CDF file to read
    variables : str or list
        Names of the variables to read or a pattern by which to match
        variable names. If not given, all variables will be read
    varformat : str or list
        Regular expression(s) used to match variable names. Mutually
        exclusive with `variables`.
    data_vars : bool
        Read only those variables with VAR_TYPE of "data". Ignored if
        `variables` is given.
    
    Returns
    -------
    ds : `xarray.Dataset`
        The data.
    '''
    if not isinstance(filename, str):
        raise ValueError('cdf_file must be a string or path.')
    if isinstance(variables, str):
        variables = [variables]
    if isinstance(varformat, str):
        varformat = [varformat]
    
    # Open the CDF file
    cdf = cdfread.CDF(filename)
    
    # Match regular expression(s)
    if variables is not None:
        varnames = variables
    
    elif varformat is not None:
        all_variables = cdf_varnames(cdf, data_vars=data_vars)
        varnames = []
        for fmt in varformat:
            varnames =+ [v for v in all_variables if re.match(fmt, v)]
    
    # Select all (data) variables
    else:
        varnames = cdf_varnames(cdf, data_vars=data_vars)
    
    # Read the data
    ds = xr.Dataset()
    variables = []
    for varname in varnames:
        ds = ds.assign({varname: cdf_load_var(cdf, varname)})
    
    # Grab the global attributes from the data file
    ds.attrs['filename'] = filename
    ds.attrs.update(cdf.globalattsget())
    
    return ds


def cdf_load_var(cdf, varname):
    '''
    Read a variable and its metadata into a xarray.DataArray
    
    Parameters
    ----------
    cdf : `cdfread.CDF`
        The CDF file object
    varname : str
        Name of the variable to read
    
    Returns
    -------
    da : `xarray.DataArray`
        The data with metadata
    '''
    
    time_types = ('CDF_EPOCH', 'CDF_EPOCH16', 'CDF_TIME_TT2000')
    tepoch = epochs.CDFepoch()
    varinq = cdf.varinq(varname)

    # Convert epochs to datetimes
    #   - None is returned if tstart and tend are outside data interval
    data = cdf.varget(variable=varname)
    if varinq['Data_Type_Description'] in time_types:
        try:
            data = np.asarray([np.datetime64(t)
                               for t in tepoch.to_datetime(data)])
        except TypeError:
            pass
    
    # Get dependent variables first because multi-dimensional
    # dependent variables (e.g. those with record variance)
    # will need to be parsed in order to set the dimensions of
    # the current variable
    coords = {}
    dims = []
    varatts = cdf.varattsget(varname)
    for dim in range(len(data.shape)):
        
        dim_name = None
        try:
            dim_name = varatts['DEPEND_{0}'.format(dim)]
        
        except KeyError:
            # Dimensions may have labels instead.
            try:
                dim_name = varatts['LABL_PTR_{0}'.format(dim)]
            
            # DEPEND_0 variables for the record varying dimension
            # typically do not depend on anything or have labels.
            #
            # DEPEND_N variables (where 1 <= N <= 3) are typically
            # one dimensional if not record varying, and two
            # dimensional if record varying. In either case, the
            # non-record-varying dimension does not have a
            # dependent variable or label because *it is* the
            # dependent variable. In this case, name the dimension
            # with the axis label.
            except KeyError:
                try:
                    dim_name = varatts['LABLAXIS']
                except KeyError:
                    dim_name = varname
                
#                 if dim == 0:
#                     pass
#                 elif dim == 1:
#                     print(varname)
#                     dim_name = varatts['LABLAXIS']
#                 else: 
#                     raise ValueError('Unknown coordinates for '
#                                      'dimension {}'.format(dim))
        
        if dim_name == varname:
            continue
        
        # Sometimes the same DEPEND_N or LABL_PTR_N variable is
        # used for multiple dimensions. A DataArray requires
        # unique coordinate names, so append "_dimN" to duplicate
        # names. This happens for, e.g., a temperature tensor
        # variable with dimensions [3,3] and labels ['x', 'y', 'z']
        # for each dimension.
        coord_name = dim_name
        if dim_name in coords:
            coord_name += '_dim{}'.format(dim)
        
        # DEPEND_# and LABL_PTR_# are pointers to other variables in the
        # CDF file. Read the data from those variables as the coordinate
        # values for this dimension.
        if dim_name in cdf.cdf_info()['zVariables']:
#            coords[coord_name] = cdf_load_var(cdf, dim_name)
            
            
            coord_data = cdf_load_var(cdf, dim_name)
            if len(coord_data.shape) == 1:
                dim_name = coord_name
            elif len(coord_data.shape) == 2:
                dim_name = coord_data.dims[-1]
            else:
                ValueError('Coordinate has unexpected number of '
                           'dimensions (>2).')
            
            coords[coord_name] = coord_data
            dims.append(dim_name)
        
        elif dim > 0:
            dims.append(dim_name)
    
    # If the variable is not record varying, the "records" dimension
    # will be shallow and need to be removed so that data has the
    # same number of dimensions as dims has elements.
    if not varinq['Rec_Vary']:
        data = data.squeeze()
    
    # Unlike coords, dims cannot be empty.
    if len(coords) == 0:
        dims.append(varname)
    
    da = xr.DataArray(data,
                      dims=dims,
                      coords=coords)
    
    # Create the variable
    da.name = varname
    da.attrs['rec_vary'] = varinq['Rec_Vary']
    da.attrs['cdf_name'] = varinq['Variable']
    da.attrs['cdf_type'] = varinq['Data_Type_Description']
    
    # Read the metadata
    cdf_attget(cdf, da)
    
    return da


def cdf_get_dims(cdf, da):
    '''
    Get the names and number of dimensions of a CDF variable by
    checking how many DEPEND_# variable attributes it has (# is a
    number ranging from 0 to 3).
    
    Parameters
    ----------
    cdf : `cdfread.CDF`
        The CDF file object
    da : `xarray.DataArray`
        The variable data and metadata
    
    Returns
    -------
    dims : list
        Names of the dependent variable dimensions
    '''
    
    dims = []
    varatts = cdf.varattsget(da.cdf_name)
    for dep in range(4):
        try:
            attrvalue = varatts['DEPEND_{0}'.format(dep)]
            dims.append(attrvalue)
        except AttributeError:
            break
    
    return dims


def cdf_attget(cdf, da):
    '''
    Read the variable's metadata. Standardize some of the variable
    attribute names.
    
    Parameters
    ----------
    cdf : `cdfread.CDF`
        The CDF file object
    da : `xarray.DataArray`
        The variable data and metadata. da.attrs is edited in-place.
    '''
    # Get variable attributes for given variable
    varatts = cdf.varattsget(da.attrs['cdf_name'])

    # Get names of all cdf variables
    cdf_varnames = cdf.cdf_info()['zVariables']

    # Follow pointers to retrieve data
    #   - Some CDFs label the axis with the variable name.
    #   - If LABLAXIS is a variable name, do not follow pointer
    for attrname, attrvalue in varatts.items():
        # These attributes are already taken care of
        if attrname.startswith(('DEPEND_', 'LABL_PTR_')):
            continue
        
        '''
        This is commented out for now because the DELTA_PLUS_VAR
        for Epoch variable can have a DEPEND_0 attribute that
        points back to Epoch, resulting in an infinite loop.
        
        # Some attribute values point to other variables
        if (isinstance(attrvalue, str)
            and (attrvalue in cdf_varnames)
            ):
            
            import pdb
            pdb.set_trace()
            attrvalue = cdf_load_var(cdf, ds, attrvalue)
        '''
        
        # Rename particular attributes
        if attrname == 'DELTA_PLUS_VAR':
            attrname = 'DELTA_PLUS'
        elif attrname == 'DELTA_MINUS_VAR':
            attrname = 'DELTA_MINUS'
        elif attrname == 'UNITS':
            attrname = 'units'
        
        da.attrs[attrname] = attrvalue


def rename_df_cols(df, old_col, new_cols):
    '''
    Each column of a multi-dimensional CDF variable gets stored as
    its own independent column in the DataFrame, with "_#" appended
    to the original variable name to indicate which column index
    the column was taken from. This function renames those columns.
    
    Parameters
    ----------
    df : `pandas.DataFrame`
        DataFrame for which the columns are to be renamed
    old_col : str
        Name of the column (sans "_#")
    new_cols : list
        New names to be given to the columns
    '''
    df.rename(columns={'{}_{}'.format(old_col, idx): new_col_name
                       for idx, new_col_name in enumerate(new_cols)},
              inplace=True
             )


def load_data(sc='mms1', instr='fgm', mode='srvy', level='l2',
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
                            offline=offline,
                            site=site)
    files = sdc.download_files()
    files = api.sort_files(files)[0]
    
    # Concatenate data along the records (time) dimension, which
    # should be equivalent to the DEPEND_0 variable name of the
    # density variable.
    data = []
    for file in files:
        data.append(cdf_to_ds(file, **kwargs))
    
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