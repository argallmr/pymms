from cdflib import cdfread, epochs
import pandas as pd
import xarray as xr
import numpy as np

def cdf_to_df(cdf_files, cdf_vars, epoch='Epoch'):
    '''
    Read variables from CDF files into a data frame
    
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


def cdf_to_ds(cdf_file, cdf_vars):
    '''
    Read variables from CDF files into a data frame
    
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
    if not isinstance(cdf_file, str):
        raise ValueError('cdf_file must be a string or path.')
    if isinstance(cdf_vars, str):
        cdf_vars = [cdf_vars]
    
    cdf = cdfread.CDF(cdf_file)
    ds = xr.Dataset()
    variables = []
    for varname in cdf_vars:
        ds = ds.assign({varname: cdf_load_var(cdf, varname)})
    
    return ds


def cdf_load_var(cdf, varname):
    
    time_types = ('CDF_EPOCH', 'CDF_EPOCH16', 'CDF_TIME_TT2000')
    tepoch = epochs.CDFepoch()
    varinq = cdf.varinq(varname)

    # Convert epochs to datetimes
    #   - None is returned if tstart and tend are outside data interval
    data = cdf.varget(variable=varname)
    if varinq['Data_Type_Description'] in time_types:
        try:
            data = epochs().to_datetime(data)
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
                if dim == 0:
                    pass
                elif dim == 1:
                    dim_name = varatts['LABLAXIS']
                else: 
                    raise ValueError('Unknown coordinates for '
                                     'dimension {}'.format(dim))
        
        if dim_name is None:
            continue
        
        # Sometimes the same DEPEND_N or LABL_PTR_N variable is
        # used for multiple dimensions. A DataArray requires
        # unique dimension names, so append "_dimN" to duplicate
        # names.
        key = dim_name
        if dim_name in coords:
            key += '_dim{}'.format(dim)
        
        # Read the dependent variable data corresponding to dim
        if dim_name in cdf.cdf_info()['zVariables']:
            coords[key] = cdf_load_var(cdf, dim_name)
        else:
            coords[key] = np.arange(data.shape[dim])
        
        # For record varying dependent variables (2D), use
        # the name of the data dimension (not the record-varying)
        # dimension to label the dimension. The coordinate will
        # be given the variable name, but the dimension is the
        # non-record-varying dimension's name/label.
        if len(coords[key].shape) == 2:
            dim_name = coords[key].dims[-1]
        elif len(coords[key].shape) > 2:
            ValueError('Coordinate has unexpected number of '
                       'dimensions (>2).')
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
    
    dims = []
    varatts = cdf.varattsget(var.cdf_name)
    for dep in range(4):
        try:
            attrvalue = varatts['DEPEND_{0}'.format(dep)]
            dims.append(attrvalue)
        except AttributeError:
            break
    
    return dims


def cdf_attget(cdf, da):
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


def _cdflib_readvar(cdf, varname, tstart, tend):
    global cdf_vars
    global file_vars

    # Data has already been read from this file
    if varname in file_vars:
        var = file_vars[varname]
    else:
        time_types = ('CDF_EPOCH', 'CDF_EPOCH16', 'CDF_TIME_TT2000')
        varinq = cdf.varinq(varname)

        # Convert epochs to datetimes
        #   - None is returned if tstart and tend are outside data interval
        data = cdf.varget(variable=varname, starttime=tstart, endtime=tend)
        if varinq['Data_Type_Description'] in time_types:
            try:
                var = metatime.MetaTime(cdflib.cdfepoch().to_datetime(data))
            except TypeError:
                var = metatime.MetaTime(data)
        else:
            var = MetaArray(data)

        # If the variable has been read from a different file, append
        if (varname in cdf_vars) and varinq['Rec_Vary']:
            d0 = cdf_vars[varname]
            try:
                var = np.append(d0, var, 0).view(type(var))
            except ValueError:
                if data is None:
                    return d0
                else:
                    raise

        # Create the variable
        var.name = varname
        var.rec_vary = varinq['Rec_Vary']
        var.cdf_name = varinq['Variable']
        var.cdf_type = varinq['Data_Type_Description']

        # List as read
        #  - Prevent infinite loop. Must save the variable in the registry
        #  so that variable attributes do not try to read the same variable
        #  again.
        cdf_vars[varname] = var
        file_vars[varname] = var

        # Read the metadata
        _cdflib_attget(cdf, var, tstart, tend)
        _cdfattrs_to_gfxkeywords(var)

    return var



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