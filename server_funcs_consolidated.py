#!/usr/bin/python3
import numpy as np
import datetime as dt
import spacepy
from spacepy import pycdf
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os.path
import pymms
from pymms import mms_utils
import pdb
import sqlite3

## Creating the pymms object
# First, we create an instance of the object that communicates with the SDC. For the sake of this
# example, we will start with data from `2015-10-16` because there are several magnetopause crossings
# and a well-studied electron diffusion region event. Also, for simplicity we will work with data from
# the MMS1 spacecraft. Data from other spacecraft can be loaded by changing the `sc` property
# below to `'mms2'`, `'mms3'`, or `'mms4'`.

# Create an instance of SDC object
sdc = pymms.MrMMS_SDC_API()

def data_export(spacecraft, level, start_date, end_date, data_download_path):
    # Define the spacecraft. We will use the variable later when accessing the CDF files.
    sc = spacecraft
    #level = 'sitl'                    # 'l2' or 'sitl'
    level = level
    start_date = start_date
    end_date = end_date
    data_root = os.path.expanduser(data_download_path)
            # Specifying data_root="~/" does not expand the tilde yet
            # However, if data_root=None, then ~/data is the default
    
    # Set attributes
    sdc.sc = sc
    sdc.start_date = start_date
    sdc.end_date = end_date
    sdc.data_root = data_root
    
    ## Working with CDF Files
    # CDF files are somewhat like netCDF or HDF5 files in that the contain data as well as metadata.
    # Data is associated with variable names and variable metadata, or variable attributes. The file itself
    # has metadata in the form of global attributes. For our purpose, we are interested in determining the
    # variable names, what they mean, then selecting the subset of variables that are relevant to us. To do that,
    # we will need to download an MMS CDF data file and make use of pycdf from the spacepy package.
    
    # Downloading an MMS CDF File
    # Here, we will give a brief example of how to download a CDF file using the `pymms` package. We pick a
    # file from the fluxgate magnetometer (FGM) team containing magnetic field data. For demonstration purposes,
    # we select a science-quality data file (`level='l2'`) when the instrument was sampling in survey
    # mode (`mode='srvy'`). [Since the Geocentric Solar Ecliptic](https://sscweb.gsfc.nasa.gov/users_guide/Appendix_C.html)
    # (GSE) coordinate system is the standard for MMS data, we will extract non-scalar data in this system.
    
    
    # First, define variables, as they will be used in creating variable names
    fgm_mode = 'srvy'
    fgm_coords = 'gse'
    fgm_instr = 'fgm'
    fgm_level = 'l2'
    
    # Set object properties and get data
    sdc.instr = fgm_instr
    sdc.mode = fgm_mode
    sdc.level = fgm_level
    files = sdc.Download()
    
    print('FGM Files:')
    print(*files, sep='\n')
    
    
    ## CDF Attributes and Variables
    
    # In order to access data in a CDF file, it is necessary to know the names of the variables contained within. Often,
    # it is also important to know additional information about the file contents or variable data. This metadata
    # is contained in the global and variable attributes.
    # The most important variable attributes are CATDESC, which describes the variable, FILLVAL, which gives the
    # value used for bad or missing data, and DEPEND_[0-3], which list the dependent variables of a data variable.
    # Typically, the dimensions of CDF variables are ordered as [nRecs, nDep1, nDep2, nDep3], where nRecs is the total
    # number of records, each record having dimensions [nDep1, nDep2, nDep3]. The value of DEPEND_0 is typically 'Epoch',
    # indicating that the CDF variable 'Epoch' is a dependency. The 'Epoch' variable contains a CDF Epoch time stamp at
    # each of the nRecs number of records. Similarly DEPEND_[1-3] variables point to other variables in the CDF file
    # that act as dependencies. If you want to plot a variable, you will need to also extract its 'DEPEND_0' variables.
    
    # Variables and attributes are described in more detail in the ISTP CDF Guidelines. Below, we demonstrate
    # how to obtain attribute and variable names and values.
    
    
    # Open the file and pr
    cdf = pycdf.CDF(files[0])
    
    # Show global attribute
    #print('Global Attributes:')
    #for gAttrName in cdf.attrs:
    #    print('\t' + gAttrName)
    
    # Show variable names
    #print('\nVariable Names:')
    #for varName in cdf:
    #    print('\t' + varName)
    
    # Select the magnetic field variable
    vname = '_'.join((sc, fgm_instr, 'b', fgm_coords, fgm_mode, fgm_level))
    
    # Show variable attributes for a particular variable
    #print('\nVariable Attributes for "' + vname + '":')
    #for vAttrName in cdf[vname].attrs:
    #    print('\t' + vAttrName)
    
    # Important variable attributes:
    #print('\nValues of Important Variable Attributes:')
    #print('\t', 'CATDESC: ', cdf[vname].attrs['CATDESC'])
    #print('\t', 'FILLVAL: ', cdf[vname].attrs['FILLVAL'])
    #print('\t', 'DEPEND_0: ', cdf[vname].attrs['DEPEND_0'])
    
    
    ## FGM
    # The FGM dataset contains magnetic field data from the fluxgate magnetometer (FGM).
    # [Since the Geocentric Solar Ecliptic](https://sscweb.gsfc.nasa.gov/users_guide/Appendix_C.html) (GSE)
    # coordinate system is the standard for MMS data, we will extract non-scalar data in this system.
    
    # Download & Read Data
    # Now we can read data and its corresponding time stamps for a CDF variable. We choose the `'mms1_fgm_b_gse_srvy_l2'`
    # variable because, as shown above, its `CATDESC` attribute describes it as the magnetic field in GSE coordinates.
    # In order to be completely general, I will build the variable names from the attributes we have already defined.
    # Variable names have the convention of `sc_instr_param_coords_optdesc_mode_level`, where `param` describes the
    # quantity and `coords` is the coordinate system whenever relevant. Other components are similar to the file name
    # conventions.
    
    # Update instrument-specific variables
    fgm_mode = 'srvy'
    if level == 'sitl':
        fgm_coords = 'dmpa'
        fgm_instr = 'dfg'
        fgm_level = 'ql'
    else:
        fgm_coords = 'gse'
        fgm_instr = 'fgm'
        fgm_level = 'l2'
    
    # Set object properties
    sdc.instr = fgm_instr
    sdc.mode = fgm_mode
    sdc.level = fgm_level
    
    # Download data
    files = sdc.Download()
    files = mms_utils.sort_files(files)[0]
    
    # Read the magnetic field and its time stamps
    if level == 'l2':
        b_vname = '_'.join((sc, fgm_instr, 'b', fgm_coords, fgm_mode, fgm_level))
    else:
        b_vname = '_'.join((sc, fgm_instr, fgm_mode, fgm_coords))
    
    fgm_t = []
    fgm_b = []
    
    print('FGM Files:')
    for file in files:
        # Open the file
        cdf = pycdf.CDF(file)
    
        # Read the data
        #   - Convert numpy arrays to lists to make appending easier
        fgm_t += list(cdf[cdf[b_vname].attrs['DEPEND_0']][:])
        fgm_b += list(cdf[b_vname][:])
    
        # Close the file
        cdf.close()
        print('  ' + file)
    
    # Convert back to numpy arrays
    fgm_t = np.array(fgm_t)
    fgm_b = np.array(fgm_b)
    
    # Compute clock and normal angles
    fgm_ca = np.rad2deg(np.arctan2(fgm_b[:,1], fgm_b[:,2]))
    fgm_tbn = np.rad2deg(np.arctan2(fgm_b[:,0], fgm_b[:,2]))
    
    ## Data Frame
    
    # Create a dictionary
    fgm_data = {
        'Time' :  fgm_t,
        'Bx' : fgm_b[:,0],
        'By' : fgm_b[:,1],
        'Bz' : fgm_b[:,2],
        'Bmag' : fgm_b[:,3],
        'clock_angle' : fgm_ca,
        'normal_angle' : fgm_tbn
    }
    
    # Convert dictionary to data from
    fgm_data = pd.DataFrame(fgm_data, columns=fgm_data.keys())
    
    
    ## EDP
    # Now for electric field and spacecraft potential data from the Electric Field Double Pobles (EDP).
    
    # Download & Read
    
    # Update instrument-specific variables
    edp_instr = 'edp'
    edp_mode = 'fast'
    edp_level = level
    dce_optdesc = 'dce'
    scpot_optdesc = 'scpot'
    
    if level == 'l2':
        edp_coords = 'gse'
    else:
        edp_coords = 'dsl'
    
    # EDP variable names
    e_vname = '_'.join((sc, 'edp', dce_optdesc, edp_coords, edp_mode, edp_level))
    scpot_vname = '_'.join((sc, 'edp', scpot_optdesc, edp_mode, edp_level))
    
    # Download DCE files
    sdc.instr = edp_instr
    sdc.mode = edp_mode
    sdc.level = edp_level
    sdc.optdesc = dce_optdesc
    dce_files = sdc.Download()
    dce_files = mms_utils.sort_files(dce_files)[0]
    
    # Download SCPOT files
    sdc.optdesc = scpot_optdesc
    scpot_files = sdc.Download()
    scpot_files = mms_utils.sort_files(scpot_files)[0]
    
    # Read the data
    edp_t = []
    edp_e = []
    edp_v = []
    print('EDP Files:')
    for ifile, file in enumerate(dce_files):
        # Open the file
        dce_cdf = pycdf.CDF(dce_files[ifile])
        scpot_cdf = pycdf.CDF(scpot_files[ifile])
    
        # Read data and replace fill value with NaN
        e = dce_cdf[e_vname][:]
        v = scpot_cdf[scpot_vname][:]
        e[e == dce_cdf[e_vname].attrs['FILLVAL']] = np.nan
        v[v == scpot_cdf[scpot_vname].attrs['FILLVAL']] = np.nan
    
        # Read the data
        #   - Convert numpy arrays to lists to make appending easier
        edp_t += list(dce_cdf[dce_cdf[e_vname].attrs['DEPEND_0']][:])
        edp_e += list(e)
        edp_v += list(v)
    
        # Close the file
        dce_cdf.close()
        scpot_cdf.close()
        print('  ' + dce_files[ifile])
        print('  ' + scpot_files[ifile])
    
    # Convert back to numpy arrays
    edp_t = np.array(edp_t)
    edp_e = np.array(edp_e)
    edp_v = np.array(edp_v)
    
    
    ## Data Frame
    
    # Create a dictionary
    edp_data = {
        'Time' :  edp_t,
        'Ex' : edp_e[:,0],
        'Ey' : edp_e[:,1],
        'Ez' : edp_e[:,2],
        'scpot' : edp_v
    }
    
    # Convert dictionary to data from
    edp_data = pd.DataFrame(edp_data, columns=edp_data.keys())
    
    ## FPI
    # Next, we will repeat the process for the Fast Plasma Instrument (FPI), which consists of the Dual Electron
    # Spectrometer (DES) and the Dual Ion Spectrometer (DIS). These measure characteristics of the electron and
    # ion plasmas, respectively. Here, we are interested in the density, velocity, and temperature.
    
    # Normally, survey mode files are a combination of fast and slow survey data and span an entire day. Because FPI
    # produces so much data, however, it is only operated in fast survey mode and its "daily files" are broken up
    # into several files of shorter time intervals.
    
    
    # DIS: Download and Read
    
    # Update instrument-specific variables
    dis_instr = 'fpi'
    dis_mode = 'fast'
    
    if level == 'sitl':
        dis_coords = 'dbcs'
        dis_level = 'ql'
        dis_optdesc = 'dis'
    else:
        dis_coords = 'gse'
        dis_level = level
        dis_optdesc = 'dis-moms'
    
    # Set attributes
    sdc.instr = dis_instr
    sdc.mode = dis_mode
    sdc.level = dis_level
    sdc.optdesc = dis_optdesc
    
    # DIS variable names
    n_vname = '_'.join((sc, 'dis', 'numberdensity', dis_mode))
    v_vname = '_'.join((sc, 'dis', 'bulkv', dis_coords, dis_mode))
    t_para_vname = '_'.join((sc, 'dis', 'temppara', dis_mode))
    t_perp_vname = '_'.join((sc, 'dis', 'tempperp', dis_mode))
    espec_vname = '_'.join((sc, 'dis', 'energyspectr', 'omni', dis_mode))
    
    # Open the file
    files = sdc.Download()
    files = mms_utils.sort_files(files)[0]
    
    # Read the data
    dis_t = []
    dis_n = []
    dis_v = []
    dis_temp_para = []
    dis_temp_perp = []
    dis_espec = []
    dis_e = []
    print('DIS Files:')
    for file in files:
        # Open the file
        cdf = pycdf.CDF(file)
    
        # Read timee and shift to center of interval
        #   - There must be a bug in the CDF package because the Epoch_plus_var variables
        #     are read as empty but really contain scalar values
        t = cdf[cdf[n_vname].attrs['DEPEND_0']][:]
    #    dt_minus = t.attrs['DELTA_MINUS_VAR']
    #    dt_plus = t.attrs['DELTA_PLUS_VAR']
        dt_minus = 0
        dt_plus = 4.5
        t += dt.timedelta(seconds=(dt_plus - dt_minus) / 2.0)
    
        # Read the data
        #   - Convert numpy arrays to lists to make appending easier
        dis_t += list(t)
        dis_n += list(cdf[n_vname][:])
        dis_v += list(cdf[v_vname][:])
        dis_temp_para += list(cdf[t_para_vname][:])
        dis_temp_perp += list(cdf[t_perp_vname][:])
        dis_espec += list(cdf[espec_vname][:])
        dis_e += list(cdf[cdf[espec_vname].attrs['DEPEND_1']][:])
    
        # Close the file
        cdf.close()
        print('  ' + file)
    
    # Convert back to numpy arrays
    dis_t = np.array(dis_t)
    dis_n = np.array(dis_n)
    dis_v = np.array(dis_v)
    dis_temp_para = np.array(dis_temp_para)
    dis_temp_perp = np.array(dis_temp_perp)
    dis_espec = np.array(dis_espec)
    dis_e = np.array(dis_e)
    
    # Compute velocity magnitude
    dis_vmag = np.sqrt(dis_v[:,0]**2.0 + dis_v[:,1]**2.0 + dis_v[:,2]**2.0)
    
    # Compute scalar temperature
    dis_temp = 1.0/3.0 * (2.0*dis_temp_perp + dis_temp_para)
    
    ## Data Frame
    
    # Create a dictionary
    dis_data = pd.DataFrame()
    dis_data['Time'] = pd.Series(dis_t)
    dis_data['N'] = pd.Series(dis_t)
    dis_data['Vx'] = pd.Series(dis_t)
    dis_data['Vy'] = pd.Series(dis_t)
    dis_data['Vz'] = pd.Series(dis_t)
    dis_data['Vmag'] = pd.Series(dis_t)
    dis_data['Tpara'] = pd.Series(dis_t)
    dis_data['Tperp'] = pd.Series(dis_t)
    dis_data['T'] = pd.Series(dis_temp)
    dis_data = {
        'Time' :  dis_t,
        'N' : dis_n,
        'Vx' : dis_v[:,0],
        'Vy' : dis_v[:,1],
        'Vz' : dis_v[:,2],
        'Vmag' : dis_vmag,
        'Tpara' : dis_temp_para,
        'Tperp' : dis_temp_perp,
        'T' : dis_temp
    #    'ESpec': dis_espec,
    #    'Energy': dis_e,
    }
    
    # Convert dictionary to data from
    dis_data = pd.DataFrame(dis_data, columns=dis_data.keys())
    
    # Add dis_espec to dis_data
    for i in range(dis_espec.shape[1]):
        dis_data['ESpec_E{:02}'.format(i)] = Series(data=dis_espec[:,i])
    
    
    ## DES
    
    # Update instrument-specific variables
    des_instr = 'fpi'
    des_mode = 'fast'
    
    if level == 'sitl':
        des_coords = 'dbcs'
        des_level = 'ql'
        des_optdesc = 'des'
    else:
        des_coords = 'gse'
        des_level = level
        des_optdesc = 'des-moms'
    
    # Set attributes
    sdc.instr = des_instr
    sdc.mode = des_mode
    sdc.level = des_level
    sdc.optdesc = des_optdesc
    
    # DIS variable names
    n_vname = '_'.join((sc, 'des', 'numberdensity', des_mode))
    v_vname = '_'.join((sc, 'des', 'bulkv', des_coords, des_mode))
    t_para_vname = '_'.join((sc, 'des', 'temppara', des_mode))
    t_perp_vname = '_'.join((sc, 'des', 'tempperp', des_mode))
    espec_vname = '_'.join((sc, 'des', 'energyspectr', 'omni', des_mode))
    pad_low_vname = '_'.join((sc, 'des', 'pitchangdist', 'lowen', des_mode))
    pad_mid_vname = '_'.join((sc, 'des', 'pitchangdist', 'miden', des_mode))
    pad_high_vname = '_'.join((sc, 'des', 'pitchangdist', 'highen', des_mode))
    
    
    # Open the file
    files = sdc.Download()
    files = mms_utils.sort_files(files)[0]
    
    # Read the data
    des_t = []
    des_n = []
    des_v = []
    des_temp_para = []
    des_temp_perp = []
    des_espec = []
    des_energy = []
    des_pad_low = []
    des_pad_mid = []
    des_pad_high = []
    des_pa = []
    print('DES Files:')
    for file in files:
        # Open the file
        cdf = pycdf.CDF(file)
    
        # Read timee and shift to center of interval
        #   - There must be a bug in the CDF package because the Epoch_plus_var variables
        #     are read as empty but really contain scalar values
        t = cdf[cdf[n_vname].attrs['DEPEND_0']][:]
    #    dt_minus = t.attrs['DELTA_MINUS_VAR']
    #    dt_plus = t.attrs['DELTA_PLUS_VAR']
        dt_minus = 0
        dt_plus = 4.5
        t += dt.timedelta(seconds=(dt_plus - dt_minus) / 2.0)
    
        # Read the data
        des_t += list(t)
        des_n += list(cdf[n_vname][:])
        des_v += list(cdf[v_vname][:])
        des_temp_para += list(cdf[t_para_vname][:])
        des_temp_perp += list(cdf[t_perp_vname][:])
        des_espec += list(cdf[espec_vname][:])
        des_energy += list(cdf[cdf[espec_vname].attrs['DEPEND_1']][:])
        des_pad_low += list(cdf[pad_low_vname][:])
        des_pad_mid += list(cdf[pad_mid_vname][:])
        des_pad_high += list(cdf[pad_high_vname][:])
        des_pa += list(cdf[cdf[pad_low_vname].attrs['DEPEND_1']][:])
    
        # Close the file
        cdf.close()
        print('  ' + file)
    
    # Convert back to numpy arrays
    des_t = np.array(des_t)
    des_n = np.array(des_n)
    des_v = np.array(des_v)
    des_temp_para = np.array(des_temp_para)
    des_temp_perp = np.array(des_temp_perp)
    des_espec = np.array(des_espec)
    des_energy = np.array(des_energy)
    des_pad_low = np.array(des_pad_low)
    des_pad_mid = np.array(des_pad_mid)
    des_pad_high = np.array(des_pad_high)
    des_pa = np.array(des_pa)
    
    # Compute velocity magnitude
    des_vmag = np.sqrt(des_v[:,0]**2.0 + des_v[:,1]**2.0 + des_v[:,2]**2.0)
    
    # Compute scalar temperature
    des_temp = 1.0/3.0*(2.0*des_temp_perp + des_temp_para)
    
    # Compute pich angle distribution
    des_pad = (des_pad_low + des_pad_mid + des_pad_high) / 3.0
    
    # Create a dictionary
    des_data = {
        'Time' :  des_t,
        'N' : des_n,
        'Vx' : des_v[:,0],
        'Vy' : des_v[:,1],
        'Vz' : des_v[:,2],
        'Vmag' : des_vmag,
        'Tpara' : des_temp_para,
        'Tperp' : des_temp_perp,
        'T' : des_temp
    #    'ESpec': dis_espec,
    #    'Energy': dis_e,
    #    'PAD': des_pad,
    #    'PA': des_pa
    }
    
    # Convert dictionary to data from
    des_data = pd.DataFrame(des_data, columns=des_data.keys())
    
    # Add des_espec to des_data
    for i in range(des_espec.shape[1]):
        des_data['ESpec_E{:02}'.format(i)] = Series(data=des_espec[:,i])
    
    # EDI - Should work, but is disabled until verified
    
    # Update instrument-specific variables
    edi_instr = 'edi'
    edi_mode = 'srvy'
    edi_optdesc = None   # Get whatever is available
    
    if level == 'sitl':
        edi_level = 'ql'
    else:
        edi_level = level
    
    # Set attributes
    sdc.instr = edi_instr
    sdc.mode = edi_mode
    sdc.level = edi_level
    sdc.optdesc = 'amb'
    
    # Figure out which data product is available
    files = sdc.FileNames()
    parts = mms_utils.parse_filename(files)
    edi_optdesc = [p[4] for p in parts]
    
    # EDI variable names
    cts1_0_vname = '_'.join((sc, edi_instr, 'flux1', '0', edi_mode, 'l2'))
    cts1_180_vname = '_'.join((sc, edi_instr, 'flux1', '180', edi_mode, 'l2'))
    
    # Open the file
    files = sdc.Download()
    files = mms_utils.sort_files(files)[0]
    
    # Read the data
    edi_t = []
    edi_cts1_0 = []
    edi_cts1_180 = []
    
    print('EDI Files:')
    for file in files:
        # Open the file
        cdf = pycdf.CDF(file)
    
        # Read the datafi
        edi_t += list(cdf[cdf[cts1_0_vname].attrs['DEPEND_0']][:])
        edi_cts1_0 += list(cdf[cts1_0_vname][:])
        edi_cts1_180 += list(cdf[cts1_180_vname][:])
    
        # Close the file
        cdf.close()
        print('  ' + file)
    
    # Convert back to numpy arrays
    edi_t = np.array(edi_t)
    edi_cts1_0 = np.array(edi_cts1_0)
    edi_cts1_180 = np.array(edi_cts1_180)
    
    #TODO: This is disabled for now, because the data is not a scalar
    
    # Create a dictionary
    edi_data = {
        'Time' : edi_t,
        'cts1_0' : edi_cts1_0,
        'cts1_180' : edi_cts1_180
    }
    
    #Convert dictionary to data from
    #print(type(edi_data))
    #print(edi_data.keys())
    #print(edi_data['Time'].shape)
    edi_data = pd.DataFrame(edi_data, columns=edi_data.keys())
    
    ## Interpolate All Values to `t_des`
    # In this step, we need to get all variables into the same time basis. We will interpolate data
    # from FGM and DIS onto the time tags of DES.
    
    # Convert datetime objects to floats
    des_t_stamp = [t.timestamp() for t in des_t]
    fgm_t_stamp = [t.timestamp() for t in fgm_t]
    dis_t_stamp = [t.timestamp() for t in dis_t]
    edp_t_stamp = [t.timestamp() for t in edp_t]
    edi_t_stamp = [t.timestamp() for t in edi_t]
    
    # Interpolate FGM data
    #   - An Nx4 array, ordered as (Bx, By, Bz, |B|)
    nTimes = len(des_t_stamp)
    nComps = np.size(fgm_b, 1)
    fgm_b_interp = np.zeros([nTimes, nComps], dtype=float)
    for idx in range(nComps):
        fgm_b_interp[:,idx] = np.interp(des_t_stamp, fgm_t_stamp, fgm_b[:,idx])
    fgm_clock_angle_interp = np.interp(des_t_stamp, fgm_t_stamp, fgm_ca)
    fgm_normal_angle_interp  = np.interp(des_t_stamp, fgm_t_stamp, fgm_tbn)
    
    # Interpolate DIS data
    dis_n_interp = np.interp(des_t_stamp, dis_t_stamp, dis_n)
    dis_temp_para_interp = np.interp(des_t_stamp, dis_t_stamp, dis_temp_para)
    dis_temp_perp_interp = np.interp(des_t_stamp, dis_t_stamp, dis_temp_perp)
    dis_temp_interp = np.interp(des_t_stamp, dis_t_stamp, dis_temp)
    # An Nx3 array, ordered as (Vx, Vy, Vz)
    nComps = np.size(dis_v, 1)
    dis_v_interp = np.zeros([nTimes, nComps])
    for idx in range(nComps):
        dis_v_interp[:,idx] = np.interp(des_t_stamp, dis_t_stamp, dis_v[:,idx])
    # An Nx32 array, ordered as (ESpec_00, ESpec_01, ... , ESpec_30, ESpec_31)
    nComps = np.size(dis_espec, 1)
    dis_espec_interp = np.zeros([nTimes, nComps])
    for idx in range(nComps):
        dis_espec_interp[:, idx] = np.interp(des_t_stamp, dis_t_stamp, dis_espec[:, idx])
    
    # Interpolate EDP data
    # An Nx3 array, ordered as (Ex, Ey, Ez)
    nComps = np.size(edp_e, 1)
    edp_e_interp = np.zeros([nTimes, nComps])
    for idx in range(nComps):
        edp_e_interp[:,idx] = np.interp(des_t_stamp, edp_t_stamp, edp_e[:,idx])
    edp_scpot_interp = np.interp(des_t_stamp, edp_t_stamp, edp_v)
        
    # Interpolate EDI data
    edi_cts1_0_interp = np.interp(des_t_stamp, edi_t_stamp, edi_cts1_0)
    edi_cts1_180_interp = np.interp(des_t_stamp, edi_t_stamp, edi_cts1_180)
    
    # Print results
    #print('Time:                   ', np.shape(des_t), des_t.dtype)
    #print('DES Density:            ', np.shape(des_n), des_n.dtype)
    #print('DES Velocity:           ', np.shape(des_v), des_v.dtype)
    #print('DES Temperature (para): ', np.shape(des_temp_para), des_temp_para.dtype)
    #print('DES Temperature (perp): ', np.shape(des_temp_perp), des_temp_perp.dtype)
    #print('FGM Magnetic Field:     ', np.shape(fgm_b_interp), fgm_b_interp.dtype)
    #print('DIS Density:            ', np.shape(dis_n_interp), dis_n_interp.dtype)
    #print('DIS Velocity:           ', np.shape(dis_v_interp), dis_v_interp.dtype)
    #print('DIS Temperature (para): ', np.shape(dis_temp_para_interp), dis_temp_para_interp.dtype)
    #print('DIS Temperature (perp): ', np.shape(dis_temp_perp_interp), dis_temp_perp_interp.dtype)
    
    ## Write a CSV file
    # Open file and write data
    data = {
        'Time'   :  des_t,
        'DES N'  : des_n,
        'DES Vx' : des_v[:,0],
        'DES Vy' : des_v[:,1],
        'DES Vz' : des_v[:,2],
        'DES T_para' : des_temp_para,
        'DES T_perp' : des_temp_perp,
        'FGM Bx' : fgm_b_interp[:,0],
        'FGM By' : fgm_b_interp[:,1],
        'FGM Bz' : fgm_b_interp[:,2],
        'FGM Bt' : fgm_b_interp[:,3],
        'FGM Clock_angle'  : fgm_clock_angle_interp, # Needed?
        'FGM Normal_angle' : fgm_normal_angle_interp, # Needed?
        'DIS N'  : dis_n_interp,
        'DIS Vx' : dis_v_interp[:,0],
        'DIS Vy' : dis_v_interp[:,1],
        'DIS Vz' : dis_v_interp[:,2],
        'DIS T_para' : dis_temp_para_interp,
        'DIS T_perp' : dis_temp_perp_interp,
        'DIS Temp'   : dis_temp_interp, # Needed?
        'EDP Ex' : edp_e_interp[:,0],
        'EDP Ey' : edp_e_interp[:,1],
        'EDP Ez' : edp_e_interp[:,2],
        'EDP Scpot'  : edp_scpot_interp, # Needed?
        'EDI cts1_0' : edi_cts1_0_interp,
        'EDI cts1_180'     : edi_cts1_180_interp 
    }
    
    # Add des_espec data en masse
    for col in range(np.size(des_espec, 1)):
        data['DES ESpec_{0:02d}'.format(col)] = des_espec[:,col]
    
    # Add dis_espec_interp data en masse
    for col in range(np.size(dis_espec_interp, 1)):
        data['DIS ESpec_{0:02d}'.format(col)] = dis_espec_interp[:,col]
    
    # Create a data frame
    data = pd.DataFrame(data, columns=data.keys())
    
    return data
    
#    Instrument-specific output disabled for mass data download
#    des_data.to_csv("~/data/des_output.csv", index=False)
#    dis_data.to_csv("~/data/dis_output.csv", index=False)
#    fgm_data.to_csv("~/data/fgm_output.csv", index=False)
#    edp_data.to_csv("~/data/edp_output.csv", index=False)
#    edi_data.to_csv("~/data/edi_output.csv", index=False)

# def checkTableExists(table_name):
#     check_table = "SELECT name FROM sqlite_master WHERE type='table' AND name='{{}}'".format(table_name)
#     c.execute(check_table)
#     result = c.fetchone()[0]
#     if result: return True;
#     else: return False;

def getRowVals(rownum, data, spacecraft):
    rowvals = "'" + str(data.loc[[rownum]].values.tolist()[0][0]) + "',"
    rowvals += "'" + spacecraft + "', "
    rowvals += ', '.join([ str(f) for f in data.loc[[rownum]].values.tolist()[0][1:]]);
    return rowvals

def createTable(spacecraft, level, start_date, end_date, table_name, data, drop = False):
    """
    Uses a global variable `c` for the SQL connection

    @param drop Whether to drop the table if it already exists
    """
    global c
    columns = list(data)
    
    # Check if table exists in DB already
    if drop:
        drop_table = 'DROP TABLE if exists mms1'
        c.execute(drop_table)
    
    # attempt to create the table. Fails if the table already exists
    try:
        # Create a new table with columns from the dataframe  
        create_table = 'CREATE TABLE ' +  table_name + ' ('
    
        # Create string containing column names and types
        colnames = 'Time TIMESTAMP, Spacecraft STRING , '
        for colname in columns[1:len(columns)-1]:
            colnames += colname.replace(' ', '_') + ' REAL, '
        colnames += columns[len(columns)-1].replace(' ', '_') + ' REAL'
        create_table += colnames + ');'
    
        c.execute(create_table)

        # Set indices
        set_primary_index = 'CREATE UNIQUE INDEX Time ON mms1(Time)'
        set_secondary_index = 'CREATE INDEX Spacecraft ON mms1(Spacecraft)'
        c.execute(set_primary_index)
        c.execute(set_secondary_index)
    except sqlite3.OperationalError as error:
        print("Warning: Creating the table failed with an error: " + format(error))
        pass
    

def insertRows(spacecraft, data):
    """ 
    Insert rows of dataframe to table
    """
    global c # TODO: rename to something more specific
    for rownum in range(1,len(data)):
        try:
            insert_row = ('INSERT INTO mms1 VALUES({});'.format(getRowVals(rownum, data, spacecraft)))
            # replace NaN values; does not seem to work
            insert_row.replace('nan', 'NULL')
            c.execute(insert_row)
        except sqlite3.OperationalError as e:
            print("Warning: error saving row: " + format(e))

def run(spacecraft, level, start_date, end_date, data):
    
    # Create a table and insert rows from the associated .csv
    #file_path = '/home/colin/pymms/sql' + '_'.join([spacecraft, level, start_date, 'to']) + end_date + '.csv'   
    createTable(spacecraft, level, start_date, end_date, 'mms1', data)
    insertRows('mms1', data) 
    
    connection.commit()
    connection.close()
    
if __name__ == "__main__":
    # Open connection to SQLite DB
    sqlite_file = '/data/mms/alldata.db'
    connection = sqlite3.connect(sqlite_file)
    c = connection.cursor()

    spacecraft = 'mms1'
    level = 'l2'
    start_date = '2015-12-07'
    end_date = '2015-12-31T23:59:59'
    data_download_path = '/data/mms/'

    data = data_export(spacecraft, level, start_date, end_date, data_download_path)
    run(spacecraft, level, start_date, end_date, data)
