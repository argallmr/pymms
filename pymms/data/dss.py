import numpy as np
import xarray as xr
from scipy.stats import binned_statistic
from pymms.data import util

def rename(data, sc, optdesc):
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
    
    if optdesc == '101':
        sunssps_vname = '_'.join((sc, '101', 'sunssps'))
        sunper_vname = '_'.join((sc, '101', 'iifsunper'))
        sunpulse_vname = '_'.join((sc, '101', 'sunpulse'))
        sunpulse_uniq_vname = '_'.join((sc, '101', 'sunpulse', 'uniq'))

        names = {'Epoch': 'time',
                 sunssps_vname: 'flag',
                 sunper_vname: 'period',
                 sunpulse_vname: 'sunpulse',
                 sunpulse_uniq_vname: 'sunpulse_uniq'}

        names = {key:val for key, val in names.items() if key in data}
        data = data.rename(names)
    
    else:
        raise ValueError('Optional descriptor not recognized: "{0}"'
                         .format(optdesc))
    
    return data


def sunpulse2phase(data, time):
    
    if data['flag'].any():
        raise ValueError('Sunpulse flags are set.')
    if len(data['sunpulse']) <= 1:
        raise ValueError('Sunpulse must have more than one value.')
    if time[0] < (data['sunpulse'][0] - data['period'][0]):
        raise ValueError('Data start time {0} is before first sun pulse {1}.'
                         .format(time[0].data, data['sunpulse'].data[0]))
    if time[-1] > (data['sunpulse'][-1] + data['period'][-1]):
        raise ValueError('Data end time {0} is after last sun pulse {1}.'
                         .format(time[-1].data, data['sunpulse'].data[-1]))
    
    
    # Add a pseudo sun pulse before and after the data by using the period
    sunpulse = np.append(np.append(data['sunpulse'][0] - data['period'][0],
                                   data['sunpulse']),
                         data['sunpulse'][-1] + data['period'][-1])
    
    # Calculate the time interval between pulses
    #   - To keep the length of the array the same, add a period to the front
    dpulse = np.append(data['period'][0], np.diff(sunpulse))
    
    # Use the sun pulses to bin the given times
    #   - scipy does not like datetime64 so time has to be converted to floats
    t_ref =  np.min([sunpulse[0].astype('datetime64[D]'),
                     time[0].data.astype('datetime64[D]')])
    t_bins = (sunpulse - t_ref).astype('float')
    time = (time - t_ref).astype('float')
    cts, bin_edges, binnumber = binned_statistic(time, time,
                                                 statistic='count',
                                                 bins=t_bins)
    
    # Determine the nearest sun pulse to each time
    #   - binned_statistic will use 0 for all points before the first sun pulse
    #     and len(time) for all points after the last sun pulse
    #   - The checks at the beginning of the program should ensure that no
    #     points fall in bin zero and only one period worth of points should fall
    #     in bin len(time).
    #   - Given that bin 1 is the first non-extrapolated bin, shift all bins by 1
    #     to convert bins to indices.
    nearest_pulse = (sunpulse[binnumber-1] - t_ref).astype('float')
    
    # Calculate the spin phase
    #   - The amount of time since the last spin pulse
    #   - Normalized by the spin period
    #   - Converted to degrees
    dphase = 360.0 * (time - nearest_pulse) / dpulse[binnumber].astype('float')
    return dphase


def despin(data, time, offset=-76.0, spinup=False):
    '''
    Create a transformation matrix to despin data.
    
    Parameters
    ----------
    data : `xarray.Dataset`
        A dataset containing sunpulse information.
    time : `xarray.DataArray`
        Times at which data to be despun are sampled.
    offset : float
        Angular offset (degrees) of data coordinate system from coordinate
        system of the Digital Sun Sensor. The default is -76 degrees, which is
        the angular offset from BCS to DSS.
    spinup : bool
        If True, the rotation matrices will spin up a dataset.
    
    Returns
    -------
    R : `xarray.DataArray`
        An Nx3x3 set of rotation matrices, where N is the length of `time`.
    '''
    
    # Calculate the spin phase
    phase = np.deg2rad(sunpulse2phase(data, time))
    
    # Reverse the rotation if we are to add spin
    if spinup:
        phase = -phase
    
    # Incorporate the offset from the data coordinate system to DSS
    offset = np.deg2rad(offset)
    sinPhase = np.sin(phase + offset)
    cosPhase = np.cos(phase + offset)
    
    # Rotate about the z-axis
    #                 |  cos  sin  0 |
    #   spun2despun = | -sin  cos  0 |
    #                 |   0    0   1 |
    spun2despun = xr.DataArray(np.zeros((len(time), 3, 3)),
                               dims=['time', 'spun', 'despun'],
                               coords={'time': time,
                                       'spun': ['x', 'y', 'z'],
                                       'despun': ['x', 'y', 'z']})
    spun2despun[:,0,0] = cosPhase
    spun2despun[:,0,1] = sinPhase
    spun2despun[:,1,0] = -sinPhase
    spun2despun[:,1,1] = cosPhase
    spun2despun[:,2,2] = 1
    
    return spun2despun


def load_sunpulse(sc='mms1',
                  start_date=None, end_date=None, rename_vars=True):
    """
    Load Digital Sun Sensor data.
    
    CDF variable names are renamed to something easier to remember and
    use. Original CDF variable names are kept as an attribute "cdf_name"
    in each individual variable.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    rename_vars : bool
        If true (default), rename the standard MMS variable names
        to something more memorable and easier to use.
    
    Returns
    -------
    dist : `xarray.Dataset`
        EDI data.
    """
    instr = 'fields'
    mode = 'hk'
    level = 'l1b'
    optdesc = '101'
    
    sunsps_vname = '_'.join((sc, '101', 'sunssps'))
    sunper_vname = '_'.join((sc, '101', 'iifsunper'))
    sunpulse_vname = '_'.join((sc, '101', 'sunpulse'))
    sunpulse_uniq_vname = '_'.join((sc, '101', 'sunpulse', 'uniq'))
    
    # Load the data
    data = util.load_data(sc=sc, instr=instr, mode=mode, level=level,
                          optdesc=optdesc, start_date=start_date, 
                          end_date=end_date, team_site=True, data_type='hk',
                          variables=[sunsps_vname, sunper_vname,
                                     sunpulse_vname, sunpulse_uniq_vname])
    
    # Convert the sun pulse period to a timedelta64
    data[sunper_vname] = data[sunper_vname].astype('timedelta64[us]')
    
    # Select only the unique sun pulse times
    #  - Spin period is 20 sec
    #  - HK is reported once every 10 sec
    attrs = data.attrs
    pulse, idx = np.unique(data[sunpulse_vname], return_index=True, axis=0)
    data = data.isel(Epoch=idx)
    
    # Rename data variables to something simpler
    if rename_vars:
        data = rename(data, sc, optdesc)
    
    # Add data descriptors to attributes
    data.attrs['sc'] = sc
    data.attrs['instr'] = instr
    data.attrs['mode'] = mode
    data.attrs['level'] = level
    data.attrs['optdesc'] = optdesc
    
    return data


if __name__ == '__main__':
    import datetime as dt
    from pymms.data import fgm, anc
    from matplotlib import pyplot as plt
    
    # Use the Torbert Science (2018) data interval to test
    sc = 'mms1'
    t0 = dt.datetime(2017, 7, 11, 22, 33, 30)
    t1 = dt.datetime(2017, 7, 11, 22, 34, 30)
    
    # Load the data
    #   - Expand the time interval for DSS so the sun pulses encompass the data
    #   - FGM L2Pre has BCS and DBCS data
    dss_data = load_sunpulse(sc=sc,
                             start_date=t0 - dt.timedelta(seconds=40),
                             end_date=t1 + dt.timedelta(seconds=40))
    fgm_data = fgm.load_data(sc=sc, instr='dfg', mode='srvy', level='l2pre',
                             start_date=t0, end_date=t1,
                             coords=('bcs', 'dmpa'), team_site=True)
    
    # Get the major principal axis vector
    anc_data = anc.load_ancillary(sc, 'defatt', t0, t1)
    
    # Rotate to MPA
    mpa = anc_data.attrs['MPA']
    z_bcs2smpa = mpa / np.linalg.norm(np.array(mpa))
    y_bcs2smpa = np.cross(z_bcs2smpa, np.array([1, 0, 0]))
    x_bcs2smpa = np.cross(y_bcs2smpa, z_bcs2smpa)
    bcs2smpa = xr.DataArray(np.stack([x_bcs2smpa, y_bcs2smpa, z_bcs2smpa],
                                      axis=1),
                            dims=['bcs', 'smpa'],
                            coords={'bcs': ['x', 'y', 'z'],
                                    'smpa': ['x', 'y', 'z']})
    
    b_bcs = fgm_data['B_BCS'][:,0:3].rename(b_index='bcs')
    b_smpa = bcs2smpa.dot(b_bcs, dims='bcs')
    
    # Despin
    spun2despun_dss = (despin(dss_data, fgm_data['time'])
                       .rename(spun='smpa', despun='dmpa')
                       )
    spun2despun_anc = (anc.despin(anc_data, fgm_data['time'])
                       .rename(spun='smpa', despun='dmpa')
                       )
                       
    import pdb
    pdb.set_trace()
    
    b_dmpa_dss = spun2despun_dss.dot(b_smpa, dims='smpa')
    b_dmpa_anc = spun2despun_anc.dot(b_smpa, dims='smpa')
    
    # Plot the results
    fig, axes = plt.subplots(nrows=3, ncols=1, squeeze=False)
    plt.subplots_adjust(left=0.15, right=0.82, top=0.95)
    
    # Bx
    ax = axes[0,0]
    fgm_data['B_DMPA'][:,0].plot(ax=ax, label='FGM DMPA')
    b_dmpa_dss[:,0].plot(ax=ax, label='DSS DMPA', linestyle='--')
    b_dmpa_anc[:,0].plot(ax=ax, label='ANC DMPA', linestyle=':')
    ax.set_title('Despinning Data')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$B_{X}$\n[nT]')
    leg = ax.legend(bbox_to_anchor=(1, 1),
                    borderaxespad=0.0,
                    frameon=False,
                    handlelength=0,
                    handletextpad=0,
                    loc='upper left')
    
    # By
    ax = axes[1,0]
    fgm_data['B_DMPA'][:,1].plot(ax=ax, label='FGM DMPA')
    b_dmpa_dss[:,1].plot(ax=ax, label='DSS DMPA', linestyle='--')
    b_dmpa_anc[:,1].plot(ax=ax, label='ANC DMPA', linestyle=':')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$B_{Y}$\n[nT]')
    leg = ax.legend(bbox_to_anchor=(1.01, 1),
                    borderaxespad=0.0,
                    frameon=False,
                    handlelength=0,
                    handletextpad=0,
                    loc='upper left')
    
    # Bz
    ax = axes[2,0]
    fgm_data['B_DMPA'][:,2].plot(ax=ax, label='FGM DMPA')
    b_dmpa_dss[:,2].plot(ax=ax, label='DSS DMPA', linestyle='--')
    b_dmpa_anc[:,2].plot(ax=ax, label='ANC DMPA', linestyle=':')
    ax.set_title('')
    ax.set_ylabel('$B_{Z}$\n[nT]')
    leg = ax.legend(bbox_to_anchor=(1.02, 1),
                    borderaxespad=0.0,
                    frameon=False,
                    handlelength=0,
                    handletextpad=0,
                    loc='upper left')
    
    plt.show()