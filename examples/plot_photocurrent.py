import numpy as np
import xarray as xr
from scipy import constants
from scipy.stats import binned_statistic
import datetime as dt
from pymms.sdc import mrmms_sdc_api as api
from pymms.data import edp, fpi, util as mms_util
from matplotlib import pyplot as plt, dates as mdates
import util

# Event parameters
sc = 'mms1'
mode = 'fast'
t0 = dt.datetime(2016, 11, 5, 20, 0, 0)
t1 = dt.datetime(2016, 11, 6, 0, 0, 0) - dt.timedelta(microseconds=1)

# Determine the start and end times of the orbit during which the event occurs
orbit = api.mission_events('orbit', t0-dt.timedelta(hours=20), t0+dt.timedelta(days=2), sc=sc)
orbit_num = orbit['start_orbit'][0]
orbit_t0 = orbit['tstart'][0]
orbit_t1 = orbit['tend'][0]

# Sunlit area of the spacecraft
#   - Roberts JGR 2012 https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JA027854
A_sunlit = 34 # m^2

# Physical constants and conversions
eV2K = constants.value('electron volt-kelvin relationship')
me = constants.m_e # kg
e = constants.e # C
kB   = constants.k # J/K

def electron_current(n, T, V):
    
    Ie = (A_sunlit * e / np.sqrt(2 * np.pi * me) 
          * n * np.sqrt(kB * T * eV2K) 
          * (1 + e*V/(kB*T*eV2K))
          )
    return Ie

def plot_photocurrent():
    
    # Download the data
    scpot = edp.load_scpot(sc=sc, mode=mode,
                           start_date=orbit_t0, end_date=orbit_t1)
    dis_moms = fpi.load_moms(sc=sc, mode=mode, optdesc='dis-moms',
                             start_date=orbit_t0, end_date=orbit_t1)
    des_moms = fpi.load_moms(sc=sc, mode=mode, optdesc='des-moms',
                             start_date=orbit_t0, end_date=orbit_t1)
    aspoc = mms_util.load_data(sc=sc, instr='aspoc', mode='srvy', level='l2',
                               start_date=orbit_t0, end_date=orbit_t1)
    
    # Bin data into FPI time stamps
    t_bins = np.append(dis_moms['time'].data
                       - dis_moms['Epoch_minus_var'].data,
                       dis_moms['time'].data[-1]
                       + dis_moms['Epoch_plus_var'].data)
    Vsc, edges, binnum = binned_statistic(scpot['time'].astype('i8'),
                                          scpot['Vsc'],
                                          statistic='mean',
                                          bins=t_bins.astype('i8'))
    
    asp1, edges, binnum = binned_statistic(aspoc['Epoch'].astype('i8'),
                                           aspoc[sc+'_asp1_ionc'],
                                           statistic='mean',
                                           bins=t_bins.astype('i8'))
    
    idx = 0
    data_bin = []
    asp2 = np.empty_like(asp1)
    asp_on = np.zeros(asp1.shape, dtype='?')
    ref_idx = binnum[0]
    
    for aspoc_idx, bin_idx in enumerate(binnum):
        if (bin_idx == 0) | (bin_idx == len(aspoc[sc+'_asp1_ionc'])):
            continue
        elif ref_idx == 0:
            ref_idx = bin_idx
        
        if bin_idx == ref_idx:
            data_bin.append(aspoc_idx)
        else:
            asp2[idx] = aspoc[sc+'_asp2_ionc'][data_bin].mean()
            asp_on[idx] = aspoc[sc+'_aspoc_status'][data_bin,3].max() > 0
            idx += 1
            ref_idx = bin_idx
            data_bin = [aspoc_idx]
    
    Vsc = xr.DataArray(Vsc, dims='time', coords={'time': dis_moms['time']})
    asp = xr.Dataset({'asp1': (['time'], asp1),
                      'asp2': (['time'], asp2),
                      'asp': (['time'], asp1 + asp2)},
                     coords={'time': dis_moms['time']})

    # Electron current
    Ie = electron_current(des_moms['density'], des_moms['t'], Vsc)
    
    # Flag the data
    flag = xr.DataArray(asp_on.astype('int'),
                        dims='time',
                        coords={'time': dis_moms['time']})
    flag += (((Ie < 1e-11) & (Vsc > 8) & (Vsc < 11)) * 2**1)
    flag += (((Vsc > 14.75) & (Ie > 7.9e-12) & (Ie < 2e-11)) * 2**2)
    flag += (((Vsc > 12.6) & (Vsc <= 14.75) & (Ie > 9.2e-12) & (Ie < 2e-11)) * 2**2)
    
    #
    # Fit the data
    #
    
    # Ie = Iph0 exp(-Vsc/Vph0)
    # y = a exp(b * x)
    # log(y) = log(a) + b*x
    b, a = np.polyfit(Vsc[flag==0], np.log(Ie[flag==0]), 1, w=np.sqrt(Ie[flag==0]))
    Ie_fit = np.exp(a) * np.exp(b*Vsc[flag==0])
    str_fit = ('Ie = {0:0.3e} exp(-Vsc/{1:0.3e})'
               .format(np.exp(a), 1/np.exp(b)))
    
    # Fit density to the spacecraft potential
    c, b, a = np.polyfit(Vsc[flag==0], des_moms['density'][flag==0], 2)
    Ne_fit2 = c*Vsc[flag==0]**2 + b*Vsc[flag==0] + a
    str_Ne_fit2 = ('Ne = {0:0.3e}Vsc^2 + {1:0.3e}Vsc^1 + {2:0.3e})'
                  .format(c, b, a))
    
    e, d, c, b, a = np.polyfit(Vsc[flag==0], des_moms['density'][flag==0], 4)
    Ne_fit4 = e*Vsc[flag==0]**4 + d*Vsc[flag==0]**3 + c*Vsc[flag==0]**2 + b*Vsc[flag==0] + a
    str_Ne_fit4 = ('Ne = {0:0.3e}Vsc^4 + {1:0.3e}Vsc^3 + {2:0.3e}Vsc^2 + {3:0.3e}Vsc + {4:0.3e})'
                  .format(e, d, c, b, a))
    
    # Fit density to the inverse of the spacecraft potential
    c, b, a = np.polyfit(1/Vsc[flag==0], des_moms['density'][flag==0], 2)
    invV_fit2 = c/Vsc[flag==0]**2 + b/Vsc[flag==0] + a
    str_invV_fit2 = ('Ne = {0:0.3e}/Vsc^2 + {1:0.3e}/Vsc + {2:0.3e})'
                   .format(c, b, a))
    
    e, d, c, b, a = np.polyfit(1/Vsc[flag==0], des_moms['density'][flag==0], 4)
    invV_fit4 = e/Vsc[flag==0]**4 + d/Vsc[flag==0]**3 + c/Vsc[flag==0]**2 + b/Vsc[flag==0] + a
    str_invV_fit4 = ('Ne = {0:0.3e}/Vsc^4 + {1:0.3e}/Vsc^3 + {2:0.3e}/Vsc^2 + {3:0.3e}/Vsc + {4:0.3e})'
                   .format(e, d, c, b, a))
    
    
    #
    # Plot: Time Series
    #
    
    # Plot current-related data
    fig1, axes1 = plt.subplots(nrows=8, ncols=1, sharex=True,
                               figsize=(5, 6.5), squeeze=False)
    plt.subplots_adjust(left=0.17, right=0.85, bottom=0.14, top=0.95)
    
    # Ion energy spectrogram
    nt = dis_moms['omnispectr'].shape[0]
    nE = dis_moms['omnispectr'].shape[1]
    x0 = mdates.date2num(dis_moms['time'])[:, np.newaxis].repeat(nE, axis=1)
    x1 = dis_moms['energy']

    y = np.where(dis_moms['omnispectr'] == 0, 1, dis_moms['omnispectr'])
    y = np.log(y[0:-1,0:-1])

    ax = axes1[0,0]
    im = ax.pcolorfast(x0, x1, y, cmap='nipy_spectral')
    ax.images.append(im)
    ax.set_title(sc.upper())
    ax.set_yscale('log')
    ax.set_ylabel('E (ion)\n(eV)')
    util.format_axes(ax, xaxis='off')
    cb = util.add_colorbar(ax, im)
    cb.set_label('$log_{10}$DEF\nkeV/($cm^{2}$ s sr keV)')
    
    # Electron energy spectrogram
    nt = des_moms['omnispectr'].shape[0]
    nE = des_moms['omnispectr'].shape[1]
    x0 = mdates.date2num(des_moms['time'])[:, np.newaxis].repeat(nE, axis=1)
    x1 = des_moms['energy']

    y = np.where(des_moms['omnispectr'] == 0, 1, des_moms['omnispectr'])
    y = np.log(y[0:-1,0:-1])

    ax = axes1[1,0]
    im = ax.pcolorfast(x0, x1, y, cmap='nipy_spectral')
    ax.images.append(im)
    ax.set_yscale('log')
    ax.set_ylabel('E (e-)\n(eV)')
    util.format_axes(ax, xaxis='off')
    cb = util.add_colorbar(ax, im)
    cb.set_label('$log_{10}$DEF\nkeV/($cm^{2}$ s sr keV)')
    
    # Density
    ax = axes1[2,0]
    l1 = dis_moms['density'].plot(ax=ax, label='$N_{i}$')
    l2 = des_moms['density'].plot(ax=ax, label='$N_{e}$')
    ax.set_title('')
    ax.set_ylabel('N\n($cm^{3}$)')
    ax.set_yscale('log')
    util.format_axes(ax, xaxis='off')
    util.add_legend(ax, [l1[0], l2[0]])
    
    # Temperature
    ax = axes1[3,0]
    l1 = dis_moms['t'].plot(ax=ax, label='$T_{i}$')
    l2 = des_moms['t'].plot(ax=ax, label='$T_{e}$')
    util.format_axes(ax, xaxis='off')
    util.add_legend(ax, [l1[0], l2[0]])
    ax.set_title('')
    ax.set_ylabel('T\n(eV)')
    ax.set_yscale('log')
    
    # Spacecraft potential
    ax = axes1[4,0]
    Vsc[~asp_on].plot(ax=ax)
    Vsc[asp_on].plot(ax=ax, color='red')
    util.format_axes(ax, xaxis='off')
    ax.set_title('')
    ax.set_ylabel('$V_{S/C}$\n(V)')
    
    # Electron current
    ax = axes1[5,0]
    Ie.plot(ax=ax)
    ax.set_title('')
    ax.set_ylabel('$I_{e}$\n($\mu A$)')
    ax.set_yscale('log')
    ax.set_ylim([1e-12, 1e-10])
    util.format_axes(ax, xaxis='off')
    
    # Aspoc Status
    ax = axes1[6,0]
    l1 = asp['asp1'].plot(ax=ax, label='$ASP_{1}$')
    l2 = asp['asp2'].plot(ax=ax, label='$ASP_{2}$')
    l3 = asp['asp'].plot(ax=ax, label='$ASP_{tot}$')
    util.format_axes(ax, xaxis='off')
    util.add_legend(ax, [l1[0], l2[0], l3[0]])
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('$I_{ASPOC}$\n($\mu A$)')
    
    # Flag
    ax = axes1[7,0]
    flag.plot(ax=ax)
    util.format_axes(ax)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('Flag')
    
    #
    # Plot: Photocurrent Fit
    #
    
    # Plot the data
    fig2, axes = plt.subplots(nrows=3, ncols=1, figsize=[5, 5.5], squeeze=False)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.94)

    # I_ph = I_ph0 exp(Vsc/Vph0)
    ax = axes[0,0]
    ax.scatter(Vsc[flag==0], Ie[flag==0], marker='o')
    ax.scatter(Vsc[np.bitwise_and(flag, 2**0)>0], Ie[np.bitwise_and(flag, 2**0)>0],
               marker='o', color='red')
    ax.scatter(Vsc[np.bitwise_and(flag, 2**1)>0], Ie[np.bitwise_and(flag, 2**1)>0],
               marker='o', color='purple')
    ax.scatter(Vsc[np.bitwise_and(flag, 2**2)>0], Ie[np.bitwise_and(flag, 2**2)>0],
               marker='o', color='green')
    ax.plot(Vsc[flag==0], Ie_fit, color='black')
    ax.set_title('Photocurrent Parameters $I_{e} = -I_{ph0} \exp(V_{S/C}/V_{ph0})$')
    ax.set_xlabel('$V_{S/C}$ (V)')
    ax.set_ylabel('$I_{e}$\n($\mu A$)')
    ax.set_yscale('log')
    ax.set_ylim([10**np.floor(np.log10(Ie.min().values)), 10**np.ceil(np.log10(Ie.max().values))])
    ax.text(0.9*ax.get_xlim()[1], 0.7*ax.get_ylim()[1], str_fit, horizontalalignment='right', verticalalignment='bottom', color='black')
    
    # Ne = \Sum a_i * V^i
    ax = axes[1,0]
    ax.scatter(Vsc[flag==0], des_moms['density'][flag==0], marker='o')
    ax.scatter(Vsc[np.bitwise_and(flag, 2**0)>0], des_moms['density'][np.bitwise_and(flag, 2**0)>0], marker='o', color='red')
    ax.scatter(Vsc[np.bitwise_and(flag, 2**1)>0], des_moms['density'][np.bitwise_and(flag, 2**1)>0], marker='o', color='purple')
    ax.scatter(Vsc[np.bitwise_and(flag, 2**2)>0], des_moms['density'][np.bitwise_and(flag, 2**2)>0], marker='o', color='green')
    ax.plot(Vsc[flag==0], Ne_fit2, color='black')
    ax.plot(Vsc[flag==0], Ne_fit4, color='grey')
    ax.set_title('')
    ax.set_xlabel('$V_{S/C}$ (V)')
    ax.set_ylabel('$N_{e}$\n($cm^{-3}$)')
    ax.text(0.98*ax.get_xlim()[1], 0.85*ax.get_ylim()[1], str_Ne_fit2, horizontalalignment='right', verticalalignment='bottom', color='black')
    ax.text(0.98*ax.get_xlim()[1], 0.75*ax.get_ylim()[1], str_Ne_fit4, horizontalalignment='right', verticalalignment='bottom', color='grey')
    
    # Ne = \Sum a_i * V^-i
    ax = axes[2,0]
    ax.scatter(1/Vsc[flag==0], des_moms['density'][flag==0], marker='o')
    ax.scatter(1/Vsc[np.bitwise_and(flag, 2**0)>0], des_moms['density'][np.bitwise_and(flag, 2**0)>0], marker='o', color='red')
    ax.scatter(1/Vsc[np.bitwise_and(flag, 2**1)>0], des_moms['density'][np.bitwise_and(flag, 2**1)>0], marker='o', color='purple')
    ax.scatter(1/Vsc[np.bitwise_and(flag, 2**2)>0], des_moms['density'][np.bitwise_and(flag, 2**2)>0], marker='o', color='green')
    ax.plot(Vsc[flag==0], invV_fit2, color='black')
    ax.plot(Vsc[flag==0], invV_fit4, color='grey')
    ax.set_title('')
    ax.set_xlabel('1/$V_{S/C}$ ($V^{-1}$)')
    ax.set_ylabel('$N_{e}$\n($cm^{-3}$)')
    ax.text(0.98*ax.get_xlim()[1], 0.85*ax.get_ylim()[1], str_invV_fit2, horizontalalignment='right', verticalalignment='bottom', color='black')
    ax.text(0.98*ax.get_xlim()[1], 0.75*ax.get_ylim()[1], str_invV_fit4, horizontalalignment='right', verticalalignment='bottom', color='grey')
    
    return [fig1, fig2], ax


if __name__ == '__main__':
    figs, axes = plot_photocurrent()
    plt.show()