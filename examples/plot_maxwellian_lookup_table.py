import util
from pymms import config
from pymms.data import fgm, fpi
import numpy as np
from matplotlib import pyplot as plt, dates as mdates, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import xarray as xr

data_root = Path(config['dropbox_root'])

def maxwellian_lookup_table(sc, mode, species, start_date, end_date,
                            lut_file=None, minimize='both'):
    '''
    Create Maxwellian look-up table and plot the difference bewtween the
    measured distribution and the equivalent Maxwellian. A Maxwellian look-up
    table consists of a 2D grid of Maxwellian distributions having a range of
    densities and temperatures that span the density and temperature ranges of
    the measured distribution functions. For each Maxwellian distribution in
    the look-up table, we calculate the density and temperature and compare it
    to the measured distribution function, from which the Maxwellians are based.
    
    Parameters
    ----------
    sc : str
        MMS spacecraft identifier: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Data rate mode: ('srvy', 'brst')
    species : str
        Particle species: ('i', 'e') for ion or electron, respectively
    start_date, end_date : `datetime.datetime`
        Start and end times of the data interval
    lut_file : str or path-like
        Path to a previously generated look-up table file
    minimize : str
        The parameter to minimize when selecting the best Maxwellian
        distribution: ('N', 't', 'both')
    '''
    instr = 'fpi'
    level = 'l2'
    optdesc = 'd'+species+'s-dist'
    
    # Name of the look-up table
    if lut_file is None:
        lut_file = data_root / '_'.join((sc, instr, mode, level,
                                         optdesc+'lookup-table',
                                         start_date.strftime('%Y%m%d_%H%M%S'),
                                         end_date.strftime('%Y%m%d_%H%M%S')))
        lut_file = lut_file.with_suffix('.ncdf')
    
    # Ensure it is Path-like
    else:
        lut_file = Path(lut_file).expanduser().absolute()
    
    # Read the data
    fpi_dist = fpi.load_dist(sc=sc, mode=mode, optdesc=optdesc,
                             start_date=start_date, end_date=end_date)
    
    # Precondition the distributions
    fpi_kwargs = fpi.precond_params(sc, mode, level, optdesc,
                                    start_date, end_date,
                                    time=fpi_dist['time'])
    f = fpi.precondition(fpi_dist['dist'], **fpi_kwargs)
    
    # Calculate Moments
    N = fpi.density(f)
    V = fpi.velocity(f, N=N)
    T = fpi.temperature(f, N=N, V=V)
    P = fpi.pressure(f, N=N, T=T)
    s = fpi.entropy(f)
    sv = fpi.vspace_entropy(f, N=N, s=s)
    t = ((T[:,0,0] + T[:,1,1] + T[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    p = ((P[:,0,0] + P[:,1,1] + P[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    s_max_moms = fpi.maxwellian_entropy(N, p)
    
    # Create equivalent Maxwellian distributions and calculate moments
    f_max = fpi.maxwellian_distribution(f, N=N, bulkv=V, T=t)
    N_max = fpi.density(f_max)
    V_max = fpi.velocity(f_max, N=N_max)
    T_max = fpi.temperature(f_max, N=N_max, V=V_max)
    P_max = fpi.pressure(f_max, N=N_max, T=T_max)
    s_max = fpi.entropy(f_max)
    sv_max = fpi.vspace_entropy(f, N=N_max, s=s_max)
    t_max = ((T_max[:,0,0] + T_max[:,1,1] + T_max[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    p_max = ((P_max[:,0,0] + P_max[:,1,1] + P_max[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])

    # Create the lookup table of Maxwellian distributions if it does not exist
    if lut_file.exists():
        print('Using Look-Up Table file:\n\t{0}'.format(lut_file))

    else:
        print('Creating Look-Up Table file:\n\t{0}'.format(lut_file))

        N_range = (0.9*N.min().values, 1.1*N.max().values)
        t_range = (0.9*t.min().values, 1.1*t.max().values)
        dims = (int(10**max(np.floor(np.abs(np.log10(N_range[1] - N_range[0]))), 1)),
                int(10**max(np.floor(np.abs(np.log10(t_range[1] - t_range[0]))), 1)))
        fpi.maxwellian_lookup(f[[0], ...], N_range, t_range,
                              dims=dims, fname=lut_file)
    
    # Read the dataset
    lut = xr.load_dataset(lut_file)
    dims = lut['N'].shape
    
    # Find the error in N and T between the Maxwellian and Measured
    # distribution
    N_grid, t_grid = np.meshgrid(lut['N_data'], lut['t_data'], indexing='ij')
    dN = (N_grid - lut['N']) / N_grid * 100.0
    dt = (t_grid - lut['t']) / t_grid * 100.0
    
    N_lut = xr.zeros_like(N)
    t_lut = xr.zeros_like(t)
    s_lut = xr.zeros_like(s)
    sv_lut = xr.zeros_like(sv)
    if minimize == 'N':
        for idx, dens in enumerate(N):
            imin = np.argmin(np.abs(lut['N'].data - dens.item()))
            irow = imin // dims[1]
            icol = imin % dims[1]
            N_lut[idx] = lut['N'][irow, icol]
            t_lut[idx] = lut['t'][irow, icol]
            s_lut[idx] = lut['s'][irow, icol]
            sv_lut[idx] = lut['sv'][irow, icol]

        
    elif minimize == 't':
        for idx, temp in enumerate(t):
            imin = np.argmin(np.abs(lut['t'].data - temp.item()))
            irow = imin // dims[1]
            icol = imin % dims[1]
            N_lut[idx] = lut['N'][irow, icol]
            t_lut[idx] = lut['t'][irow, icol]
            s_lut[idx] = lut['s'][irow, icol]
            sv_lut[idx] = lut['sv'][irow, icol]
    
    elif minimize == 'both':
        for idx, (dens, temp) in enumerate(zip(N, t)):
            imin = np.argmin(np.sqrt((lut['t'].data - temp.item())**2
                                     + (lut['N'].data - dens.item())**2
                                     ))
            irow = imin // dims[1]
            icol = imin % dims[1]
            N_lut[idx] = lut['N'][irow, icol]
            t_lut[idx] = lut['t'][irow, icol]
            s_lut[idx] = lut['s'][irow, icol]
            sv_lut[idx] = lut['sv'][irow, icol]
    
    # Create a time-series dataset of Maxwellian data by interpolating the
    # lookup-table onto the timestamps of the data
    elif minimize == 'data':
        lut_interp = lut.interp({'N_data': N, 't_data': t}, method='linear')
        N_lut = lut_interp['N']
        t_lut = lut_interp['t']
        s_lut = lut_interp['s']
        sv_lut = lut_interp['sv']
    
    # Create the figure
    fig = plt.figure(figsize=(6,7.5))
    
    # Figure positions
    #   - Start with the position of a plot in the first row of a one-column
    #     set of plots.
    #   - Define the width and height spacing between each row
    #   - The first row will actually be broken into two columns with a gap
    #     between the first and second rows to allow for axis labels
    #   - Subsequent plots will be offset from the first by multiples of
    #     height and hspace
    left, bottom, width, height = 0.15, 0.8, 0.7, 0.14
    wspace, hspace, gap = 0.2, 0.02, 0.07
    row1_width = 0.26

    # Error in the look-up table density
    #   - Error is independent of density, so plot as 1D line plot
    ax = fig.add_subplot(521)
    dN[0,:].plot(ax=ax)
    ax.set_title('')
    ax.set_xlabel('$T_{'+species+'}$ (eV)')
    ax.set_ylabel('$\Delta n_{'+species+'}/n_{'+species+'}$ (%)')
    ax.set_position((left, bottom, row1_width, height))
    util.format_axes(ax, time=False)

    '''
    img = dN.T.plot(ax=ax, cmap=cm.get_cmap('rainbow', 10), add_colorbar=False)
    ax.set_title('')
    ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
    ax.set_ylabel('$T_{'+species+'}$ (eV)')
    
    # Create a colorbar that is aware of the image's new position
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.1)
    fig.add_axes(cax)
    cb = fig.colorbar(img, cax=cax, orientation="vertical")
    cb.set_label('$\Delta n_{'+species+'}/n_{'+species+'}$ (%)')
    cb.ax.minorticks_on()
    '''

    # Error in the look-up table temperature
    #   - Error is independent of density, so plot as 1D line plot
    ax = fig.add_subplot(522)
    dt[0,:].plot(ax=ax)
    ax.set_title('')
    ax.set_xlabel('$T_{'+species+'}$ (eV)')
    ax.set_ylabel('$\Delta T_{'+species+'}/T_{'+species+'}$ (%)')
    ax.set_position((left+row1_width+wspace, bottom, row1_width, height))
    util.format_axes(ax, time=False)
    
    '''
    img = dt.T.plot(ax=ax, cmap=cm.get_cmap('rainbow', 10), add_colorbar=False)
    ax.set_title('')
    ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
    ax.set_ylabel('$T_{'+species+'}$ (eV)')
    ax.set_position((left+row1_width+wspace, bottom, row1_width, height))
    util.format_axes(ax, time=False)
    
    # Create a colorbar that is aware of the image's new position
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.1)
    fig.add_axes(cax)
    cb = fig.colorbar(img, cax=cax, orientation="vertical")
    cb.set_label('$\Delta T_{'+species+'}/T_{'+species+'}$ (%)')
    cb.ax.minorticks_on()
    '''

    # Error in the adjusted look-up table
    dN_max = (N - N_max) / N * 100.0
    dN_adj = (N - N_lut) / N * 100.0
    ax = fig.add_subplot(512)
    l1 = dN_max.plot(ax=ax, label='$\Delta n_{'+species+',Max}/n_{'+species+',Max}$')
    l2 = dN_adj.plot(ax=ax, label='$\Delta n_{'+species+',adj}/n_{'+species+',adj}$')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('$\Delta n_{'+species+'}/n_{'+species+'}$ (%)')
    ax.set_position((left, bottom-gap-height, width, height))
    util.format_axes(ax, xaxis='off')
    util.add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)
    
    # Deviation in temperature
    dt_max = (t - t_max) / t * 100.0
    dt_adj = (t - t_lut) / t * 100.0
    ax = fig.add_subplot(513)
    l1 = dt_max.plot(ax=ax, label='$\Delta T_{'+species+',Max}/T_{'+species+',Max}$')
    l2 = dt_adj.plot(ax=ax, label='$\Delta T_{'+species+',adj}/T_{'+species+',adj}$')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('$\Delta T_{'+species+'}/T_{'+species+'}$ (%)')
    ax.set_ylim(-1,2.5)
    ax.set_position((left, bottom-gap-2*height-hspace, width, height))
    util.format_axes(ax, xaxis='off')
    util.add_legend(ax, [l1[0], l2[0]], corner='NE', horizontal=True)
    
    # Deviation in entropy
    ds_moms = (s - s_max_moms) / s * 100.0
    ds_m = (s - s_max) / s * 100.0
    ds_adj = (s - s_lut) / s * 100.0
    ax = fig.add_subplot(514)
    l1 = ds_m.plot(ax=ax, label='$\Delta s_{'+species+',Max}/s_{'+species+',Max}$')
    l2 = ds_adj.plot(ax=ax, label='$\Delta s_{'+species+',adj}/s_{'+species+',adj}$')
    l3 = ds_moms.plot(ax=ax, label='$\Delta s_{'+species+',moms}/s_{'+species+',moms}$')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('$\Delta s_{'+species+'}/s_{'+species+'}$ (%)')
    ax.set_ylim(-9,2.5)
    ax.set_position((left, bottom-gap-3*height-2*hspace, width, height))
    util.format_axes(ax, xaxis='off')
    util.add_legend(ax, [l1[0], l2[0], l3[0]], corner='SE', horizontal=True)
    
    # Deviation in velocity-space entropy
    dsv_max = (sv - sv_max) / sv * 100.0
    dsv_adj = (sv - sv_lut) / sv * 100.0
    ax = fig.add_subplot(515)
    l1 = dsv_max.plot(ax=ax, label='$\Delta s_{V,'+species+',Max}/s_{V,'+species+',Max}$')
    l2 = dsv_adj.plot(ax=ax, label='$\Delta s_{V,'+species+',adj}/s_{V,'+species+',adj}$')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('$\Delta s_{V,'+species+'}/s_{V,'+species+'}$ (%)')
    ax.set_position((left, bottom-gap-4*height-3*hspace, width, height))
    util.format_axes(ax)
    util.add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)

    fig.suptitle('Maxwellian Look-up Table')
    
    plt.setp(fig.axes[2:], xlim=mdates.date2num([start_date, end_date]))

    return fig, fig.axes


if __name__ == '__main__':
    import argparse
    import datetime as dt
    from os import path
    
    parser = argparse.ArgumentParser(
        description='Plot parameters associated with kinetic entropy.'
        )
    
    parser.add_argument('sc', 
                        type=str,
                        help='Spacecraft Identifier')
    
    parser.add_argument('mode', 
                        type=str,
                        help='Data rate mode')
    
    parser.add_argument('species', 
                        type=str,
                        help='Particle species (i, e)')
    
    parser.add_argument('start_date', 
                        type=str,
                        help='Start date of the data interval: '
                             '"YYYY-MM-DDTHH:MM:SS""'
                        )
    
    parser.add_argument('end_date', 
                        type=str,
                        help='Start date of the data interval: '
                             '"YYYY-MM-DDTHH:MM:SS""'
                        )
    
    parser.add_argument('-l', '--lookup',
                        type=str,
                        help='Path to Maxwellian look-up table'
                        )
                        
    parser.add_argument('-m', '--minimization',
                        type=str,
                        choices=('both', 'N', 't', 'data'),
                        default='both',
                        help='Minimization scheme for look-up table.'
                        )
                        
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--dir',
                       type=str,
                       help='Path to output destination'
                       )
                        
    group.add_argument('-f', '--filename',
                       type=str,
                       help='Output file name'
                       )
                        
    parser.add_argument('-e', '--extension',
                        default='png',
                        type=str,
                        help='Output file type',
                        )
                        
    parser.add_argument('-n', '--no-show',
                        help='Do not show the plot.',
                        action='store_true')

    args = parser.parse_args()
    t0 = dt.datetime.strptime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    t1 = dt.datetime.strptime(args.end_date, '%Y-%m-%dT%H:%M:%S')
    
    fig, axes = maxwellian_lookup_table(args.sc, args.mode, args.species, t0, t1,
                                        lut_file=args.lookup,
                                        minimize=args.minimization)
    
    # Save to directory
    if args.dir is not None:
        optdesc = '-'.join(('lut', 'errors', args.minimization))
        if t0.date() == t1.date():
            fname = '_'.join((args.sc, 'fpi', args.mode, 'l2', optdesc,
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%H%M%S')))
        else:
            fname = '_'.join((args.sc, 'fpi', args.mode, 'l2', optdesc,
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%Y%m%d'), t1.strftime('%H%M%S')))
        plt.savefig(path.join(args.dir, fname + '.' + args.extension))
    
    # Save to file
    if args.filename is not None:
        plt.savefig(args.filename)
    
    # Show on screen
    if not args.no_show:
        plt.show()
