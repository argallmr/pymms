import util
from pymms import config
from pymms.data import fgm, fpi
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from pathlib import Path
from xarray import load_dataset

data_root = Path(config['dropbox_root'])

def maxwellian_lookup_table(sc, mode, species, start_date, end_date,
                            lut_file=None):
    
    instr = 'fpi'
    level = 'l2'
    optdesc = 'd'+species+'s-dist'
    
    # Name of the look-up table
    if lut_file is None:
        lut_file = data_root / '_'.join((sc, instr, mode, level,
                                         optdesc+'lookup-table',
                                         start_date.strftime('%Y%m%d_%H%M%S'),
                                         end_date.strftime('%Y%m%d_%H%M%S')))
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
    if not lut_file.exists():
        N_range = (0.9*N.min().values, 1.1*N.max().values)
        t_range = (0.9*t.min().values, 1.1*t.max().values)
        dims = (10**max(np.floor(np.abs(np.log10(N_range[1] - N_range[0]))), 1),
                10**max(np.floor(np.abs(np.log10(t_range[1] - t_range[0]))), 1))
        fpi.maxwellian_lookup(f[[0], ...], N_range, t_range,
                              dims=dims, fname=lut_file)
    
    # Read the dataset
    lut = load_dataset(lut_file)
    
    # Create a time-series dataset of Maxwellian data by interpolating the
    # lookup-table onto the timestamps of the data
    ds_max = lut.interp({'N_data': N, 't_data': t}, method='linear')
    
    # Find the error in N and T between the Maxwellian and Measured
    # distribution
    N_grid, t_grid = np.meshgrid(lut['N_data'], lut['t_data'], indexing='ij')
    dN = (N_grid - lut['N']) / N_grid * 100.0
    dt = (t_grid - lut['t']) / t_grid * 100.0
    
    # Create the figure
    fig = plt.figure(figsize=(6,8))

    # Error in the look-up table
    ax = fig.add_subplot(521)
    img = dN.T.plot(ax=ax, cmap=cm.get_cmap('nipy_spectral', 20))
    ax.set_title('Error in Maxwellian Distribution')
    ax.set_xlabel('N ($cm^{-3}$)')
    ax.set_ylabel('T (eV)')
    cb = img.colorbar
    cb.set_label('$\Delta N$ (%)')

    ax = fig.add_subplot(522)
    img = dt.T.plot(ax=ax, cmap=cm.get_cmap('nipy_spectral', 20))
    ax.set_title('Error in Maxwellian Distribution')
    ax.set_xlabel('N ($cm^{-3}$)')
    ax.set_ylabel('T (eV)')
    cb = img.colorbar
    cb.set_label('$\Delta T$ (%)')

    # Error in the adjusted look-up table
    dN_max = (N - N_max) / N * 100.0
    dN_adj = (N - ds_max['N']) / N * 100.0
    ax = fig.add_subplot(512)
    dN_max.plot(ax=ax, label='$\Delta N_{Max}$')
    dN_adj.plot(ax=ax, label='$\Delta N_{adj}$')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('$\Delta N$ (%)')
    ax.legend(bbox_to_anchor=(1.05, 1),
              borderaxespad=0.0,
              frameon=False,
              handlelength=0,
              handletextpad=0,
              loc='upper left')

    dt_max = (t - t_max) / t * 100.0
    dt_adj = (t - ds_max['t']) / t * 100.0
    ax = fig.add_subplot(513)
    dt_max.plot(ax=ax, label='$\Delta t_{Max}$')
    dt_adj.plot(ax=ax, label='$\Delta t_{adj}$')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('$\Delta t$ (%)')

    ds_m = (s - s_max) / s * 100.0
    ds_adj = (s - ds_max['s']) / s * 100.0
    ax = fig.add_subplot(514)
    ds_m.plot(ax=ax, label='$\Delta s_{Max}$')
    ds_adj.plot(ax=ax, label='$\Delta s_{adj}$')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('$\Delta$ s (%)')

    dsv_max = (sv - sv_max) / sv * 100.0
    dsv_adj = (sv - ds_max['sv']) / sv * 100.0
    ax = fig.add_subplot(515)
    dsv_max.plot(ax=ax, label='$\Delta sv_{Max}$')
    dsv_adj.plot(ax=ax, label='$\Delta sv_{adj}$')
    ax.set_title('')
    ax.set_ylabel('$\Delta s_{V}$ (%)')

    fig.suptitle('Error in Maxwellian')
    plt.subplots_adjust(left=0.2, right=0.85, top=0.95, hspace=0.4)
    plt.show()
    
#    plt.setp(axes, xlim=xlim)
    return fig, axes


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
                        
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--dir',
                       type=str,
                       help='Path to output destination',
                       )
                        
    group.add_argument('-f', '--filename',
                       type=str,
                       help='Output file name',
                       )
                        
    parser.add_argument('-n', '--no-show',
                        help='Do not show the plot.',
                        action='store_true')
    
    parser.add_argument('-l', '--lookup',
                        type=str,
                        help='Path to Maxwellian look-up table',
                        )

    args = parser.parse_args()
    t0 = dt.datetime.strptime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    t1 = dt.datetime.strptime(args.end_date, '%Y-%m-%dT%H:%M:%S')
    
    fig, axes = maxwellian_lookup_table(args.sc, args.mode, args.species, t0, t1,
                                        lut_file=args.lookup)
    
    # Save to directory
    if args.dir is not None:
        if t0.date() == t1.date():
            fname = '_'.join((args.sc, 'fpi', args.mode, 'l2', 'kinetic-entropy',
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%H%M%S')))
        else:
            fname = '_'.join((args.sc, 'fpi', args.mode, 'l2', 'kinetic-entropy',
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%Y%m%d'), t1.strftime('%H%M%S')))
        plt.savefig(path.join(args.dir, fname + '.png'))
    
    # Save to file
    if args.filename is not None:
        plt.savefig(args.filename)
    
    # Show on screen
    if not args.no_show:
        plt.show()
