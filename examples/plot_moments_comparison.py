import util
from pymms.data import fpi, edp
from matplotlib import pyplot as plt
from matplotlib import dates as mdates


def compare_moments(sc, mode, species, start_date, end_date,
                    scpot_correction=False, ephoto_correction=False):
    '''
    Compare moments derived from the 3D velocity distribution functions
    from three sources: official FPI moments, derived herein, and those
    of an equivalent Maxwellian distribution derived herein.
    
    Parameters
    ----------
    sc : str
        Spacecraft identifier. Choices are ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Data rate mode. Choices are ('fast', 'brst')
    species : str
        Particle species. Choices are ('i', 'e') for ions and electrons,
        respectively
    start_date, end_date : `datetime.datetime`
        Start and end dates and times of the time interval
    scpot_correction : bool
        Apply spacecraft potential correction to the distribution functions.
    ephoto : bool
        Subtract photo-electrons. Applicable to DES data only.
    
    Returns
    -------
    fig : `matplotlib.figure`
        Figure in which graphics are displayed
    ax : list
        List of `matplotlib.pyplot.axes` objects
    '''
    # Read the data
    moms_xr = fpi.load_moms(sc, mode, species, start_date, end_date)
    dist_xr = fpi.load_dist(sc, mode, species, start_date, end_date,
                            ephoto=ephoto_correction)
    
    # Spacecraft potential correction
    scpot = None
    if scpot_correction:
        edp_mode = mode if mode == 'brst' else 'fast'
        scpot = edp.load_scpot(sc, edp_mode, start_date, end_date)
        scpot = scpot.interp_like(moms_xr, method='nearest')
    
    # Create an equivalent Maxwellian distribution
    max_xr = fpi.maxwellian_distribution(dist_xr,
                                         moms_xr['density'],
                                         moms_xr['velocity'],
                                         moms_xr['t'])
    
    # Density
    ni_xr = fpi.density(dist_xr, scpot=scpot)
    ni_max_dist = fpi.density(max_xr, scpot=scpot)
    
    # Entropy
    s_xr = fpi.entropy(dist_xr, scpot=scpot)
    s_max_dist = fpi.entropy(max_xr, scpot=scpot)
    s_max = fpi.maxwellian_entropy(moms_xr['density'], moms_xr['p'])
    
    # Velocity
    v_xr = fpi.velocity(dist_xr, N=ni_xr, scpot=scpot)
    v_max_dist = fpi.velocity(max_xr, N=ni_max_dist, scpot=scpot)
    
    # Temperature
    T_xr = fpi.temperature(dist_xr, N=ni_xr, V=v_xr, scpot=scpot)
    T_max_dist = fpi.temperature(max_xr, N=ni_max_dist, V=v_max_dist, scpot=scpot)
    
    # Pressure
    P_xr = fpi.pressure(dist_xr, N=ni_xr, T=T_xr)
    P_max_dist = fpi.pressure(max_xr, N=ni_max_dist, T=T_max_dist)
    
    # Scalar pressure
    p_scalar_xr = (P_xr[:,0,0] + P_xr[:,1,1] + P_xr[:,2,2]) / 3.0
    p_scalar_max_dist = (P_max_dist[:,0,0] + P_max_dist[:,1,1] + P_max_dist[:,2,2]) / 3.0
    
    p_scalar_xr = p_scalar_xr.drop(['t_index_dim1', 't_index_dim2'])
    p_scalar_max_dist = p_scalar_max_dist.drop(['t_index_dim1', 't_index_dim2'])
    
    # Epsilon
    e_xr = fpi.epsilon(dist_xr, dist_max=max_xr, N=ni_xr)
    
    nrows = 6
    ncols = 3
    figsize = (10.0, 5.5)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize, squeeze=False)
    
    '''
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    # Denisty
    ax = axes[0,0]
    lines = []
    moms_xr['density'].plot(ax=ax, label='moms')
    ni_xr.plot(ax=ax, label='dist')
    ni_max_dist.plot(ax=ax, label='max')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel='N\n($cm^{-3}$)'
    
    # Create the legend outside the right-most axes
    leg = ax.legend(bbox_to_anchor=(1.05, 1),
                    borderaxespad=0.0,
                    frameon=False,
                    handlelength=0,
                    handletextpad=0,
                    loc='upper left')
    
    # Color the text the same as the lines
    for line, text in zip(lines, leg.get_texts()):
        text.set_color(line.get_color())
    '''
    
    # Density
    ax = axes[0,0]
    moms_xr['density'].attrs['label'] = 'moms'
    ni_xr.attrs['label'] = 'dist'
    ni_max_dist.attrs['label'] = 'max'
    ax = util.plot([moms_xr['density'], ni_xr, ni_max_dist],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off', ylabel='N\n($cm^{-3}$)')
    
    # Entropy
    ax = axes[1,0]
    ax = util.plot([s_max, s_xr, s_max_dist],
                   ax=ax, labels=['moms', 'dist', 'max dist'],
                   xaxis='off',
                   ylabel='S\n[J/K/$m^{3}$ ln($s^{3}/m^{6}$)]'
                   )
    
    # Epsilon
    ax = axes[1,0].twinx()
    e_xr.plot(ax=ax, color='r')
    ax.spines['right'].set_color('red')
    ax.yaxis.label.set_color('red')
    ax.tick_params(axis='y', colors='red')
    ax.set_title('')
    ax.set_xticks([])
    ax.set_xlabel('')
    ax.set_ylabel('$\epsilon$\n$(s/m)^{3/2}$')
    
    # Vx
    ax = axes[2,0]
    ax = util.plot([moms_xr['velocity'][:,0], v_xr[:,0], v_max_dist[:,0]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off',
                   ylabel='Vx\n(km/s)'
                   )
    
    # Vy
    ax = axes[3,0]
    ax = util.plot([moms_xr['velocity'][:,1], v_xr[:,1], v_max_dist[:,1]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off',
                   ylabel='Vy\n(km/s)'
                   )
    
    # Vz
    ax = axes[4,0]
    ax = util.plot([moms_xr['velocity'][:,2], v_xr[:,2], v_max_dist[:,2]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off',
                   ylabel='Vz\n(km/s)'
                   )
    
    # Scalar Pressure
    ax = axes[5,0]
    ax = util.plot([moms_xr['p'], p_scalar_xr, p_scalar_max_dist],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xlabel='', ylabel='p\n(nPa)'
                   )
    
    # T_xx
    ax = axes[0,1]
    ax = util.plot([moms_xr['temptensor'][:,0,0], T_xr[:,0,0], T_max_dist[:,0,0]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off', ylabel='Txx\n(eV)'
                   )
    
    # T_yy
    ax = axes[1,1]
    ax = util.plot([moms_xr['temptensor'][:,1,1], T_xr[:,1,1], T_max_dist[:,1,1]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off', ylabel='Tyy\n(eV)'
                   )
    
    # T_zz
    ax = axes[2,1]
    ax = util.plot([moms_xr['temptensor'][:,2,2], T_xr[:,2,2], T_max_dist[:,2,2]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off', ylabel='Tzz\n(eV)'
                   )
    
    # T_xy
    ax = axes[3,1]
    ax = util.plot([moms_xr['temptensor'][:,0,1], T_xr[:,0,1], T_max_dist[:,0,1]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off', ylabel='Txy\n(eV)'
                   )
    
    # T_xz
    ax = axes[4,1]
    ax = util.plot([moms_xr['temptensor'][:,0,2], T_xr[:,0,2], T_max_dist[:,0,2]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off', ylabel='Txz\n(eV)'
                   )
    
    # T_yz
    ax = axes[5,1]
    ax = util.plot([moms_xr['temptensor'][:,1,2], T_xr[:,1,2], T_max_dist[:,1,2]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xlabel='', ylabel='Txz\n(eV)'
                   )
    
    # P_xx
    ax = axes[0, 2]
    ax = util.plot([moms_xr['prestensor'][:,0,0], P_xr[:,0,0], P_max_dist[:,0,0]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off', ylabel='Pxx\n(nPa)'
                   )
    
    # P_yy
    ax = axes[1, 2]
    ax = util.plot([moms_xr['prestensor'][:,1,1], P_xr[:,1,1], P_max_dist[:,1,1]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off', ylabel='Pyy\n(nPa)'
                   )
    
    # P_zz
    ax = axes[2, 2]
    ax = util.plot([moms_xr['prestensor'][:,2,2], P_xr[:,2,2], P_max_dist[:,2,2]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off', ylabel='Pzz\n(nPa)'
                   )
    
    # P_xy
    ax = axes[3, 2]
    ax = util.plot([moms_xr['prestensor'][:,0,1], P_xr[:,0,1], P_max_dist[:,0,1]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off', ylabel='Pxy\n(nPa)'
                   )
    
    # P_xz
    ax = axes[4, 2]
    ax = util.plot([moms_xr['prestensor'][:,0,2], P_xr[:,0,2], P_max_dist[:,0,2]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xaxis='off', ylabel='Pxz\n(nPa)'
                   )
    
    # P_yz
    ax = axes[5, 2]
    ax = util.plot([moms_xr['prestensor'][:,1,2], P_xr[:,1,2], P_max_dist[:,1,2]],
                   ax=ax, labels=['moms', 'dist', 'max'],
                   xlabel='', ylabel='Pyz\n(nPa)'
                   )
    
    fig.suptitle('Comparing FPI Moments, Integrated Distribution, Equivalent Maxwellian')
    plt.subplots_adjust(left=0.1, right=0.90, top=0.95, bottom=0.12,
                        hspace=0.3, wspace=0.8)
    return fig, axes


if __name__ == '__main__':
    import argparse
    from os import path
    import datetime as dt
    
    # Define acceptable parameters
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
                        help='Particle species'
                        )
    
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
                        
    parser.add_argument('-V', '--scpot',
                        help='Spacecraft potential correction',
                        action='store_true'
                        )
                        
    parser.add_argument('-P', '--ephoto',
                        help='Photo-electron correction',
                        action='store_true'
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

    # Gather input arguments
    args = parser.parse_args()
    t0 = dt.datetime.strptime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    t1 = dt.datetime.strptime(args.end_date, '%Y-%m-%dT%H:%M:%S')
    
    # Generate the figure
    fig, axes = compare_moments(args.sc, args.mode, args.species, t0, t1,
                                scpot_correction=args.scpot,
                                ephoto_correction=args.ephoto)
    
    # Save to directory
    if args.dir is not None:
        optdesc = 'moms'
        if args.scpot:
            optdesc += '-Vsc'
        if args.ephoto:
            optdesc += '-ephoto'
        
        if t0.date() == t1.date():
            fname = '_'.join((args.sc, 'd'+args.species+'s', args.mode, 'l2',
                              optdesc,
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%H%M%S')))
        else:
            fname = '_'.join((args.sc, 'd'+args.species+'s', args.mode, 'l2',
                              optdesc,
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%Y%m%d'), t1.strftime('%H%M%S')))
        plt.savefig(path.join(args.dir, fname + '.png'))
    
    # Save to file
    if args.filename is not None:
        plt.savefig(args.filename)
    
    # Show on screen
    if not args.no_show:
        plt.show()