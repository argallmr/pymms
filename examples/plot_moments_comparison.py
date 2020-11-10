def compare_moments(sc, mode, species, start_date, end_date,
                    scpot_correction=False):
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
    
    Returns
    -------
    fig : `matplotlib.figure`
        Figure in which graphics are displayed
    ax : list
        List of `matplotlib.pyplot.axes` objects
    '''
    # Read the data
    moms_xr = load_moms(sc, mode, species, start_date, end_date)
    dist_xr = load_dist(sc, mode, species, start_date, end_date)
    
    # Spacecraft potential correction
    scpot = None
    if scpot_correction:
        edp_mode = mode if mode == 'brst' else 'fast'
        scpot = edp.load_scpot(sc, edp_mode, start_date, end_date)
        scpot = scpot.interp_like(moms_xr, method='nearest')
    
    # Compute scalar temperature
#    t_scalar = t_scalar.drop_vars(['cart_index_dim1', 'cart_index_dim2'])
    
    # Compute scalar pressure
#    p_scalar = p_scalar.drop_vars(['cart_index_dim1', 'cart_index_dim2'])
    
    # Create an equivalent Maxwellian distribution
    max_xr = maxwellian_distribution(dist_xr,
                                     moms_xr['density'],
                                     moms_xr['velocity'],
                                     moms_xr['t'])
    
    # Density
    ni_xr = density(dist_xr, scpot=scpot)
    ni_max_dist = density(max_xr, scpot=scpot)
    
    # Entropy
    s_xr = entropy(dist_xr, scpot=scpot)
    s_max_dist = entropy(max_xr, scpot=scpot)
    s_max = maxwellian_entropy(moms_xr['density'], moms_xr['p'])
    
    # Velocity
    v_xr = velocity(dist_xr, N=ni_xr, scpot=scpot)
    v_max_dist = velocity(max_xr, N=ni_max_dist, scpot=scpot)
    
    # Temperature
    T_xr = temperature(dist_xr, N=ni_xr, V=v_xr, scpot=scpot)
    T_max_dist = temperature(max_xr, N=ni_max_dist, V=v_max_dist, scpot=scpot)
    
    # Pressure
    P_xr = pressure(dist_xr, N=ni_xr, T=T_xr)
    P_max_dist = pressure(max_xr, N=ni_max_dist, T=T_max_dist)
    
    # Scalar pressure
    p_scalar_xr = (P_xr[:,0,0] + P_xr[:,1,1] + P_xr[:,2,2]) / 3.0
    p_scalar_max_dist = (P_max_dist[:,0,0] + P_max_dist[:,1,1] + P_max_dist[:,2,2]) / 3.0
    
    p_scalar_xr = p_scalar_xr.drop(['t_index_dim1', 't_index_dim2'])
    p_scalar_max_dist = p_scalar_max_dist.drop(['t_index_dim1', 't_index_dim2'])
    
    # Epsilon
    e_xr = epsilon(dist_xr, dist_max=max_xr, N=ni_xr)
    
    nrows = 6
    ncols = 3
    figsize = (10.0, 5.5)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize, squeeze=False)
    
    # Denisty
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
    
    parser_short = argparse.ArgumentParser(
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

    args = parser.parse_args()
    t0 = dt.datetime.strftime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    t1 = dt.datetime.strftime(args.end_date, '%Y-%m-%dT%H:%M:%S')
    
    fig, axes = plot_entropy(args.sc, args.mode, args.species, t0, t1)

    plt.show()