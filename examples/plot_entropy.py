def kinetic_entropy(sc, mode, start_date, end_date, **kwargs):
    
    # Read the data
    b = fgm.load_data(sc, mode, start_date, end_date)
    dis_moms = load_moms(sc, mode, 'i', start_date, end_date)
    des_moms = load_moms(sc, mode, 'e', start_date, end_date)
    dis_dist = load_dist(sc, mode, 'i', start_date, end_date)
    des_dist = load_dist(sc, mode, 'e', start_date, end_date)
    
    # Entropy
    Si_dist = entropy(dis_dist, **kwargs)
    Se_dist = entropy(des_dist, **kwargs)
    
    Si_moms = maxwellian_entropy(dis_moms['density'], dis_moms['p'])
    Se_moms = maxwellian_entropy(des_moms['density'], des_moms['p'])
    
    # M-bar
    mi_bar = np.abs(Si_dist - Si_moms) / Si_moms
    me_bar = np.abs(Se_dist - Se_moms) / Se_moms
    
    # Epsilon
    ei = epsilon(dis_dist, N=dis_moms['density'],
                           V=dis_moms['velocity'],
                           T=dis_moms['t'],
                           **kwargs)
    ee = epsilon(des_dist, N=des_moms['density'],
                           V=des_moms['velocity'],
                           T=des_moms['t'],
                           **kwargs)
    
    # Anisotropy
    Ai = dis_moms['temppara'] / dis_moms['tempperp'] - 1
    Ae = des_moms['temppara'] / des_moms['tempperp'] - 1
    
    nrows = 9
    ncols = 1
    figsize = (5.5, 7.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize, squeeze=False)
    
    # B
    ax = axes[0,0]
    ax = util.plot([b['B'][:,3], b['B'][:,0], b['B'][:,1], b['B'][:,2]],
                   ax=ax, labels=['|B|', 'Bx', 'By', 'Bz'],
                   xaxis='off', ylabel='B\n(nT)'
                   )
    
    # N
    ax = axes[1,0]
    ax = util.plot([dis_moms['density'], des_moms['density']],
                   ax=ax, labels=['Ni', 'Ne'],
                   xaxis='off', ylabel='N\n($cm^{-3}$)'
                   )
    
    '''
    ax = ax.twinx()
    ei.plot(ax=ax, corlor='r')
    ax.spines['right'].set_color('red')
    ax.set_ylabel('$\epsilon$\n$(s/m)^{3/2}$')
    ax.tick_params(axis='y', colors='red')
    ax.yaxis.label.set_color('red')
    '''
    
    # Ion entropy
    ax = axes[2,0]
    ax = util.plot([Si_moms, Si_dist],
                   ax=ax, labels=['$s_{i,V,max}$', '$s_{i,V}$'],
                   xaxis='off', ylabel='s\n$J/K/m^{3}$ $ln(s^{3}/m^{6})$'
                   )
    
    # Electron entropy
    ax = axes[3,0]
    ax = util.plot([Se_moms, Se_dist],
                   ax=ax, labels=['$s_{e,V,max}$', '$s_{e,V}$'],
                   xaxis='off', ylabel='s'
                   )
    
    '''
    ax = ax.twinx()
    ee.plot(ax=ax, corlor='r')
    ax.spines['right'].set_color('red')
    ax.set_ylabel('$\epsilon$')
    ax.tick_params(axis='y', colors='red')
    ax.yaxis.label.set_color('red')
    '''
    
    # M-bar
    ax = axes[4,0]
    ax = util.plot(mi_bar,
                   ax=ax, legend=False,
                   xaxis='off', ylabel='$\overline{M}_{i}$'
                   )
    
    # Epsilon
    ax = ax.twinx()
    ei.plot(ax=ax, color='g')
    ax.set_ylabel('$\epsilon_{i}$\n$(s/m)^{3/2}$')
    ax.yaxis.label.set_color('g')
    ax.tick_params(axis='y', colors='g')
    ax.set_xticklabels([])
    
    # M-bar
    ax = axes[5,0]
    ax = util.plot(me_bar,
                   ax=ax, legend=False,
                   xaxis='off', ylabel='$\overline{M}_{e}$'
                   )
    
    # Epsilon
    ax = ax.twinx()
    ee.plot(ax=ax, color='r')
    ax.set_ylabel('$\epsilon_{e}$\n$(s/m)^{3/2}$')
    ax.yaxis.label.set_color('r')
    ax.tick_params(axis='y', colors='r')
    ax.set_xticklabels([])
    
    # Ion temperature
    ax = axes[6,0]
    ax = util.plot([dis_moms['temppara'], dis_moms['tempperp'], dis_moms['t']],
                   ax=ax, labels=['$T_{\parallel}$', '$T_{\perp}$, $T$'],
                   xaxis='off', ylabel='$T_{i}$\n(eV)'
                   )
    
    # Electron temperature
    ax = axes[7,0]
    ax = util.plot([des_moms['temppara'], des_moms['tempperp'], des_moms['t']],
                   ax=ax, labels=['$T_{\parallel}$', '$T_{\perp}$, $T$'],
                   xaxis='off', ylabel='$T_{e}$\n(eV)'
                   )
    
    # Anisotropy
    ax = axes[8,0]
    ax = util.plot([Ai, Ae],
                   ax=ax, labels=['$A_{i}$', '$A_{e}$'],
                   ylabel='A'
                   )
    
    fig.suptitle('Plasma Parameters for Kinetic and Boltzmann Entropy')
    plt.subplots_adjust(left=0.2, right=0.85, top=0.95, hspace=0.4)
#    plt.setp(axes, xlim=xlim)
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
    
    fig, axes = plot_entropy(args.sc, args.mode, t0, t1)

    plt.show()
