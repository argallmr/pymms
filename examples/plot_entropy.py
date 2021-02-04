import util
from pymms.data import fgm, fpi
import numpy as np
from matplotlib import pyplot as plt

def kinetic_entropy(sc, mode, start_date, end_date, **kwargs):
    
    # Read the data
    b = fgm.load_data(sc, mode, start_date, end_date)
    dis_moms = fpi.load_moms(sc, mode, 'i', start_date, end_date)
    des_moms = fpi.load_moms(sc, mode, 'e', start_date, end_date)
    dis_dist = fpi.load_dist(sc, mode, 'i', start_date, end_date)
    des_dist = fpi.load_dist(sc, mode, 'e', start_date, end_date)
    
    # Equivalent Maxwellian distribution
    dis_max_dist = fpi.maxwellian_distribution(dis_dist,
                                               N=dis_moms['density'],
                                               bulkv=dis_moms['velocity'],
                                               T=dis_moms['t'])
    des_max_dist = fpi.maxwellian_distribution(des_dist,
                                               N=des_moms['density'],
                                               bulkv=des_moms['velocity'],
                                               T=des_moms['t'])
    
    # Entropy density
    si_dist = fpi.entropy(dis_dist, **kwargs)
    se_dist = fpi.entropy(des_dist, **kwargs)
    
#    si_max = fpi.maxwellian_entropy(dis_moms['density'], dis_moms['p'])
#    se_max = fpi.maxwellian_entropy(des_moms['density'], des_moms['p'])
    si_max = fpi.entropy(dis_max_dist, **kwargs)
    se_max = fpi.entropy(des_max_dist, **kwargs)
    
    
    # Velcoity space entropy density
    siv_dist = fpi.vspace_entropy(dis_dist,
                                  N=dis_moms['density'],
                                  s=si_dist,
                                  **kwargs)
    sev_dist = fpi.vspace_entropy(des_dist,
                                  N=des_moms['density'],
                                  s=se_dist,
                                  **kwargs)
    
    siv_max = fpi.vspace_entropy(des_max_dist,
                                 N=dis_moms['density'],
                                 s=si_max,
                                 **kwargs)
    sev_max = fpi.vspace_entropy(des_max_dist,
                                 N=des_moms['density'],
                                 s=se_max,
                                 **kwargs)
    
    # M-bar
    mi_bar = np.abs(si_max - si_dist) / si_max
    me_bar = np.abs(se_max - se_dist) / se_max
    
    miv_bar = np.abs(siv_max - siv_dist) / siv_max
    mev_bar = np.abs(sev_max - sev_dist) / sev_max
    
    # Epsilon
    ei = fpi.epsilon(dis_dist, N=dis_moms['density'],
                               V=dis_moms['velocity'],
                               T=dis_moms['t'],
                               **kwargs)
    ee = fpi.epsilon(des_dist, N=des_moms['density'],
                               V=des_moms['velocity'],
                               T=des_moms['t'],
                               **kwargs)
    
    # Setup the plot
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
    
    # Ion entropy density
    ax = axes[1,0]
    ax = util.plot([si_max, si_dist],
                   ax=ax, labels=['$s_{i,max}$', '$s_{i}$'],
                   xaxis='off', ylabel='s\n$J/K/m^{3}$ $ln(s^{3}/m^{6})$'
                   )
    
    # Electron entropy density
    ax = axes[2,0]
    ax = util.plot([se_max, se_dist],
                   ax=ax, labels=['$s_{e,max}$', '$s_{e}$'],
                   xaxis='off', ylabel='s'
                   )
    
    # Ion M-bar
    ax = axes[3,0]
    ax = util.plot(mi_bar,
                   ax=ax, legend=False,
                   xaxis='off', ylabel='$\overline{M}_{i}$'
                   )
    
    # Ion Epsilon
    ax = ax.twinx()
    ei.plot(ax=ax, color='g')
    ax.set_ylabel('$\epsilon_{i}$\n$(s/m)^{3/2}$')
    ax.yaxis.label.set_color('g')
    ax.tick_params(axis='y', colors='g')
    ax.set_xticklabels([])
    
    # Electron M-bar
    ax = axes[4,0]
    ax = util.plot(me_bar,
                   ax=ax, legend=False,
                   xaxis='off', ylabel='$\overline{M}_{e}$'
                   )
    
    # Electron Epsilon
    ax = ax.twinx()
    ee.plot(ax=ax, color='r')
    ax.set_ylabel('$\epsilon_{e}$\n$(s/m)^{3/2}$')
    ax.yaxis.label.set_color('r')
    ax.tick_params(axis='y', colors='r')
    ax.set_xticklabels([])
    
    # Velocity space ion entropy density
    ax = axes[5,0]
    ax = util.plot([siv_max, siv_dist],
                   ax=ax, labels=['$s_{i,V,max}$', '$s_{i,V}$'],
                   xaxis='off', ylabel='$s_{V}$\n$J/K/m^{3}$ $ln()$'
                   )
    
    # Velocity space electron entropy density
    ax = axes[6,0]
    ax = util.plot([sev_max, sev_dist],
                   ax=ax, labels=['$s_{e,V,max}$', '$s_{e,V}$'],
                   xaxis='off', ylabel='$s_{V}$'
                   )
    
    # Velocity space ion M-bar
    ax = axes[7,0]
    ax = util.plot(miv_bar,
                   ax=ax, legend=False,
                   xaxis='off', ylabel='$\overline{M}_{i,V}$'
                   )
    
    # Velocity space electron M-bar
    ax = axes[8,0]
    ax = util.plot(mev_bar,
                   ax=ax, legend=False,
                   ylabel='$\overline{M}_{e,V}$'
                   )
    
    
    fig.suptitle('Total and Velocity Space Entropy Density')
    plt.subplots_adjust(left=0.2, right=0.85, top=0.95, hspace=0.4)
#    plt.setp(axes, xlim=xlim)
    return fig, axes


if __name__ == '__main__':
    import argparse
    import datetime as dt
    
    parser = argparse.ArgumentParser(
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
    t0 = dt.datetime.strptime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    t1 = dt.datetime.strptime(args.end_date, '%Y-%m-%dT%H:%M:%S')
    
    fig, axes = kinetic_entropy(args.sc, args.mode, t0, t1)

    plt.show()
