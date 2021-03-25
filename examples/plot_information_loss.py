import datetime as dt
from matplotlib import pyplot as plt
from pymms.data import fpi, edp
from scipy import constants
import re

kB = constants.k # J/K


def information_loss(sc, instr, mode, start_date, end_date):

    # Load the data
    fpi_moms = fpi.load_moms(sc=sc, mode=mode, optdesc=instr+'-moms',
                             start_date=start_date, end_date=end_date)
    fpi_dist = fpi.load_dist(sc=sc, mode=mode, optdesc=instr+'-dist',
                             start_date=start_date, end_date=end_date)
    
    # Precondition the distributions
    kwargs = fpi.precond_params(sc, mode, 'l2', instr+'-dist',
                                start_date, end_date,
                                time=fpi_dist['time'])
    f = fpi.precondition(fpi_dist['dist'], **kwargs)
    
    # Calculate moments
    #  - Use calculated moments for the Maxwellian distribution
    N = fpi.density(f)
    V = fpi.velocity(f, N=N)
    T = fpi.temperature(f, N=N, V=V)
    P = fpi.pressure(f, N=N, T=T)
    t = ((T[:,0,0] + T[:,1,1] + T[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    p = ((P[:,0,0] + P[:,1,1] + P[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    f_max = fpi.maxwellian_distribution(f, N=N, bulkv=V, T=t)
    
    # Maxwellian Entropy
    sMm = fpi.maxwellian_entropy(fpi_moms['density'], fpi_moms['p'])
    sMi = fpi.maxwellian_entropy(N, p)
    sMd = fpi.entropy(f_max)
    
    # Velocity space entropy
    #   - There are three options for calculating the v-space entropy of
    #     the Maxwellian distribution: using
    #        1) FPI integrated moments,
    #        2) Custom moments of the measured distribution
    #        3) Custom moments of the equivalent Maxwellian distribution
    #     Because the Maxwellian is built with discrete v-space bins, its
    #     density, velocity, and temperature do not match that of the
    #     measured distribution on which it is based. If NiM is used, the
    #     M-bar term will be negative, which is unphysical, so here we use
    #     the density of the measured distribution and the entropy of the
    #     equivalent Maxwellian.
    s = fpi.entropy(f)
    sV = fpi.vspace_entropy(f, N=N, s=s)
    sVM = fpi.vspace_entropy(f_max, N=N, s=sMd)
    
    MbarKP = 1e-6 * (sMd - s) / (3/2 * kB * N)
    Mbar1 = (sVM - sV) / sVM

    # Calculate information loss
    num, denom = fpi.information_loss(f_max, f, N=N, T=t)
    Mbar2 = (MbarKP - num) / denom
    
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(6, 7), squeeze=False)
    
    # s
    ax = axes[0,0]
    s.plot(ax=ax, label='s')
    sMm.plot(ax=ax, label='$s_{M,moms}$')
    sMi.plot(ax=ax, label='$s_{M,int}$')
    sMd.plot(ax=ax, label='$s_{M,f}$')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('s\n(J/K/$cm^{3}$)')
    ax.set_title('')
    ax.legend()
    
    # sV
    ax = axes[1,0]
    sV.plot(ax=ax, label='$s_{V}$')
    sVM.plot(ax=ax, label='$s_{M,V}$')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$s_{V}$\n(J/K/$cm^{3}$)')
    ax.set_title('')
    ax.legend()

    ax = axes[2,0]
    num.plot(ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('Num')
    ax.set_title('')

    ax = axes[3,0]
    (1/denom).plot(ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('1/Denom')
    ax.set_title('')

    ax = axes[4,0]
    (num/denom).plot(ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('Num/Denom')
    ax.set_title('')

    ax = axes[5,0]
    MbarKP.plot(ax=ax, label='$\overline{M}_{KP}$')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$\overline{M}_{KP}$')
    ax.set_title('')

    ax = axes[6,0]
    Mbar1.plot(ax=ax, label='$\overline{M}_{1}$')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$\overline{M}_{1}$')
    ax.set_title('')

    ax = axes[7,0]
    Mbar2.plot(ax=ax, label='$\overline{M}_{2}$')
    ax.set_ylabel('$\overline{M}_{2}$')
    ax.set_title('')

    fig.suptitle('$\overline{M}_{1} = (s_{V,M} - s_{V})/s_{V,M}$\n'
                 '$\overline{M}_{2} = (\overline{M}_{KP} - Num)/Denom$')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.12, hspace=0.3)

    return fig, axes


if __name__ == '__main__':
    import argparse
    import datetime as dt
    from os import path
    
    # Define acceptable parameters
    parser = argparse.ArgumentParser(
        description=('Plot information loss due to binning of velocity space.')
        )
    
    parser.add_argument('sc', 
                        type=str,
                        help='Spacecraft Identifier')
    
    parser.add_argument('instr', 
                        type=str,
                        help='FPI instrument (dis | des)')
    
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
    fig, axes = information_loss(args.sc, args.instr, args.mode, t0, t1)
    
    # Save to directory
    if args.dir is not None:
        if t0.date() == t1.date():
            fname = '_'.join((args.sc, args.instr, args.mode, 'l2', 'info-loss',
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%H%M%S')))
        else:
            fname = '_'.join((args.sc, args.instr, args.mode, 'l2', 'info-loss',
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%Y%m%d'), t1.strftime('%H%M%S')))
        plt.savefig(path.join(args.dir, fname + '.png'))
    
    # Save to file
    if args.filename is not None:
        plt.savefig(args.filename)
    
    # Show on screen
    if not args.no_show:
        plt.show()
