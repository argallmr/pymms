import numpy as np
import re
from matplotlib import pyplot as plt

import util
from pymms.data import fgm, fpi, edp

def vspace_entropy(sc, mode, start_date, end_date):
    
    # Read the data
    b = fgm.load_data(sc, mode, start_date, end_date)
    dis_moms = fpi.load_moms(sc, mode, optdesc='dis-moms',
                             start_date=start_date, end_date=end_date)
    des_moms = fpi.load_moms(sc, mode, optdesc='des-moms',
                             start_date=start_date, end_date=end_date)
    dis_dist = fpi.load_dist(sc, mode, optdesc='dis-dist',
                             start_date=start_date, end_date=end_date)
    des_dist = fpi.load_dist(sc, mode, optdesc='des-dist',
                             start_date=start_date, end_date=end_date)
    
    # Precondition the distributions
    dis_kwargs = fpi.precond_params(sc, mode, 'l2', 'dis-dist',
                                    start_date, end_date,
                                    time=dis_dist['time'])
    des_kwargs = fpi.precond_params(sc, mode, 'l2', 'des-dist',
                                    start_date, end_date,
                                    time=des_dist['time'])
    f_dis = fpi.precondition(dis_dist['dist'], **dis_kwargs)
    f_des = fpi.precondition(des_dist['dist'], **des_kwargs)
    
    # Calculate moments
    #  - Use calculated moments for the Maxwellian distribution
    Ni = fpi.density(f_dis)
    Vi = fpi.velocity(f_dis, N=Ni)
    Ti = fpi.temperature(f_dis, N=Ni, V=Vi)
    Pi = fpi.pressure(f_dis, N=Ni, T=Ti)
    ti = ((Ti[:,0,0] + Ti[:,1,1] + Ti[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    pi = ((Pi[:,0,0] + Pi[:,1,1] + Pi[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    
    Ne = fpi.density(f_des)
    Ve = fpi.velocity(f_des, N=Ne)
    Te = fpi.temperature(f_des, N=Ne, V=Ve)
    Pe = fpi.pressure(f_des, N=Ne, T=Te)
    te = ((Te[:,0,0] + Te[:,1,1] + Te[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    pe = ((Pe[:,0,0] + Pe[:,1,1] + Pe[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    
    # Equivalent (preconditioned) Maxwellian distributions
    # fi_max = fpi.maxwellian_distribution(f_dis, N=Ni, bulkv=Vi, T=ti)
    # fe_max = fpi.maxwellian_distribution(f_des, N=Ne, bulkv=Ve, T=te)
    fi_max = fpi.maxwellian_distribution(f_dis, N=Ni, bulkv=Vi, T=ti)
    fe_max = fpi.maxwellian_distribution(f_des, N=Ne, bulkv=Ve, T=te)
    
    # Analytically derived Maxwellian entropy density
    # si_max = fpi.maxwellian_entropy(Ni, pi)
    # se_max = fpi.maxwellian_entropy(Ne, pe)
    si_max = fpi.entropy(fi_max)
    se_max = fpi.entropy(fe_max)
    
    # Velocity space entropy density
    siv_dist = fpi.vspace_entropy(f_dis)
    sev_dist = fpi.vspace_entropy(f_des)
    
    # The Maxwellian is already preconditioned
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
    siv_max = fpi.vspace_entropy(fi_max, N=Ni, s=si_max)
    sev_max = fpi.vspace_entropy(fe_max, N=Ne, s=se_max)
    
    # M-bar
    miv_bar = (siv_max - siv_dist) / siv_max
    mev_bar = (sev_max - sev_dist) / sev_max
    
    # Anisotropy
    Ai = dis_moms['temppara'] / dis_moms['tempperp'] - 1
    Ae = des_moms['temppara'] / des_moms['tempperp'] - 1
    
    nrows = 8
    ncols = 1
    figsize = (5.5, 7.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize, squeeze=False)
    
    # B
    ax = axes[0,0]
    ax = util.plot([b['B_GSE'][:,3], b['B_GSE'][:,0], b['B_GSE'][:,1], b['B_GSE'][:,2]],
                   ax=ax, labels=['|B|', 'Bx', 'By', 'Bz'],
                   xaxis='off', ylabel='B\n(nT)'
                   )
    
    # N
    ax = axes[1,0]
    ax = util.plot([dis_moms['density'], des_moms['density']],
                   ax=ax, labels=['Ni', 'Ne'],
                   xaxis='off', ylabel='N\n($cm^{-3}$)'
                   )
    
    # Velocity space ion entropy density
    ax = axes[2,0]
    ax = util.plot([siv_max, siv_dist],
                   ax=ax, labels=['$s_{i,V,max}$', '$s_{i,V}$'],
                   xaxis='off', ylabel='$s_{V}$\n$J/K/m^{3}$'
                   )
    
    # Velocity space electron entropy density
    ax = axes[3,0]
    ax = util.plot([sev_max, sev_dist],
                   ax=ax, labels=['$s_{e,V,max}$', '$s_{e,V}$'],
                   xaxis='off', ylabel='$s_{V}$'
                   )
    
    # Velocity space M-bar
    ax = axes[4,0]
    ax = util.plot([miv_bar, mev_bar],
                   ax=ax, labels=['$\overline{M}_{i,V}$', '$\overline{M}_{e,V}$'],
                   xaxis='off', ylabel='$\overline{M}_{V}$'
                   )
    
    # Ion temperature
    ax = axes[5,0]
    ax = util.plot([dis_moms['temppara'], dis_moms['tempperp'], dis_moms['t']],
                   ax=ax, labels=['$T_{\parallel}$', '$T_{\perp}$', 'T'],
                   xaxis='off', ylabel='$T_{i}$\n(eV)'
                   )
    
    # Electron temperature
    ax = axes[6,0]
    ax = util.plot([des_moms['temppara'], des_moms['tempperp'], des_moms['t']],
                   ax=ax, labels=['$T_{\parallel}$', '$T_{\perp}$', 'T'],
                   xaxis='off', ylabel='$T_{e}$\n(eV)'
                   )
    
    # Anisotropy
    ax = axes[7,0]
    ax = util.plot([Ai, Ae],
                   ax=ax, labels=['$A_{i}$', '$A_{e}$'],
                   ylabel='A'
                   )
    
    fig.suptitle('Velocity Space Entropy')
    plt.subplots_adjust(left=0.2, right=0.85, top=0.95, hspace=0.4)
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
    fig, axes = vspace_entropy(args.sc, args.mode, t0, t1)
    
    # Save to directory
    if args.dir is not None:
        if t0.date() == t1.date():
            fname = '_'.join((args.sc, 'fpi', args.mode, 'l2', 'sv',
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%H%M%S')))
        else:
            fname = '_'.join((args.sc, 'fpi', args.mode, 'l2', 'sv',
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%Y%m%d'), t1.strftime('%H%M%S')))
        plt.savefig(path.join(args.dir, fname + '.png'))
    
    # Save to file
    if args.filename is not None:
        plt.savefig(args.filename)
    
    # Show on screen
    if not args.no_show:
        plt.show()
