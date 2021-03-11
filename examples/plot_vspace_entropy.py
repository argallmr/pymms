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
    
    
    # Integration limits for moments
    dis_kwargs = {}
    des_kwargs = {}
    regex = re.compile('([0-9]+.[0-9]+)')
    dis_kwargs['E0'] = float(regex.match(dis_moms.attrs['Energy_e0']
                                         ).group(1))
    dis_kwargs['E_low'] = float(regex.match(dis_moms.attrs['Lower_energy_integration_limit']
                                            ).group(1))
    des_kwargs['E0'] = float(regex.match(des_moms.attrs['Energy_e0']
                                         ).group(1))
    des_kwargs['E_low'] = float(regex.match(des_moms.attrs['Lower_energy_integration_limit']
                                            ).group(1))
    
    # Spacecraft potential correction
    edp_mode = mode if mode == 'brst' else 'fast'
    scpot = edp.load_scpot(sc, edp_mode, start_date, end_date)
    dis_kwargs['scpot'] = scpot.interp_like(dis_moms, method='nearest')
    des_kwargs['scpot'] = scpot.interp_like(des_moms, method='nearest')
    
    # Philosopy
    #   - For the Maxwellian distributions, use the FPI moments data
    #     whenever possible
    #   - For the integrated moments, do not mix them with FPI moments
    
    # Equivalent Maxwellian distribution
    dis_max_dist = fpi.maxwellian_distribution(dis_dist['dist'],
                                               N=dis_moms['density'],
                                               bulkv=dis_moms['velocity'],
                                               T=dis_moms['t'])
    des_max_dist = fpi.maxwellian_distribution(des_dist['dist'],
                                               N=des_moms['density'],
                                               bulkv=des_moms['velocity'],
                                               T=des_moms['t'])
    
    # Maxwellian entropy density
    #   - Calculated from FPI moments to stick with philosophy
#    si_dist = fpi.entropy(dis_dist, scpot=scpot_dis)
#    se_dist = fpi.entropy(des_dist, scpot=scpot_des)
    
    si_max = fpi.maxwellian_entropy(dis_moms['density'], dis_moms['p'])
    se_max = fpi.maxwellian_entropy(des_moms['density'], des_moms['p'])
    
    # Velocity space entropy density
    siv_dist = fpi.vspace_entropy(dis_dist['dist'],
#                                  N=dis_moms['density'],
#                                  s=si_dist,
                                  **dis_kwargs)
    sev_dist = fpi.vspace_entropy(des_dist['dist'],
#                                  N=des_moms['density'],
#                                  s=se_dist,
                                  **des_kwargs)
    
    siv_max = fpi.vspace_entropy(dis_max_dist,
                                 N=dis_moms['density'],
                                 s=si_max)
    sev_max = fpi.vspace_entropy(des_max_dist,
                                 N=des_moms['density'],
                                 s=se_max)
    
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
    try:
        ax = util.plot([b['B_GSE'][:,3], b['B_GSE'][:,0], b['B_GSE'][:,1], b['B_GSE'][:,2]],
                       ax=ax, labels=['|B|', 'Bx', 'By', 'Bz'],
                       xaxis='off', ylabel='B\n(nT)'
                       )
    except:
        import pdb
        pdb.set_trace()
    
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
#    plt.setp(axes, xlim=xlim)
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
