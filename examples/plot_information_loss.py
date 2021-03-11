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
    
    # Integration limits
    regex = re.compile('([0-9]+.[0-9]+)')
    E0 = float(regex.match(fpi_moms.attrs['Energy_e0']).group(1))
    E_low = float(regex.match(fpi_moms.attrs['Lower_energy_integration_limit']).group(1))
    
    # Spacecraft potential correction
    edp_mode = mode if mode == 'brst' else 'fast'
    scpot = edp.load_scpot(sc, edp_mode, t0, t1)
    scpot = scpot.interp_like(fpi_moms, method='nearest')
    
    moms_kwargs = {'E0': E0,
                   'E_low': E_low,
                   'scpot': scpot}
    
    # Calculate moments
    #  - Use calculated moments for the Maxwellian distribution
    N = fpi.density(fpi_dist['dist'], **moms_kwargs)
    V = fpi.velocity(fpi_dist['dist'], N=N, **moms_kwargs)
    T = fpi.temperature(fpi_dist['dist'], N=N, V=V, **moms_kwargs)
    P = fpi.pressure(fpi_dist['dist'], N=N, T=T, **moms_kwargs)
    t = ((T[:,0,0] + T[:,1,1] + T[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    p = ((P[:,0,0] + P[:,1,1] + P[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    max_dist = fpi.maxwellian_distribution(fpi_dist['dist'], N=N, bulkv=V, T=t)

    # Maxwellian Entropy
    sMm = fpi.maxwellian_entropy(fpi_moms['density'], fpi_moms['p'])
    sMi = fpi.maxwellian_entropy(N, p)
    sMd = fpi.entropy(max_dist, E_low=0) #, E0=E0, E_low=E_low)
    sMd_Vsc = fpi.entropy(max_dist, **moms_kwargs)
    
    # Velocity space entropy
    s = fpi.entropy(fpi_dist['dist'], **moms_kwargs)
    sV = fpi.vspace_entropy(fpi_dist['dist'], N=N, s=s, **moms_kwargs)
    sVM = fpi.vspace_entropy(max_dist, N=N, s=sMd, E0=E0, E_low=E_low)
    sVM_Vsc = fpi.vspace_entropy(max_dist, N=N, s=sMd_Vsc, **moms_kwargs)
    
    MbarKP = 1e-6 * (sMd_Vsc - s) / (3/2 * kB * N)
    Mbar1 = (sVM_Vsc - sV) / sVM_Vsc

    # Calculate information loss
    num, denom = fpi.information_loss(max_dist, fpi_dist['dist'], N=N, T=t, **moms_kwargs)
    Mbar2 = (MbarKP - num) / denom
    
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(6, 7), squeeze=False)
    
    # s
    ax = axes[0,0]
    s.plot(ax=ax, label='s')
    sMm.plot(ax=ax, label='$s_{M,moms}$')
    sMi.plot(ax=ax, label='$s_{M,int}$')
    sMd.plot(ax=ax, label='$s_{M,f}$')
    sMd_Vsc.plot(ax=ax, label='$s_{M,f,Vsc}$')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('s\n(J/K/$cm^{3}$)')
    ax.set_title('')
    ax.legend()
    
    # sV
    ax = axes[1,0]
    sV.plot(ax=ax, label='$s_{V}$')
    sVM.plot(ax=ax, label='$s_{M,V}$')
    sVM_Vsc.plot(ax=ax, label='$s_{M,V,Vsc}$')
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
