import datetime as dt
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt, dates as mdates
from scipy import constants
import re
from pymms.data import fpi, edp
import util

kB = constants.k # J/K


def information_loss(sc, instr, mode, start_date, end_date, lut_file):

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
    s = fpi.entropy(f)
    sV = fpi.vspace_entropy(f, N=N, s=s)
    
    # Analytical form of the Maxwellian entropy
    #   - FPI moments (_moms) and integrated moments (_int)
    sM_moms = fpi.maxwellian_entropy(fpi_moms['density'], fpi_moms['p'])
    sM_int = fpi.maxwellian_entropy(N, p)
    
    # Use calculated moments for the Maxwellian distribution
    if lut_file is None:
        f_max = fpi.maxwellian_distribution(f, N=N, bulkv=V, T=t)
    
        # Maxwellian Entropy integrated from the equivalent
        # Maxwellian distribution (_dist)
        sM_dist = fpi.entropy(f_max)
    
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
        sVM = fpi.vspace_entropy(f_max, N=N, s=sM_dist)
    
    # Use a look-up table for the Maxwellian parameters
    else:
        # Read the dataset
        lut = xr.load_dataset(lut_file)
        dims = lut['N'].shape
        
        # Allocate memory
        NM = xr.zeros_like(N)
        tM = xr.zeros_like(N)
        sM_dist = xr.zeros_like(N)
        sVM = xr.zeros_like(N)
        f_max = xr.zeros_like(f)
        
        # Minimize error in density and temperature
        for idx, (dens, temp) in enumerate(zip(N, t)):
            imin = np.argmin(np.sqrt((lut['t'].data - temp.item())**2
                                     + (lut['N'].data - dens.item())**2
                                     ))
            irow = imin // dims[1]
            icol = imin % dims[1]
            NM[idx] = lut['N'][irow, icol]
            tM[idx] = lut['t'][irow, icol]
            sM_dist[idx] = lut['s'][irow, icol]
            sVM[idx] = lut['sv'][irow, icol]
            f_max[idx, ...] = lut['f'][irow, icol, ...]
    
    MbarKP = 1e-6 * (sM_dist - s) / (3/2 * kB * N)
    MbarKP_sV = 1e-6 * (sVM - sV) / (3/2 * kB * N)
    Mbar1 = (sVM - sV) / sVM

    # Calculate information loss. Contains:
    #   - a bias term in the numerator and
    #   - a regularization term in the denominator
    #   - (MbarKP - num) / denom --> R*(MbarKP + B)
    num, denom = fpi.information_loss(f_max, f, N=N, T=t)
    Mbar2 = (MbarKP - num) / denom
    
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(4.5, 7), squeeze=False)
    
    # s
    ax = axes[0,0]
    l1 = sM_dist.plot(ax=ax, label='$s_{M}$')
    l2 = s.plot(ax=ax, label='s')
#    l2 = sM_moms.plot(ax=ax, label='$s_{M,moms}$')
#    l3 = sM_int.plot(ax=ax, label='$s_{M,int}$')
#    l4 = sM_dist.plot(ax=ax, label='$s_{M,f}$')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('s\n(J/K/$cm^{3}$)')
    ax.set_title('')
    util.format_axes(ax, xaxis='off')
    util.add_legend(ax, [l1[0], l2[0]], outside=True)
    
    # sV
    ax = axes[1,0]
    l1 = sVM.plot(ax=ax, label='$s_{M,V}$')
    l2 = sV.plot(ax=ax, label='$s_{V}$')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$s_{V}$\n(J/K/$cm^{3}$)')
    ax.set_title('')
    util.format_axes(ax, xaxis='off')
    util.add_legend(ax, [l1[0], l2[0]], outside=True)

    ax = axes[2,0]
    (-num).plot(ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$B_{cg}$')
    ax.set_title('')
    util.format_axes(ax, xaxis='off')

    ax = axes[3,0]
    (1/denom).plot(ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('R')
    ax.set_title('')
    util.format_axes(ax, xaxis='off')

    ax = axes[4,0]
    (-num/denom).plot(ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$RB_{cg}$')
    ax.set_title('')
    util.format_axes(ax, xaxis='off')

    ax = axes[5,0]
    l1 = MbarKP_sV.plot(ax=ax, label='$\overline{M}_{KP,s_{V}}$')
    l2 = MbarKP.plot(ax=ax, label='$\overline{M}_{KP,s}$')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$\overline{M}_{KP}$')
    ax.set_title('')
    util.format_axes(ax, xaxis='off')
    util.add_legend(ax, [l1[0], l2[0]], outside=True)

#    ax = axes[6,0]
#    Mbar1.plot(ax=ax, label='$\overline{M}_{1}$')
#    ax.set_xlabel('')
#    ax.set_xticklabels([])
#    ax.set_ylabel('$\overline{M}_{1}$')
#    ax.set_title('')

    ax = axes[6,0]
    l1 = Mbar1.plot(ax=ax, label='$\overline{M}_{1}$')
    l2 = Mbar2.plot(ax=ax, label='$\overline{M}_{2}$')
    ax.set_xlabel('')
    ax.set_ylabel('$\overline{M}$')
    ax.set_title('')
    util.format_axes(ax)
    util.add_legend(ax, [l1[0], l2[0]], outside=True)
    

    fig.suptitle('$\overline{M}_{1} = (s_{V,M} - s_{V})/s_{V,M}$\n'
                 '$\overline{M}_{2} = R (\overline{M}_{KP} + B_{cg})$')
    plt.subplots_adjust(left=0.17, right=0.86, top=0.9, bottom=0.12, hspace=0.3)

    plt.setp(axes, xlim=mdates.date2num([start_date, end_date]))
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
                        
    parser.add_argument('-l', '--lookup-file',
                        help='File containing a look-up table of Maxwellian '
                             'distributions.')
                        
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--dir',
                       type=str,
                       help='Path to output destination',
                       )
                        
    group.add_argument('-f', '--filename',
                       type=str,
                       help='Output file name',
                       )
                        
    parser.add_argument('-e', '--extension',
                        default='png',
                        type=str,
                        help='Output file type',
                        )
                        
    parser.add_argument('-n', '--no-show',
                        help='Do not show the plot.',
                        action='store_true')
    
    # Gather input arguments
    args = parser.parse_args()
    t0 = dt.datetime.strptime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    t1 = dt.datetime.strptime(args.end_date, '%Y-%m-%dT%H:%M:%S')
    
    # Generate the figure
    fig, axes = information_loss(args.sc, args.instr, args.mode, t0, t1,
                                 args.lookup_file)
    
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
        plt.savefig(path.join(args.dir, fname + '.' + args.extension))
    
    # Save to file
    if args.filename is not None:
        plt.savefig(args.filename)
    
    # Show on screen
    if not args.no_show:
        plt.show()
