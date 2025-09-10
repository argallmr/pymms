import numpy as np
import datetime as dt
from scipy import constants as c
from pymms.data import fgm, fpi
from matplotlib import pyplot as plt
import util

kB = c.physical_constants['Boltzmann constant'] # J/K
me = c.physical_constants['electron mass'] # kg
mp = c.physical_constants['proton mass'] # kg
eV2J = c.physical_constants['electron volt-joule relationship'] # J/eV
eV2K = c.physical_constants['electron volt-kelvin relationship'] # K/eV

def thermal_velocity(m, t):
    vth = np.sqrt((2 * kB * t*eV2K) / m) # m/s
    return vth

def maxwellian_1d(m, t, v, v_bulk):
    vth = thermal_velocity(m, t) # m/s
    f = np.sqrt(1/(np.pi * vth**2)) * np.exp(-(v-v_bulk)**2/vth)
    return f # s/m

def distribution(sc, instr, mode, time, **kwargs):
    
    #
    # Gather data
    #
    optdesc = instr+'-dist'
    if mode == 'brst':
        t0 = time - dt.timedelta(seconds=0.5)
        t1 = time + dt.timedelta(seconds=0.5)
    else:
        t0 = time - dt.timedelta(seconds=10)
        t1 = time + dt.timedelta(seconds=10)

    b_data = fgm.load_data(sc=sc, mode=mode, start_date=t0, end_date=t1)
    des_dist = fpi.load_dist(sc=sc, mode=mode, optdesc=optdesc,
                             start_date=t0, end_date=t1)
    des_moms = fpi.load_moms(sc=sc, mode=mode, optdesc=instr+'-moms',
                             start_date=t0, end_date=t1)
    fpi_kwargs = fpi.precond_params(sc=sc, mode=mode, level='l2', optdesc=optdesc,
                                    start_date=t0, end_date=t1, time=des_dist['time'])
    
    # Subtracting the photoelectron model results in negative phase space
    # densities. Set these values to zero
    des_dist['dist'].data = des_dist['dist'].where(des_dist['dist'] >= 0, 0)

    #
    # Pick Distribution and Precondition
    #
    
    # Pick a specific time to create a look-up table
    #   - Select the spacecraft potential and distribution function at that time
    scpot = fpi_kwargs.pop('scpot')
    Vsci = scpot.sel(time=time, method='nearest').data
    t_fpi = des_dist['time'].sel(time=time, method='nearest')
    f_fpi = des_dist['dist'].sel(time=time, method='nearest')

    # Create a distribution function object from the measured distribution
    #   - Provide the preconditioning keywords so that the original and
    #     preconditioned data are present
    f = fpi.Distribution_Function.from_fpi(f_fpi, scpot=Vsci, time=t_fpi.data, **fpi_kwargs)
    f.precondition()

    #
    # Create a transformation matrix
    #

    # Interpolate FGM to DES time stamps
    b_des = b_data['B_GSE'].interp_like(des_dist['time'])

    # Pick vectors to define the parallel and perpendicular directions
    #   - Par = B
    #   - Perp = V
    par = b_des.sel(time=t_fpi, method='nearest').data[0:3]
    perp = des_moms['velocity'].sel(time=t_fpi, method='nearest')

    fig, ax = f.plot_par_perp_1D(par, perp, )

    plt.show()
    return fig, ax

    # Rotate the distribution function
    f_new = f.rotate(par, perp, cs='vxb', orientations=['pp1', 'p1p2'])

    # Reduce the distribution to 1D
    f_par = f_new[0].reduce('v', component='x')
    f_perp1 = f_new[0].reduce('v', component='y')
    f_perp2 = f_new[1].reduce('v', component='y')

    f_max = f_new[0].maxwellian()
    f_max_par = f_max.reduce('v', component='x')

    # Plot the distributions
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
    plt.subplots_adjust()

    ax = axes[0,0]
    f_par.plot(ax=ax, color='red', label='$f_{\parallel}$', marker='o')
    f_perp1.plot(ax=ax, color='blue', label='$f_{\perp_{1}}$', marker='o')
    f_perp2.plot(ax=ax, color='green', label='$f_{\perp_{2}}$', marker='o')
    f_max_par.plot(ax=ax, color='magenta', label='$f_{M}$', marker='+')
    util.add_legend(ax, outside=True)

    ax.set_xscale('symlog', linscale=0.1, linthresh=1e3)
    ax.set_xlabel('v (km/s)')
    ax.set_yscale('log')
    ax.set_ylabel('f ($s^{3}/cm^{6}$)')
    ax.set_ylim(f_par[f_par > 0].min(), f_par.max())

    f_time = f.time.astype('datetime64[us]').astype(dt.datetime).item()
    title = '{0}\n{1}'.format(sc.upper(), f_time.strftime('%Y-%m-%d %H:%M:%S.%f'))
    ax.set_title(title)

    return fig, axes


if __name__ == '__main__':
    import argparse
    import datetime as dt
    from os import path
    
    parser = argparse.ArgumentParser(
        description='Plot an overview of MMS data.'
        )
    
    parser.add_argument('sc', 
                        type=str,
                        help='Spacecraft Identifier')
    
    parser.add_argument('instr', 
                        type=str,
                        help='Instrument (des, dis)')
    
    parser.add_argument('mode', 
                        type=str,
                        help='Data rate mode')
    
    parser.add_argument('time', 
                        type=str,
                        help='Date and time of the distribution to be: '
                             '"YYYY-MM-DDTHH:MM:SS.fff"'
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
                        
    parser.add_argument('--vscale',
                        help='A power of ten by which to scale the data.',
                        type=int)
    parser.add_argument('--cmax',
                        type=float,
                        help='Maxmum colorvalue to display')
    parser.add_argument('--cmin',
                        type=float,
                        help='Minimum color value to display (must be used with cmax)')
    parser.add_argument('--vlim',
                        type=float,
                        help='Maximum velocity to display on the graph. Applied after vscale.')
    parser.add_argument('-c', '--contours',
                        help='Draw contours on the plot.',
                        action='store_true')
    parser.add_argument('-z', '--horizontal',
                        help='Plot the distributions horizontally instead of vertically.',
                        action='store_true')
    parser.add_argument('--nlevels',
                        type=int,
                        default=20,
                        help='Number of contour levels to draw.')
    parser.add_argument('-n', '--no-show',
                        help='Do not show the plot.',
                        action='store_true')

    args = parser.parse_args()
    time = dt.datetime.strptime(args.time, '%Y-%m-%dT%H:%M:%S.%f')

    clim=None
    if (args.cmin is not None) and (args.cmax is not None):
        clim = (args.cmin, args.cmax)

    # Create the plot
    fig, axes = distribution(args.sc, args.instr, args.mode, time,
                             clim=clim, vlim=args.vlim, vscale=args.vscale,
                             contours=args.contours, nlevels=args.nlevels,
                             horizontal=args.horizontal)
    
    # Save to directory
    if args.dir is not None:
        fname = '_'.join((args.sc, args.instr, args.mode, 'l2', 'dist',
                          time.strftime('%Y%m%d'), time.strftime('%H%M%S')))
        plt.savefig(path.join(args.dir, fname + '.png'))
    
    # Save to file
    if args.filename is not None:
        plt.savefig(args.filename)
    
    # Show on screen
    if not args.no_show:
        plt.show()