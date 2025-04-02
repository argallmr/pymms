from pymms.data import fgm, fpi
from matplotlib import pyplot as plt

def distribution(sc, instr, mode, time):
    
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
    kwargs = fpi.precond_params(sc=sc, mode=mode, level='l2', optdesc=optdesc,
                                start_date=t0, end_date=t1, time=des_dist['time'])
    
    # Subtracting the photoelectron model results in negative phase space
    # densities. Set these values to zero
    des_dist['dist'].data = des_dist['dist'].where(des_dist['dist'] >= 0, 0)

    #
    # Pick Distribution and Precondition
    #
    
    # Pick a specific time to create a look-up table
    #   - Select the spacecraft potential and distribution function at that time
    scpot = kwargs.pop('scpot')
    Vsci = scpot.sel(time=time, method='nearest').data
    f_fpi = des_dist['dist'].sel(time=time, method='nearest')

    # Create a distribution function object from the measured distribution
    #   - Provide the preconditioning keywords so that the original and
    #     preconditioned data are present
    f = fpi.Distribution_Function.from_fpi(f_fpi, scpot=Vsci, **kwargs)
    f.precondition()

    #
    # Create a transformation matrix
    #

    # Interpolate FGM to DES time stamps
    b_des = b_data['B_GSE'].interp_like(des_dist['time'])

    # Pick vectors to define the parallel and perpendicular directions
    #   - Par = B
    #   - Perp = V
    par = b_des.sel(time=time, method='nearest').data[0:3]
    perp = des_moms['velocity'].sel(time=time, method='nearest')

    # Rotate and plot the distribution
    fig, axes = f.plot_par_perp(par, perp, cs='vxb')

    # Set the figure title
    #   - This can be simplified in matplotlib v3.8 with fig.get_suptitle()
    ymd = time.strftime('%Y-%m-%d')
    hms = time.strftime('%H:%M:%S.%f')
    suptitle = ' '.join((sc.upper(), instr.upper(), ymd, hms))
    fig.suptitle(suptitle, x=0.5, y=0.92, horizontalalignment='center')

    plt.show(block=False)


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
                        
    parser.add_argument('-n', '--no-show',
                        help='Do not show the plot.',
                        action='store_true')

    args = parser.parse_args()
    time = dt.datetime.strptime(args.time, '%Y-%m-%dT%H:%M:%S.%f')

    # Create the plot
    distribution(args.sc, args.instr, args.mode, time)
    
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