from pymms.data import edp, fgm, fpi
import numpy as np
from matplotlib import pyplot as plt, dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def overview(sc, mode, start_date, end_date, **kwargs):
    
    # Read the data
    b = fgm.load_data(sc=sc, mode=mode,
                      start_date=start_date, end_date=end_date)
    e = edp.load_data(sc=sc, mode=mode,
                      start_date=start_date, end_date=end_date)
    dis_moms = fpi.load_moms(sc=sc, mode=mode, optdesc='dis-moms',
                             start_date=start_date, end_date=end_date)
    des_moms = fpi.load_moms(sc=sc, mode=mode, optdesc='des-moms',
                             start_date=start_date, end_date=end_date)
    
    nrows = 7
    ncols = 1
    figsize = (6.0, 7.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize, squeeze=False)
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    
    # B
    ax = axes[0,0]
    b['B_GSE'][:,3].plot(ax=ax, label='|B|')
    b['B_GSE'][:,0].plot(ax=ax, label='Bx')
    b['B_GSE'][:,1].plot(ax=ax, label='By')
    b['B_GSE'][:,2].plot(ax=ax, label='Bz')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('B [nT]')
    ax.set_title('')
    
    # Create the legend outside the right-most axes
    leg = ax.legend(bbox_to_anchor=(1.05, 1),
                    borderaxespad=0.0,
                    frameon=False,
                    handlelength=0,
                    handletextpad=0,
                    loc='upper left')
    
    # Color the legend text the same color as the lines
    for line, text in zip(ax.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    # Ion energy spectrogram
    nt = dis_moms['omnispectr'].shape[0]
    nE = dis_moms['omnispectr'].shape[1]
    x0 = mdates.date2num(dis_moms['time'])[:, np.newaxis].repeat(nE, axis=1)
    x1 = dis_moms['energy']

    y = np.where(dis_moms['omnispectr'] == 0, 1, dis_moms['omnispectr'])
    y = np.log(y[0:-1,0:-1])

    ax = axes[1,0]
    im = ax.pcolorfast(x0, x1, y, cmap='nipy_spectral')
    ax.images.append(im)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylabel('E ion\n(eV)')

    cbaxes = inset_axes(ax,
                        width='2%', height='100%', loc=4,
                        bbox_to_anchor=(0, 0, 1.05, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
    cb = plt.colorbar(im, cax=cbaxes, orientation='vertical')
    cb.set_label('$log_{10}$DEF\nkeV/($cm^{2}$ s sr keV)')

    # Electron energy spectrogram
    nt = des_moms['omnispectr'].shape[0]
    nE = des_moms['omnispectr'].shape[1]
    x0 = mdates.date2num(des_moms['time'])[:, np.newaxis].repeat(nE, axis=1)
    x1 = des_moms['energy']

    y = np.where(des_moms['omnispectr'] == 0, 1, des_moms['omnispectr'])
    y = np.log(y[0:-1,0:-1])

    ax = axes[2,0]
    im = ax.pcolorfast(x0, x1, y, cmap='nipy_spectral')
    ax.images.append(im)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylabel('E e-\n(eV)')

    cbaxes = inset_axes(ax,
                        width='2%', height='100%', loc=4,
                        bbox_to_anchor=(0, 0, 1.05, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
    cb = plt.colorbar(im, cax=cbaxes, orientation='vertical')
    cb.set_label('$log_{10}$DEF')
    
    
    # N
    ax = axes[3,0]
    dis_moms['density'].plot(ax=ax, label='Ni')
    des_moms['density'].plot(ax=ax, label='Ne')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('N\n($cm^{-3}$)')
    ax.set_title('')
    
    leg = ax.legend(bbox_to_anchor=(1.05, 1),
                    borderaxespad=0.0,
                    frameon=False,
                    handlelength=0,
                    handletextpad=0,
                    loc='upper left')
    
    for line, text in zip(ax.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    # Vi
    ax = axes[4,0]
    dis_moms['velocity'][:,0].plot(ax=ax, label='Vx')
    dis_moms['velocity'][:,1].plot(ax=ax, label='Vy')
    dis_moms['velocity'][:,2].plot(ax=ax, label='Vz')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('Vi\n(km/s)')
    ax.set_title('')
    
    leg = ax.legend(bbox_to_anchor=(1.05, 1),
                    borderaxespad=0.0,
                    frameon=False,
                    handlelength=0,
                    handletextpad=0,
                    loc='upper left')
    
    for line, text in zip(ax.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    # Ve
    ax = axes[5,0]
    des_moms['velocity'][:,0].plot(ax=ax, label='Vx')
    des_moms['velocity'][:,1].plot(ax=ax, label='Vy')
    des_moms['velocity'][:,2].plot(ax=ax, label='Vz')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('Ve\n(km/s)')
    ax.set_title('')
    
    leg = ax.legend(bbox_to_anchor=(1.05, 1),
                    borderaxespad=0.0,
                    frameon=False,
                    handlelength=0,
                    handletextpad=0,
                    loc='upper left')
    
    for line, text in zip(ax.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    # E
    ax = axes[6,0]
    e['E_GSE'][:,0].plot(ax=ax, label='Ex')
    e['E_GSE'][:,1].plot(ax=ax, label='Ey')
    e['E_GSE'][:,2].plot(ax=ax, label='Ez')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('E\n(mV/m)')
    ax.set_title('')
    
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    leg = ax.legend(bbox_to_anchor=(1.05, 1),
                    borderaxespad=0.0,
                    frameon=False,
                    handlelength=0,
                    handletextpad=0,
                    loc='upper left')
    
    for line, text in zip(ax.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    
    fig.suptitle(sc.upper())
    plt.subplots_adjust(left=0.2, right=0.8, top=0.95, hspace=0.2)
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

    args = parser.parse_args()
    t0 = dt.datetime.strptime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    t1 = dt.datetime.strptime(args.end_date, '%Y-%m-%dT%H:%M:%S')
    
    fig, axes = overview(args.sc, args.mode, t0, t1)
    
    # Save to directory
    if args.dir is not None:
        optdesc = 'overview'
        if t0.date() == t1.date():
            fname = '_'.join((args.sc, 'instr', args.mode, 'l2',
                              optdesc,
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%H%M%S')))
        else:
            fname = '_'.join((args.sc, 'instr', args.mode, 'l2',
                              optdesc,
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%Y%m%d'), t1.strftime('%H%M%S')))
        plt.savefig(path.join(args.dir, fname + '.png'))
    
    # Save to file
    if args.filename is not None:
        plt.savefig(args.filename)
    
    # Show on screen
    if not args.no_show:
        plt.show()