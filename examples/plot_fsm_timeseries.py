import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt

import util
from pymms.data import fgm, scm, fsm


def fsm_timeseries(sc, start_date, end_date):
    
    # FSM only has burst mode data
    mode = 'brst'

    # Load the data
    fgm_data = fgm.load_data(sc, mode, start_date, end_date)
    scm_data = scm.load_data(sc, mode, 'l2',
                             start_date=start_date, end_date=end_date)
    fsm_data = fsm.load_data(sc=sc, start_date=start_date, end_date=end_date)
    
    # High-pass filter FSM data at 1 Hz to compare with SCM
    f_Nyquist = 0.5 / (np.diff(fsm_data['time']).mean().astype(int) / 1e9)
    order = 3 # filter order
    fc = 1.0 # 1 Hz cut-off frequency
    fc_norm = fc / f_Nyquist
    b, a = butter(order, fc_norm, btype='high', analog=False)
    fsm_data['dB_GSE'] = xr.DataArray(filtfilt(b, a, fsm_data['B_GSE'], axis=0),
                                               dims=['time', 'b_index'])

    # Plot the data
    fig, axes = plt.subplots(nrows=4, ncols=2, squeeze=False,
                             figsize=(8.5, 6))

    # Bx
    ax = axes[0,0]
    ax = util.plot([fsm_data['B_GSE'].loc[:,'x'],
                    fgm_data['B_GSE'].loc[:,'x']],
                   ax=ax, labels=['FGM', 'FSM'],
                   xaxis='off', ylabel='Bx\n(nT)'
                   )

    # By
    ax = axes[1,0]
    ax = util.plot([fsm_data['B_GSE'].loc[:,'z'],
                    fgm_data['B_GSE'].loc[:,'z']],
                   ax=ax, labels=['FGM', 'FSM'],
                   xaxis='off', ylabel='By\n(nT)'
                   )

    # Bz
    ax = axes[2,0]
    ax = util.plot([fsm_data['B_GSE'].loc[:,'z'],
                    fgm_data['B_GSE'].loc[:,'z']],
                   ax=ax, labels=['FGM', 'FSM'],
                   xaxis='off', ylabel='Bz\n(nT)'
                   )

    # |B|
    ax = axes[3,0]
    ax = util.plot([fsm_data['|B|'],
                    fgm_data['B_GSE'].loc[:,'t']],
                   ax=ax, labels=['FGM', 'FSM'],
                   ylabel='|B|\n(nT)'
                   )

    # dBx
    ax = axes[0,1]
    ax = util.plot([fsm_data['dB_GSE'].loc[:,'x'],
                    scm_data['B_GSE'].loc[:,'x']],
                   ax=ax, labels=['FSM', 'SCM'],
                   xaxis='off', ylabel='$\delta$Bx\n(nT)'
                   )

    # dBy
    ax = axes[1,1]
    ax = util.plot([fsm_data['dB_GSE'].loc[:,'y'],
                    scm_data['B_GSE'].loc[:,'y']],
                   ax=ax, labels=['FSM', 'SCM'],
                   xaxis='off', ylabel='$\delta$By\n(nT)'
                   )

    # dBz
    ax = axes[2,1]
    ax = util.plot([fsm_data['dB_GSE'].loc[:,'z'],
                    scm_data['B_GSE'].loc[:,'z']],
                   ax=ax, labels=['FSM', 'SCM'],
                   ylabel='$\delta$Bz\n(nT)'
                   )
    
    fig.suptitle('FGM, SCM, & FSM Timeseries Comparison')
    plt.subplots_adjust(left=0.1, right=0.90, top=0.95, bottom=0.12,
                        hspace=0.2, wspace=0.4)
    
    # |dB|
    ax = axes[3,1]
    ax.axis('off')
    
    return fig, axes


if __name__ == '__main__':
    import argparse
    import datetime as dt
    
    parser = argparse.ArgumentParser(
        description='Compare FGM, SCM, and FSM timeseries.'
        )
    
    parser.add_argument('sc', 
                        type=str,
                        help='Spacecraft Identifier')
    
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
    
    # Generate the figure
    fig, axes = fsm_timeseries(args.sc, t0, t1)
    
    # Save to directory
    if args.dir is not None:
        if t0.date() == t1.date():
            fname = '_'.join((args.sc, 'fsm', 'brst', 'l2', '8khz-ts',
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%H%M%S')))
        else:
            fname = '_'.join((args.sc, 'fsm', 'brst', 'l2', '8khz-ts',
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%Y%m%d'), t1.strftime('%H%M%S')))
        plt.savefig(path.join(args.dir, fname + '.png'))
    
    # Save to file
    if args.filename is not None:
        plt.savefig(args.filename)
    
    # Show on screen
    if not args.no_show:
        plt.show()
