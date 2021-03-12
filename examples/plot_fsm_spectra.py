import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch

import util
from pymms.data import fgm, scm, fsm


def fsm_timeseries(sc, start_date, end_date):
    
    # FSM only has burst mode data
    mode = 'brst'

    # Load the data
    fgm_data = fgm.load_data(sc=sc, mode=mode,
                             start_date=start_date, end_date=end_date)
    scm_data = scm.load_data(sc=sc, mode=mode, level='l2',
                             start_date=start_date, end_date=end_date)
    fsm_data = fsm.load_data(sc=sc, start_date=start_date, end_date=end_date)
    
    # Determine sample rate
    fs_fgm = (np.diff(fgm_data['time']).mean().astype(int) / 1e9)**(-1)
    fs_scm = (np.diff(scm_data['time']).mean().astype(int) / 1e9)**(-1)
    fs_fsm = (np.diff(fsm_data['time']).mean().astype(int) / 1e9)**(-1)
    
    # Length of FFT -- aim for 10 second increments
    duration = (end_date - start_date).total_seconds()
    tperseg = 10.0
    nseg = duration // tperseg
    if nseg == 0:
        nseg = 1
        tperseg = duration
    nperseg_fgm = tperseg * fs_fgm
    nperseg_scm = tperseg * fs_scm
    nperseg_fsm = tperseg * fs_fsm
    
    # PWelch
    f_fgm, pxx_fgm = welch(fgm_data['B_GSE'], axis=0,
                           fs=fs_fgm, window='hanning', nperseg=nperseg_fgm,
                           detrend='linear', return_onesided=True,
                           scaling='density')
    f_scm, pxx_scm = welch(scm_data['B_GSE'], axis=0,
                           fs=fs_scm, window='hanning', nperseg=nperseg_scm,
                           detrend='linear', return_onesided=True,
                           scaling='density')
    f_fsm, pxx_fsm = welch(fsm_data['B_GSE'], axis=0,
                           fs=fs_fsm, window='hanning', nperseg=nperseg_fsm,
                           detrend='linear', return_onesided=True,
                           scaling='density')
    
    # Plot the data
    fig, axes = plt.subplots(nrows=3, ncols=1, squeeze=False,
                             figsize=(8.5, 5))

    # Bx PSD
    ax = axes[0,0]
    ax.plot(f_fsm, np.sqrt(pxx_fsm[:,0]), label='FSM')
    ax.plot(f_scm, np.sqrt(pxx_scm[:,0]), label='SCM')
    ax.plot(f_fgm, np.sqrt(pxx_fgm[:,0]), label='FGM')
    ax.set_xscale('log')
    ax.set_xlabel('f (Hz)')
    ax.set_yscale('log')
    ax.set_ylabel('Bx PSD\n(nT/$\sqrt{Hz}$)')
    ax.legend()

    # By PSD
    ax = axes[1,0]
    ax.plot(f_fsm, np.sqrt(pxx_fsm[:,1]), label='FSM')
    ax.plot(f_scm, np.sqrt(pxx_scm[:,1]), label='SCM')
    ax.plot(f_fgm, np.sqrt(pxx_fgm[:,1]), label='FGM')
    ax.set_xscale('log')
    ax.set_xlabel('f (Hz)')
    ax.set_yscale('log')
    ax.set_ylabel('By PSD\n(nT/$\sqrt{Hz}$)')
    ax.legend()

    # Bz PSD
    ax = axes[2,0]
    ax.plot(f_fsm, np.sqrt(pxx_fsm[:,2]), label='FSM')
    ax.plot(f_scm, np.sqrt(pxx_scm[:,2]), label='SCM')
    ax.plot(f_fgm, np.sqrt(pxx_fgm[:,2]), label='FGM')
    ax.set_xscale('log')
    ax.set_xlabel('f (Hz)')
    ax.set_yscale('log')
    ax.set_ylabel('Bz PSD\n(nT/$\sqrt{Hz}$)')
    ax.legend()
    
    # Timestamp
    if start_date.date() == end_date.date():
        tstamp = ' '.join((start_date.strftime('%Y-%m-%d'),
                           start_date.strftime('%H:%M:%S'), '-',
                           end_date.strftime('%H:%M:%S')))
    else:
        tstamp = ' '.join((start_date.strftime('%Y-%m-%d'),
                           start_date.strftime('%H:%M:%S'), '-',
                           end_date.strftime('%Y-%m-%d'),
                           end_date.strftime('%H:%M:%S')))
    
    
    fig.suptitle('FGM, SCM, & FSM Spectra Comparison\n{0}'
                 .format(tstamp))
    plt.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.12,
                        hspace=0.2, wspace=0.4)

    return fig, axes


if __name__ == '__main__':
    import argparse
    import datetime as dt
    
    parser = argparse.ArgumentParser(
        description=('Compare FGM, SCM, and FSM power spectra. Time series '
                     'are broken into intervals of 10s duration (or duration '
                     'of data interval), detrended with a linear function, '
                     'windowed with a hanning window, and averaged together.')
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
            fname = '_'.join((args.sc, 'fsm', 'brst', 'l3', '8khz-psd',
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%H%M%S')))
        else:
            fname = '_'.join((args.sc, 'fsm', 'brst', 'l3', '8khz-psd',
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%Y%m%d'), t1.strftime('%H%M%S')))
        plt.savefig(path.join(args.dir, fname + '.png'))
    
    # Save to file
    if args.filename is not None:
        plt.savefig(args.filename)
    
    # Show on screen
    if not args.no_show:
        plt.show()
