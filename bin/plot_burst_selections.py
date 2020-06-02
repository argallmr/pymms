import datetime as dt
import pymms
from pymms.sdc import mrmms_sdc_api as sdc
from pymms.sdc import selections as sel
from metaarray import metabase, metaarray, metatime
from matplotlib import pyplot as plt
import pathlib


def time_to_orbit(time, sc='mms1', delta=10):
    '''
    Identify the orbit in which a time falls.
    
    Parameters
    ----------
    time : `datetime.datetime`
        Time within the orbit
    sc : str
        Spacecraft identifier
    delta : int
        Number of days around around the time of interest in
        which to search for the orbit. Should be the duration
        of at least one orbit.
    
    Returns
    -------
    orbit : int
        Orbit during which `time` occurs
    '''
    # sdc.mission_events filters by date, and the dates are right-exclusive:
    # [tstart, tstop). For it to return data on the date of `time`, `time`
    # must be rounded up to the next day. Start the time interval greater
    # than one orbit prior than the start time. The desired orbit should then
    # be the last orbit in the list
    tstop = dt.datetime.combine(time.date() + dt.timedelta(days=delta),
                                dt.time(0, 0, 0))
    tstart = tstop - dt.timedelta(days=2*delta)
    orbits = sdc.mission_events('orbit', tstart, tstop, sc=sc)
    
    orbit = None
    for idx in range(len(orbits['tstart'])):
        if (time > orbits['tstart'][idx]) and (time < orbits['tend'][idx]):
            orbit = orbits['start_orbit'][idx]
    if orbit is None:
        ValueError('Did not find correct orbit!')
    
    return orbit
    

def get_sroi(start, sc='mms1'):
    '''
    Get the start and stop times of the SROIs, the sub-regions of interest
    within the orbit.
    
    Parameters
    ----------
    start : `datetime.datetime` or int
        Time within an orbit or an orbit number. If time, note that the
        duration of the SROIs are shorter than that of the orbit so it is
        possible that `start` is not bounded by the start and end of the
        SROIs themselves.
    sc : str
        Spacecraft identifier
    
    Returns
    -------
    tstart, tend : `datetime.datetime`
        Start and end time of the SROIs
    '''
    # Convert a time stamp to an orbit number
    if isinstance(start, dt.datetime):
        start = time_to_orbit(start, sc=sc)

    # Get the Sub-Regions of Interest
    sroi = sdc.mission_events('sroi', start, start, sc=sc)

    return sroi['tstart'], sroi['tend']


def plot_selections_in_sroi(sc, tstart, 
                            tstop=dt.datetime.now(),
                            outdir=None,
                            **kwargs):
    
    # Get orbit range
    start_orbit = time_to_orbit(tstart)
    stop_orbit = time_to_orbit(tstop)
    
    outdir = pathlib.Path(outdir)
    if not outdir.exists():
        outdir.mkdir()
    fname_fmt = 'burst_selections_orbit-{0}_sroi-{1}.png'

    # Step through each orbit
    for offset in range(stop_orbit-start_orbit+1):
        # Get the SROI start and end times
        orbit = start_orbit + offset
        sroi = sdc.mission_events('sroi', int(orbit), int(orbit), sc=sc)
        
        for i in (0,2):
            try:
                fig, axes = plot_burst_selections(sc,
                                                  sroi['tstart'][i],
                                                  sroi['tend'][i],
                                                  **kwargs
                                                  )
            except Exception as e:
                print('Failed on orbit-{0} SROI-{1}'.format(orbit, i+1))
                print(e)
                continue
            
            # Update title and selections limits
            axes[0][0].set_title('{0} Orbit {1} SROI{2}'
                                 .format(sc.upper(), orbit, i+1))
            
            # Save the figure
            if outdir is not None:
                plt.savefig(outdir / fname_fmt.format(orbit, i+1))
            plt.close(fig)


def plot_sroi(sc, tstart, sroi=1):
    tstart, tend = get_sroi(tstart, sc)
    fig, axes = plot_burst_selections(sc, tstart[sroi-1], tend[sroi-1])
    
    #fig.set_size_inches(6.5, 8)
    plt.show()


def plot_burst_selections(sc, start_date, end_date,
                          outdir=(pymms.config['data_root']
                                  + '/figures/burst_selections/'),
                          sitl_file=None,
                          abs_file=None,
                          gls_file=None,
                          img_fmt=None
                          ):
    figsize=(5.5, 7)
    mode = 'srvy'
    level = 'l2'

    # FGM
    b_vname = '_'.join((sc, 'fgm', 'b', 'gse', mode, level))
    api = sdc.MrMMS_SDC_API(sc, 'fgm', mode, level,
                            start_date=start_date, end_date=end_date)
    files = api.download_files()
    files = sdc.sort_files(files)[0]
    fgm_data = metaarray.from_pycdf(files, b_vname,
                                    tstart=start_date, tend=end_date)

    # FPI DIS
    fpi_mode = 'fast'
    ni_vname = '_'.join((sc, 'dis', 'numberdensity', fpi_mode))
    espec_i_vname = '_'.join((sc, 'dis', 'energyspectr', 'omni', fpi_mode))
    api = sdc.MrMMS_SDC_API(sc, 'fpi', fpi_mode, level,
                            optdesc='dis-moms',
                            start_date=start_date, end_date=end_date)
    files = api.download_files()
    files = sdc.sort_files(files)[0]
    
    ni_data = metaarray.from_pycdf(files, ni_vname,
                                   tstart=start_date, tend=end_date)
    especi_data = metaarray.from_pycdf(files, espec_i_vname,
                                       tstart=start_date, tend=end_date)

    # FPI DES
    ne_vname = '_'.join((sc, 'des', 'numberdensity', fpi_mode))
    espec_e_vname = '_'.join((sc, 'des', 'energyspectr', 'omni', fpi_mode))
    api = sdc.MrMMS_SDC_API(sc, 'fpi', fpi_mode, level,
                            optdesc='des-moms',
                            start_date=start_date, end_date=end_date)
    files = api.download_files()
    files = sdc.sort_files(files)[0]
    ne_data = metaarray.from_pycdf(files, ne_vname,
                                   tstart=start_date, tend=end_date)
    espece_data = metaarray.from_pycdf(files, espec_e_vname,
                                       tstart=start_date, tend=end_date)
    
    
    # Grab selections
    if abs_file is None:
        abs_data = sel.selections('abs', start_date, end_date,
                                  sort=True, combine=True, latest=True)
    else:
        abs_data = sel.read_csv(abs_file,
                                start_time=start_date, stop_time=end_date)
    
    if sitl_file is None:
        sitl_data = sel.selections('sitl+back', start_date, end_date,
                                   sort=True, combine=True, latest=True)
    else:
        sitl_data = sel.read_csv(sitl_file,
                                 start_time=start_date, stop_time=end_date)
    
    if gls_file is None:
        gls_data = sel.selections('gls', start_date, end_date,
                                  sort=True, combine=True, latest=True)
    else:
        gls_data = sel.read_csv(gls_file,
                                start_time=start_date, stop_time=end_date)

    # SITL data time series
    t_abs = [start_date]
    x_abs = [0]
    for selection in abs_data:
        t_abs.extend([selection.start_time, selection.start_time,
                      selection.stop_time, selection.stop_time])
        x_abs.extend([0, selection.fom, selection.fom, 0])
    t_abs.append(end_date)
    x_abs.append(0)
    abs = metaarray.MetaArray(x_abs, x0=metatime.MetaTime(t_abs))
        

    t_sitl = [start_date]
    x_sitl = [0]
    for selection in sitl_data:
        t_sitl.extend([selection.start_time, selection.start_time,
                       selection.stop_time, selection.stop_time])
        x_sitl.extend([0, selection.fom, selection.fom, 0])
    t_sitl.append(end_date)
    x_sitl.append(0)
    sitl = metaarray.MetaArray(x_sitl, x0=metatime.MetaTime(t_sitl))

    t_gls = [start_date]
    x_gls = [0]
    for selection in gls_data:
        t_gls.extend([selection.start_time, selection.start_time,
                      selection.stop_time, selection.stop_time])
        x_gls.extend([0, selection.fom, selection.fom, 0])
    t_gls.append(end_date)
    x_gls.append(0)
    gls = metaarray.MetaArray(x_gls, x0=metatime.MetaTime(t_gls))
    
    # Set attributes to make plot pretty
    especi_data.plot_title = sc.upper()
    especi_data.title = 'DEF'
    especi_data.x1.title = '$E_{ion}$\n(eV)'
    espece_data.title = 'DEF\n(keV/(cm^2 s sr keV))'
    espece_data.x1.title = '$E_{e-}$\n(eV)'
    fgm_data.title = 'B\n(nT)'
    fgm_data.label = ['Bx', 'By', 'Bz', '|B|']
    ni_data.title = 'N\n($cm^{-3}$)'
    ne_data.title = 'N\n($cm^{-3}$)'
    abs.lim = (0, 200)
    abs.title = 'ABS'
    gls.lim = (0, 200)
    gls.title = 'GLS'
    sitl.lim = (0, 200)
    sitl.title = 'SITL'
    
    # Plot
    fig, axes = metabase.MetaCache.plot(
        (especi_data, espece_data, fgm_data, ni_data, abs, gls, sitl),
        figsize=figsize
        )
    plt.subplots_adjust(left=0.2, right=0.80, top=0.93)
    
    # Save the figure
    if img_fmt is not None:
        # Make sure the output directory exists
        outdir = pathlib.Path(outdir)
        if not outdir.exists():
            outdir.mkdir(parents=True)
        
        # Create the file name
        fname = ('burst_selections_{0}_{1}.{2}'
                 .format(dt.datetime.strftime(start_date, '%Y%m%d%H%M%S'),
                         dt.datetime.strftime(end_date, '%Y%m%d%H%M%S'),
                         img_fmt)
                 )
        
        plt.savefig(outdir / fname)
    
    return fig, axes


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot burst selections.')
    
    parser.add_argument('sc', 
                        type=str,
                        help='MMS spacecraft identifier')
    
    parser.add_argument('tstart', 
                        type=str,
                        help='Start of time interval, formatted as ' \
                             '%%Y-%%m-%%dT%%H:%%M:%%S')
    
    parser.add_argument('tend', 
                        type=str,
                        help='End of time interval, formatted as ' \
                             '%%Y-%%m-%%dT%%H:%%M:%%S')
    
    parser.add_argument('--sroi', 
                        action='store_true',
                        help='Make one plot per SROI')
    
    parser.add_argument('-d', '--directory',
                        type=str,
                        help='Directory in which to save figures',
                        default=(pymms.config['gls_root']
                                 + '/figures/burst_selections'))
    
    parser.add_argument('-s', '--sitl-file',
                        type=str,
                        help='CSV file containing SITL selections')
    
    parser.add_argument('-a', '--abs-file',
                        type=str,
                        help='CSV file containing ABS selections')
    
    parser.add_argument('-g', '--gls-file',
                        type=str,
                        help='CSV file containing GLS selections')
    
    parser.add_argument('-f', '--fig-type',
                        type=str,
                        help='Type of image to create (png, jpg, etc.)')
    
    args = parser.parse_args()
    
    start_date = dt.datetime.strptime(args.tstart, '%Y-%m-%dT%H:%M:%S')
    end_date = dt.datetime.strptime(args.tend, '%Y-%m-%dT%H:%M:%S')
    
    if args.sroi:
        func = plot_selections_in_sroi
    else:
        func = plot_burst_selections
    
    fig_axes = func(args.sc, start_date, end_date,
                    outdir=args.directory,
                    sitl_file=args.sitl_file,
                    abs_file=args.abs_file,
                    gls_file=args.gls_file,
                    img_fmt=args.fig_type)

    plt.show()