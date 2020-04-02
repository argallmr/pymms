import datetime as dt
from pymms.pymms import mrmms_sdc_api as sdc
from pymms.pymms import selections as sel
from pyarray.metaarray import metabase, metaarray, metatime
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
                            tstop=dt.datetime.now(), outdir=None):
    
    if tstop is None:
        tstop = dt.datetime.now()
    
    # Get orbit range
    start_orbit = time_to_orbit(tstart)
    stop_orbit = time_to_orbit(tstop)
    
    outdir = pathlib.Path(outdir)
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
                                                  sroi['tend'][i]
                                                  )
            except Exception as e:
                print('Failed on orbit-{0} SROI-{1}'.format(orbit, i+1))
                print(e)
                continue
        
            plt.subplots_adjust(left=0.15, right=0.85, top=0.93)
            if outdir is not None:
                plt.savefig(outdir / fname_fmt.format(orbit, i+1))
            plt.close(fig)


def download_ql_data(t0, t1):
    t0 = dt.datetime(2020, 1, 17, 19, 30)
    t1 = dt.datetime(2020, 1, 17, 21, 0)

    t0 = dt.datetime(2019, 12, 16, 18, 0, 0)
    t1 = dt.datetime(2019, 12, 17, 08, 0, 0)

    start_date = dt.datetime.combine(t0.date(), dt.time(0, 0, 0))
    end_date = dt.datetime.combine(t1.date() + dt.timedelta(days=1), dt.time(0, 0, 0))
    api = sdc.MrMMS_SDC_API('mms1', 'afg', 'srvy', 'ql',
                            start_date=start_date, end_date=end_date)
    afg_files = api.download()

    api.instr = 'edp'
    api.mode = 'fast'
    api.optdesc = 'dce'
    edp_files = api.download()

    api.instr = 'fpi'
    api.optdesc = 'des'
    des_files = api.download()

    api.optdesc = 'dis'
    dis_files = api.download()


def plot_sroi(sc, tstart, sroi=1):
    tstart, tend = get_sroi(tstart, sc)
    fig, axes = plot_burst_selections(sc, tstart[sroi-1], tend[sroi-1])
    
    #fig.set_size_inches(6.5, 8)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.93)
    plt.show()


def plot_burst_selections(sc, start_date, end_date,
                          figsize=(5.5, 7)):
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
#    abs_files = sdc.sitl_selections('abs_selections',
#                                    start_date=start_date, end_date=end_date)
#    gls_files = sdc.sitl_selections('gls_selections', gls_type='mp-dl-unh',
#                                    start_date=start_date, end_date=end_date)

    # Read the files
    abs_data = sel.read_csv('/Users/argall/Desktop/abs_test.csv',
                            start_time=start_date, stop_time=end_date)
    sitl_data = sel.read_csv('/Users/argall/Desktop/sitl_test.csv',
                             start_time=start_date, stop_time=end_date)
    gls_data = sel.read_csv('/Users/argall/Desktop/gls_test.csv',
                            start_time=start_date, stop_time=end_date)

    # SITL data time series
    t_abs = []
    x_abs = []
    for selection in abs_data:
        t_abs.extend([selection.start_time, selection.start_time,
                      selection.stop_time, selection.stop_time])
        x_abs.extend([0, selection.fom, selection.fom, 0])
    if len(abs_data) == 0:
        t_abs = [start_date, end_date]
        x_abs = [0, 0]
    abs = metaarray.MetaArray(x_abs, x0=metatime.MetaTime(t_abs))
        

    t_sitl = []
    x_sitl = []
    for selection in sitl_data:
        t_sitl.extend([selection.start_time, selection.start_time,
                       selection.stop_time, selection.stop_time])
        x_sitl.extend([0, selection.fom, selection.fom, 0])
    if len(sitl_data) == 0:
        t_sitl = [start_date, end_date]
        x_sitl = [0, 0]
    sitl = metaarray.MetaArray(x_sitl, x0=metatime.MetaTime(t_sitl))

    t_gls = []
    x_gls = []
    for selection in gls_data:
        t_gls.extend([selection.start_time, selection.start_time,
                      selection.stop_time, selection.stop_time])
        x_gls.extend([0, selection.fom, selection.fom, 0])
    if len(gls_data) == 0:
        t_gls = [start_date, end_date]
        x_gls = [0, 0]
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
    abs.title = 'ABS'
    gls.title = 'GLS'
    gls.lim = (0, 200)
    sitl.title = 'SITL'
    
    # Plot
    fig, axes = metabase.MetaCache.plot(
        (especi_data, espece_data, fgm_data, ni_data, abs, gls, sitl),
        figsize=figsize
        )
    plt.subplots_adjust(left=0.15, right=0.85, top=0.93)
    return fig, axes


def read_mec_position(sc, start_date, end_date, optdesc='epht89d'):
    mode = 'srvy'
    level = 'l2'
    
    # MEC
    tepoch = epochs.CDFepoch()
    t_vname = 'Epoch'
    r_vname = '_'.join((sc, 'mec', 'r', 'gse'))
    api = sdc.MrMMS_SDC_API(sc, 'mec', mode, level, optdesc=optdesc,
                            start_date=start_date, end_date=end_date)
    files = api.download_files()
    t_mec = np.empty(0, dtype='datetime64')
    r_mec = np.empty((0, 3), dtype='float')
    for file in files:
        cdf = cdfread.CDF(file)
        time = cdf.varget(t_vname)
        t_mec = np.append(t_mec, tepoch.to_datetime(time, to_np=True), 0)
        r_mec = np.append(r_mec, cdf.varget(r_vname), 0)
    
    # Filter based on time interval
    dt_start = dt.datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S.%f')
    dt_end = dt.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%S.%f')
    istart = np.searchsorted(t_mec, dt_start)
    iend = np.searchsorted(t_mec, dt_end)
    
    return t_mec[istart:iend], r_mec[istart:iend,:]


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
    
    start_date = dt.datetime.strptime(args.tstart, '%Y-%m-%dT%H:%M:%S')
    end_date = dt.datetime.strptime(args.tend, '%Y-%m-%dT%H:%M:%S')
    
    fig, axes = plot_burst_selections(sc, start_date, end_date)
    plt.show()