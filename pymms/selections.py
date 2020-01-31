import pathlib
import datetime as dt
import numpy as np
import re
from . import mrmms_sdc_api as sdc
from cdflib import cdfread, epochs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

tai_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])

class BurstSegment:
    def __init__(self, fom, tstart, tstop, discussion,
                 sourceid=None, file=None):
        '''
        Create an object representing a burst data segment.

        Parameters
        ----------
        fom : int, float
            Figure of merit given to the selection
        tstart : int, str, `datetime.datetime`
            The start time of the burst segment, given as a `datetime` object,
            a TAI time in seconds since 1 Jan. 1958, or as a string formatted
            as `yyyy-MM-dd hh:mm:SS`. Times are converted to `datetimes`.
        tstop : int, str, `datetime.datetime`
            The stop time of the burst segment, similar to `tstart`.
        discussion : str
            Description of the segment provided by the SITL
        sourceid : str
            Username of the SITL that made the selection
        file : str
            Name of the file containing the selection
        '''

        # Convert tstart to datetime
        if isinstance(tstart, str):
            tstart = dt.datetime.strptime(tstart, '%Y-%m-%d %H:%M:%S')
        elif isinstance(tstart, int):
            tstart = self.__class__.tai_to_datetime(tstart)

        # Convert tstop to datetime
        if isinstance(tstop, str):
            tstop = dt.datetime.strptime(tstop, '%Y-%m-%d %H:%M:%S')
        elif isinstance(tstop, int):
            tstop = self.__class__.tai_to_datetime(tstop)

        if file is not None:
            file = pathlib.Path(file)

        self.discussion = discussion
        self.file = file
        self.fom = fom
        self.sourceid = sourceid
        self.tstart = tstart
        self.tstop = tstop

    def __str__(self):
        return '{0}   {1}   {2:3.0f}   {3:>8}   {4}'.format(
            self.tstart, self.tstop, self.fom, self.sourceid, self.discussion
            )

    def __repr__(self):
        return 'selections.BurstSegment({0}, {1}, {2:3.0f}, {3}, {4})'.format(
            self.tstart, self.tstop, self.fom, self.sourceid, self.discussion
            )

    def file_start_time(self):
        '''
        Extract the time from the burst selection file name.

        Returns
        -------
        start_time : str
             the file start time as `datetime.datetime`
        '''
        return dt.datetime.strptime(self.file.stem.split('_')[-1],
                                    '%Y-%m-%d-%H-%M-%S'
                                    )

    @staticmethod
    def datetime_to_list(t):
        return [t.year, t.month, t.day,
                t.hour, t.minute, t.second,
                t.microsecond // 1000, t.microsecond % 1000, 0
                ]

    @classmethod
    def tai_to_datetime(cls, t):
        tepoch  = epochs.CDFepoch()
        return tepoch.to_datetime(t * int(1e9) + tai_1958)

    @classmethod
    def datetime_to_tai(cls, t):
        t_list = cls.datetime_to_list(t)
        return int((epochs.CDFepoch.compute_tt2000(t_list) - tai_1958) // 1e9)

    @property
    def taistarttime(self):
        return self.__class__.datetime_to_tai(self.tstart)

    @property
    def taiendtime(self):
        return self.__class__.datetime_to_tai(self.tstop)

    @property
    def start_time(self):
        return self.tstart.strftime('%Y-%m-%d %H:%M:%S')

    @property
    def stop_time(self):
        return self.tstop.strftime('%Y-%m-%d %H:%M:%S')


def sort(data):
    '''
    Sort abs, sitl, or gls selections into ascending order.

    Parameters
    ----------
    data : dict
        Selections to be sorted. Must have keys 'start_time', 'end_time',
        'fom', 'discussion', 'tstart', and 'tstop'.
    '''
    return sorted(data, key=lambda x: x.tstart)


def remove_duplicates(data):
    '''
    If SITL or GLS selections are submitted multiple times,
    there can be multiple copies of the same selection, or
    altered selections that overlap with what was selected
    previously. Find overlapping segments and select those
    from the most recent file, as indicated by the file name.

    Parameters
    ----------
    data : list of `BurstSegment`
        Selections from which to prude duplicates. Segments
        must be sorted by `tstart`.
    '''

    idx = 0
    i = idx
    noverlap = 0
    result = []
    for idx in range(len(data)):
        if idx < i:
            continue

        ref_seg = data[idx]
        t0_ref = ref_seg.taistarttime
        t1_ref = ref_seg.taiendtime
        dt_ref = t1_ref - t0_ref

        try:
            i = idx + 1
            while ((data[i].taistarttime - t0_ref) < dt_ref):
                noverlap += 1
                test_seg = data[i]
                fstart_ref = ref_seg.file_start_time()
                fstart_test = test_seg.file_start_time()

                if data[i].file_start_time() > data[idx].file_start_time():
                    ref_seg = test_seg
                i += 1
        except IndexError:
            pass

        result.append(ref_seg)

    print('# Overlapping Segments: {}'.format(noverlap))
    return result


def combine(data, delta_t=0):
    '''
    Combine contiguous burst selections into single selections.

    Parameters
    ----------
    data : list of `BurstSegment`
        Selections to be combined.
    delta_t : int
        Time interval between adjacent selections. For selections
        returned by `pymms.sitl_selections()`, this is 0. For selections
        returned by `pymms.burst_data_segment()`, this is 10.
    '''
    # Any time delta > delta_t sec indicates the end of a contiguous interval
    t_deltas = [(seg1.tstart - seg0.tstop).total_seconds()
                for seg1, seg0 in zip(data[1:], data[:-1])
                ]
    t_deltas.append(1000)
    icontig = 0  # Current contiguous interval
    result = []

    # Check if adjacent elements are continuous in time
    #   - Use itertools.islice to select start index without copying array
    for idx, t_delta in enumerate(t_deltas):
        # Contiguous segments are separated by delta_t seconds
        if t_delta == delta_t:
            # And unique segments have the same fom and discussion
            if (data[icontig].fom == data[idx+1].fom) and \
                    (data[icontig].discussion == data[idx+1].discussion):
                continue

        # End of a contiguous interval
        data[icontig].tstop = data[idx].tstop

        # Next interval
        icontig = icontig + 1

        # Move data for new contiguous segments to beginning of array
        try:
            data[icontig] = data[idx+1]
        except IndexError:
            pass

    # Truncate data beyond last contiguous interval
    del data[icontig:]


def re_filter(data, re_filter):
    '''
    Filter burst selections by their discussion string.

    Parameters
    ----------
    data : dict
        Selections to be combined. Must have key 'discussion'.
    '''
    return [seg for seg in data if re.search(re_filter, seg.discussion)]


def print_selections(data):
    '''
    Print details of the burst selections.

    Parameters
    ----------
    data : `BurstSegment` or list of `BurstSegment`
        Selections to be printed. Must have keys 'tstart', 'tstop',
        'fom', 'sourceid', and 'discussion'
    '''
    print('{0:>19}   {1:>19}   {2}   {3:>8}   {4}'
          .format('TSTART', 'TSTOP', 'FOM', 'SOURCE', 'DISCUSSION')
          )

    if isinstance(data, list):
        for selection in data:
            print(selection)
    else:
        print(data)


def get_sroi(start):

    starttime = None
    starttime = None
    start_orbit = None
    end_orbit = None

    # Start of interval
    if isinstance(start, dt.datetime):
        starttime = start
    elif isinstance(start, int):
        start_orbit = start
    else:
        raise ValueError('start must be an int or datetime.')

    # End of interval
    if starttime is not None:
        endtime = starttime + dt.timedelta(days=10)
    elif start_orbit is not None:
        end_orbit = start_orbit

    # Get the SROI
    sroi = sdc.mission_events(start_date=starttime, end_date=endtime,
                              start_orbit=start_orbit, end_orbit=end_orbit,
                              source='BDM', event_type='science_roi')

    return sroi['tstart'][0], sroi['tend'][0]


def selection_overlap(ref, tests):
    out = {'dt': ref.tstop - ref.tstart,
           'dt_next': dt.timedelta(days=7000),
           'n_selections': 0,
           't_overlap': dt.timedelta(seconds=0.0),
           't_overselect': dt.timedelta(seconds=0.0),
           'pct_overlap': 0.0,
           'pct_overselect': 0.0
           }

    # Find which selections overlap with the given entry and by how much
    tdelta = dt.timedelta(days=7000)
    for test in tests:

        if ((test.tstart <= ref.tstop) and
            (test.tstop >= ref.tstart)
            ):
            out['n_selections'] += 1
            out['t_overlap'] += (min(test.tstop, ref.tstop)
                                 - max(test.tstart, ref.tstart)
                                 )

        # Time to nearest interval
        out['dt_next'] = min(out['dt_next'], abs(test.tstart - ref.tstart))

    # Overlap and over-selection statistics
    if out['n_selections'] > 0:
        out['t_overselect'] = out['dt'] - out['t_overlap']
        out['pct_overlap'] = out['t_overlap'] / out['dt'] * 100.0
        out['pct_overselect'] = out['t_overselect'] / out['dt'] * 100.0
    else:
        out['t_overselect'] = out['dt']
        out['pct_overselect'] = 100.0

    return out


def metric():
    starttime = dt.datetime(2019, 10, 17)

    # Find SROI
    #start_date, end_date = gls_get_sroi(starttime)
    start_date = dt.datetime(2019, 10, 19)
    #end_date = start_date + dt.timedelta(days=5)
    end_date = dt.datetime.combine(dt.date.today(), dt.time())

    # Grab selections
#    abs_files = sdc.sitl_selections('abs_selections',
#                                    start_date=start_date, end_date=end_date)
#    sitl_files = sdc.sitl_selections('sitl_selections',
#                                     start_date=start_date, end_date=end_date)
    gls_files = sdc.sitl_selections('gls_selections', gls_type='mp-dl-unh',
                                    start_date=start_date, end_date=end_date)

    # Read the files
#    abs_data = sdc.read_selections_from_sav(abs_files)
#    sitl_data = sdc.read_selections(sitl_files)
    sitl_data = sdc.burst_data_segments(start_date, end_date)
    gls_data = sdc.read_selections(gls_files)

    # Take only the magnetopause crossings
#    sitl_data = sort(sitl_data)
#    sitl_data = remove_duplicates(sitl_data)
    combine(sitl_data, delta_t=10)
#    sitl_data = re_filter(sitl_data, '(MP|Magnetopause)')

    gls_data = sort(gls_data)
    gls_data = remove_duplicates(gls_data)
    combine(gls_data)

    # Find overlap between GLS and SITL
    results = []
    for segment in gls_data:
        if (segment.tstart <= sitl_data[-1].tstop) & \
                (segment.tstop >= sitl_data[0].tstart):
            results.append(selection_overlap(segment, sitl_data))

    # Aggregate results
    total_selected = sum(result['n_selections'] > 0 for result in results)
    pct_selected = total_selected / len(results) * 100.0
    pct_overlap = [selection['pct_overlap'] for selection in results]

    # Nearest selection
    dt_offset = [result['dt_next'].total_seconds() for result in results if result['n_selections'] == 0]

    # Create a figure
    fig, axes = plt.subplots(nrows=1, ncols=2)

    # Histogram selections
    hh = axes[0].hist(pct_overlap, bins=25, range=(0, 125))
    axes[0].set_xlabel('% Overlap Between GLS and SITL Segment')
    axes[0].set_ylabel('Occurrence')
    axes[0].set_title('{0:4.1f}% of {1:d} GLS Segments Also Selected by SITL'.format(pct_selected, total_selected))

    # Histogram missed selections
    hh = axes[1].hist(dt_offset,  bins=110, range=(0, 11000))
    axes[1].set_xlabel('Offset from GLS to Closest SITL Selection (s)')
    axes[1].set_ylabel('Occurrence')
    axes[1].set_title('{0:4.1f}% of {1:d} GLS Segments Not Selected by SITL'.format(100-pct_selected, total_selected))
    plt.show()


def plot_context():
    sc = 'mms1'
    mode = 'srvy'
    level = 'l2'
    starttime = dt.datetime(2019, 10, 17)

    # Find SROI
    start_date, end_date = gls_get_sroi(starttime)

    # Grab selections
    abs_files = sdc.sitl_selections('abs_selections',
                                    start_date=start_date, end_date=end_date)
    sitl_files = sdc.sitl_selections('sitl_selections',
                                     start_date=start_date, end_date=end_date)
    gls_files = sdc.sitl_selections('gls_selections', gls_type='mp-dl-unh',
                                    start_date=start_date, end_date=end_date)

    # Read the files
    abs_data = sdc.read_eva_fom_structure(abs_files[0])
    sitl_data = sdc.read_eva_fom_structure(sitl_files[0])
    gls_data = sdc.read_gls_csv(gls_files)

    # SITL data time series
    t_abs = []
    x_abs = []
    for tstart, tstop, fom in zip(abs_data['tstart'], abs_data['tstop'], abs_data['fom']):
        t_abs.extend([tstart, tstart, tstop, tstop])
        x_abs.extend([0, fom, fom, 0])

    t_sitl = []
    x_sitl = []
    for tstart, tstop, fom in zip(sitl_data['tstart'], sitl_data['tstop'], sitl_data['fom']):
        t_sitl.extend([tstart, tstart, tstop, tstop])
        x_sitl.extend([0, fom, fom, 0])

    t_gls = []
    x_gls = []
    for tstart, tstop, fom in zip(gls_data['tstart'], gls_data['tstop'], gls_data['fom']):
        t_gls.extend([tstart, tstart, tstop, tstop])
        x_gls.extend([0, fom, fom, 0])

    # FGM
    tepoch = epochs.CDFepoch()
    t_vname = 'Epoch'
    b_vname = '_'.join((sc, 'fgm', 'b', 'gse', mode, level))
    api = sdc.MrMMS_SDC_API(sc, 'fgm', mode, level,
                            start_date=start_date, end_date=end_date)
    files = api.download_files()
    t_fgm = np.empty(0, dtype='datetime64')
    b_fgm = np.empty((0,4), dtype='float')
    for file in files:
        cdf = cdfread.CDF(file)
        time = cdf.varget(t_vname)
        t_fgm = np.append(t_fgm, tepoch.to_datetime(time, to_np=True), 0)
        b_fgm = np.append(b_fgm, cdf.varget(b_vname), 0)

    # FPI DIS
    fpi_mode = 'fast'
    api = sdc.MrMMS_SDC_API(sc, 'fpi', fpi_mode, level,
                            optdesc='dis-moms',
                            start_date=start_date, end_date=end_date)
    files = api.download_files()
    ti = np.empty(0, dtype='datetime64')
    ni = np.empty(0)
    espec_i = np.empty((0,32))
    Ei = np.empty((0,32))
    ti = np.empty(0)
    t_vname = 'Epoch'
    ni_vname = '_'.join((sc, 'dis', 'numberdensity', fpi_mode))
    espec_i_vname = '_'.join((sc, 'dis', 'energyspectr', 'omni', fpi_mode))
    Ei_vname = '_'.join((sc, 'dis', 'energy', fpi_mode))
    for file in files:
        cdf = cdfread.CDF(file)
#        tepoch = epochs.CDFepoch()
        time = cdf.varget(t_vname)
        ti = np.append(ti, tepoch.to_datetime(time, to_np=True), 0)
        ni = np.append(ni, cdf.varget(ni_vname), 0)
        espec_i = np.append(espec_i, cdf.varget(espec_i_vname), 0)
        Ei = np.append(Ei, cdf.varget(Ei_vname), 0)

    # FPI DES
    fpi_mode = 'fast'
    api.optdesc = 'des-moms'
    files = api.download_files()
    te = np.empty(0, dtype='datetime64')
    ne = np.empty(0)
    espec_e = np.empty((0,32))
    Ee = np.empty((0,32))
    te = np.empty(0)
    t_vname = 'Epoch'
    ne_vname = '_'.join((sc, 'des', 'numberdensity', fpi_mode))
    espec_e_vname = '_'.join((sc, 'des', 'energyspectr', 'omni', fpi_mode))
    Ee_vname = '_'.join((sc, 'des', 'energy', fpi_mode))
    for file in files:
        cdf = cdfread.CDF(file)
#        tepoch = epochs.CDFepoch()
        time = cdf.varget(t_vname)
        te = np.append(te, tepoch.to_datetime(time, to_np=True), 0)
        ne = np.append(ne, cdf.varget(ne_vname), 0)
        espec_e = np.append(espec_e, cdf.varget(espec_e_vname), 0)
        Ee = np.append(Ee, cdf.varget(Ee_vname), 0)
        cdf.close()

    # Create the figure
    fig, axes = plt.subplots(ncols=1, nrows=7, figsize=(8,9), sharex=True)

    # Inset axes for colorbar
    axins1 = inset_axes(axes[0],
                    width="5%",
                    height="80%",
                    loc='right',
                    bbox_to_anchor=(1.05, 0, 1, 1))
    axins2 = inset_axes(axes[1],
                    width="5%",
                    height="80%",
                    loc='right',
                    bbox_to_anchor=(1.05, 0, 1, 1))

    # FFT parameters -- resolve the oxygen gyrofrequency
    im1 = axes[0].pcolormesh(np.tile(ti, (32, 1)).T, Ei, np.log10(espec_i), cmap='nipy_spectral')
    axes[0].set_xticklabels([])
    axes[0].set_ylabel('ion E\n(eV)')
    axes[0].set_yscale('log')
    cbar1 = fig.colorbar(im1, cax=axins1)
    cbar1.set_label('Flux')
    axes[0].set_title('{} SITL Selections'.format(sc.upper()))

    im2 = axes[1].pcolormesh(np.tile(te, (32,1)).T, Ee, np.log10(espec_e), cmap='nipy_spectral')
    axes[1].set_xticklabels([])
    axes[1].set_ylabel('elec E\n(ev)')
    axes[1].set_yscale('log')
    cbar2 = fig.colorbar(im2, ax=axins2)
    cbar2.set_label('Flux')

    axes[2].plot(ti, ni, color='blue', label='Ni')
    axes[2].plot(te, ne, color='red', label='Ne')
    axes[2].set_xticklabels([])
    axes[2].set_ylabel('N\n(cm^3)')

    axes[3].plot(t_fgm, b_fgm, label=['Bx', 'By', 'Bz', '|B|'])
    axes[3].set_xticklabels([])
    axes[3].set_ylabel('B\n(nT)')
#    L_items = axes[3].get_legend().get_texts()
#    L_items[0].set_text('Bx')
#    L_items[1].set_text('By')
#    L_items[2].set_text('Bz')
#    L_items[3].set_text('|B|')

    axes[4].plot(t_abs, x_abs)
    axes[4].set_xticklabels([])
    axes[4].set_ylabel('ABS')

    axes[5].plot(t_sitl, x_sitl)
    axes[5].set_xticklabels([])
    axes[5].set_ylabel('SITL')

    axes[6].plot(t_gls, x_gls)
    axes[6].set_ylabel('GLS')

    plt.setp(axes[6].xaxis.get_majorticklabels(), rotation=45)

    plt.show()

    pdb.set_trace()

    return

if __name__ == 'main':
    from heliopy import config
    import pathlib

    # Inputs
    sc = sys.argv[0]
    start_date = sys.argv[1]
    if len(sys.argv) == 3:
        dir = sys.argv[2]
    else:
        dir = pathlib.Path(config['download_dir']) / 'figures' / 'mms'

    start_date = dt.datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')

    # Plot the data
    fig = plot_context(sc, start_date, dir=dir)