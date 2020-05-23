import pathlib
import re
import csv
import datetime as dt
from . import mrmms_sdc_api as sdc
from cdflib import cdfread, epochs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

tai_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])

class BurstSegment:
    def __init__(self, tstart, tstop, fom, discussion,
                 sourceid=None, createtime=None):
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

        self.discussion = discussion
        self.createtime = createtime
        self.fom = fom
        self.sourceid = sourceid
        self.tstart = tstart
        self.tstop = tstop

    def __str__(self):
        return ('{0}   {1}   {2:3.0f}   {3}'
                .format(self.tstart, self.tstop, self.fom, self.discussion)
                )

    def __repr__(self):
        return ('selections.BurstSegment({0}, {1}, {2:3.0f}, {3})'
                .format(self.tstart, self.tstop, self.fom, self.discussion)
                )

    @staticmethod
    def datetime_to_list(t):
        return [t.year, t.month, t.day,
                t.hour, t.minute, t.second,
                t.microsecond // 1000, t.microsecond % 1000, 0
                ]

    @classmethod
    def datetime_to_tai(cls, t):
        t_list = cls.datetime_to_list(t)
        return int((epochs.CDFepoch.compute_tt2000(t_list) - tai_1958) // 1e9)

    @classmethod
    def tai_to_datetime(cls, t):
        tepoch  = epochs.CDFepoch()
        return tepoch.to_datetime(t * int(1e9) + tai_1958)

    @property
    def start_time(self):
        return self.tstart.strftime('%Y-%m-%d %H:%M:%S')

    @property
    def stop_time(self):
        return self.tstop.strftime('%Y-%m-%d %H:%M:%S')

    @property
    def taistarttime(self):
        return self.__class__.datetime_to_tai(self.tstart)

    @property
    def taiendtime(self):
        return self.__class__.datetime_to_tai(self.tstop)


def _burst_data_segments_to_burst_segment(data):
    '''
    Turn selections created by `MrMMS_SDC_API.burst_data_segements` and turn
    them into `BurstSegment` instances.

    Parameters
    ----------
    data : dict
        Data associated with each burst segment

    Returns
    -------
    result : list of `BurstSegment`
        Data converted to `BurstSegment` instances
    '''
    # Look at createtime and finishtime keys to see if either can
    # substitute for a file name time stamp
    
    result = []
    for tstart, tend, fom, discussion, sourceid, createtime, status in \
        zip(data['tstart'], data['tstop'], data['fom'],
            data['discussion'], data['sourceid'], data['createtime'],
            data['status']
            ):
        segment = BurstSegment(tstart, tend, fom, discussion,
                               sourceid=sourceid,
                               createtime=createtime)
        segment .status = status
        result.append(segment)
    return result


def _get_selections(type, start, stop,
                    sort=False, combine=False, latest=True, unique=False,
                    metadata=False, filter=None):

    if latest and unique:
        raise ValueError('latest and unique keywords '
                         'are mutually exclusive.')

    # Burst selections can be made multiple times within the orbit.
    # Multiple submissions will be appear as repeated entires with
    # different create times, but only the last submission is the
    # official submission. To ensure that the last submission is
    # returned, look for selections submitted through the following
    # orbit as well.
    orbit_start = sdc.time_to_orbit(start)
    orbit_stop = sdc.time_to_orbit(stop) + 1

    # Get the selections
    data = sdc.burst_selections(type, orbit_start, orbit_stop)

    # Turn the data into BurstSegments. Adjacent segments returned
    # by `sdc.sitl_selections` have a 0 second gap between stop and
    # start times. Those returned by `sdc.burst_data_segments` are
    # separated by 10 seconds.
    if type in ('abs', 'sitl', 'gls', 'mp-dl-unh'):
        delta_t = 0.0
        converter = _sitl_selections_to_burst_segment
    elif type == 'sitl+back':
        delta_t = 10.0
        converter = _burst_data_segments_to_burst_segment
    else:
        raise ValueError('Invalid selections type {}'.format(type))

    # Convert data into BurstSegments
    data = converter(data)

    # Get metadata associated with orbit, sroi, and metadata
    if metadata:
        _get_segment_data(data, orbit_start, orbit_stop)

    # The official selections are those from the last submission
    # containing selections within the science_roi. Get rid of
    # all submissions except the last.
    if latest:
        data = _latest_segments(data, orbit_start, orbit_stop,
                                sitl=(type == 'sitl+back'))

    # Get rid of extra selections obtained by changing the
    # start and stop interval
    data = [segment
            for segment in data
            if (segment.tstart >= start) \
            and (segment.tstop <= stop)]

    # Additional handling of data
    if combine:
        combine_segments(data, delta_t)
    if sort:
        data = sort_segments(data)
    if unique:
        data = remove_duplicate_segments(data)
    if filter is not None:
        data = filter_segments(data, filter)

    return data


def _get_segment_data(data, orbit_start, orbit_stop, sc='mms1'):
    '''
    Add metadata associated with the orbit, SROI, and SITL window
    to each burst segment.

    Parameters
    ----------
    data : list of BurstSegments
        Burst selections. Selections are altered in-place to
        have new attributes:
        Attribute            Description
        ==================   ======================
        orbit                Orbit number
        orbit_tstart         Orbit start time
        orbit_tstop          Orbit stop time
        sroi                 SROI number
        sroi_tstart          SROI start time
        sroi_tstop           SROI stop time
        sitl_window_tstart   SITL window start time
        sitl_window_tstop    SITL window stop time
        ==================   ======================
    orbit_start, orbit_stop : int or np.integer
        Orbit interval in which selections were made
    sc : str
        Spacecraft identifier
    '''
    idx = 0
    for iorbit in range(orbit_start, orbit_stop+1):
        # The start and end times of the sub-regions of interest.
        # These are the times in which selections can be made for
        # any given orbit.
        orbit = sdc.mission_events('orbit', iorbit, iorbit, sc=sc)
        sroi = sdc.mission_events('sroi', iorbit, iorbit, sc=sc)
        window = sdc.mission_events('sitl_window', iorbit, iorbit)
        tstart = min(sroi['tstart'])
        tend = max(sroi['tend'])

        # Find the burst segments that were selected within the
        # current SROI

        while idx < len(data):
            segment = data[idx]

            # Filter out selections from the previous orbit(s)
            # Stop when we get to the next orbit
            if segment.tstop < tstart:
                idx += 1
                continue
            if segment.tstart > tend:
                break

            # Keep segments from the same submission (create time).
            # If there is a new submission within the orbit, take
            # selections from the new submission and discard those
            # from the old.
            segment.orbit = iorbit
            segment.orbit_tstart = orbit['tstart'][0]
            segment.orbit_tstop = orbit['tend'][0]

            sroi_data = _get_sroi_number(sroi, segment.tstart, segment.tstop)
            segment.sroi = sroi_data[0]
            segment.sroi_tstart = sroi_data[1]
            segment.sroi_tstop = sroi_data[2]

            # The SITL Window is not defined as of orbit 1098, when
            # a SITL-defined window was implemented
            if iorbit < 1098:
                segment.sitl_window_tstart = window['tstart']
                segment.sitl_window_tstop = window['tend']

            idx += 1


def _get_sroi_number(sroi, tstart, tstop):
    '''
    Determine which sub-region of interest (SROI) in which a given
    time interval resides.

    Parameters
    ----------
    sroi : dict
        SROI information for a specific orbit
    tstart, tstop : `datetime.datetime`
        Time interval

    Returns
    -------
    result : tuple
       The SROI number and start and stop times.
    '''
    sroi_num = 0
    for sroi_tstart, sroi_tstop in zip(sroi['tstart'], sroi['tend']):
        sroi_num += 1
        if (tstart >= sroi_tstart) and (tstop <= sroi_tstop):
            break

    return sroi_num, sroi_tstart, sroi_tstop


def _latest_segments(data, orbit_start, orbit_stop, sitl=False):
    '''
    Return the last burst selections submission from each orbit.

    Burst selections can be submitted multiple times but only
    the latest file serves as the official selections file.

    For the SITL, the SITL makes selections within the SITL
    window. If data is downlinked after the SITL window closes,
    selections on that data are submitted separately into the
    back structure and should be appended to the last submission
    that took place within the SITL window.

    Parameters
    ----------
    data : list of BurstSegments
        Burst segment selections
    orbit_start, orbit_stop : int or np.integer
        Orbits in which selections were made
    sitl : bool
        If true, the burst selections were made by the SITL and
        there are back structure submissions to take into
        consideration

    Returns
    -------
    results : list of BurstSegments
        The latest burst selections
    '''
    result = []
    for orbit in range(orbit_start, orbit_stop+1):
        # The start and end times of the sub-regions of interest.
        # These are the times in which selections can be made for
        # any given orbit.
        
        # SROI information is not available before orbit 239.
        # The first SROI is defined on 2015-11-06, which is only a
        # couple of months after formal science data collection began
        # on 2015-09-01.
        
        # However, similar information is available starting at 
        # 2015-08-10 (orbit 151) in the science_roi event type. It's
        # nearly equivalent because there was only a single SROI per
        # orbit during the earlier mission phases, but science_roi is
        # the span across all spacecraft.
        if orbit < 239:
            sroi = sdc.mission_events('science_roi', orbit, orbit)
        else:
            sroi = sdc.mission_events('sroi', orbit, orbit)
        tstart = min(sroi['tstart'])
        tend = max(sroi['tend'])

        # The SITL Window is not defined as of orbit 1098, when
        # a SITL-defined window was implemented
        if orbit >= 1098:
            sitl = False

        # Need to know when the SITL window closes in order to
        # keep submissions to the back structure.
        if sitl:
            sitl_window = sdc.mission_events('sitl_window', orbit, orbit)

        # Find the burst segments that were selected within the
        # current SROI
        create_time = None
        orbit_segments = []
        for idx, segment in enumerate(data):
            # Filter out selections from the previous orbit(s)
            # Stop when we get to the next orbit
            if segment.tstop < tstart:
                continue
            if segment.tstart > tend:
                break

            # Initialize with the first segment within the orbit.
            # Create times are the same for all selections within
            # a single submission. There may be several submissions
            # per orbit.
            if create_time is None:
                create_time = segment.createtime

            # Keep segments from the same submission (create time).
            # If there is a new submission within the orbit, take
            # selections from the new submission and discard those
            # from the old.
            #
            # Submissions to the back structure occur after the
            # SITL window has closed and are in addition to whatever
            # the latest submission was.
            #
            # GLS and ABS selections can occur after the SITL window
            # closes, but those are treated the same as selections
            # made within the SITL window.
            if create_time == segment.createtime:
                orbit_segments.append(segment)
            elif segment.createtime > create_time:
                if sitl and (segment.createtime > sitl_window['tend'][0]):
                    orbit_segments.append(segment)
                else:
                    create_time = segment.createtime
                    orbit_segments = [segment]
            else:
                continue

        # Truncate the segments and append this orbit's submissions
        # to the overall results.
        data = data[idx:]
        result.extend(orbit_segments)

    return result


def _mission_events_to_burst_segment(data):
    '''
    Turn selections created by `MrMMS_SDC_API.mission_events` and turn
    them into `BurstSegment` instances.

    Parameters
    ----------
    data : dict
        Data associated with each burst segment

    Returns
    -------
    result : list of `BurstSegment`
        Data converted to `BurstSegment` instances
    '''
    raise NotImplementedError


def _sitl_selections_to_burst_segment(data):
    '''
    Turn selections created by `MrMMS_SDC_API.sitl_selections` and turn
    them into `BurstSegment` instances.

    Parameters
    ----------
    data : dict
        Data associated with each burst segment

    Returns
    -------
    result : list of `BurstSegment`
        Data converted to `BurstSegment` instances
    '''
    result = []
    for idx in range(len(data['fom'])):
        result.append(BurstSegment(data['tstart'][idx], data['tstop'][idx],
                                   data['fom'][idx], data['discussion'][idx],
                                   sourceid=data['sourceid'][idx],
                                   createtime=data['createtime'][idx],
                                   )
                      )
    return result


def combine_segments(data, delta_t=0):
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
        if t_delta <= delta_t:
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


def filter_segments(data, filter):
    '''
    Filter burst selections by their discussion string.

    Parameters
    ----------
    data : dict
        Selections to be combined. Must have key 'discussion'.
    '''
    return [seg
            for seg in data
            if re.search(filter, seg.discussion, re.IGNORECASE)]


def plot_metric(ref_data, test_data, fig, labels, location,
                nbins=10):
    '''
    Visualize the overlap between segments.

    Parameters
    ----------
    ref_data : list of `BurstSegment`s
        Reference burst segments
    test_data : list of `BurstSegment`s
        Test burst segments. Determine which test segments
        overlap with the reference segments and by how much
    labels : tuple of str
        Labels for the reference and test segments
    location : tuple
        Location of the figure (row, col, nrows, ncols)
    nbins : int
        Number of histogram bins to create

    Returns:
    --------
    ax : `matplotlib.pyplot.Axes`
        Axes in which data is displayed
    ref_test_data : list of `BurstSegment`s
        Reference data that falls within the [start, stop] times
        of the test data.
    '''

    # Determine by how much the test data overlaps with
    # the reference data.
    ref_test = []
    ref_test_data = []
    ref_test = [selection_overlap(segment, test_data)
                for segment in ref_data]

    # Overlap statistics
    #   - Number of segments selected
    #   - Percentage of segments selected
    #   - Percent overlap from each segment
    ref_test_selected = sum(selection['n_selections'] > 0
                            for selection in ref_test)
    ref_test_pct_selected = ref_test_selected / len(ref_test) * 100.0
    ref_test_pct_overlap = [selection['pct_overlap'] for selection in ref_test]

    # Calculate the plot index from the (row,col) subplot location
    plot_idx = lambda rowcol, ncols : (rowcol[0]-1)*ncols + rowcol[1]

    # Create a figure
    ax = fig.add_subplot(location[2], location[3],
                         plot_idx(location[0:2], location[3]))
    hh = ax.hist(ref_test_pct_overlap, bins=nbins, range=(0, 100))
    #ax.set_xlabel('% Overlap Between {0} and {1} Segments'.format(*labels))
    if location[0] == location[2]:
        ax.set_xlabel('% Overlap per Segment')
    if location[1] == 1:
        ax.set_ylabel('Occurrence')
    ax.text(0.5, 0.98, '{0:4.1f}% of {1:d}'
              .format(ref_test_pct_selected, len(ref_test)),
              verticalalignment='top', horizontalalignment='center',
              transform=ax.transAxes)
    ax.set_title('{0} Segments\nSelected by {1}'.format(*labels))

    return ax, ref_test


def metric(sroi=None, output_dir=None, figtype=None):
    do_sroi=False
    if sroi in (1, 2, 3):
        do_sroi=True

    if output_dir is None:
        output_dir = pathlib.Path('~/').expanduser()
    else:
        output_dir = pathlib.Path(output_dir).expanduser()

    starttime = dt.datetime(2019, 10, 17)

    # Find SROI
    #start_date, end_date = gls_get_sroi(starttime)
    start_date = dt.datetime(2019, 10, 19)
    #end_date = start_date + dt.timedelta(days=5)
    end_date = dt.datetime.now()

    abs_data = selections('abs', start_date, end_date,
                          latest=True, combine=True, metadata=do_sroi)

    gls_data = selections('mp-dl-unh', start_date, end_date,
                          latest=True, combine=True, metadata=do_sroi)

    sitl_data = selections('sitl+back', start_date, end_date,
                           latest=True, combine=True, metadata=do_sroi)

    # Filter by SROI
    if do_sroi:
        abs_data = [s for s in abs_data if s.sroi == sroi]
        sitl_data = [s for s in sitl_data if s.sroi == sroi]
        gls_data = [s for s in gls_data if s.sroi == sroi]

    sitl_mp_data = filter_segments(sitl_data, '(MP|Magnetopause)')

    # Create a figure
    nbins = 10
    nrows = 4
    ncols = 3
    fig = plt.figure(figsize=(8.5, 10))
    fig.subplots_adjust(hspace=0.55, wspace=0.3)

    # GLS-SITL Comparison
    ax, gls_sitl = plot_metric(gls_data, sitl_data, fig,
                               ('GLS', 'SITL'), (1, 1, nrows, ncols),
                               nbins=nbins)
    ax, sitl_gls = plot_metric(sitl_data, gls_data, fig,
                               ('SITL', 'GLS'), (2, 1, nrows, ncols),
                               nbins=nbins)
    ax, gls_sitl_mp = plot_metric(gls_data, sitl_mp_data, fig,
                                  ('GLS', 'SITL MP'), (3, 1, nrows, ncols),
                                  nbins=nbins)
    ax, sitl_mp_gls = plot_metric(sitl_mp_data, gls_data, fig,
                                  ('SITL MP', 'GLS'), (4, 1, nrows, ncols),
                                  nbins=nbins)

    # ABS-SITL Comparison
    ax, abs_sitl = plot_metric(abs_data, sitl_data, fig,
                               ('ABS', 'SITL'), (1, 2, nrows, ncols),
                               nbins=nbins)
    ax, sitl_abs = plot_metric(sitl_data, abs_data, fig,
                               ('SITL', 'ABS'), (2, 2, nrows, ncols),
                               nbins=nbins)
    ax, abs_sitl_mp = plot_metric(abs_data, sitl_mp_data, fig,
                                  ('ABS', 'SITL MP'), (3, 2, nrows, ncols),
                                  nbins=nbins)
    ax, sitl_mp_abs = plot_metric(sitl_mp_data, abs_data, fig,
                                  ('SITL MP', 'ABS'), (4, 2, nrows, ncols),
                                  nbins=nbins)

    # GLS-ABS Comparison
    abs_mp_data = [abs_data[idx]
                   for idx, s in enumerate(abs_sitl_mp)
                   if s['n_selections'] > 0]

    ax, gls_abs = plot_metric(gls_data, abs_data, fig,
                              ('GLS', 'ABS'), (1, 3, nrows, ncols),
                              nbins=nbins)
    ax, abs_gls = plot_metric(abs_data, gls_data, fig,
                              ('ABS', 'GLS'), (2, 3, nrows, ncols),
                              nbins=nbins)
    ax, gls_abs_mp = plot_metric(gls_data, abs_mp_data, fig,
                                 ('GLS', 'ABS MP'), (3, 3, nrows, ncols),
                                 nbins=nbins)
    ax, abs_mp_gls = plot_metric(abs_mp_data, gls_data, fig,
                                 ('ABS MP', 'GLS'), (4, 3, nrows, ncols),
                                 nbins=nbins)

    # Save the figure
    if figtype is not None:
        sroi_str = ''
        if do_sroi:
            sroi_str = '_sroi{0:d}'.format(sroi)
        filename = (output_dir
                    / '_'.join(('selections_metric' + sroi_str,
                                start_date.strftime('%Y%m%d%H%M%S'),
                                end_date.strftime('%Y%m%d%H%M%S')
                                )))
        filename = filename.with_suffix('.' + figtype)
        plt.savefig(filename.expanduser())

    plt.show()


def print_segments(data, full=False):
    '''
    Print details of the burst selections.

    Parameters
    ----------
    data : `BurstSegment` or list of `BurstSegment`
        Selections to be printed. Must have keys 'tstart', 'tstop',
        'fom', 'sourceid', and 'discussion'
    '''
    if full:
        source_len = max(len(s.sourceid) for s in data)
        source_len = max(source_len, 8)
        header_fmt = '{0:>19}   {1:>19}   {2:>19}   {3:>5}   ' \
                     '{4:>19}   {5:>'+str(source_len)+'}   {6}'
        data_fmt = '{0:>19}   {1:>19}   {2:>19}   {3:5.1f}   ' \
                   '{4:>19}, {5:>'+str(source_len)+'}   {6}'
        print(header_fmt.format('TSTART', 'TSTOP', 'CREATETIME',
                                'FOM', 'STATUS', 'SOURCEID', 'DISCUSSION'
                                )
              )
        for s in data:
            try:
                status = s.status
            except AttributeError:
                status = ''
            
            createtime = dt.datetime.strftime(s.createtime,
                                              '%Y-%m-%d %H:%M:%S')
            print(data_fmt.format(s.start_time, s.stop_time,
                                  createtime, s.fom, status, s.sourceid,
                                  s.discussion)
                  )
        return


    print('{0:>19}   {1:>19}   {2}   {3}'
          .format('TSTART', 'TSTOP', 'FOM', 'DISCUSSION')
          )

    if isinstance(data, list):
        for selection in data:
            print(selection)
    else:
        print(data)


def read_csv(filename, start_time=None, stop_time=None, header=True):
    '''
    Read a CSV file with burst segment selections.

    Parameters
    ----------
    filename : str
        The name of the file to which `data` is to be read
    start_time : str or `datetime.datetime`
        Filter results to contain segments selected on or
        after this time. Possible only if `header` is True
        and if a column is named `'start_time'`
    stop_time : str or `datetime.datetime`
        Filter results to contain segments selected on or
        before this time. Possible only if `header` is True
        and if a column is named `'stop_time'`
    header : bool
        If `True`, the csv file has a header indicating the
        names of each column. If `header` is `False`, the
        assumed column names are 'start_time', 'stop_time',
        'fom', 'sourceid', 'discussion', 'createtime'.

    Returns
    -------
    data : list of `BurstSegment`
        Burst segments read from the csv file
    '''
    # Convert time itnerval to datetimes if needed
    if isinstance(start_time, str):
        start_time = dt.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    if isinstance(stop_time, str):
        stop_time = dt.datetime.strptime(stop_time, '%Y-%m-%d %H:%M:%S')

    file = pathlib.Path(filename)
    data = []

    # Read the file
    with open(file.expanduser(), 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)

        # Take column names from file header
        if header:
            keys = next(csvreader)

        # Read the rows
        for row in csvreader:
            # Select the data within the time interval
            if start_time is not None:
                tstart = dt.datetime.strptime(row[0],
                                              '%Y-%m-%d %H:%M:%S')
                if tstart < start_time:
                    continue
            if stop_time is not None:
                tstop = dt.datetime.strptime(row[1],
                                             '%Y-%m-%d %H:%M:%S')
                if tstop > stop_time:
                    continue  # BREAK if sorted!!

            # Initialize segment with required fields then add
            # additional fields after
            data.append(BurstSegment(row[0], row[1], float(row[2]), row[4],
                                     sourceid=row[3], createtime=row[5]
                                     )
                        )

    return data


def remove_duplicate_segments(data):
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

    Returns
    -------
    results : list of `BurstSegments`
        Unique burst segments.
    '''
    results = data.copy()
    overwrite_times = set()
    for idx, segment in enumerate(data):
        iahead = idx + 1

        # Segments should already be sorted. Future segments
        # overlap with current segment if the future segement
        # start time is closer to the current segment start
        # time than is the current segment's end time.
        while ((iahead < len(data)) and
               ((data[iahead].tstart - segment.tstart) <
                (segment.tstop - segment.tstart))
               ):

            # Remove the segment with the earlier create time
            if segment.createtime < data[iahead].createtime:
                remove_segment = segment
                overwrite_time = segment.createtime
            else:
                remove_segment = data[iahead]
                overwrite_time = data[iahead].createtime

            # The segment may have already been removed if
            # there are more than one other segments that
            # overlap with it.
            try:
                results.remove(remove_segment)
                overwrite_times.add(overwrite_time)
            except ValueError:
                pass

            iahead += 1

    # Remove all segments with create times in the overwrite set
    # Note that the model results do not appear to be reproducible
    # so that every time the model is run, a different set of
    # selections are created. Using the filter below, data from
    # whole days can be removed if two processes create overlapping
    # segments at the edges of their time intervals.
#     results = [segment
#                for segment in results
#                if segment.createtime not in overwrite_times
#                ]

    print('# Segments Removed: {}'.format(len(data) - len(results)))
    return results


def remove_duplicate_segments_v1(data):
    idx = 0  # current index
    i = idx  # look-ahead index
    noverlap = 0
    result = []
    for idx in range(len(data)):
        if idx < i:
            continue
        print(idx)

        # Reference segment is the current segment
        ref_seg = data[idx]
        t0_ref = ref_seg.taistarttime
        t1_ref = ref_seg.taiendtime
        dt_ref = t1_ref - t0_ref

        if idx == len(data) - 2:
            import pdb
            pdb.set_trace()

        try:
            # Test segment are after the reference segment. The
            # test segment overlaps with the reference segment
            # if its start time is closer to the reference start
            # time than is the reference end time. If there is
            # overlap, keep the segment that was created more
            # recently.
            i = idx + 1
            while ((data[i].taistarttime - t0_ref) < dt_ref):
                noverlap += 1
                test_seg = data[i]
                if data[i].createtime > data[idx].createtime:
                    ref_seg = test_seg
                i += 1
        except IndexError:
            pass

        result.append(ref_seg)

    print('# Overlapping Segments: {}'.format(noverlap))
    return result


def selection_overlap(ref, tests):
    '''
    Gather overlap statistics.
    
    Parameters
    ----------
    ref : `selections.BurstSegment`
        The reference burst segment.
    tests : list of `selections.BurstSegment`
        The burst segements against which the reference segment is compared.
    
    Returns
    -------
    out : dict
        Data regarding how much the reference segment overlaps with the
        test segments.
    '''
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


def selections(type, start, stop,
               sort=False, combine=False, latest=True, unique=False,
               metadata=False, filter=None):
    '''
    Factory function for burst data selections.

    Parameters
    ----------
    type : str
        Type of burst data selections to retrieve. Options are 'abs',
        'abs-all', 'sitl', 'sitl+back', 'gls', 'mp-dl-unh'.
    start, stop : `datetime.datetime`
        Interval over which to retrieve data
    sort : bool
        Sort burst segments by time. Submissions to the back structure and
        multiple submissions or process executions in one SITL window can
        cause multiples of the same selection or out-of-order selections.
    combine : bool
        Combine adjacent selection into one selection. Due to downlink
        limitations, long time duration burst segments must be broken into
        smaller chunks.
    latest : bool
        For each SROI, keep only those times with the most recent time
        of creation. Cannot be used with `unique`.
    metadata : bool
        Retrieve the orbit number and time interval, SROI number and time
        interval, and SITL window time interval.
    unique : bool
        Return only unique segments. See `sort`. Also, a SITL may adjust
        the time interval of their selections, so different submissions will
        have duplicates selections but with different time stamps. This is
        accounted for.
    filter : str
        Filter the burst segments by applying the regular expression to
        the segment's discussions string.

    Returns
    -------
    results : list of `BurstSegment`
        Each burst segment.
    '''
    return _get_selections(type, start, stop,
                           sort=sort, combine=combine, unique=unique,
                           latest=latest, metadata=metadata,
                           filter=filter)


def sort_segments(data, createtime=False):
    '''
    Sort abs, sitl, or gls selections into ascending order.

    Parameters
    ----------
    data : list of `BurstSegement`
        Selections to be sorted. Must have keys 'start_time', 'end_time',
        'fom', 'discussion', 'tstart', and 'tstop'.
    createtime: bool
        Sort by time created instead of by selection start time.

    Returns
    -------
    results : list of `BurstSegement`
        Inputs sorted by time
    '''
    if createtime:
        return sorted(data, key=lambda x: x.createtime)
    return sorted(data, key=lambda x: x.tstart)


def write_csv(filename, data, append=False):
    '''
    Write a CSV file with burst data segment selections.

    Parameters
    ----------
    filename : str
        The name of the file to which `data` is to be written
    data : list of `BurstSegment`
        Burst segments to be written to the csv file
    append : bool
        If True, `data` will be appended to the end of the file.
    '''
    header = ('start_time', 'stop_time', 'fom', 'sourceid',
              'discussion', 'createtime')

    mode = 'w'
    if append:
        mode = 'a'

    file = pathlib.Path(filename)
    with open(file.expanduser(), mode, newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        if not append:
            csvwriter.writerow(header)

        for segment in data:
            csvwriter.writerow([segment.start_time,
                                segment.stop_time,
                                segment.fom,
                                segment.sourceid,
                                segment.discussion,
                                segment.createtime
                                ]
                               )


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
