import csv
import pathlib
import datetime as dt
from pymms.sdc import selections as sel

# NOTES:
#  - sdc.sitl_selections filters files the same way that the science files
#    are filtered. The file with the time stamp just prior to the requested
#    interval start time is kept, all previous are discarded. This is ok
#    because we are interested in only the most recent submissions. However,
#    data prior to the requested time could be contained within the file.
#  - sdc.burst_data_segments returns only those segments that lie within the
#    time interval. End points are inclusive. Segments are still out of
#    order.
#  - sdc.mission_events 'Burst', 'burst_segment' needs work. There are layers
#    of information.

def get_start_time(filename):
    '''
    In a file of burst segment selections, find the time after which no
    selections have been recorded.
    
    Parameters
    ----------
    filename : str
        Name of the file
    
    Returns
    -------
    tstart : `datetime.datetime`
        End time of the last burst segment. Used to search ahead for
        new data.
    '''
    data = sel.read_csv(filename)
    last_create_time = dt.datetime.strptime(data[-1].createtime, '%Y-%m-%d %H:%M:%S')
    data = [segment
            for segment in data
            if (dt.datetime.strptime(segment.createtime, '%Y-%m-%d %H:%M:%S')
                < last_create_time)
            ]
    
    return data


def burst_selections(file, type, tstart=None, tstop=None, **kwargs):
    '''
    Retrieve burst selections and write them to a file.
    
    Parameters
    ----------
    file : str
        Name of the file to be written. If the file exists,
        `tstart` will be amended and new data will be appended
        to the end.
    type : str
        Type of selections to retrieve. Options include:
        'abs', 'sitl+back', or 'gls'. See
        `pymms.selections.selections` for more.
    tstart : `datetime.datetime`
        Start of the data interval. If `file` exists, the
        start time will default to the end of the last
        interval recorded. The end of the data interval
        is the current time.
    \*\*kwargs
        Any keyword accepted by `pymms.selections.selections`
    '''
    file = pathlib.Path(file)
    old_data = []
    if not file.parent.exists():
        file.parent.mkdir()
    if file.exists():
        old_data = get_start_time(file)
        tstart = old_data[-1].tstop
    if tstop is None:
        tstop = dt.datetime.now()
    
    # Get the data interval. If the file exists, start from
    # the last time in the file. Go through the current time.
    
    # Read the data
    new_data = sel.selections(type, tstart, tstop, **kwargs)
    i = 0
    while new_data[i].tstart < tstart:
        i += 1
    old_data.extend(new_data[i:])
    
    # Write the data
    sel.write_csv(file, old_data)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Write burst selection to file.')
    
    parser.add_argument('file',
                        type=str,
                        help='Name of file to be written. ' \
                             'If the file exists, new data will be ' \
                             'appended.'
                        )
    
    parser.add_argument('type',
                        type=str,
                        help=('Type of selections. Options '
                              'are "abs", "sitl", "sitl+back", or "gls_*", '
                              'where "*" indicates the name of the ground loop '
                              'algorithm.')
                        )
    
    parser.add_argument('tstart', 
                        type=str,
                        help='Start of time interval, formatted as ' \
                             '%%Y-%%m-%%dT%%H:%%M:%%S')
    
    parser.add_argument('-a', '--all',
                        action='store_true',
                        help='Include duplicate and over-written selections.')
    
    parser.add_argument('-f', '--filter',
                        type=str,
                        action='store',
                        default='',
                        help='Filter results by applying regular expression ' \
                             'to the discussion string.')
    
    parser.add_argument('-c', '--case-sensitive',
                        type=str,
                        action='store_true',
                        help='Make the filter case sensitive.')
    
    parser.add_argument('-s', '--split',
                        action='store_true',
                        help='Segements that were split remain split.')
    
    parser.add_argument('-u', '--unsorted',
                        action='store_true',
                        help='Leave segments unsorted.')
    
    args = parser.parse_args()
    
    # Translate script arguments to function keywords
    tstart = dt.datetime.strptime(args.tstart, '%Y-%m-%dT%H:%M:%S')
    sort = not args.unsorted
    combine = not args.split
    unique = not args.all
    filter = args.filter
    if filter == '':
        filter = None
    
    # Run the program
    burst_selections(args.file, args.type, tstart,
                     sort=sort, combine=combine, filter=filter,
                     case_sensitive=args.case_sensitive)