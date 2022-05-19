import numpy as np
import xarray as xr
from pandas import read_csv
import datetime as dt
import re
from pathlib import Path
import requests
from tqdm import tqdm

from pymms.sdc import mrmms_sdc_api as api
from pymms.data import util
from pymms import config

data_root = Path(config['data_root'])

info_type = ['download', 'file_name', 'file_info', 'file_version']
sdc_home = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/'
data_type = 'ancillary'

class ANCDownloader(util.Downloader):
    
    def __init__(self, sc, product, start_date, end_date, version=None):
        
        self.sc = sc
        self.product = product
        self.start_date = start_date
        self.end_date = end_date
        self.version = version
        self.files = None
        self._session = requests.Session()

    def check_response(self, response):
        '''
        Check the status code for a requests response and perform
        and appropriate action (e.g. log-in, raise error, etc.)

        Parameters
        ----------
        response : `requests.response`
            Response from the SDC

        Returns
        -------
        r : `requests.response`
            Updated response
        '''

        # OK
        if response.status_code == 200:
            r = response

        # Authentication required
        elif response.status_code == 401:
            print('Log-in Required')

            maxAttempts = 4
            nAttempts = 1
            while nAttempts <= maxAttempts:
                # First time through will automatically use the
                # log-in information from the config file. If that
                # information is wrong/None, ask explicitly
                if nAttempts == 1:
                    self.login(mms_username, mms_password)
                else:
                    self.login()

                # Remake the request
                #   - Ideally, self._session.send(response.request)
                #   - However, the prepared request lacks the
                #     authentication data
                if response.request.method == 'POST':
                    query = parse_qs(response.request.body)
                    r = self._session.post(response.request.url, data=query)
                else:
                    r = self._session.get(response.request.url)

                # Another attempt
                if r.ok:
                    break
                else:
                    print('Incorrect username or password. %d tries '
                          'remaining.' % maxAttempts-nAttempts)
                    nAttempts += 1

            # Failed log-in
            if nAttempts > maxAttempts:
                raise ConnectionError('Failed log-in.')

        else:
            raise ConnectionError(response.reason)

        # Return the resulting request
        return r
    
    def download(self, interval):
        
        # Get information on the files that were found
        #   - To do that, specify the specific files.
        #     This sets all other properties to None
        #   - Save the state of the object as it currently
        #     is so that it can be restored
        #   - Setting FILES will indirectly cause SITE='public'.
        #     Keep track of SITE.
        state = {}
        state['sc'] = self.sc
        state['product'] = self.product
        state['version'] = self.version
        state['files'] = self.files

        # Get file name and size
        info = self.file_info(interval)['files'][0]
        self.files = info['file_name']

        # Build the URL sans query
        url = self.url('download', query=False)

        # Amount to download per iteration
        block_size = 1024*128

        # Create the destination directory
        file = self.name2path(info['file_name'])
        if not file.parent.exists():
            file.parent.mkdir(parents=True)

        # downloading: https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
        # progress bar: https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
        try:
            r = self._session.get(url,
                                  params={'file': info['file_name']},
                                  stream=True)
            with tqdm(total=info['file_size'],
                           unit='B',
                           unit_scale=True,
                           unit_divisor=1024
                           ) as pbar:
                with open(file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            pbar.update(block_size)
        except:
            if file.exists():
                file.unlink()
            for key in state:
                self.files = None
                setattr(self, key, state[key])
            raise

        # Restore the entry state
        self.files = None
        for key in state:
            setattr(self, key, state[key])

        return file
    
    @staticmethod
    def df2ds(df):
        
        # Create a DataArray from the DataFrome
        ds = xr.Dataset()
        
        # Time in mission elapsed seconds
        ds['Elapsed Sec'] = xr.DataArray(df['Elapsed Sec'].values,
                                          dims=['time'],
                                          coords={'time': df.index.values},
                                          name='elapsedsec')
        df.drop('Elapsed Sec', axis='columns', inplace=True)
        
        # Quaternions to GEI
        ds['q'] = xr.DataArray(np.stack([df['q1'].values, df['q2'].values,
                                         df['q3'].values, df['qc'].values],
                                        axis=1),
                               dims=['time', 'quaternions'],
                               coords={'time': df.index.values, 
                                       'quaternions': ['q1', 'q2', 'q3', 'qc']},
                               name='quaternions')
        df.drop(['q1', 'q2', 'q3', 'qc'], axis='columns', inplace=True)
        
        # Spin vector
        ds['w'] = xr.DataArray(np.stack([df['wX'].values, df['wY'].values,
                                         df['wZ'].values, df['w-Phase'].values],
                                        axis=1),
                               dims=['time', 'body'],
                               coords={'time': df.index.values, 
                                       'body': ['X', 'Y', 'Z', 'Phase']},
                               name='spinvector')
        df.drop(['wX', 'wY', 'wZ', 'w-Phase'], axis='columns', inplace=True)
        
        # Body z-axis
        ds['z'] = xr.DataArray(np.stack([df['z-RA'].values,
                                         df['z-Dec'].values,
                                         df['z-Phase'].values],
                                        axis=1),
                               dims=['time', 'celest'],
                               coords={'time': df.index.values, 
                                       'celest': ['RA', 'Dec', 'Phase']},
                               name='bodyvector')
        df.drop(['z-RA', 'z-Dec', 'z-Phase'], axis='columns', inplace=True)
        
        # Angular momentum axis
        ds['L'] = xr.DataArray(np.stack([df['L-RA'].values,
                                         df['L-Dec'].values,
                                         df['L-Phase'].values],
                                        axis=1),
                               dims=['time', 'celest'],
                               coords={'time': df.index.values, 
                                       'celest': ['RA', 'Dec', 'Phase']},
                               name='angmomvector')
        df.drop(['L-RA', 'L-Dec', 'L-Phase'], axis='columns', inplace=True)
        
        # Principal axis of inertia
        ds['P'] = xr.DataArray(np.stack([df['P-RA'].values,
                                         df['P-Dec'].values,
                                         df['P-Phase'].values],
                                        axis=1),
                               dims=['time', 'celest'],
                               coords={'time': df.index.values, 
                                       'celest': ['RA', 'Dec', 'Phase']},
                               name='principalvector')
        df.drop(['P-RA', 'P-Dec', 'P-Phase'], axis='columns', inplace=True)
        
        # Nutation
        ds['Nut'] = xr.DataArray(df['Nut'].values,
                               dims=['time'],
                               coords={'time': df.index.values},
                               name='nutationangle')
        df.drop('Nut', axis='columns', inplace=True)
        
        # Quality flag
        ds['QF'] = xr.DataArray(df['QF'].values,
                                dims=['time'],
                                coords={'time': df.index.values},
                                name='qualityflag')
        df.drop('QF', axis='columns', inplace=True)
        
        return ds

    def file_info(self, interval):
        '''
        Obtain file information from the SDC.

        Returns
        -------
        file_info : list
                    Information about each file.
        '''
        response = self.post('file_info', interval)
        return response.json()
        
    def fname(self, interval=None, sc=None, product=None,
              start_date=None, end_date=None, version=None):
        '''
        Create the file name associated with a given interval.
        
        Parameters
        ----------
        interval : tuple of datetime.datetime
            Start and end time of the data interval
        
        Returns
        -------
        filename : str
            File name
        '''
        
        # In general, the version number is not known
        #   - Query the SDC for the corresponding file name so that it
        #     returns the proper version number.
        if interval is not None:
            response = self.post('file_names', interval)
            files = [Path(file).name
                     for file in response.text.split(',')]
            
            # Return a single file name as a string
            if len(files) == 1:
                files = files[0]
            # Files are returned in reverse chronological order (newest to
            # oldest). Reverse it so that times are in-order when read.
            else:
                files.reverse()
            
            return files
        
        # If the user wants to make their own file name.
        else:
            return '_'.join((sc, product,
                             interval[0].strftime('%Y%j'),
                             interval[1].strftime('%Y%j'),
                             version))
    
    def intervals(self):
        '''
        Break the time interval down into a set of intervals associated
        with individual file names.
        
        Parameters
        ----------
        starttime, endtime : datetime.datetime
            Start and end times of the data interval
        
        Returns
        -------
        intervals : list of tuples
            Time intervals (starttime, endtime) associated with individual
            data files
        '''
        files = self.fname((self.start_date, self.end_date))
        if isinstance(files, str):
            files = [files]
        
        file_parts = [parse_file_name(file, to_datetime=True)
                      for file in files]
        
        intervals = [(parts[2], parts[3]) for parts in file_parts]
        
        return intervals

    
    def load_local_file(self, interval):
        
        # File to load
        file = self.local_path(interval)
        
        # A function to parse the times in the file
        date_parser = lambda x: dt.datetime.strptime(x, '%Y-%jT%H:%M:%S.%f')
        
        # Open and read the file
        with open(file, 'r') as f:
            attrs = self.parse_header(f)
            line = f.readline()
            df = read_csv(f, 
                          delim_whitespace=True,
                          header=None,
                          names=['time', 'Elapsed Sec', 'q1', 'q2', 'q3', 'qc',
                                 'wX', 'wY', 'wZ', 'w-Phase', 'z-RA', 'z-Dec',
                                 'z-Phase', 'L-RA', 'L-Dec', 'L-Phase', 'P-RA',
                                 'P-Dec', 'P-Phase', 'Nut', 'QF'],
                          index_col=0,
                          parse_dates=True,
                          date_parser=date_parser,
                          skipfooter=1,
                          engine='python')
        
        # Convert the dataframe to a dataset
        ds = self.df2ds(df)
        ds.attrs = attrs
        
        # Remove duplicate time stamps
        #   - Some timestamps are duplicated
        #   - The associated values are not identical but they are not outliers
        _, idx = np.unique(ds['time'], return_index=True)
        ds = ds.isel(time=idx)
        
        return ds
    
    def load(self):
        
        intervals = self.intervals()

        data = []
        for interval in intervals:
            if not self.local_file_exists(interval):
                file = self.download(interval)
            data.append(self.load_local_file(interval))
        
        # Concatenate data from multiple files
        data = xr.concat(data, dim='time')
        
        # Adjacent files overlap so we want to remove duplicate times
        if len(intervals) > 1:
            _, idx = np.unique(data['time'], return_index=True)
            data = data.isel(time=idx)
        
        return data
    
    def local_dir(self, interval):
        '''
        Local directory for a given interval. This is relative to the
        PyMMS data directory.
        
        Parameters
        ----------
        interval : tuple of datetime.datetime
            Start and end time of the data interval
        
        Returns
        -------
        dir : pathlib.Path
            Local directory
        '''
        return (data_root / 'ancillary' / self.sc / self.product
                / interval[0].strftime('%Y')
                / interval[0].strftime('%m')
                )

    @staticmethod
    def name2path(file):
        file_parts = parse_file_name(file, to_datetime=True)
        
        return (data_root / 'ancillary' / file_parts[0].lower()
                / file_parts[1].lower()
                / file_parts[2].strftime('%Y')
                / file_parts[2].strftime('%m')
                / file
                )
    
    @staticmethod
    def parse_header_line(line):
        keyval = line.split(' = ')
        key = keyval[0].strip()
        values = [s for s in keyval[1].split(' ') if s != '']
        
        parsed_values = []
        for v in values:
            try:
                v = float(v)
            except ValueError:
                try:
                    v = dt.datetime.strptime(v, '%Y-%jT%H:%M:%S')
                except ValueError:
                    try:
                        v = dt.datetime.strptime(v, '%Y-%jT%H:%M:%S.%f')
                    except ValueError:
                        pass
            parsed_values.append(v)
        
        if len(parsed_values) == 1:
            parsed_values = parsed_values[0]
        elif isinstance(parsed_values[0], float):
            parsed_values = np.array(parsed_values)
        
        return key, parsed_values
    
    @staticmethod
    def parse_header(f):
        header = {}
        header_counter = 0
        while True:
            line = f.readline().strip('\n')
            header_counter +=1
            
            if line.startswith('COMMENT'):
                if 'Major principal axis of inertia' in line:
                    key, value = ANCDownloader.parse_header_line(line)
                    key = 'MPA'
                else:
                    continue
            
            elif line.startswith('DATA_START'):
                break
            
            else:
                try:
                    key, value = ANCDownloader.parse_header_line(line)
                except IndexError:
                    continue
            
            header[key] = value
        
        header['NLINES'] = header_counter
        return header

    def post(self, info_type, interval):
        '''
        Retrieve data from the SDC.

        Returns
        -------
        r : `session.response`
            Response to the request posted to the SDC.
        '''
        # Build the URL sans query
        url = self.url(info_type, query=False)

        # Check on query
        r = self._session.post(url, data=self.query(interval))

        # Check if everything is ok
        if not r.ok:
            r = self.check_response(r)

        # Return the response for the requested URL
        return r

    def query(self, interval):
        '''
        build a dictionary of key-value pairs that serve as the URL
        query string.

        Returns
        -------
        query : dict
            URL query
        '''

        # Adjust end date
        #   - The query takes '%Y-%m-%d' but the object allows
        #     '%Y-%m-%dT%H:%M:%S'
        #   - Further, the query is half-exclusive: [start, end)
        #   - If the dates are the same but the times are different, then
        #     files between self.start_date and self.end_date will not be
        #     found
        #   - In these circumstances, increase the end date by one day
        end_date = interval[1].strftime('%Y-%m-%d')
        if interval[0].date() == interval[1].date() or \
           interval[1].time() != dt.time(0, 0, 0):
            end_date = (interval[1] + dt.timedelta(1)
                        ).strftime('%Y-%m-%d')

        query = {}
        query['sc_id'] = self.sc
        query['product'] = self.product
        query['start_date'] = interval[0].strftime('%Y-%m-%d')
        query['end_date'] = end_date

        return query

    def url(self, info_type, interval=None, query=True):
        """
        Build a URL to query the SDC.

        Parameters
        ----------
        query : bool
            If True (default), add the query string to the url.

        Returns
        -------
        url : str
            URL used to retrieve information from the SDC.
        """
        sep = '/'
        url = sep.join((sdc_home, info_type, data_type))

        # Build query from parts of file names
        if query:
            query_string = '?'
            qdict = self.query(interval)
            for key in qdict:
                query_string += key + '=' + qdict[key] + '&'

            # Combine URL with query string
            url += query_string

        return url

    def version_info(self, interval):
        '''
        Obtain version information from the SDC.

        Returns
        -------
        vinfo : dict
            Version information regarding the requested files
        '''
        response = self.post('version_info', interval)
        return response.json()


def sort_file_names(files):

    # Make sure we have a list-like iterable
    if isinstance(files, str) | (len(files) == 1):
        return files
    
    # Parse the file names
    parts = [parse_file_name(file, to_datetime=True) for file in files]
    
    # Extract the start dates and sort them
    start_dates = np.array([parts[2] for part in parts], dtype='datetime64')
    indices = start_dates.argsort()
    
    # Return the sorted list
    return [files[idx] for idx in indices]


def filter_time(files, start_time, end_time):
    
    if isinstance(files, str):
        files = [files]
    
    # Extract the start and stop times from the file names
    file_parts = [parse_file_name(file) for file in files]
    
    # Return only those files that fall within the requested time interval
    return [file
            for file, parts in zip(file, file_parts)
            if (parts[2] <= end_time) & (parts[3] >= start_time)]


def parse_file_name(file, to_datetime=False):
    regex = re.compile('(MMS[1-4])_([A-Z]+)_([0-9]{7})_([0-9]{7}).V([0-9]{2})')
    m = regex.search(file)
    
    if to_datetime:
        # Convert the time to a datetime
        tstart = dt.datetime.strptime(m.group(3), '%Y%j')
        tstop = dt.datetime.strptime(m.group(4), '%Y%j')
        return m[1], m[2], tstart, tstop, m[5]
    else:
        return m.groups(1, 2, 3, 4, 5)


def interp_phase(phase, time):

    # Unwrap the phase so that it is monotonically increasing
    #   - Interpolating later will not have to skip from 2*pi to 0
    phase = xr.DataArray(np.unwrap(phase),
                         dims=['time'],
                         coords={'time': phase['time']},
                         name='unwrapped-phase')
    
    # Interpolate the phase to the given timestamps
    return phase.interp_like(time)


def despin(defatt, time, vector='L', offset=0.0, spinup=False):

    # Interpolate the phase to the given time stamps
    phase = interp_phase(np.deg2rad(defatt[vector].loc[:, 'Phase']), time)

    # Negate the phase to spin up
    if spinup:
        phase = -phase

    # Caluclate sine and cosine only once
    sinPhase = np.sin(phase + offset)
    cosPhase = np.cos(phase + offset)

    # Create the rotation matrix
    #   - PHASE is a rotation about the Z-axis to align the
    #     x-axis with the sun.
    #         |  cos  sin  0 |
    #     T = | -sin  cos  0 |
    #         |   0    0   1 |
    spun2despun = xr.DataArray(np.zeros((len(time), 3, 3)),
                               dims=['time', 'spun', 'despun'],
                               coords={'time': time,
                                       'spun': ['x', 'y', 'z'],
                                       'despun': ['x', 'y', 'z']})
    spun2despun[:,0,0] =  cosPhase
    spun2despun[:,0,1] =  sinPhase
    spun2despun[:,1,0] = -sinPhase
    spun2despun[:,1,1] =  cosPhase
    spun2despun[:,2,2] =  1

    return spun2despun


def load_ancillary(sc='mms1', product='defatt',
                   start_date=None, end_date=None):
    """
    Load Ancillary data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    product : str
        Ancillary data product to load.
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    
    Returns
    -------
    data : `xarray.Dataset`
        Ancillary data.
    """
    
    anc = ANCDownloader(sc, product, start_date, end_date)
    data = anc.load()
    data = data.sel(indexers={'time': slice(start_date, end_date)})
    
    # Add data descriptors to attributes
    data.attrs['sc'] = sc
    data.attrs['product'] = product
    
    return data