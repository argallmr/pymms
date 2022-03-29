import glob
import os
import io
import re
import requests
import csv
import pymms
from tqdm import tqdm
import datetime as dt
import numpy as np
from cdflib import epochs
from urllib.parse import parse_qs
import urllib3
import warnings
from scipy.io import readsav
from getpass import getpass

data_root = pymms.config['data_root']
dropbox_root = pymms.config['dropbox_root']
mirror_root = pymms.config['mirror_root']

mms_username = pymms.config['username']
mms_password = pymms.config['password']


class MrMMS_SDC_API:
    """Interface with NASA's MMS SDC API

    Interface with the Science Data Center (SDC) API of the
    Magnetospheric Multiscale (MMS) mission.
    https://lasp.colorado.edu/mms/sdc/public/

    Params:
        sc (str,list):       Spacecraft IDs ('mms1', 'mms2', 'mms3', 'mms4')
        instr (str,list):    Instrument IDs
        mode (str,list):     Data rate mode ('slow', 'fast', 'srvy', 'brst')
        level (str,list):    Data quality level ('l1a', 'l1b', 'sitl', 'l2pre', 'l2', 'l3')
        data_type (str):     Type of data ('ancillary', 'hk', 'science')
        end_date (str):      End date of data interval, formatted as either %Y-%m-%d or
                             %Y-%m-%dT%H:%M:%S.
        files (str,list):    File names. If set, automatically sets `sc`, `instr`, `mode`,
                             `level`, `optdesc` and `version` to None.
        offline (bool):      Do not search for file information online.
        optdesc (str,list):  Optional file name descriptor
        site (str):          SDC site to use ('public', 'private'). Setting `level`
                             automatically sets `site`. If `level` is 'l2' or 'l3', then
                             `site`='public' otherwise `site`='private'.
        start_date (str):    Start date of data interval, formatted as either %Y-%m-%d or
                             %Y-%m-%dT%H:%M:%S.
        version (str,list):  File version numbers.
    """

    def __init__(self, sc=None, instr=None, mode=None, level=None,
                 data_type='science',
                 end_date=None,
                 files=None,
                 offline=False,
                 optdesc=None,
                 product=None,
                 site='public',
                 start_date=None,
                 version=None):

        # Set attributes
        #   - Put site before level because level will auto-set site
        #   - Put files last because it will reset most fields
        self.site = site

        self.data_type = data_type
        self.product = product
        self.end_date = end_date
        self.instr = instr
        self.level = level
        self.mode = mode
        self.offline = offline
        self.optdesc = optdesc
        self.sc = sc
        self.start_date = start_date
        self.version = version

        self.files = files

        self._data_root  = data_root
        self._dropbox_root = dropbox_root
        self._mirror_root = mirror_root
        self._sdc_home  = 'https://lasp.colorado.edu/mms/sdc'
        self._info_type = 'download'

        # Create a persistent session
        self._session = requests.Session()
        if (mms_username is not None) and (mms_password is not None):
            self._session.auth = (mms_username, mms_password)

    def __str__(self):
        return self.url()

    # https://stackoverflow.com/questions/17576009/python-class-property-use-setter-but-evade-getter
    def __setattr__(self, name, value):
        """Control attribute values as they are set."""

        # TYPE OF INFO
        #   - Unset other complementary options
        #   - Ensure that at least one of (download | file_names |
        #     version_info | file_info) are true
        if name == 'data_type':
            if 'gls_selections' in value:
                if value[15:] not in ('mp-dl-unh',):
                    raise ValueError('Unknown GLS Selections type.')
            elif value not in ('ancillary', 'hk', 'science',
                               'abs_selections', 'sitl_selections',
                               'bdm_sitl_changes'):
                raise ValueError('Invalid value {} for attribute'
                                 ' "{}".'.format(value, name))

            # Unset attributes related to data_type = 'science'
            if 'selections' in value:
                self.sc = None
                self.instr = None
                self.mode = None
                self.level = None
                self.optdesc = None
                self.version = None

        elif name == 'files':
            if value is not None:
                # Keep track of site because setting
                # self.level = None will set self.site = 'public'
                site = self.site
                self.sc = None
                self.instr = None
                self.mode = None
                self.level = None
                self.optdesc = None
                self.version = None
                self.site = site

        elif name == 'level':
            # L2 and L3 are the only public data levels
            if value in [None, 'l2', 'l3']:
                self.site = 'public'
            else:
                self.site = 'private'

        elif name == 'site':
            # Team site is most commonly referred to as the "team",
            # or "private" site, but in the URL is referred to as the
            # "sitl" site. Accept any of these values.
            if value in ('private', 'team', 'sitl'):
                value = 'sitl'
            elif value == 'public':
                value = 'public'
            else:
                raise ValueError('Invalid value for attribute {}.'
                                 .format(name)
                                 )

        elif name in ('start_date', 'end_date'):
            # Convert string to datetime object
            if isinstance(value, str):
                try:
                    value = dt.datetime.strptime(value[0:19],
                                                 '%Y-%m-%dT%H:%M:%S'
                                                 )
                except ValueError:
                    try:
                        value = dt.datetime.strptime(value, '%Y-%m-%d')
                    except ValueError:
                        raise

        # Set the value
        super(MrMMS_SDC_API, self).__setattr__(name, value)

    def url(self, query=True):
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
        url = sep.join((self._sdc_home, self.site, 'files', 'api', 'v1',
                        self._info_type, self.data_type))

        # Build query from parts of file names
        if query:
            query_string = '?'
            qdict = self.query()
            for key in qdict:
                query_string += key + '=' + qdict[key] + '&'

            # Combine URL with query string
            url += query_string

        return url

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

    def download_files(self):
        '''
        Download files from the SDC. First, search the local file
        system to see if they have already been downloaded.

        Returns
        -------
        local_files : list
            Names of the local files.
        '''

        # Get available files
        local_files, remote_files = self.search()
        
        if self.offline:
            return local_files

        # Download remote files
        #   - file_info() does not want the remote path
        if len(remote_files) > 0:
            remote_files = [file.split('/')[-1] for file in remote_files]
            downloaded_files = self.download_from_sdc(remote_files)
            local_files.extend(downloaded_files)

        return local_files

    def download_from_sdc(self, file_names):
        '''
        Download multiple files from the SDC. To prevent downloading the
        same file multiple times and to properly filter by file start time
        see the download_files method.

        Parameters
        ----------
        file_names : str, list
            File names of the data files to be downloaded. See
            the file_names method.

        Returns
        -------
        local_files : list
            Names of the local files. Remote files downloaded
            only if they do not already exist locally
        '''

        # Make sure files is a list
        if isinstance(file_names, str):
            file_names = [file_names]

        # Get information on the files that were found
        #   - To do that, specify the specific files.
        #     This sets all other properties to None
        #   - Save the state of the object as it currently
        #     is so that it can be restored
        #   - Setting FILES will indirectly cause SITE='public'.
        #     Keep track of SITE.
        state = {}
        state['sc'] = self.sc
        state['instr'] = self.instr
        state['mode'] = self.mode
        state['level'] = self.level
        state['optdesc'] = self.optdesc
        state['version'] = self.version
        state['files'] = self.files

        # Get file name and size
        self.files = file_names
        file_info = self.file_info()

        # Build the URL sans query
        self._info_type = 'download'
        url = self.url(query=False)

        # Amount to download per iteration
        block_size = 1024*128
        local_file_names = []

        # Download each file individually
        for info in file_info['files']:
            # Create the destination directory
            file = self.name2path(info['file_name'])
            if not os.path.isdir(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))

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
                if os.path.isfile(file):
                    os.remove(file)
                for key in state:
                    self.files = None
                    setattr(self, key, state[key])
                raise

            local_file_names.append(file)

        # Restore the entry state
        self.files = None
        for key in state:
            setattr(self, key, state[key])

        return local_file_names


    def download_from_sdc_v1(self, file_names):
        '''
        Download multiple files from the SDC. To prevent downloading the
        same file multiple times and to properly filter by file start time
        see the download_files method.

        This version of the program calls `self.file_info()` for each
        file name given, whereas `download_from_sdc_v1` calls it once
        for all files. In the event of many files, `self.get()` was
        altered to use `requests.post()` instead of `requests.get()`
        if the url was too long (i.e. too many files).

        Parameters
        ----------
        file_names : str, list
            File names of the data files to be downloaded. See
            the file_names method.

        Returns
        -------
        local_files : list
            Names of the local files. Remote files downloaded
            only if they do not already exist locally
        '''

        # Make sure files is a list
        if isinstance(file_names, str):
            file_names = [file_names]

        # Get information on the files that were found
        #   - To do that, specify the specific files.
        #     This sets all other properties to None
        #   - Save the state of the object as it currently
        #     is so that it can be restored
        #   - Setting FILES will indirectly cause SITE='public'.
        #     Keep track of SITE.
        site = self.site
        state = {}
        state['sc'] = self.sc
        state['instr'] = self.instr
        state['mode'] = self.mode
        state['level'] = self.level
        state['optdesc'] = self.optdesc
        state['version'] = self.version
        state['files'] = self.files

        # Build the URL sans query
        self.site = site
        self._info_type = 'download'
        url = self.url(query=False)

        # Amount to download per iteration
        block_size = 1024*128
        local_file_names = []

        # Download each file individually
        for file_name in file_names:
            self.files = file_name
            info = self.file_info()['files'][0]

            # Create the destination directory
            file = self.name2path(info['file_name'])
            if not os.path.isdir(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))

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
                if os.path.isfile(file):
                    os.remove(file)
                for key in state:
                    self.files = None
                    setattr(self, key, state[key])
                raise

            local_file_names.append(file)

        # Restore the entry state
        self.files = None
        for key in state:
            setattr(self, key, state[key])

        return local_file_names

    def download(self):
        '''
        Download multiple files. First, search the local file system
        to see if any of the files have been downloaded previously.

        Returns
        -------
        local_files : list
            Names of the local files. Remote files downloaded
            only if they do not already exist locally
        '''
        warnings.warn('This method will be removed in the future. Use the get method.',
                      DeprecationWarning)

        self._info_type = 'download'
        # Build the URL sans query
        url = self.url(query=False)

        # Get available files
        local_files, remote_files = self.search()
        if self.offline:
            return local_files

        # Get information on the files that were found
        #   - To do that, specify the specific files. This sets all other
        #     properties to None
        #   - Save the state of the object as it currently is so that it can
        #     be restored
        #   - Setting FILES will indirectly cause SITE='public'. Keep track
        #     of SITE.
        site = self.site
        state = {}
        state['sc'] = self.sc
        state['instr'] = self.instr
        state['mode'] = self.mode
        state['level'] = self.level
        state['optdesc'] = self.optdesc
        state['version'] = self.version
        state['files'] = self.files
        self.files = [file.split('/')[-1] for file in remote_files]

        self.site = site
        file_info = self.file_info()

        # Amount to download per iteration
        block_size = 1024*128

        # Download each file individually
        for info in file_info['files']:
            # Create the destination directory
            file = self.name2path(info['file_name'])
            if not os.path.isdir(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))

            # Downloading and progress bar:
            # https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
            # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
            try:
                r = self._session.post(url,
                                       data={'file': info['file_name']},
                                       stream=True)
                with tqdm(total=info['file_size'], unit='B', unit_scale=True,
                     unit_divisor=1024) as pbar:
                    with open(file, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=block_size):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                                pbar.update(block_size)
            except:
                if os.path.isfile(file):
                    os.remove(file)
                for key in state:
                    self.files = None
                    setattr(self, key, state[key])
                raise

            local_files.append(file)

        self.files = None
        for key in state:
            setattr(self, key, state[key])

        return local_files

    def file_info(self):
        '''
        Obtain file information from the SDC.

        Returns
        -------
        file_info : list
                    Information about each file.
        '''
        self._info_type = 'file_info'
        response = self.get()
        return response.json()

    def file_names(self):
        '''
        Obtain file names from the SDC. Note that the SDC accepts only
        start and end dates, not datetimes. Therefore the files returned
        by this function may lie outside the time interval of interest.
        For a more precise list of file names, use the search method or
        filter the files with filter_time.

        Returns
        -------
        file_names : list
            Names of the requested files.
        '''
        self._info_type = 'file_names'
        response = self.get()

        # If no files were found, the empty string is the response
        # Return [] instead of [''] so that len() is zero.
        if response.text == '':
            return []
        return response.text.split(',')

    def get(self):
        '''
        Retrieve information from the SDC.

        Returns
        -------
        r : `session.response`
            Response to the request posted to the SDC.
        '''
        # Build the URL sans query
        url = self.url(query=False)

        # Check on query
        #   - Use POST if the URL is too long
        r = self._session.get(url, params=self.query())
        if r.status_code == 414:
            r = self._session.post(url, data=self.query())

        # Check if everything is ok
        if not r.ok:
            r = self.check_response(r)

        # Return the response for the requested URL
        return r

    def local_file_names(self, mirror=False):
        '''
        Search for MMS files on the local system. Files must be
        located in an MMS-like directory structure.

        Parameters
        ----------
        mirror : bool
            If True, the local data directory is used as the
            root directory. Otherwise the mirror directory is
            used.

        Returns
        -------
        local_files : list
            Names of the local files
        '''

        # Search the mirror or local directory
        if mirror:
            data_root = self._mirror_root
        else:
            data_root = self._data_root

        # If no start or end date have been defined,
        #   - Start at beginning of mission
        #   - End at today's date
        start_date = self.start_date
        end_date = self.end_date

        # Create all dates between start_date and end_date
        deltat = dt.timedelta(days=1)
        dates = []
        while start_date <= end_date:
            dates.append(start_date.strftime('%Y%m%d'))
            start_date += deltat

        # Paths in which to look for files
        #   - Files of all versions and times within interval
        if 'selections' in self.data_type:
            paths = construct_path(data_type=self.data_type,
                                   root=data_root, files=True)
        else:
            paths = construct_path(self.sc, self.instr, self.mode, self.level,
                                   dates, optdesc=self.optdesc,
                                   root=data_root, files=True)

        # Search
        result = []
        pwd = os.getcwd()
        for path in paths:
            root = os.path.dirname(path)

            try:
                os.chdir(root)
            except FileNotFoundError:
                continue
            except:
                os.chdir(pwd)
                raise

            for file in glob.glob(os.path.basename(path)):
                result.append(os.path.join(root, file))

        os.chdir(pwd)

        return result

    def login(self, username=None, password=None):
        '''
        Log-In to the SDC

        Parameters
        ----------
        username (str):     Account username
        password (str):     Account password
        '''

        # Ask for inputs
        if username is None:
            username = input('username: ')

        if password is None:
            password = input('password: ')

        # Save credentials
        self._session.auth = (username, password)

    def name2path(self, filename):
        '''
        Convert remote file names to local file name.

        Directories of a remote file name are separated by the '/' character,
        as in a web address.

        Parameters
        ----------
        filename : str
            File name for which the local path is desired.

        Returns
        -------
        path : str
            Equivalent local file name. This is the location to
            which local files are downloaded.
        '''
        parts = filename.split('_')

        # burst data selection directories and file names are structured as
        #   - dirname:  sitl/[type]_selections/
        #   - basename: [type]_selections_[optdesc]_YYYY-MM-DD-hh-mm-ss.sav
        # To get year, index from end to skip optional descriptor
        if parts[1] == 'selections':
            path = os.path.join(self._data_root, 'sitl',
                                '_'.join(parts[0:2]),
                                filename)

        # Burst directories and file names are structured as:
        #   - dirname:  sc/instr/mode/level[/optdesc]/YYYY/MM/DD/
        #   - basename: sc_instr_mode_level[_optdesc]_YYYYMMDDhhmmss_vX.Y.Z.cdf
        # Index from end to catch the optional descriptor, if it exists
        elif parts[2] == 'brst':
            path = os.path.join(self._data_root, *parts[0:-2],
                                parts[-2][0:4], parts[-2][4:6],
                                parts[-2][6:8], filename)

        # Survey (slow,fast,srvy) directories and file names are structured as:
        #   - dirname:  sc/instr/mode/level[/optdesc]/YYYY/MM/
        #   - basename: sc_instr_mode_level[_optdesc]_YYYYMMDD_vX.Y.Z.cdf
        # Index from end to catch the optional descriptor, if it exists
        else:
            path = os.path.join(self._data_root, *parts[0:-2],
                                parts[-2][0:4], parts[-2][4:6], filename)

        return path

    def parse_file_names(self, filename):
        '''
        Parse an official MMS file name. MMS file names are formatted as
            sc_instr_mode_level[_optdesc]_tstart_vX.Y.Z.cdf
        where
            sc:       spacecraft id
            instr:    instrument id
            mode:     data rate mode
            level:    data level
            optdesc:  optional filename descriptor
            tstart:   start time of file
            vX.Y.Z:   file version, with X, Y, and Z version numbers

        Parameters
        ----------
        filename : str
            An MMS file name

        Returns
        -------
        parts : tuple
            A tuples ordered as
                (sc, instr, mode, level, optdesc, tstart, version)
            If opdesc is not present in the file name, the output will
            contain the empty string ('').
        '''
        parts = os.path.basename(filename).split('_')

        # If the file does not have an optional descriptor,
        # put an empty string in its place.
        if len(parts) == 6:
            parts.insert(-2, '')

        # Remove the file extension ``.cdf''
        parts[-1] = parts[-1][0:-4]
        return tuple(parts)

    def post(self):
        '''
        Retrieve data from the SDC.

        Returns
        -------
        r : `session.response`
            Response to the request posted to the SDC.
        '''
        # Build the URL sans query
        url = self.url(query=False)

        # Check on query
        r = self._session.post(url, data=self.query())

        # Check if everything is ok
        if not r.ok:
            r = self.check_response(r)

        # Return the response for the requested URL
        return r

    def query(self):
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
        if self.end_date is not None:
            end_date = self.end_date.strftime('%Y-%m-%d')
            if self.start_date.date() == self.end_date.date() or \
               self.end_date.time() != dt.time(0, 0, 0):
                end_date = (self.end_date + dt.timedelta(1)
                            ).strftime('%Y-%m-%d')

        query = {}
        if self.sc is not None:
            query['sc_id'] = self.sc if isinstance(self.sc, str) \
                                     else ','.join(self.sc)
        if self.instr is not None:
            query['instrument_id'] = self.instr \
                                     if isinstance(self.instr, str) \
                                     else ','.join(self.instr)
        if self.mode is not None:
            query['data_rate_mode'] = self.mode if isinstance(self.mode, str) \
                                                else ','.join(self.mode)
        if self.level is not None:
            query['data_level'] = self.level if isinstance(self.level, str) \
                                             else ','.join(self.level)
        if self.optdesc is not None:
            query['descriptor'] = self.optdesc \
                                  if isinstance(self.optdesc, str) \
                                  else ','.join(self.optdesc)
        if self.product is not None:
            query['product'] = self.product \
                               if isinstance(self.product, str) \
                               else ','.join(self.product)
        if self.version is not None:
            query['version'] = self.version if isinstance(self.version, str) \
                                            else ','.join(self.version)
        if self.files is not None:
            query['files'] = self.files if isinstance(self.files, str) \
                                        else ','.join(self.files)
        if self.start_date is not None:
            query['start_date'] = self.start_date.strftime('%Y-%m-%d')
        if self.end_date is not None:
            query['end_date'] = end_date

        return query

    def remote2localnames(self, remote_names):
        '''
        Convert remote file names to local file names.

        Directories of a remote file name are separated by the '/' character,
        as in a web address.

        Parameters
        ----------
        remote_names : list
            Remote file names returned by FileNames.

        Returns
        -------
        local_names : list
            Equivalent local file name. This is the location to
            which local files are downloaded.
        '''
        # os.path.join() requires string arguments
        #   - str.split() return list.
        #   - Unpack with *: https://docs.python.org/2/tutorial/controlflow.html#unpacking-argument-lists
        local_names = list()
        for file in remote_names:
            local_names.append(os.path.join(self._data_root,
                               *file.split('/')[2:]))

        if (len(remote_names) == 1) & (type(remote_names) == 'str'):
            local_names = local_names[0]

        return local_names

    def search(self):
        '''
        Search for files locally and at the SDC.

        Returns
        -------
        files : tuple
            Local and remote files within the interval, returned as
            (local, remote), where `local` and `remote` are lists.
        '''

        # Search locally if offline
        if self.offline:
            remote_files = []
            local_files = self.local_file_names()
            local_files = filter_time(local_files,
                                      self.start_date,
                                      self.end_date
                                      )

        # Search remote first
        #   - SDC is definitive source of files
        #   - Returns most recent version
        else:
            # Because file names contain only the start time of
            # the data interval, filter_time will always return
            # at least one file -- the file that starts before
            # the time interval. Filter files before splitting
            # them into local and remote files so that a lingering
            # remote file will not remain if all valid files are
            # local.
            local_files = []
            remote_files = self.file_names()
            remote_files = filter_time(remote_files,
                                       self.start_date,
                                       self.end_date
                                       )

            # Search for the equivalent local file names
            local_files = self.remote2localnames(remote_files)
            idx = [i for i, local in enumerate(local_files)
                   if os.path.isfile(local)
                   ]

            # Filter based on location
            local_files = [local_files[i] for i in idx]
            remote_files = [remote_files[i] for i in range(len(remote_files))
                            if i not in idx
                            ]

        return (local_files, remote_files)

    def version_info(self):
        '''
        Obtain version information from the SDC.

        Returns
        -------
        vinfo : dict
            Version information regarding the requested files
        '''
        self._info_type = 'version_info'
        response = self.post()
        return response.json()


def _datetime_to_list(datetime):
    return [datetime.year, datetime.month, datetime.day,
            datetime.hour, datetime.minute, datetime.second,
            datetime.microsecond // 1000, datetime.microsecond % 1000, 0
            ]


def datetime_to_tai(t_datetime):
    # Convert datetime to TAI
    #   - TAI timestaps are TAI seconds elapsed since 1958-01-01
    return tt2000_to_tai(datetime_to_tt2000(t_datetime))


def datetime_to_tt2000(t_datetime):
    # Convert datetime to TT2000
    #   - TT2000 are TAI nanoseconds elapsed since 2000-01-01
    t_list = _datetime_to_list(t_datetime)
    return epochs.CDFepoch.compute_tt2000(t_list)


def tai_to_tt2000(t_tai):
    # Convert TAI to TT2000
    #   - TAI timestaps are TAI seconds elapsed since 1958-01-01
    #   - TT2000 are TAI nanoseconds elapsed since 2000-01-01
    t_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])
    return np.asarray(t_tai) * int(1e9) + t_1958


def tai_to_datetime(t_tai):
    # Convert TAI to datetime
    #   - TAI timestaps are TAI seconds elapsed since 1958-01-01
    return tt2000_to_datetime(tai_to_tt2000(t_tai))


def tt2000_to_tai(t_tt2000):
    # Convert TT2000 to TAI
    #   - TAI timestaps are TAI seconds elapsed since 1958-01-01
    #   - TT2000 are TAI nanoseconds elapsed since 2000-01-01
    t_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])
    return (t_tt2000 - t_1958) // int(1e9)


def tt2000_to_datetime(t_tt2000):
    # Convert datetime to TT2000
    #   - TT2000 are TAI nanoseconds elapsed since 2000-01-01
    tepoch = epochs.CDFepoch()
    return tepoch.to_datetime(t_tt2000)


def _response_text_to_dict(text):
    # Read first line as dict keys. Cut text from TAI keys
    f = io.StringIO(text)
    reader = csv.reader(f, delimiter=',')

    # Create a dictionary from the header
    data = dict()
    for key in next(reader):

        # See sitl_selections()
        if key.startswith(('start_time', 'end_time')):
            match = re.search('((start|end)_time)_utc', key)
            key = match.group(1)

        # See burst_data_segments()
        elif key.startswith('TAI'):
            match = re.search('(TAI(START|END)TIME)', key)
            key = match.group(1)

        data[key.lower()] = []

    # Read remaining lines into columns
    keys = data.keys()
    for row in reader:
        for key, value in zip(keys, row):
            data[key].append(value)

    return data


def burst_data_segments(start_date, end_date,
                        team=False, username=None):
    """
    Get information about burst data segments. Burst segments that
    were selected in the back structure are available through this
    service, but not through `sitl_selections()`. Also, the time
    between contiguous segments is 10 seconds.

    Parameters
    ----------
    start_date : `datetime`
        Start date of time interval for which information is desired.
    end_date : `datetime`
        End date of time interval for which information is desired.
    team : bool=False
        If set, information will be taken from the team site
        (login required). Otherwise, it is take from the public site.

    Returns
    -------
    data : dict
        Dictionary of information about burst data segments
            datasegmentid
            taistarttime    - Start time of burst segment in
                              TAI sec since 1958-01-01
            taiendtime      - End time of burst segment in
                              TAI sec since 1958-01-01
            parametersetid
            fom             - Figure of merit given to the burst segment
            ispending
            inplaylist
            status          - Download status of the segment
            numevalcycles
            sourceid        - Username of SITL who selected the segment
            createtime      - Time the selections were submitted as datetime (?)
            finishtime      - Time the selections were downlinked as datetime (?)
            obs1numbufs
            obs2numbufs
            obs3numbufs
            obs4numbufs
            obs1allocbufs
            obs2allocbufs
            obs3allocbufs
            obs4allocbufs
            obs1remfiles
            obs2remfiles
            obs3remfiles
            obs4remfiles
            discussion      - Description given to segment by SITL
            dt              - Duration of burst segment in seconds
            tstart          - Start time of burst segment as datetime
            tstop           - End time of burst segment as datetime
    """

    # Convert times to TAI since 1958
    t0 = _datetime_to_list(start_date)
    t1 = _datetime_to_list(end_date)
    t_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])
    t0 = int((epochs.CDFepoch.compute_tt2000(t0) - t_1958) // 1e9)
    t1 = int((epochs.CDFepoch.compute_tt2000(t1) - t_1958) // 1e9)

    # URL
    url_path = 'https://lasp.colorado.edu/mms/sdc/'
    url_path += 'sitl/latis/dap/' if team else 'public/service/latis/'
    url_path += 'mms_burst_data_segment.csv'

    # Query string
    query = {}
    query['TAISTARTTIME>'] = '{0:d}'.format(t0)
    query['TAIENDTIME<'] = '{0:d}'.format(t1)

    # Post the query
#    cookies = None
#    if team:
#        cookies = sdc_login(username)

    # Get the log-in information
    sesh = requests.Session()
    r = sesh.get(url_path, params=query)
    if r.status_code != 200:
        raise ConnectionError('{}: {}'.format(r.status_code, r.reason))

    # Read first line as dict keys. Cut text from TAI keys
    data = _response_text_to_dict(r.text)

    # Convert to useful types
    types = ['int16', 'int64', 'int64', 'str', 'float32', 'int8',
             'int8', 'str', 'int32', 'str', 'datetime', 'datetime',
             'int32', 'int32', 'int32', 'int32', 'int32', 'int32',
             'int32', 'int32', 'int32', 'int32', 'int32', 'str']
    for key, type in zip(data, types):
        if type == 'str':
            pass
        elif type == 'datetime':
            data[key] = [dt.datetime.strptime(value,
                                              '%Y-%m-%d %H:%M:%S'
                                              )
                         if value != '' else value
                         for value in data[key]
                         ]
        else:
            data[key] = np.asarray(data[key], dtype=type)

    # Add useful tags
    #   - Number of seconds elapsed
    #   - TAISTARTIME as datetime
    #   - TAIENDTIME as datetime
    data['dt'] = data['taiendtime'] - data['taistarttime']

    # Convert TAISTART/ENDTIME to datetimes
    #    NOTE! If data['TAISTARTTIME'] is a scalar, this will not work
    #          unless everything after "in" is turned into a list
    data['tstart'] = [dt.datetime(
                         *value[0:6], value[6]*1000+value[7]
                         )
                      for value in
                      epochs.CDFepoch.breakdown_tt2000(
                          data['taistarttime']*int(1e9)+t_1958
                          )
                      ]
    data['tstop'] = [dt.datetime(
                        *value[0:6], value[6]*1000+value[7]
                        )
                     for value in
                     epochs.CDFepoch.breakdown_tt2000(
                         data['taiendtime']*int(1e9)+t_1958
                         )
                     ]
    data['start_time'] = [tstart.strftime('%Y-%m-%d %H:%M:%S')
                          for tstart in data['tstart']]
    data['stop_time'] = [tend.strftime('%Y-%m-%d %H:%M:%S')
                         for tend in data['tstop']]

    return data


def burst_selections(selection_type, start, stop):
    '''
    A factory function for retrieving burst selection data.
    
    Parameters
    ----------
    type : str
        The type of data to retrieve. Options include:
        Type       Source                     Description
        =========  =========================  =======================================
        abs        download_selections_files  ABS selections
        sitl       download_selections_files  SITL selections
        sitl+back  burst_data_segments        SITL and backstructure selections
        gls        download_selections_files  ground loop selections from 'mp-dl-unh'
        mp-dl-unh  download_selections_files  ground loop selections from 'mp-dl-unh'
        =========  ========================   =======================================
    start, stop : `datetime.datetime`
        Time interval for which data is to be retrieved
    
    Returns
    -------
    data : struct
        The requested data
    '''
    if isinstance(start, (int, np.integer)):
        orbit = mission_events('orbit', start, start)
        start = min(orbit['tstart'])
    if isinstance(stop, (int, np.integer)):
        orbit = mission_events('orbit', stop, stop)
        stop = max(orbit['tend'])
    
    data_retriever = _get_selection_retriever(selection_type)
    return data_retriever(start, stop)

def _get_selection_retriever(selection_type):
    '''
    Creator function for mission events data.
    
    Parameters
    ----------
    selections_type : str
        Type of data desired
    
    Returns
    -------
    func : function
        Function to generate the data
    '''
    if selection_type == 'abs':
        return _get_abs_data
    elif selection_type == 'sitl':
        return _get_sitl_data
    elif selection_type == 'sitl+back':
        return burst_data_segments
    elif selection_type in ('gls', 'mp-dl-unh'):
        return _get_gls_data
    else:
        raise ValueError('Burst selection type {} not recognized'
                         .format(selection_type))


def _get_abs_data(start, stop):
    '''
    Download and read Automated Burst Selections sav files.
    '''
    abs_files = download_selections_files('abs_selections',
                                          start_date=start, end_date=stop)
    return _read_fom_structures(abs_files)


def _get_sitl_data(start, stop):
    '''
    Download and read SITL selections sav files.
    '''
    sitl_files = download_selections_files('sitl_selections',
                                           start_date=start, end_date=stop)
    return _read_fom_structures(sitl_files)


def _get_gls_data(start, stop):
    '''
    Download and read Ground Loop Selections csv files.
    '''
    gls_files = download_selections_files('gls_selections',
                                          gls_type='mp-dl-unh',
                                          start_date=start, end_date=stop)
    
    # Prepare to loop over files
    if isinstance(gls_files, str):
        gls_files = [gls_files]
    
    # Statistics of bad selections
    fskip = 0  # number of files skipped
    nskip = 0  # number of selections skipped
    nexpand = 0  # number of selections expanded
    result = dict()
    
    # Read multiple files
    for file in gls_files:
        data = read_gls_csv(file)
        
        # Accumulative sum of errors
        fskip += data['errors']['fskip']
        nskip += data['errors']['nskip']
        nexpand += data['errors']['nexpand']
        if data['errors']['fskip']:
            continue
        del data['errors']
        
        # Extend results from all files. Keep track of the file
        # names since entries can change. The most recent file
        # contains the correct selections information.
        if len(result) == 0:
            result = data
            result['file'] = [file] * len(result['fom'])
        else:
            result['file'].extend([file] * len(result['fom']))
            for key, value in data.items():
                result[key].extend(value)
    
    # Display bad data
    if (fskip > 0) | (nskip > 0) | (nexpand > 0):
        print('GLS Selection Adjustments:')
        print('  # files skipped:    {}'.format(fskip))
        print('  # entries skipped:  {}'.format(nskip))
        print('  # entries expanded: {}'.format(nexpand))
    
    return result


def _read_fom_structures(files):
    '''
    Read multiple IDL sav files containing ABS or SITL selections.
    '''
    # Read data from all files
    result = dict()
    for file in files:
        data = read_eva_fom_structure(file)
        if data['valid'] == 0:
            print('Skipping invalid file {0}'.format(file))
            continue
        
        # Turn scalars into lists so they can be accumulated
        # across multiple files.
        #
        # Keep track of file name because the same selections
        # (or updated versions of the same selections) can be
        # stored in multiple files, if they were submitted to
        # the SDC multiple times.
        if len(result) == 0:
            result = {key:
                      (value
                       if isinstance(value, list)
                       else [value]
                       )
                      for key, value in data.items()
                      }
            result['file'] = [file] * len(data['fom'])
        
        # Append or extend data from subsequent files
        else:
            result['file'].extend([file] * len(data['fom']))
            for key, value in data.items():
                if isinstance(value, list):
                    result[key].extend(value)
                else:
                    result[key].append(value)
    
    return result


def construct_file_names(*args, data_type='science', **kwargs):
    '''
    Construct a file name compliant with MMS file name format guidelines.

    MMS file names follow the convention
        sc_instr_mode_level[_optdesc]_tstart_vX.Y.Z.cdf

    Parameters
    ----------
        *args : dict
            Arguments to be passed along.
        data_type : str
            Type of file names to construct. Options are:
            science or *_selections. If science, inputs are
            passed to construct_science_file_names. If
            *_selections, inputs are passed to
            construct_selections_file_names.
        **kwargs : dict
            Keywords to be passed along.

    Returns
    -------
        fnames : list
            File names constructed from inputs.
    '''

    if data_type == 'science':
        fnames = construct_science_file_names(*args, **kwargs)
    elif 'selections' in data_type:
        fnames = construct_selections_file_names(data_type, **kwargs)

    return fnames


def construct_selections_file_names(data_type, tstart='*', gls_type=None):
    '''
    Construct a SITL selections file name compliant with
    MMS file name format guidelines.

    MMS SITL selection file names follow the convention
        data_type_[gls_type]_tstart.sav

    Parameters
    ----------
        data_type : str, list, tuple
            Type of selections. Options are abs_selections
            sitl_selections, or gls_selections.
        tstart : str, list
            Start time of data file. The format is
            YYYY-MM-DD-hh-mm-ss. If not given, the default is "*".
        gls_type : str, list
            Type of ground-loop selections. Possible values are:
            mp-dl-unh.

    Returns
    -------
        fnames : list
            File names constructed from inputs.
    '''

    # Convert inputs to iterable lists
    if isinstance(data_type, str):
        data_type = [data_type]
    if isinstance(gls_type, str):
        gls_type = [gls_type]
    if isinstance(tstart, str):
        tstart = [tstart]

    # Accept tuples, as those returned by Construct_Filename
    if isinstance(data_type, tuple):
        data_type = [file[0] for file in data_type]
        tstart = [file[-1] for file in data_type]

        if len(data_type > 2):
            gls_type = [file[1] for file in data_type]
        else:
            gls_type = None

    # Create the file names
    if gls_type is None:
        fnames = ['_'.join((d, g, t+'.sav'))
                  for d in data_type
                  for t in tstart
                  ]

    else:
        fnames = ['_'.join((d, g, t+'.sav'))
                  for d in data_type
                  for g in gls_type
                  for t in tstart
                  ]

    return fnames


def construct_science_file_names(sc, instr=None, mode=None, level=None,
                                 tstart='*', version='*', optdesc=None):
    '''
    Construct a science file name compliant with MMS
    file name format guidelines.

    MMS science file names follow the convention
        sc_instr_mode_level[_optdesc]_tstart_vX.Y.Z.cdf

    Parameters
    ----------
        sc : str, list, tuple
            Spacecraft ID(s)
        instr : str, list
            Instrument ID(s)
        mode : str, list
            Data rate mode(s). Options include slow, fast, srvy, brst
        level : str, list
            Data level(s). Options include l1a, l1b, l2pre, l2, l3
        tstart : str, list
            Start time of data file. In general, the format is
            YYYYMMDDhhmmss for "brst" mode and YYYYMMDD for "srvy"
            mode (though there are exceptions). If not given, the
            default is "*".
        version : str, list
            File version, formatted as "X.Y.Z", where X, Y, and Z
            are integer version numbers.
        optdesc : str, list
            Optional file name descriptor. If multiple parts,
            they should be separated by hyphens ("-"), not under-
            scores ("_").

    Returns
    -------
        fnames : str, list
            File names constructed from inputs.
    '''

    # Convert all to lists
    if isinstance(sc, str):
        sc = [sc]
    if isinstance(instr, str):
        instr = [instr]
    if isinstance(mode, str):
        mode = [mode]
    if isinstance(level, str):
        level = [level]
    if isinstance(tstart, str):
        tstart = [tstart]
    if isinstance(version, str):
        version = [version]
    if optdesc is not None and isinstance(optdesc, str):
        optdesc = [optdesc]

    # Accept tuples, as those returned by Construct_Filename
    if type(sc) == 'tuple':
        sc_ids = [file[0] for file in sc]
        instr = [file[1] for file in sc]
        mode = [file[2] for file in sc]
        level = [file[3] for file in sc]
        tstart = [file[-2] for file in sc]
        version = [file[-1] for file in sc]

        if len(sc) > 6:
            optdesc = [file[4] for file in sc]
        else:
            optdesc = None
    else:
        sc_ids = sc

    if optdesc is None:
        fnames = ['_'.join((s, i, m, l, t, 'v'+v+'.cdf'))
                  for s in sc_ids
                  for i in instr
                  for m in mode
                  for l in level
                  for t in tstart
                  for v in version
                  ]
    else:
        fnames = ['_'.join((s, i, m, l, o, t, 'v'+v+'.cdf'))
                  for s in sc_ids
                  for i in instr
                  for m in mode
                  for l in level
                  for o in optdesc
                  for t in tstart
                  for v in version
                  ]
    return fnames


def construct_path(*args, data_type='science', **kwargs):
    '''
    Construct a directory structure compliant with MMS path guidelines.

    MMS paths follow the convention
        selections: sitl/type_selections_[gls_type_]
        brst: sc/instr/mode/level[/optdesc]/<year>/<month>/<day>
        srvy: sc/instr/mode/level[/optdesc]/<year>/<month>

    Parameters
    ----------
        *args : dict
            Arguments to be passed along.
        data_type : str
            Type of file names to construct. Options are:
            science or *_selections. If science, inputs are
            passed to construct_science_file_names. If
            *_selections, inputs are passed to
            construct_selections_file_names.
        **kwargs : dict
            Keywords to be passed along.

    Returns
    -------
    paths : list
        Paths constructed from inputs.
    '''

    if data_type == 'science':
        paths = construct_science_path(*args, **kwargs)
    elif 'selections' in data_type:
        paths = construct_selections_path(data_type, **kwargs)
    else:
        raise ValueError('Invalid value for keyword data_type')

    return paths


def construct_selections_path(data_type, tstart='*', gls_type=None,
                              root='', files=False):
    '''
    Construct a directory structure compliant with MMS path
    guidelines for SITL selections.

    MMS SITL selections paths follow the convention
        sitl/[data_type]_selections[_gls_type]/

    Parameters
    ----------
        data_type : str, list, tuple
            Type of selections. Options are abs_selections
            sitl_selections, or gls_selections.
        tstart : str, list
            Start time of data file. The format is
            YYYY-MM-DD-hh-mm-ss. If not given, the default is "*".
        gls_type : str, list
            Type of ground-loop selections. Possible values are:
            mp-dl-unh.
        root : str
            Root of the SDC-like directory structure.
        files : bool
            If True, file names are associated with each path.

    Returns
    -------
    paths : list
        Paths constructed from inputs.
    '''

    # Convert inputs to iterable lists
    if isinstance(data_type, str):
        data_type = [data_type]
    if isinstance(gls_type, str):
        gls_type = [gls_type]
    if isinstance(tstart, str):
        tstart = [tstart]

    # Accept tuples, as those returned by Construct_Filename
    if isinstance(data_type, tuple):
        data_type = [file[0] for file in data_type]
        tstart = [file[-1] for file in data_type]

        if len(data_type > 2):
            gls_type = [file[1] for file in data_type]
        else:
            gls_type = None

    # Paths + Files
    if files:
        if gls_type is None:
            paths = [os.path.join(root, 'sitl', d, '_'.join((d, t+'.sav')))
                     for d in data_type
                     for t in tstart
                     ]
        else:
            paths = [os.path.join(root, 'sitl', d, '_'.join((d, g, t+'.sav')))
                     for d in data_type
                     for g in gls_type
                     for t in tstart
                     ]

    # Paths
    else:
        if gls_type is None:
            paths = [os.path.join(root, 'sitl', d)
                     for d in data_type
                     ]
        else:
            paths = [os.path.join(root, 'sitl', d)
                     for d in data_type
                     ]

    return paths


def construct_science_path(sc, instr=None, mode=None, level=None, tstart='*',
                           optdesc=None, root='', files=False):
    '''
    Construct a directory structure compliant with
    MMS path guidelines for science files.

    MMS science paths follow the convention
        brst: sc/instr/mode/level[/optdesc]/<year>/<month>/<day>
        srvy: sc/instr/mode/level[/optdesc]/<year>/<month>

    Parameters
    ----------
        sc : str, list, tuple
            Spacecraft ID(s)
        instr : str, list
            Instrument ID(s)
        mode : str, list
            Data rate mode(s). Options include slow, fast, srvy, brst
        level : str, list
            Data level(s). Options include l1a, l1b, l2pre, l2, l3
        tstart : str, list
            Start time of data file, formatted as a date: '%Y%m%d'.
            If not given, all dates from 20150901 to today's date are
            used.
        optdesc : str, list
            Optional file name descriptor. If multiple parts,
            they should be separated by hyphens ("-"), not under-
            scores ("_").
        root : str
            Root directory at which the directory structure begins.
        files : bool
            If True, file names will be generated and appended to the
            paths. The file tstart will be "YYYYMMDD*" (i.e. the date
            with an asterisk) and the version number will be "*".

    Returns
    -------
    fnames : str, list
        File names constructed from inputs.
    '''

    # Convert all to lists
    if isinstance(sc, str):
        sc = [sc]
    if isinstance(instr, str):
        instr = [instr]
    if isinstance(mode, str):
        mode = [mode]
    if isinstance(level, str):
        level = [level]
    if isinstance(tstart, str):
        tstart = [tstart]
    if optdesc is not None and isinstance(optdesc, str):
        optdesc = [optdesc]

    # Accept tuples, as those returned by construct_filename
    if type(sc) == 'tuple':
        sc_ids = [file[0] for file in sc]
        instr = [file[1] for file in sc]
        mode = [file[2] for file in sc]
        level = [file[3] for file in sc]
        tstart = [file[-2] for file in sc]

        if len(sc) > 6:
            optdesc = [file[4] for file in sc]
        else:
            optdesc = None
    else:
        sc_ids = sc

    # Paths + Files
    if files:
        if optdesc is None:
            paths = [os.path.join(root, s, i, m, l, t[0:4], t[4:6], t[6:8],
                                  '_'.join((s, i, m, l, t+'*', 'v*.cdf'))
                                  )
                     if m == 'brst'
                     else
                     os.path.join(root, s, i, m, l, t[0:4], t[4:6],
                                  '_'.join((s, i, m, l, t+'*', 'v*.cdf'))
                                  )
                     for s in sc_ids
                     for i in instr
                     for m in mode
                     for l in level
                     for t in tstart
                     ]
        else:
            paths = [os.path.join(root, s, i, m, l, o, t[0:4], t[4:6], t[6:8],
                                  '_'.join((s, i, m, l, o, t+'*', 'v*.cdf'))
                                  )
                     if m == 'brst'
                     else
                     os.path.join(root, s, i, m, l, o, t[0:4], t[4:6],
                                  '_'.join((s, i, m, l, o, t+'*', 'v*.cdf'))
                                  )
                     for s in sc_ids
                     for i in instr
                     for m in mode
                     for l in level
                     for o in optdesc
                     for t in tstart
                     ]

    # Paths
    else:
        if optdesc is None:
            paths = [os.path.join(root, s, i, m, l, t[0:4], t[4:6], t[6:8])
                     if m == 'brst' else
                     os.path.join(root, s, i, m, l, t[0:4], t[4:6])
                     for s in sc_ids
                     for i in instr
                     for m in mode
                     for l in level
                     for t in tstart
                     ]
        else:
            paths = [os.path.join(root, s, i, m, l, o, t[0:4], t[4:6], t[6:8])
                     if m == 'brst' else
                     os.path.join(root, s, i, m, l, o, t[0:4], t[4:6])
                     for s in sc_ids
                     for i in instr
                     for m in mode
                     for l in level
                     for o in optdesc
                     for t in tstart
                     ]

    return paths


def download_selections_files(data_type='abs_selections',
                              start_date=None, end_date=None,
                              gls_type=None):
    """
    Download SITL selections from the SDC.

    Parameters
    ----------
    data_type : str
        Type of SITL selections to download. Options are
            'abs_selections', 'sitl_selections', 'gls_selections'
    gls_type : str
        Type of gls_selections. Options are
            'mp-dl-unh'
    start_date : `dt.datetime` or str
        Start date of data interval
    end_date : `dt.datetime` or str
        End date of data interval

    Returns
    -------
    local_files : list
        Names of the selection files that were downloaded. Files
        can be read using mms.read_eva_fom_structure()
    """

    if gls_type is not None:
        data_type = '_'.join((data_type, gls_type))

    # Setup the API
    api = MrMMS_SDC_API()
    api.data_type = data_type
    api.start_date = start_date
    api.end_date = end_date

    # Download the files
    local_files = api.download_files()
    return local_files


def file_start_time(file_name):
    '''
    Extract the start time from a file name.

    Parameters
    ----------
        file_name : str
            File name from which the start time is extracted.

    Returns
    -------
        fstart : `datetime.datetime`
            Start time of the file, extracted from the file name
    '''

    try:
        # Selections: YYYY-MM-DD-hh-mm-ss
        fstart = re.search('[0-9]{4}(-[0-9]{2}){5}', file_name).group(0)
        fstart = dt.datetime.strptime(fstart, '%Y-%m-%d-%H-%M-%S')
    except AttributeError:
        try:
            # Brst: YYYYMMDDhhmmss
            fstart = re.search('20[0-9]{2}'           # Year
                               '(0[0-9]|1[0-2])'      # Month
                               '([0-2][0-9]|3[0-1])'  # Day
                               '([0-1][0-9]|2[0-4])'  # Hour
                               '[0-5][0-9]'           # Minute
                               '([0-5][0-9]|60)',     # Second
                               file_name).group(0)
            fstart = dt.datetime.strptime(fstart, '%Y%m%d%H%M%S')
        except AttributeError:
            try:
                # Srvy: YYYYMMDD
                fstart = re.search('20[0-9]{2}'            # Year
                                   '(0[0-9]|1[0-2])'       # Month
                                   '([0-2][0-9]|3[0-1])',  # Day
                                   file_name).group(0)
                fstart = dt.datetime.strptime(fstart, '%Y%m%d')
            except AttributeError:
                raise AttributeError('File start time not identified in: \n'
                                     '  "{}"'.format(file_name))

    return fstart


def filename2path(fname, root=''):
    """
    Convert an MMS file name to an MMS path.

    MMS paths take the form

        sc/instr/mode/level[/optdesc]/YYYY/MM[/DD/]

    where the optional descriptor [/optdesc] is included if it is also in the
    file name and day directory [/DD] is included if mode='brst'.

    Parameters
    ----------
    fname : str
        File name to be turned into a path.
    root : str
        Absolute directory

    Returns
    -------
    path : list
        Path to the data file.
    """

    parts = parse_file_name(fname)

    # data_type = '*_selections'
    if 'selections' in parts[0]:
        path = os.path.join(root, parts[0])

    # data_type = 'science'
    else:
        # Create the directory structure
        #   sc/instr/mode/level[/optdesc]/YYYY/MM/
        path = os.path.join(root, *parts[0:5], parts[5][0:4], parts[5][4:6])

        # Burst files require the DAY directory
        #   sc/instr/mode/level[/optdesc]/YYYY/MM/DD/
        if parts[2] == 'brst':
            path = os.path.join(path, parts[5][6:8])

    path = os.path.join(path, fname)

    return path


def filter_time(fnames, start_date, end_date):
    """
    Filter files by their start times.

    Parameters
    ----------
    fnames : str, list
        File names to be filtered.
    start_date : str
        Start date of time interval, formatted as '%Y-%m-%dT%H:%M:%S'
    end_date : str
        End date of time interval, formatted as '%Y-%m-%dT%H:%M:%S'

    Returns
    -------
    paths : list
        Path to the data file.
    """

    # Make sure file names are iterable. Allocate output array
    files = fnames
    if isinstance(files, str):
        files = [files]

    # If dates are strings, convert them to datetimes
    if isinstance(start_date, str):
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')
    if isinstance(end_date, str):
        end_date = dt.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%S')

    # Parse the time out of the file name
    fstart = [file_start_time(file) for file in files]

    # Sort the files by start time
    isort = sorted(range(len(fstart)), key=lambda k: fstart[k])
    fstart = [fstart[i] for i in isort]
    files = [files[i] for i in isort]

    # End time
    #   - Any files that start on or before END_DATE can be kept
    idx = [i for i, t in enumerate(fstart) if t <= end_date]
    if len(idx) > 0:
        fstart = [fstart[i] for i in idx]
        files = [files[i] for i in idx]
    else:
        fstart = []
        files = []

    # Start time
    #   - Any file with TSTART <= START_DATE can potentially have data
    #     in our time interval of interest.
    #   - Assume the start time of one file marks the end time of the
    #     previous file.
    #   - With this, we look for the file that begins just prior to START_DATE
    #     and throw away any files that start before it.
    idx = [i for i, t in enumerate(fstart) if t >= start_date]
    if (len(idx) == 0) and \
            (len(fstart) > 0) and \
            (fstart[-1].date() == start_date.date()):
        idx = [len(fstart)-1]

    elif (len(idx) != 0) and \
            ((idx[0] != 0) and (fstart[idx[0]] != start_date)):
        idx.insert(0, idx[0]-1)

    if len(idx) > 0:
        fstart = [fstart[i] for i in idx]
        files = [files[i] for i in idx]
    else:
        fstart = []
        files = []

    return files


def filter_version(files, latest=None, version=None, min_version=None):
    '''
    Filter file names according to their version numbers.

    Parameters
    ----------
    files : str, list
        File names to be turned into paths.
    latest : bool
        If True, the latest version of each file type is
        returned. if `version` and `min_version` are not
        set, this is the default.
    version : str
        Only files with this version are returned.
    min_version : str
        All files with version greater or equal to this
        are returned.

    Returns
    -------
    filtered_files : list
        The files remaining after applying filter conditions.
    '''

    if version is None and min is None:
        latest = True
    if ((version is None) + (min_version is None) + (latest is None)) > 1:
        ValueError('latest, version, and min are mutually exclusive.')

    # Output list
    filtered_files = []
    
    # Extract the version
    parts = [parse_file_name(file) for file in files]
    versions = [part[-1] for part in parts]

    # The latest version of each file type
    if latest:
        # Parse file names and identify unique file types
        #   - File types include all parts of file name except version number
        bases = ['_'.join(part[0:-2]) for part in parts]
        uniq_bases = list(set(bases))

        # Filter according to unique file type
        for idx, uniq_base in enumerate(uniq_bases):
            test_idx = [i
                        for i, test_base in enumerate(bases)
                        if test_base == uniq_base]
            file_ref = files[idx]
            vXYZ_ref = [int(v) for v in versions[idx].split('.')]

            filtered_files.append(file_ref)
            for i in test_idx:
                vXYZ = [int(v) for v in versions[i].split('.')]
                if ((vXYZ[0] > vXYZ_ref[0]) or
                        (vXYZ[0] == vXYZ_ref[0] and
                         vXYZ[1] > vXYZ_ref[1]) or
                        (vXYZ[0] == vXYZ_ref[0] and
                         vXYZ[1] == vXYZ_ref[1] and
                         vXYZ[2] > vXYZ_ref[2])):
                    filtered_files[-1] = files[i]

    # All files with version number greater or equal to MIN_VERSION
    elif min_version is not None:
        vXYZ_min = [int(v) for v in min_version.split('.')]
        for idx, v in enumerate(versions):
            vXYZ = [int(vstr) for vstr in v.split('.')]
            if ((vXYZ[0] > vXYZ_min[0]) or
                    ((vXYZ[0] == vXYZ_min[0]) and
                     (vXYZ[1] > vXYZ_min[1])) or
                    ((vXYZ[0] == vXYZ_min[0]) and
                     (vXYZ[1] == vXYZ_min[1]) and
                     (vXYZ[2] >= vXYZ_min[2]))):
                filtered_files.append(files[idx])

    # All files with a particular version number
    elif version is not None:
        vXYZ_ref = [int(v) for v in version.split('.')]
        for idx, v in enumerate(versions):
            vXYZ = [int(vstr) for vstr in v.split('.')]
            if (vXYZ[0] == vXYZ_ref[0] and
                    vXYZ[1] == vXYZ_ref[1] and
                    vXYZ[2] == vXYZ_ref[2]):
                filtered_files.append(files[idx])

    return filtered_files


def mission_events(event_type, start, stop, sc=None):
    """
    Download MMS mission events. See the filters on the webpage
    for more ideas.
        https://lasp.colorado.edu/mms/sdc/public/about/events/#/

    Parameters
    ----------
    event_type : str
        Type of event. Options are 'apogee', 'dsn_contact', 'orbit',
        'perigee', 'science_roi', 'shadow', 'sitl_window', 'sroi'.
    start, stop : `datetime.datetime`, int
        Start and end of the data interval, specified as a time or
        orbit range.
    sc : str
        Spacecraft ID (mms, mms1, mms2, mms3, mms4) for which event
        information is to be returned.

    Returns
    -------
    data : dict
        Information about each event.
            start_time     - Start time (UTC) of event %Y-%m-%dT%H:%M:%S.%f
            end_time       - End time (UTC) of event %Y-%m-%dT%H:%M:%S.%f
            event_type     - Type of event
            sc_id          - Spacecraft to which the event applies
            source         - Source of event
            description    - Description of event
            discussion
            start_orbit    - Orbit on which the event started
            end_orbit      - Orbit on which the event ended
            tag
            id
            tstart         - Start time of event as datetime
            tend           - end time of event as datetime
    """
    event_func = _get_mission_events(event_type)
    return event_func(start, stop, sc)

def _get_mission_events(event_type):
    if event_type == 'apogee':
        return _get_apogee
    elif event_type == 'dsn_contact':
        return _get_dsn_contact
    elif event_type == 'orbit':
        return _get_orbit
    elif event_type == 'perigee':
        return _get_perigee
    elif event_type == 'science_roi':
        return _get_science_roi
    elif event_type == 'shadow':
        return _get_shadow
    elif event_type == 'sitl_window':
        return _get_sitl_window
    elif event_type == 'sroi':
        return _get_sroi

def _get_apogee(start, stop, sc):
    '''
    Apogee information between `start` and `stop` and associated
    with spacecraft `sc`.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='Timeline', event_type='apogee')

def _get_dsn_contact(start, stop, sc):
    '''
    Science region of interest information between `start` and `stop`
    and associated with spacecraft `sc`. Defines the limits of when
    fast survey and burst data can be available each orbit.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='Timeline', event_type='dsn_contact')

def _get_orbit(start, stop, sc):
    '''
    Orbital information between `start` and `stop` and associated
    with spacecraft `sc`.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='Timeline', event_type='orbit')

def _get_perigee(start, stop, sc):
    '''
    Perigee information between `start` and `stop` and associated
    with spacecraft `sc`.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='Timeline', event_type='perigee')

def _get_science_roi(start, stop, sc):
    '''
    Science region of interest information between `start` and `stop`
    and associated with spacecraft `sc`. Defines the limits of when
    fast survey and burst data can be available each orbit.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='BDM', event_type='science_roi')

def _get_shadow(start, stop, sc):
    '''
    Earth shadow information between `start` and `stop` and associated
    with spacecraft `sc`.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='POC', event_type='shadow')

def _get_sroi(start, stop, sc):
    '''
    Sub-region of interest information between `start` and `stop`
    and associated with spacecraft `sc`. There can be several
    SROIs per science_roi.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='POC', event_type='SROI')

def _get_sitl_window(start, stop, sc):
    '''
    SITL window information between `start` and `stop` and associated
    with spacecraft `sc`. Defines when the SITL can submit selections.
    '''
    return _mission_data(start, stop, sc=sc,
                         source='BDM', event_type='sitl_window')


def _mission_data(start, stop, sc=None,
                  source=None, event_type=None):
    """
    Download MMS mission events. See the filters on the webpage
    for more ideas.
        https://lasp.colorado.edu/mms/sdc/public/about/events/#/

    NOTE: some sources, such as 'burst_segment', return a format
          that is not yet parsed properly.

    Parameters
    ----------
    start, stop : `datetime.datetime`, int
        Start and end of the data interval, specified as a time or
        orbit range.
    sc : str
        Spacecraft ID (mms, mms1, mms2, mms3, mms4) for which event
        information is to be returned.
    source : str
        Source of the mission event. Options include
            'Timeline', 'Burst', 'BDM', 'SITL'
    event_type : str
        Type of mission event. Options include
            BDM: sitl_window, evaluate_metadata, science_roi

    Returns
    -------
    data : dict
        Information about each event.
            start_time     - Start time (UTC) of event %Y-%m-%dT%H:%M:%S.%f
            end_time       - End time (UTC) of event %Y-%m-%dT%H:%M:%S.%f
            event_type     - Type of event
            sc_id          - Spacecraft to which the event applies
            source         - Source of event
            description    - Description of event
            discussion
            start_orbit    - Orbit on which the event started
            end_orbit      - Orbit on which the event ended
            tag
            id
            tstart         - Start time of event as datetime
            tend           - end time of event as datetime
    """
    url = 'https://lasp.colorado.edu/' \
          'mms/sdc/public/service/latis/mms_events_view.csv'
    
    start_date = None
    end_date = None
    start_orbit = None
    end_orbit = None
    
    # mission_events() returns numpy integers, so check for
    # those, too
    if isinstance(start, (int, np.integer)):
        start_orbit = start
    else:
        start_date = start
    if isinstance(stop, (int, np.integer)):
        end_orbit = stop
    else:
        end_date = stop
    
    query = {}
    if start_date is not None:
        query['start_time_utc>'] = start_date.strftime('%Y-%m-%d')
    if end_date is not None:
        query['end_time_utc<'] = end_date.strftime('%Y-%m-%d')

    if start_orbit is not None:
        query['start_orbit>'] = start_orbit
    if end_orbit is not None:
        query['end_orbit<'] = end_orbit

    if sc is not None:
        query['sc_id'] = sc
    if source is not None:
        query['source'] = source
    if event_type is not None:
        query['event_type'] = event_type

    resp = requests.get(url, params=query)
    data = _response_text_to_dict(resp.text)
    
    # Convert to useful types
    types = ['str', 'str', 'str', 'str', 'str', 'str', 'str',
             'int32', 'int32', 'str', 'int32']
    for items in zip(data, types):
        if items[1] == 'str':
            pass
        else:
            data[items[0]] = np.asarray(data[items[0]], dtype=items[1])

    # Add useful tags
    #   - Number of seconds elapsed
    #   - TAISTARTIME as datetime
    #   - TAIENDTIME as datetime

    # NOTE! If data['TAISTARTTIME'] is a scalar, this will not work
    #       unless everything after "in" is turned into a list
    data['tstart'] = [dt.datetime.strptime(
                          value, '%Y-%m-%dT%H:%M:%S.%f'
                          )
                      for value in data['start_time']
                      ]
    data['tend'] = [dt.datetime.strptime(
                        value, '%Y-%m-%dT%H:%M:%S.%f'
                        )
                    for value in data['end_time']
                    ]

    return data
    


def mission_events_v1(start_date=None, end_date=None,
                      start_orbit=None, end_orbit=None,
                      sc=None,
                      source=None, event_type=None):
    """
    Download MMS mission events. See the filters on the webpage
    for more ideas.
        https://lasp.colorado.edu/mms/sdc/public/about/events/#/

    NOTE: some sources, such as 'burst_segment', return a format
          that is not yet parsed properly.

    Parameters
    ----------
    start_date, end_date : `datetime.datetime`
        Start and end date of time interval. The interval is right-
        exclusive: [start_date, end_date). The time interval must
        encompass the desired data (e.g. orbit begin and end times)
        for it to be returned.
    start_orbit, end_orbit : `datetime.datetime`
        Start and end orbit of data interval. If provided with `start_date`
        or `end_date`, the two must overlap for any data to be returned.
    sc : str
        Spacecraft ID (mms, mms1, mms2, mms3, mms4) for which event
        information is to be returned.
    source : str
        Source of the mission event. Options include
            'Timeline', 'Burst', 'BDM', 'SITL'
    event_type : str
        Type of mission event. Options include
            BDM: sitl_window, evaluate_metadata, science_roi

    Returns
    -------
    data : dict
        Information about each event.
            start_time     - Start time (UTC) of event %Y-%m-%dT%H:%M:%S.%f
            end_time       - End time (UTC) of event %Y-%m-%dT%H:%M:%S.%f
            event_type     - Type of event
            sc_id          - Spacecraft to which the event applies
            source         - Source of event
            description    - Description of event
            discussion
            start_orbit    - Orbit on which the event started
            end_orbit      - Orbit on which the event ended
            tag
            id
            tstart         - Start time of event as datetime
            tend           - end time of event as datetime
    """
    url = 'https://lasp.colorado.edu/' \
          'mms/sdc/public/service/latis/mms_events_view.csv'
    
    query = {}
    if start_date is not None:
        query['start_time_utc>'] = start_date.strftime('%Y-%m-%d')
    if end_date is not None:
        query['end_time_utc<'] = end_date.strftime('%Y-%m-%d')

    if start_orbit is not None:
        query['start_orbit>'] = start_orbit
    if end_orbit is not None:
        query['end_orbit<'] = end_orbit

    if sc is not None:
        query['sc_id'] = sc
    if source is not None:
        query['source'] = source
    if event_type is not None:
        query['event_type'] = event_type

    resp = requests.get(url, params=query)
    data = _response_text_to_dict(resp.text)
    
    # Convert to useful types
    types = ['str', 'str', 'str', 'str', 'str', 'str', 'str',
             'int32', 'int32', 'str', 'int32']
    for items in zip(data, types):
        if items[1] == 'str':
            pass
        else:
            data[items[0]] = np.asarray(data[items[0]], dtype=items[1])

    # Add useful tags
    #   - Number of seconds elapsed
    #   - TAISTARTIME as datetime
    #   - TAIENDTIME as datetime
#    data["start_time_utc"] = data.pop("start_time_utc "
#                                      "(yyyy-mm-dd'T'hh:mm:ss.sss)"
#                                      )
#    data["end_time_utc"] = data.pop("end_time_utc "
#                                    "(yyyy-mm-dd'T'hh:mm:ss.sss)"
#                                    )

    # NOTE! If data['TAISTARTTIME'] is a scalar, this will not work
    #       unless everything after "in" is turned into a list
    data['tstart'] = [dt.datetime.strptime(
                          value, '%Y-%m-%dT%H:%M:%S.%f'
                          )
                      for value in data['start_time']
                      ]
    data['tend'] = [dt.datetime.strptime(
                        value, '%Y-%m-%dT%H:%M:%S.%f'
                        )
                    for value in data['end_time']
                    ]

    return data


def parse_file_name(fname):
    """
    Parse a file name compliant with MMS file name format guidelines.

    Parameters
    ----------
    fname : str
        File name to be parsed.

    Returns
    -------
    parts : tuple
        The tuple elements are:
            [0]: Spacecraft IDs
            [1]: Instrument IDs
            [2]: Data rate modes
            [3]: Data levels
            [4]: Optional descriptor (empty string if not present)
            [5]: Start times
            [6]: File version number
    """

    parts = os.path.basename(fname).split('_')

    # data_type = '*_selections'
    if 'selections' in fname:
        # datatype_glstype_YYYY-mm-dd-HH-MM-SS.sav
        if len(parts) == 3:
            gls_type = ''
        else:
            gls_type = parts[2]

        # (data_type, [gls_type,] start_date)
        out = ('_'.join(parts[0:2]), gls_type, parts[-1][0:-4])

    # data_type = 'science'
    else:
        # sc_instr_mode_level_[optdesc]_fstart_vVersion.cdf
        if len(parts) == 6:
            optdesc = ''
        else:
            optdesc = parts[4]

        # (sc, instr, mode, level, [optdesc,] start_date, version)
        out = (*parts[0:4], optdesc, parts[-2], parts[-1][1:-4])

    return out


def parse_time(times):
    """
    Parse the start time of MMS file names.

    Parameters
    ----------
    times : str, list
        Start times of file names.

    Returns
    -------
    parts : list
        A list of tuples. The tuple elements are:
            [0]: Year
            [1]: Month
            [2]: Day
            [3]: Hour
            [4]: Minute
            [5]: Second
    """
    if isinstance(times, str):
        times = [times]

    # Three types:
    #    srvy        YYYYMMDD
    #    brst        YYYYMMDDhhmmss
    #    selections  YYYY-MM-DD-hh-mm-ss
    parts = [None]*len(times)
    for idx, time in enumerate(times):
        if len(time) == 19:
            parts[idx] = (time[0:4], time[5:7], time[8:10],
                          time[11:13], time[14:16], time[17:]
                          )
        elif len(time) == 14:
            parts[idx] = (time[0:4], time[4:6], time[6:8],
                          time[8:10], time[10:12], time[12:14]
                          )
        else:
            parts[idx] = (time[0:4], time[4:6], time[6:8], '00', '00', '00')

    return parts


def read_eva_fom_structure(sav_filename):
    '''
    Returns a dictionary that mirrors the SITL selections fomstr structure
    that is in the IDL .sav file.

    Parameters
    ----------
    sav_filename : str
        Name of the IDL sav file containing the SITL selections

    Returns
    -------
    data : dict
        The FOM structure.
            valid                    : 1 if the fom structure is valid, 0 otherwise
            error                    : Error string for invalid fom structures
            algversion
            sourceid                 : username of the SITL that made the selections
            cyclestart
            numcycles
            nsegs                    : number of burst segments
            start                    : index into timestamps of start time for each burst segment
            stop                     : index into timestamps of stop time for each burst segment
            seglengths
            fom                      : figure of merit for each burst segment
            nubffs
            mdq                      : mission data quality
            timestamps               : timestamp (TAI seconds since 1958) of each mdq
            targetbuffs
            fomave
            targetratio
            minsegmentsize
            maxsegmentsize
            pad
            searchratio
            fomwindowsize
            fomslope
            fomskew
            fombias
            metadatainfo
            oldestavailableburstdata :
            metadataevaltime
            discussion               : description of each burst segment given by the SITL
            note                     : note given by SITL to data within SITL window
            datetimestamps           : timestamps converted to datetimes
            start_time               : start time of the burst segment
            end_time                 : end time of the burst segment
            tstart                   : datetime timestamp of the start of each burst segment
            tstop                    : datetime timestamp of the end of each burst segment
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sav = readsav(sav_filename)

    assert 'fomstr' in sav, 'save file does not have a fomstr structure'
    fomstr = sav['fomstr']
    
    # Handle invalid structures
    #   - example: abs_selections_2017-10-29-09-25-34.sav
    if fomstr.valid[0] == 0:
        d = {'valid': int(fomstr.valid[0]),
             'error': fomstr.error[0].decode('utf-8'),
             'errno': int(fomstr.errno[0])
            }
        return d
    
    d = {'valid': int(fomstr.valid[0]),
         'error': fomstr.error[0],
         'algversion': fomstr.algversion[0].decode('utf-8'),
         'sourceid': [x.decode('utf-8') for x in fomstr.sourceid[0]],
         'cyclestart': int(fomstr.cyclestart[0]),
         'numcycles': int(fomstr.numcycles[0]),
         'nsegs': int(fomstr.nsegs[0]),
         'start': fomstr.start[0].tolist(),
         'stop': fomstr.stop[0].tolist(),
         'seglengths': fomstr.seglengths[0].tolist(),
         'fom': fomstr.fom[0].tolist(),
         'nbuffs': int(fomstr.nbuffs[0]),
         'mdq': fomstr.mdq[0].tolist(),
         'timestamps': fomstr.timestamps[0].tolist(),
         'targetbuffs': int(fomstr.targetbuffs[0]),
         'fomave': float(fomstr.fomave[0]),
         'targetratio': float(fomstr.targetratio[0]),
         'minsegmentsize': float(fomstr.minsegmentsize[0]),
         'maxsegmentsize': float(fomstr.maxsegmentsize[0]),
         'pad': int(fomstr.pad[0]),
         'searchratio': float(fomstr.searchratio[0]),
         'fomwindowsize': int(fomstr.fomwindowsize[0]),
         'fomslope': float(fomstr.fomslope[0]),
         'fomskew': float(fomstr.fomskew[0]),
         'fombias': float(fomstr.fombias[0]),
         'metadatainfo': fomstr.metadatainfo[0].decode('utf-8'),
         'oldestavailableburstdata': fomstr.oldestavailableburstdata[0].decode('utf-8'),
         'metadataevaltime': fomstr.metadataevaltime[0].decode('utf-8')
         }
    try:
        d['discussion'] = [x.decode('utf-8') for x in fomstr.discussion[0]]
    except AttributeError:
        d['discussion'] = ['ABS Selections'] * len(d['start'])
    # Some characters cannot be decoded. One example:
    #   - b'CS edge B>5nT Vy,Vz 100km/s, diffB peak 6\xe0nA/m2, energetic particles, E waves'
    except UnicodeDecodeError:
        discussion = []
        for x in fomstr.discussion[0]:
            try:
                discussion.append(x.decode('utf-8'))
            except UnicodeDecodeError:
                discussion.append(str(x))
        d['discussion'] = discussion
    
    try:
        d['note'] = fomstr.note[0].decode('utf-8')
    except AttributeError:
        d['note'] = 'ABS Selections'
    # Some characters cannot be decoded. One example:
    #   - MMS cros\x82\x93ed the magnetopause
    except UnicodeDecodeError:
        d['note'] = str(fomstr.note[0])

    # Convert TAI to datetime
    #   - timestaps are TAI seconds elapsed since 1958-01-01
    #   - tt2000 are nanoseconds elapsed since 2000-01-01
    t_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])
    tepoch  = epochs.CDFepoch()
    d['datetimestamps'] = tepoch.to_datetime(
                              np.asarray(d['timestamps']) * int(1e9) +
                              t_1958
                              )

    # FOM structure (copy procedure from IDL/SPEDAS/EVA)
    #   - eva_sitl_load_soca_simple
    #   - eva_sitl_strct_read
    #   - mms_convert_from_tai2unix
    #   - mms_tai2unix
    if 'fomslope' in d:
        if d['stop'][d['nsegs']-1] >= d['numcycles']:
            raise ValueError('Number of segments should be <= # cycles.')

        taistarttime = []
        taiendtime = []
        tstart = []
        tstop = []
        t_fom = [d['datetimestamps'][0]]
        fom = [0]
        dtai_last = (d['timestamps'][d['numcycles']-1]
                     - d['timestamps'][d['numcycles']-2])
        dt_last = (d['datetimestamps'][d['numcycles']-1]
                   - d['datetimestamps'][d['numcycles']-2])

        # Extract the start and stop times of the FOM values
        # Create a time series for FOM values
        #   - Indices 'start' and 'stop' are zero-based
        #   - timestamps mark start time of each burst buffer
        #   - Extend stop time to next timestamp to encompass the entire
        #     burst interval
        for idx in range(d['nsegs']):
            taistarttime.append(d['timestamps'][d['start'][idx]])
            tstart.append(d['datetimestamps'][d['start'][idx]])
            if d['stop'][idx] < d['numcycles']-1:
                taiendtime.append(d['timestamps'][d['stop'][idx]+1])
                tstop.append(d['datetimestamps'][d['stop'][idx]+1])
            else:
                taiendtime.append(d['timestamps'][d['numcycles']-1] + dtai_last)
                tstop.append(d['datetimestamps'][d['numcycles']-1] + dt_last)

        # Append the last time stamp to the time series
        t_fom.append(d['datetimestamps'][d['numcycles']-1] + dt_last)
        fom.append(0)

    # BAK structure
    else:
        raise NotImplemented('BAK structure has not been implemented')
        nsegs = len(d['fom'])  # BAK

    # Add to output structure
    d['taistarttime'] = taistarttime
    d['taiendtime'] = taiendtime
    d['start_time'] = [t.strftime('%Y-%m-%d %H:%M:%S') for t in tstart]
    d['stop_time'] = [t.strftime('%Y-%m-%d %H:%M:%S') for t in tstop]
    d['tstart'] = tstart
    d['tstop'] = tstop
    d['createtime'] = [file_start_time(sav_filename)] * d['nsegs']

    return d


def read_gls_csv(filename):
    """
    Read a ground loop selections (gls) CSV file.

    Parameters
    ----------
    filename : str
        Name of the CSV file to be read

    Returns
    -------
    data : dict
        Data contained in the CSV file
    """
    # Dictionary to hold data from csv file
    keys = ['start_time', 'stop_time', 'sourceid', 'fom', 'discussion',
            'taistarttime', 'taiendtime', 'tstart', 'tstop', 'createtime']
    data = {key: [] for key in keys}

    # CSV files have their generation time in the file name.
    # Multiple CSV files may have been created for the same
    # data interval, which results in duplicate data. Use
    # a set to keep only unique data entries.
    tset = set()
    nold = 0

    # Constant for converting times to TAI seconds since 1958
    t_1958 = epochs.CDFepoch.compute_tt2000([1958, 1, 1, 0, 0, 0, 0, 0, 0])

    # Parse each row of all files
    skip_file = False
    nentry_skip = 0
    nentry_expand = 0
    with open(filename) as f:
        fstart = file_start_time(filename)
        
        reader = csv.reader(f)
        for row in reader:
            tstart = dt.datetime.strptime(
                         row[0], '%Y-%m-%d %H:%M:%S'
                         )
            tstop = dt.datetime.strptime(
                        row[1], '%Y-%m-%d %H:%M:%S'
                        )

            # Convert times to TAI seconds since 1958
            t0 = _datetime_to_list(tstart)
            t1 = _datetime_to_list(tstop)
            t0 = int((epochs.CDFepoch.compute_tt2000(t0) - t_1958) // 1e9)
            t1 = int((epochs.CDFepoch.compute_tt2000(t1) - t_1958) // 1e9)

            # Ensure selections have a minimum length of 10 seconds
            if (t1 - t0) == 0:
                t1 += int(10)
                tstop += dt.timedelta(seconds=10)
                row[1] = dt.datetime.strftime(
                             tstop, '%Y-%m-%d %H:%M:%S'
                             )
                nentry_expand += 1

            # Some burst segments are unrealistically long
            #   - Usually, the longest have one selection per file
            if ((t1 - t0) > 3600):
                with open(filename) as f_test:
                    nrows = sum(1 for row in f_test)
                if nrows == 1:
                    skip_file = True
                    break

            # Some entries have negative durations
            if (t1 - t0) < 0:
                nentry_skip += 1
                continue

            # Store data
            data['taistarttime'].append(t0)
            data['taiendtime'].append(t1)
            data['start_time'].append(row[0])
            data['stop_time'].append(row[1])
            data['fom'].append(float(row[2]))
            data['discussion'].append(','.join(row[3:]))
            data['tstart'].append(tstart)
            data['tstop'].append(tstop)
            data['createtime'].append(fstart)

        # Source ID is the name of the GLS model
        parts = parse_file_name(filename)
        data['sourceid'].extend([parts[1]] * len(data['fom']))

        # Errors
        data['errors'] = {'fskip': skip_file,
                          'nexpand': nentry_expand,
                          'nskip': nentry_skip
                          }

    return data


def _sdc_parse_form(r):
    '''Parse key-value pairs from the log-in form

    Parameters
    ----------
    r (object):    requests.response object.

    Returns
    -------
    form (dict):   key-value pairs parsed from the form.
    '''
    # Find action URL
    pstart = r.text.find('<form')
    pend = r.text.find('>', pstart)
    paction = r.text.find('action', pstart, pend)
    pquote1 = r.text.find('"', pstart, pend)
    pquote2 = r.text.find('"', pquote1+1, pend)
    url5 = r.text[pquote1+1:pquote2]
    url5 = url5.replace('&#x3a;', ':')
    url5 = url5.replace('&#x2f;', '/')

    # Parse values from the form
    pinput = r.text.find('<input', pend+1)
    inputs = {}
    while pinput != -1:
        # Parse the name-value pair
        pend = r.text.find('/>', pinput)

        # Name
        pname = r.text.find('name', pinput, pend)
        pquote1 = r.text.find('"', pname, pend)
        pquote2 = r.text.find('"', pquote1+1, pend)
        name = r.text[pquote1+1:pquote2]

        # Value
        if pname != -1:
            pvalue = r.text.find('value', pquote2+1, pend)
            pquote1 = r.text.find('"', pvalue, pend)
            pquote2 = r.text.find('"', pquote1+1, pend)
            value = r.text[pquote1+1:pquote2]
            value = value.replace('&#x3a;', ':')

            # Extract the values
            inputs[name] = value

        # Next iteraction
        pinput = r.text.find('<input', pend+1)

    form = {'url': url5,
            'payload': inputs}

    return form


def sdc_login(username):
    '''
    Log-In to the MMS Science Data Center.

    Parameters:
    -----------
    username : str
        Account username.
    password : str
        Account password.

    Returns:
    --------
    Cookies : dict
        Session cookies for continued access to the SDC. Can
        be passed to an instance of requests.Session.
    '''

    # Ask for the password
    password = getpass()

    # Disable warnings because we are not going to obtain certificates
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Attempt to access the site
    #   - Each of the redirects are stored in the history attribute
    url0 = 'https://lasp.colorado.edu/mms/sdc/team/'
    r = requests.get(url0, verify=False)

    # Extract cookies and url
    cookies = r.cookies
    for response in r.history:
        cookies.update(response.cookies)

        try:
            url = response.headers['Location']
        except:
            pass

    # Submit login information
    payload = {'j_username': username, 'j_password': password}
    r = requests.post(url, cookies=cookies, data=payload, verify=False)

    # After submitting info, we land on a page with a form
    #   - Parse form and submit values to continue
    form = _sdc_parse_form(r)
    r = requests.post(form['url'],
                      cookies=cookies,
                      data=form['payload'],
                      verify=False
                      )

    # Update cookies to get session information
#    cookies = r.cookies
    for response in r.history:
        cookies.update(response.cookies)

    return cookies


def sort_files(files):
    """
    Sort MMS file names by data product and time.

    Parameters:
    files : str, list
        Files to be sorted

    Returns
    -------
    sorted : tuple
        Sorted file names. Each tuple element corresponds to
        a unique data product.
    """

    # File types and start times
    parts = [parse_file_name(file) for file in files]
    bases = ['_'.join(p[0:5]) for p in parts]
    tstart = [p[-2] for p in parts]

    # Sort everything
    idx = sorted(range(len(tstart)), key=lambda k: tstart[k])
    bases = [bases[i] for i in idx]
    files = [files[i] for i in idx]

    # Find unique file types
    fsort = []
    uniq_bases = list(set(bases))
    for ub in uniq_bases:
        fsort.append([files[i] for i, b in enumerate(bases) if b == ub])

    return tuple(fsort)


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
    orbits = mission_events('orbit', tstart, tstop, sc=sc)
    
    orbit = None
    for idx in range(len(orbits['tstart'])):
        if (time > orbits['tstart'][idx]) and (time < orbits['tend'][idx]):
            orbit = orbits['start_orbit'][idx]
    if orbit is None:
        ValueError('Did not find correct orbit!')
    
    return orbit


if __name__ == '__main__':
    '''Download data'''

    # Inputs common to each calling sequence
    sc = sys.argv[0]
    instr = sys.argv[1]
    mode = sys.argv[2]
    level = sys.argv[3]

    # Basic dataset
    if len(sys.argv) == 7:
        optdesc = None
        start_date = sys.argv[4]
        end_date = sys.argv[5]

    # Optional descriptor given
    elif len(sys.argv) == 8:
        optdesc = sys.argv[4]
        start_date = sys.argv[5]
        end_date = sys.argv[6]

    # Error
    else:
        raise TypeError('Incorrect number if inputs.')

    # Create the request
    api = MrMMS_SDC_API(sc, instr, mode, level,
                        optdesc=optdesc, start_date=start_date, end_date=end_date)

    # Download the data
    files = api.download_files()

