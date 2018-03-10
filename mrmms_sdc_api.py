import os, requests
import pdb # https://pythonconquerstheuniverse.wordpress.com/2009/09/10/debugging-in-python/
from tqdm import tqdm

class MrMMS_SDC_API:
    """Class to interface with the Science Data Center (SDC) API of the
    Magnetospheric Multiscale (MMS) mission.
    https://lasp.colorado.edu/mms/sdc/public/"""
    
    def __init__(self, sc=None, instr=None, mode=None, level=None,
                 anc_product=None,
                 data_type='science',
                 data_root=None,
                 end_date=None,
                 files=None,
                 optdesc=None,
                 site='public',
                 start_date=None,
                 version=None):
        
        # Set attributes
        self.anc_product = anc_product
        self.data_type = data_type
        self.end_date = end_date
        self.files = files
        self.instr = instr
        self.level = level
        self.mode = mode
        self.optdesc = optdesc
        self.sc = sc
        self.site = site
        self.start_date = start_date
        self.version = version
        
        # Setup download directory
        #   - $HOME/data/mms/
        if data_root is None:
            data_root = os.path.join(os.path.expanduser('~'), 'data', 'mms')
            if not os.path.isdir(data_root):
                os.makedirs(data_root, exist_ok=True)
        
        self._data_root = data_root
        self._sdc_home  = 'https://lasp.colorado.edu/mms/sdc'
        self._info_type = 'download'
    
    def BuildQuery(self):
        """Build a URL to query the SDC."""
        sep = '/'
        home = 'https://lasp.colorado.edu/mms/sdc'
        url = sep.join( (home, self.site, 'files', 'api', 'v1', 
                         self._info_type, self.data_type) )
        
        # Build query from parts of file names
        query = '?'
        if self.sc is not None:
            query += 'sc_id=' + self.sc + '&'
        if self.instr is not None:
            query += 'instrument_id=' + self.instr + '&'
        if self.mode is not None:
            query += 'data_rate_mode=' + self.mode + '&'
        if self.level is not None:
            query += 'data_level=' + self.level + '&'
        if self.optdesc is not None:
            query += 'descriptor=' + self.optdesc + '&'
        if self.version is not None:
            query += 'version=' + self.version + '&'
        if self.start_date is not None:
            query += 'start_date=' + self.start_date + '&'
        if self.end_date is not None:
            query += 'end_date=' + self.end_date + '&'
        
        # Combine URL with query string
        url += query
        return url
    
    def Download(self):
        self._info_type = 'download'
        # Build the URL sans query
        url = '/'.join((self._sdc_home, self.site, 'files', 'api', 'v1',
                        self._info_type, self.data_type))
        
        # Get available files
        file_info = self.FileInfo()
        local_files = []
        
        # Download each file individually
        for info in file_info['files']:
            # Create the destination directory
            file = self.name2path(info['file_name'])
            if not os.path.isdir(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file))
            
            # downloading: https://stackoverflow.com/questions/16694907/how-to-download-large-file-in-python-with-requests-py
            # progress bar: https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
            try:
                r = requests.get(url,
                                 params={'file': info['file_name']}, 
                                 stream=True)
                with open(file, 'wb') as f:
                    for chunk in tqdm(r.iter_content(chunk_size=1024),
                                      total=info['file_size']/1024,
                                      unit='k',
                                      unit_scale=True):
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
            except:
                if os.path.isfile(file):
                    os.remove(file)
                raise
            
            local_files.append(file)
        
        if len(local_files) == 1:
            local_files = local_files[0]
        
        return local_files
    
    def FileInfo(self):
        """Obtain file information from the SDC."""
        self._info_type = 'file_info'
        response = self.Get()
        return response.json()
        
    def FileNames(self):
        """Obtain file names from the SDC."""
        self._info_type = 'file_names'
        response = self.Get()
        return response.text.split(',')
    
    def Local_FileNames(self):
        print('Finding local file names.')
        from pymms import MrMMS_Construct_Filename, MrMMS_Filename2Path
        
        tstart = self.start_date[0:4] + self.start_date[5:7] + self.start_date[8:10]
        fnames = MrMMS_Construct_Filename(self.sc, self.instr, self.mode,
                                          self.level, tstart,
                                          optdesc=self.optdesc)
        paths = MrMMS_Filename2Path(fnames, self._data_root)
        
        return paths
        
    
    def Get(self):
        """Retrieve data from the SDC."""
        # Build the URL sans query
        url = '/'.join((self._sdc_home, self.site, 'files', 'api', 'v1',
                        self._info_type, self.data_type))
        
        # Return the response for the requested URL
        return requests.get(url, params=self.query)
    
    def name2path(self, filename):
        """Convert remote file names to local file name.
        
        Directories of a remote file name are separated by the '/' character,
        as in a web address.
        
        Parameters
        ----------
        filename:  str
                   File name for which the local path is desired.
        
        Returns
        -------
        local_name:  Equivalent local file name. This is the location to
                     which local files are downloaded.
        """
        parts = filename.split('_')
        
        # Survey directories and file names are structured as:
        #   - dirname:  sc/instr/mode/level[/optdesc]/YYYY/MM/
        #   - basename: sc_instr_mode_level[_optdesc]_YYYYMMDD_vX.Y.Z.cdf
        # Index from end to catch the optional descriptor, if it exists
        if parts[2] == 'srvy':
            path = os.path.join(self._data_root, *parts[0:-2],
                                parts[-2][0:4], parts[-2][4:6], filename)
        
        # Burst directories and file names are structured as:
        #   - dirname:  sc/instr/mode/level[/optdesc]/YYYY/MM/DD/
        #   - basename: sc_instr_mode_level[_optdesc]_YYYYMMDDhhmmss_vX.Y.Z.cdf
        # Index from end to catch the optional descriptor, if it exists
        else:
            path = os.path.join(self._data_root, *parts[0:-2],
                                parts[-2][0:4], parts[-2][4:6],
                                parts[-2][6:8], filename)
        return path
    
    def ParseFileNames(self, filename):
        """Parse file names.
        
        Parse official MMS file names. MMS file names are formatted as
            sc_instr_mode_level[_optdesc]_tstart_vX.Y.Z.cdf
        where
            sc:       spacecraft id
            instr:    instrument id
            mode:     data rate mode
            level:    data level
            optdesc:  optional filename descriptor
            tstart:   start time of file
            vX.Y.Z    file version, with X, Y, and Z version numbers
        
        Params
        ------
        filename :  str
                    An MMS file name
        
        Returns
        -------
        parts :  A tuples ordered as
                 (sc, instr, mode, level, optdesc, tstart, version)
                 If opdesc is not present in the file name, the output will
                 contain the empty string ('').
        """
        parts = os.path.basename(filename).split('_')
        
        # If the file does not have an optional descriptor, 
        # put an empty string in its place.
        if len(parts) == 6:
            parts.insert(-2, '')
            
        # Remove the file extension ``.cdf''
        parts[-1] = parts[-1][0:-4]
        return tuple(parts)
    
    def remote2localnames(self, remote_names):
        """Convert remote file names to local file names.
        
        Directories of a remote file name are separated by the '/' character,
        as in a web address.
        
        Parameters:
        remote_names -- Remote file names returned by FileNames.
        
        Returns:
        local_names -- Equivalent local file name. This is the location to
                       which local files are downloaded.
        """
        # os.path.join() requires string arguments, but str.split() return list.
        #   - Unpack with *: https://docs.python.org/2/tutorial/controlflow.html#unpacking-argument-lists
        local_names =list()
        for file in remote_names:
            local_names.append(os.path.join(self._data_root, *file.split('/')[2:]))
        
        if (len(remote_names) == 1) & (type(remote_names) == 'str'):
            local_names = local_names[0]
        
        return local_names
    
    def VersionInfo(self):
        """Obtain version information from the SDC."""
        self._info_type = 'version_info'
        response = self.Get()
        return response.json()
    
    # https://stackoverflow.com/questions/17576009/python-class-property-use-setter-but-evade-getter
    def __setattr__(self, name, value):
        
        # TYPE OF INFO
        #   - Unset other complementary options
        #   - Ensure that at least one of (download | file_names | 
        #     version_info | file_info) are true
        if name == 'anc_product':
            self.data_type = 'ancillary'
        elif name == 'data_type':
            if value not in ('ancillary', 'hk', 'science'):
                raise ValueError('Invalid value for attribute "' + name + '".')
        
        # Set the value
        super(MrMMS_SDC_API, self).__setattr__(name, value)
    
    @property
    def site(self):
        return self._site
    
    @site.setter
    def site(self, value):
        if (value == 'team') | (value == 'team_site') | (value == 'sitl'):
            self._site = 'sitl'
        elif (value == 'public') | (value == 'public_site'):
            self._site = 'public'
        else:
            raise ValueError('Invalid value for the "site" attribute')
    
    @property
    def query(self):
        query = {'sc_id': self.sc,
                 'instrument_id': self.instr,
                 'data_rate_mode': self.mode,
                 'data_level': self.level,
                 'descriptor': self.optdesc,
                 'version': self.version,
                 'start_date': self.start_date,
                 'end_date': self.end_date}
        return query