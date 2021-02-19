from pymms.sdc import mrmms_sdc_api as api
from . import fgm, edp
import datetime as dt
import numpy as np
import xarray as xr
from scipy import constants
import warnings

#ePhoto_Downloader
import re
import requests
import pathlib
from pymms import config
from . import util
from tqdm import tqdm

# prep_ephoto
from cdflib import cdfread

model_url = 'https://lasp.colorado.edu/mms/sdc/public/data/models/fpi/'
data_root = pathlib.Path(config['data_root'])


class ePhoto_Downloader(util.Downloader):
    '''
    Class to download FPI photoelectron distribution functions.
        *download
        *fname
        *intervals
        load
        load_local_file
        *local_file_exists
        *local_path
        *local_dir
    '''
    def __init__(self, sc='mms1', instr='fpi', mode='fast', level='l2',
                 starttime='2017-07-11T22:33:30',
                 endtime='2017-07-11T22:34:30',
                 optdesc=None):
        '''
        Instatiate an PHI Photoelectron Downloader object.
        
        Parameters
        ----------
        sc : str, default='mms1'
            Spacecraft identifier. Must be one of
            ("mms1", "mms2", "mms3", "mms4")
        instr : str, default='fpi'
            Instrument. Must be "fpi"
        mode : str, default='fast'
            Data rate mode. Must be one of ("fast", "srvy", "brst")
        level : str, default="l2"
            Data level. Must be one of
            ("l1b", "sitl", "ql", "l2", "trig")
        starttime, endtime : `datetime.datetime`
            Start and end time of the data interval
        optdesc : str
            Optional descriptor. Must begin with "dis-" or "des-" and end
            with one of ("dist", "moms", "partmoms")
        '''
        self.sc = sc
        self.instr = instr
        self.mode = mode
        self.level = level
        self.optdesc = optdesc
        self.starttime = starttime
        self.endtime = endtime
        self.optdesc = optdesc
    
    def download(self, filename):
    
        remote_file = model_url + '/' + fname
        local_file = self.local_dir() / fname
        
        r = requests.get(remote_file, stream=True, allow_redirects=True)
        total_size = int(r.headers.get('content-length'))
        initial_pos = 0
        
        # Download 
        with open(local_fname, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True,
                      desc=remote_fname, initial=initial_pos,
                      ascii=True) as pbar:
                    
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    
        return local_file
    
    def fname(self, stepper, version=None):
        '''
        Return a model file name.
        
        Parameters
        ----------
        stepper : str
            Stepper ID for the file name. Options are:
            ('0-2', '3-5', '6-8', '12-14', '15-17')
        version : str
            File version number formatted as 'X.Y.Z', where X, Y, and Z
            are integers. If not given, a file name with the appropriate
            stepper id will be searched for on the web.
        
        Returns
        -------
        fname : str
            File name of the photoelectron model
        '''
        # Validate the stepper value
        steppers = ('0-2', '3-5', '6-8', '12-14', '15-17')
        if stepper not in steppers:
            raise ValueError('Stepper {0} is not in {1}'
                             .format(stepper, steppers))
        
        # If no version was given, use a regular expression to try
        # and capture the file from the model file listing
        if version is None:
            v = '[0-9]+.[0-9]+.[0-9]+'
        else:
            v = version
        
        # Build the file name
        fname = '_'.join(('mms', self.instr, self.mode, self.level,
                          'des-bgdist', 'v'+v, 'p'+stepper))

        # If no version was given, search online for possible files.
        if version is None:
            model_fnames = self.model_listing()
            files = [f
                     for f in model_fnames
                     if bool(re.match(fname, f))]
            
            if len(files) == 1:
                fname = files[0]
            else:
                raise ValueError("One file expected. {0} found: {1}"
                                 .format(len(files), files))
        else:
            fname += '.cdf'
        
        return fname
    
    @staticmethod
    def fname_stepper(fname):
        '''
        Extract the stepper ID from the file name.
        '''
        return fname[fname.rfind('p')+1:fname.rfind('.')]
    
    @staticmethod
    def fname_version(fname):
        '''
        Extract the version number from the file name.
        '''
        return fname.split('_')[5][1:]
    
    def load(self, stepper, version=None):
        '''
        Load data
        '''
        if version is None:
            filename = self.fname(stepper)
            version = filename.split('_')[5][1:]
        file_path = self.local_path(stepper, version)
        
        # Download the file
        if not file_path.exists():
            filename = self.download(filename)
        
        # Load all of the data variables from the file
        ds = util.cdf_to_ds(str(file_path))
        return ds
    
    def local_dir(self):
        '''
        Local directory where model files are saved. This is relative
        to the PyMMS data root.
        
        Returns
        -------
        dir : pathlib.Path
            Local directory
        '''
        return pathlib.Path('data', 'models', 'fpi')
    
    def local_file_exists(self, stepper, version):
        '''
        Check if a local file exists.
        
        Parameters
        ----------
        
        Returns
        -------
        exists : bool
            True if local file exists. False otherwise.
        '''
        return self.local_path(stepper, version).exists()
    
    def local_path(self, stepper, version):
        '''
        Absolute path to a single file.
        
        Parameters
        ----------
        
        Returns
        -------
        path : str
            Absolute file path
        '''
        local_path = self.local_dir() / self.fname(stepper, version=version)
        return data_root / local_path
    
    def model_listing(self):
        '''
        Retrieve a listing of photoelectron model files.
        
        Returns
        -------
        files : list
            Names of model files available at the SDC
        '''
        # Find the file names
        #   - Location where they are stored
        #   - Pattern matching the file names
        #   - Download the page to serve as a directory listing
        #   - Parse the page for file names
        fpattern = ('<a href="(mms_fpi_(brst|fast)_l2_des-bgdist_'
                    'v[0-9]+.[0-9]+.[0-9]+_p[0-9]+-[0-9]+.cdf)">')
        response = requests.get(model_url)
        
        return [match.group(1)
                for match in re.finditer(fpattern, response.text)]
    
    @property
    def sc(self):
        return self._sc
    @sc.setter
    def sc(self, sc):
        '''
        Check that a valid spacecraft ID was given.
    
        Parameters
        ----------
        sc : str
            Spacecraft identifier
        '''
        if sc not in ('mms1', 'mms2', 'mms3', 'mms4'):
            raise ValueError('Spacecraft ID {0} must be one of '
                             '("mms1", "mms2", "mms3", "mms4").'
                             .format(sc))
        self._sc = sc
    
    @property
    def instr(self):
        return self._instr
    @instr.setter
    def instr(self, instr):
        '''
        Instrument.
        
        Parameters
        ----------
        instr : str
            Data rate mode. Must be ("fpi").
        '''
        if instr != 'fpi':
            raise ValueError('Instrument {0} must "fpi"'.format(instr))
        self._instr = instr
    
    @property
    def mode(self):
        return self._mode
    @mode.setter
    def mode(self, mode):
        '''
        Set the mode property.
        
        Parameters
        ----------
        mode : str
            Data rate mode. Must be ("fast", "srvy", "brst"). "srvy"
            is translated to "fast"
        '''
        if mode == 'srvy':
            mode == 'fast'
        elif mode not in ('fast', 'brst'):
            raise ValueError('Data rate mode {0} must be one of '
                             '("fast", "brst").'.format(mode))
        self._mode = mode
    
    @property
    def level(self):
        return self._level
    @level.setter
    def level(self, level):
        '''
        Set the data level property.
        
        Parameters
        ----------
        level : str
            Data rate mode. Must be ("l1b", "sitl", "ql", "l2", "trig")
        '''
        if level == 'srvy':
            level == 'fast'
        elif level not in ("l1b", "sitl", "ql", "l2", "trig"):
            raise ValueError('Data rate mode {0} must be one of '
                             '("l1b", "sitl", "ql", "l2", "trig").'
                             .format(level))
        self._level = level
    
    @property
    def starttime(self):
        return self._starttime
    @starttime.setter
    def starttime(self, starttime):
        # Convert string to datetime64 object
        self._starttime = np.datetime64(starttime, 's')
    
    @property
    def endtime(self):
        return self._endtime
    @starttime.setter
    def endtime(self, endtime):
        # Convert string to datetime object
        self._starttime = np.datetime64(endtime, 's')
    


def center_timestamps(fpi_data):
    '''
    FPI time stamps are at the beginning of the sampling interval.
    Adjust the timestamp to the center of the interval.
    
    Parameters
    ----------
    fpi_data : `xarray.Dataset`
        Dataset containing the time coordinates to be centered
    
    Returns
    -------
    new_data : `xarray.Dataset`
        A new dataset with the time coordinates centered
    '''
    
    t_delta = np.timedelta64(int(1e9 * (fpi_data['Epoch_plus_var'].data
                                        + fpi_data['Epoch_minus_var'].data)
                                 / 2.0), 'ns')
    
    data = fpi_data.assign_coords({'time': fpi_data['time'] + t_delta})
    data['time'].attrs = fpi_data['attrs']
    data['Epoch_plus_var'] = t_delta
    data['Epoch_minus_var'] = t_delta
    
    return data


def check_spacecraft(sc):
    '''
    Check that a valid spacecraft ID was given.
    
    Parameters
    ----------
    sc : str
        Spacecraft identifier
    '''
    if sc not in ('mms1', 'mms2', 'mms3', 'mms4'):
        raise ValueError('{} is not a recongized SC ID. '
                         'Must be ("mms1", "mms2", "mms3", "mms4")'
                         .format(sc))

def check_mode(mode):
    '''
    Check that a valid data rate mode was given.
    
    Parameters
    ----------
    mode : str
        Data rate mode. Can be ('brst', 'srvy', 'fast'). If 'srvy' is
        given, it is changed to 'fast'.
    
    Returns
    -------
    mode : str
        A valid data rate mode for FPI
    '''
    
    modes = ('brst', 'fast')
    if mode == 'srvy':
        mode = 'fast'
    
    if mode not in modes:
        raise ValueError('Mode "{0}" is not in {1}'.format(mode, modes))

    return mode


def check_species(species):
    '''
    Check that a valid particle species was given.
    
    Parameters
    ----------
    species : str
        Particle species: 'e' or 'i'.
    
    Returns
    -------
    mode : str
        A valid data rate mode for FPI
    '''
    if species not in ('e', 'i'):
        raise ValueError('{} is not a recongized species. '
                         'Must be ("i", "e")')


def download_ephoto_models():
    '''
    Download photoelectron model distribution functions.
    
    The file names of the photoelectron models contain the stepper-ids. Which
    stepper-id is in use is found externally, in the appropriate dis-moms or
    des-moms
    '''
    
    # Find the file names
    #   - Location where they are stored
    #   - Pattern matching the file names
    #   - Download the page to serve as a directory listing
    url = 'https://lasp.colorado.edu/mms/sdc/public/data/models/fpi/'
    fpattern = ('mms_fpi_(brst|fast)_l2_d(i|e)s-bgdist_'
                'v[0-9]+.[0-9]+.[0-9]+_p[0-9]+-[0-9]+.cdf')
    response = requests.get(url)
    
    # Local repository
    local_dir = data_root.joinpath(*url.split('/')[6:9])
    if not local_dir.exists():
        local_dir.mkdir(parents=True)
    local_files = []
    
    # Parse the page and download the files
    for match in re.finditer(fpattern, response.text):
        
        # Remote file
        remote_fname = match.group(0)
        remote_file = '/'.join((url, remote_fname))
        
        # Local file after download
        local_fname = local_dir / remote_fname
        
        
        r = requests.get(remote_file, stream=True, allow_redirects=True)
        total_size = int(r.headers.get('content-length'))
        initial_pos = 0
        
        # Download 
        with open(local_fname, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True,
                      desc=remote_fname, initial=initial_pos,
                      ascii=True) as pbar:
                    
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        local_files.append(local_fname)
    
    return local_files


def prep_ephoto(sdc, startdelphi, parity=None):
    '''
    Prepare the photo electron distributions
    '''
    # Download the moments file
    sdc.optdesc = 'des-moms'
    moms_files = sdc.download_files()
    
    moms_files = api.sort_files(moms_files)[0]
    cdf = cdfread.CDF(moms_files[0])
    scl = np.float(cdf.attget('Photoelectron_model_scaling_factor', entry=0)['Data'])
    fphe = cdf.attget('Photoelectron_model_filenames', entry=0)['Data']
    cdf.close()
    
    # Check to see if the file name and scaling factor change
    # If it does, the implementation will have to change to be
    # applied on a per-file basis
    for file in moms_files[1:]:
        cdf = cdfread.CDF(file)
        if scl != cdf.attget('Photoelectron_model_scaling_factor'):
            raise ValueError('Scale factor changes between files.')
        if fphe != cdf.attget('Photoelectron_model_filenames'):
            raise ValueError('Photoelectron mode file name changes.')
        cdf.close()
    
    # Extract the stepper number
    stepper = ePhoto_Downloader.fname_stepper(fphe)
    version = ePhoto_Downloader.fname_version(fphe)
    
    # Load the photo-electron model file
    ePhoto = ePhoto_Downloader(mode=sdc.mode)
    f_photo = ePhoto.load(stepper, version)
    
    # Map the measured startdelphi to the model startdelphi
    idx = np.int16(np.floor(startdelphi/16))
    if sdc.mode == 'brst':
        f_p0_vname = '_'.join(('mms', 'des', 'bgdist', 'p0', sdc.mode))
        f_p1_vname = '_'.join(('mms', 'des', 'bgdist', 'p1', sdc.mode))
        sdp_vname = '_'.join(('mms', 'des', 'startdelphi', 'counts', sdc.mode))
        
        f0 = f_photo[f_p0_vname][idx,:,:,:]
        f1 = f_photo[f_p1_vname][idx,:,:,:]
        
        f0 = f0.rename({sdp_vname: 'Epoch'}).assign_coords({'Epoch': startdelphi['Epoch']})
        f1 = f1.rename({sdp_vname: 'Epoch'}).assign_coords({'Epoch': startdelphi['Epoch']})
        
        # Select the proper parity
        f_model = f0.where(parity == 0, f1)
    else:
        f_vname = '_'.join(('mms', 'des', 'bgdist', sdc.mode))
        sdp_vname = '_'.join(('mms', 'des', 'startdelphi', 'counts', sdc.mode))
        
        f_model = (f_photo[f_vname][idx,:,:,:]
                   .rename({sdp_vname: 'Epoch'})
                   .assign_coords({'Epoch': startdelphi['Epoch']})
                   )

    return scl * f_model


def load_dist(sc, mode, species, start_date, end_date,
              ephoto=True):
    """
    Load FPI distribution function data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    species : str
        Particle species: ('i', 'e') for ions and electrons, respectively.
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    ephoto : bool
        Remove photo electrons from the distribution. Applies only to
        des data. Requires downloading the des-moms files
    
    Returns
    -------
    dist : `metaarray.metaarray`
        Particle distribution function.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    check_species(species)
    
    # File and variable name parameters
    instr = 'd{0}s'.format(species)
    optdesc = instr+'-dist'
    dist_vname = '_'.join((sc, instr, 'dist', mode))
    epoch_vname = 'Epoch'
    phi_vname = '_'.join((sc, instr, 'phi', mode))
    theta_vname = '_'.join((sc, instr, 'theta', mode))
    energy_vname = '_'.join((sc, instr, 'energy', mode))
    startdelphi_vname = '_'.join((sc, instr, 'startdelphi', 'count', mode))
    parity_vname = '_'.join((sc, instr, 'steptable', 'parity', mode))
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, 'fpi', mode, 'l2',
                            optdesc=optdesc,
                            start_date=start_date,
                            end_date=end_date)
    fpi_files = sdc.download_files()
    fpi_files = api.sort_files(fpi_files)[0]
    
    # Concatenate data along the records (time) dimension, which
    # should be equivalent to the DEPEND_0 variable name of the
    # density variable.
    fpi_data = []
    variables = [dist_vname]
    if (species == 'e') & ephoto:
        variables.append(startdelphi_vname)
        if mode == 'brst':
            variables.append(parity_vname)
    
    for file in fpi_files:
        fpi_data.append(util.cdf_to_ds(file, variables))
    fpi_data = xr.concat(fpi_data, dim=fpi_data[0][dist_vname].dims[0])
    dist = fpi_data[dist_vname]
    
    # Subtract photoelectrons
    if ephoto & (species == 'e'):
        if mode == 'brst':
            phi_rename = 'phi'
            f_model = prep_ephoto(sdc,
                                  fpi_data[startdelphi_vname],
                                  fpi_data[parity_vname])
        else:
            phi_rename = phi_vname
            f_model = prep_ephoto(sdc,
                                  fpi_data[startdelphi_vname])
        
        # Re-assign coordinates so that the model can be subtracted
        # from the distribution. Note that the energy tables for
        # parity 0 and parity 1 are no longer in the des-dist files
        # or the model files, so it is impossible to reconstruct the
        # coordinates. Stealing them from the distribution itself
        # should be fine, though, because we used the measured
        # distribution as a template.
        sector_index_vname = '_'.join(('mms', 'des', 'sector', 'index', mode))
        pixel_index_vname = '_'.join(('mms', 'des', 'pixel', 'index', mode))
        energy_index_vname = '_'.join(('mms', 'des', 'energy', 'index', mode))
        
        f_model = (f_model
                   .rename({sector_index_vname: phi_rename,
                            pixel_index_vname: theta_vname,
                            energy_index_vname: 'energy'})
                   .assign_coords({phi_vname: dist[phi_vname],
                                   theta_vname: dist[theta_vname],
                                   energy_vname: dist[energy_vname]})
                   .drop_vars(['phi', 'energy'], errors='ignore')
                   )
        
        dist -= f_model
    
    # Rename coordinates
    #   - Phi is record varying in burst but not in survey data,
    #     so the coordinates are different 
    coord_rename_dict = {epoch_vname: 'time',
                         phi_vname: 'phi',
                         theta_vname: 'theta',
                         energy_vname: 'energy',
                         'energy': 'energy_index'}
    if mode == 'brst':
        coord_rename_dict['phi'] = 'phi_index'
    dist = dist.rename(coord_rename_dict)
    
    # Select the appropriate time interval
    dist = dist.sel(time=slice(start_date, end_date))
    
    dist.attrs['sc'] = sc
    dist.attrs['mode'] = mode
    dist.attrs['species'] = species
    return dist


def load_moms(sc, mode, species, start_date, end_date):
    """
    Load FPI distribution function data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    species : str
        Particle species: ('i', 'e') for ions and electrons, respectively.
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    
    Returns
    -------
    dist : `metaarray.metaarray`
        Particle distribution function.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    check_species(species)
    
    # File and variable name parameters
    instr = 'd{0}s'.format(species)
    optdesc = instr+'-moms'
    epoch_vname = 'Epoch'
    n_vname = '_'.join((sc, instr, 'numberdensity', mode))
    v_vname = '_'.join((sc, instr, 'bulkv', 'dbcs', mode))
    p_vname = '_'.join((sc, instr, 'prestensor', 'dbcs', mode))
    t_vname = '_'.join((sc, instr, 'temptensor', 'dbcs', mode))
    q_vname = '_'.join((sc, instr, 'heatq', 'dbcs', mode))
    t_para_vname = '_'.join((sc, instr, 'temppara', mode))
    t_perp_vname = '_'.join((sc, instr, 'tempperp', mode))
    v_labl_vname = '_'.join((sc, instr, 'bulkv', 'dbcs', 'label', mode))
    q_labl_vname = '_'.join((sc, instr, 'heatq', 'dbcs', 'label', mode))
    espectr_vname = '_'.join((sc, instr, 'energyspectr', 'omni', mode))
    cart1_labl_vname = '_'.join((sc, instr, 'cartrep', mode))
    cart2_labl_vname = '_'.join((sc, instr, 'cartrep', mode, 'dim2'))
    e_labl_vname = '_'.join((sc, instr, 'energy', mode))
    varnames = [n_vname, v_vname, p_vname, t_vname, q_vname,
                t_para_vname, t_perp_vname, espectr_vname]
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, 'fpi', mode, 'l2',
                            optdesc=optdesc,
                            start_date=start_date,
                            end_date=end_date)
    fpi_files = sdc.download_files()
    fpi_files = api.sort_files(fpi_files)[0]
    
    # Concatenate data along the records (time) dimension, which
    # should be equivalent to the DEPEND_0 variable name of the
    # density variable.
    fpi_data = []
    for file in fpi_files:
        fpi_data.append(util.cdf_to_ds(file, varnames))
    fpi_data = xr.concat(fpi_data, dim=fpi_data[0][n_vname].dims[0])
    
    fpi_data = fpi_data.rename({epoch_vname: 'time',
                                n_vname: 'density',
                                v_vname: 'velocity',
                                p_vname: 'prestensor',
                                t_vname: 'temptensor',
                                q_vname: 'heatflux',
                                t_para_vname: 'temppara',
                                t_perp_vname: 'tempperp',
                                v_labl_vname: 'velocity_index',
                                q_labl_vname: 'heatflux_index',
                                espectr_vname: 'omnispectr',
                                cart1_labl_vname: 'cart_index_dim1',
                                cart2_labl_vname: 'cart_index_dim2',
                                'energy': 'energy_index',
                                e_labl_vname: 'energy'})
    fpi_data = fpi_data.sel(time=slice(start_date, end_date))
    
    fpi_data = fpi_data.assign(t=(fpi_data['temptensor'][:,0,0] 
                                  + fpi_data['temptensor'][:,1,1]
                                  + fpi_data['temptensor'][:,2,2]
                                  ) / 3.0,
                               p=(fpi_data['prestensor'][:,0,0] 
                                  + fpi_data['prestensor'][:,1,1]
                                  + fpi_data['prestensor'][:,2,2]
                                  ) / 3.0
                               )
    
    # Adjust the time stamp
    fpi_data = center_timestamps(fpi_data)
    
    for name, value in fpi_data.items():
        value.attrs['sc'] = sc
        value.attrs['mode'] = mode
        value.attrs['species'] = species
    
    return fpi_data


def load_moms_pd(sc, mode, species, start_date, end_date):
    """
    Load FPI moments as a Pandas DataFrame.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    species : str
        Particle species: ('i', 'e') for ions and electrons, respectively.
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    
    Returns
    -------
    moms : `pandas.DataFrame`
        Moments of the distribution function.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    check_species(species)
    
    # File and variable name parameters
    instr = 'd{0}s'.format(species)
    optdesc = instr+'-moms'
    n_vname = '_'.join((sc, instr, 'numberdensity', mode))
    v_vname = '_'.join((sc, instr, 'bulkv', 'dbcs', mode))
    p_vname = '_'.join((sc, instr, 'prestensor', 'dbcs', mode))
    t_vname = '_'.join((sc, instr, 'temptensor', 'dbcs', mode))
    q_vname = '_'.join((sc, instr, 'heatq', 'dbcs', mode))
    t_para_vname = '_'.join((sc, instr, 'temppara', mode))
    t_perp_vname = '_'.join((sc, instr, 'tempperp', mode))
    varnames = [n_vname, v_vname, p_vname, t_vname, q_vname,
                t_para_vname, t_perp_vname]
    
    # Download the data
    sdc = api.MrMMS_SDC_API(sc, 'fpi', mode, 'l2',
                            optdesc=optdesc,
                            start_date=start_date,
                            end_date=end_date)
    fpi_files = sdc.download_files()
    fpi_files = api.sort_files(fpi_files)[0]
    
    # Read the data from files
    fpi_df = util.cdf_to_df(fpi_files, varnames)
    
    # Calculate Maxwellian Entropy
    if maxwell_entropy:
        data.append(mexwellian_entropy(data[0]))
    
    # Rename columns
    fpi_df.rename(columns={n_vname: 'N'}, inplace=True)
    fpi_df.rename(columns={t_para_vname: 'T_para'}, inplace=True)
    fpi_df.rename(columns={t_perp_vname: 'T_perp'}, inplace=True)
    util.rename_df_cols(fpi_df, v_vname, ('Vx', 'Vy', 'Vz'))
    util.rename_df_cols(fpi_df, q_vname, ('Q_xx', 'Q_yy', 'Q_zz'))
    util.rename_df_cols(fpi_df, t_vname,
                        ('T_xx', 'T_xy', 'T_xz',
                         'T_yx', 'T_yy', 'T_yz',
                         'T_zx', 'T_zy', 'T_zz'
                         ))
    util.rename_df_cols(fpi_df, p_vname,
                        ('P_xx', 'P_xy', 'P_xz',
                         'P_yx', 'P_yy', 'P_yz',
                         'P_zx', 'P_zy', 'P_zz'
                         ))

    # Drop redundant components of the pressure and temperature tensors
    fpi_df.drop(columns=['T_yx', 'T_zx', 'T_zy',
                         'P_yx', 'P_zx', 'P_zy'],
                inplace=True
                )
    
    # Scalar temperature and pressure
    fpi_df['t'] = (fpi_df['T_xx'] + fpi_df['T_yy'] + fpi_df['T_zz'])/3.0
    fpi_df['p'] = (fpi_df['P_xx'] + fpi_df['P_yy'] + fpi_df['P_zz'])/3.0
    fpi_df.sc = sc
    fpi_df.mode = mode
    fpi_df.species = species
    
    return fpi_df


def maxwellian_distribution(dist, N=None, bulkv=None, T=None):
    """
    Given a measured velocity distribution function, create a Maxwellian
    distribution function with the same density, bulk velociy, and
    temperature.
    
    Parameters
    ----------
    dist : `xarray.DataSet`
        A time series of 3D velocity distribution functions
    N : `xarray.DataArray`
        Number density computed from `dist`.
    bulkv : `xarray.DataArray`
        Bulk velocity computed from `dist`.
    T : `xarray.DataArray`
        Scalar temperature computed from `dist`
    
    Returns
    -------
    f_max : `xarray.DataSet`
        Maxwellian distribution function.
    """
    
    eV2K = constants.value('electron volt-kelvin relationship')
    eV2J = constants.eV
    kB   = constants.k
    mass = species_to_mass(dist.attrs['species'])
    
    if density is None:
        N = density(dist)
    if bulkv is None:
        bulkv = velocity(dist, N=N)
    if T is None:
        T = temperature(dist, N=N, V=bulkv)
    
    
    phi = np.deg2rad(dist['phi'])
    theta = np.deg2rad(dist['theta'])
    velocity = np.sqrt(2.0 * eV2J / mass * dist['energy'])  # m/s
    
    vxsqr = (-velocity * np.sin(theta) * np.cos(phi) - (1e3*bulkv[:,0]))**2
    vysqr = (-velocity * np.sin(theta) * np.sin(phi) - (1e3*bulkv[:,1]))**2
    vzsqr = (-velocity * np.cos(theta) - (1e3*bulkv[:,2]))**2
    
    f_out = (1e-6 * N 
             * (mass / (2 * np.pi * kB * eV2K * T))**(3.0/2.0)
             * np.exp(-mass * (vxsqr + vysqr + vzsqr)
                      / (2.0 * kB * eV2K * T))
             )
    f_out = f_out.drop('velocity_index')
    
    try:
        f_out = f_out.transpose('time', 'phi', 'theta', 'energy_index')
    except ValueError:
        f_out = f_out.transpose('time', 'phi_index', 'theta', 'energy_index')
    
    f_out.name = 'Equivalent Maxwellian distribution'
    f_out.attrs['sc'] = dist.attrs['sc']
    f_out.attrs['mode'] = dist.attrs['mode']
    f_out.attrs['species'] = dist.attrs['species']
    f_out.attrs['long_name'] = ('Maxwellian distribution constructed from '
                                'the density, velocity, and temperature of '
                                'the measured distribution function.')
    f_out.attrs['standard_name'] = 'maxwellian_distribution'
    f_out.attrs['units'] = 's^3/cm^6'
    
    return f_out


def maxwellian_entropy(N, P):
    """
    Calculate the maxwellian entropy of a distribution.
    
    Parameters
    ----------
    N : `xarray.DataArray`
        Number density.
    P : `xarray.DataArray`
        Scalar pressure.
    
    Returns
    -------
    Sb : `xarray.DataArray`
        Maxwellian entropy
    """
    J2eV = constants.value('joule-electron volt relationship')
    kB   = constants.k
    mass = species_to_mass(N.attrs['species'])
    
    Sb = (-kB * 1e6 * N
          * (np.log((1e19 * mass * N**(5.0/3.0)
                    / 2 / np.pi / P)**(3/2)
                   )
             - 3/2
             )
          )
    
    Sb.name = 'S{}'.format(N.attrs['species'])
    Sb.attrs['species'] = N.attrs['species']
    Sb.attrs['long_name'] = 'Boltzmann entropy for a given density and pressure.'
    Sb.attrs['standard_name'] = 'Boltzmann_entropy'
    Sb.attrs['units'] = 'J/K/m^3 ln(s^3/m^6)'
    return Sb


def moments(dist, moment, **kwargs):
    """
    Calculate the moments a velocity distribution function.
    
    Parameters
    ----------
    dist : `xarray.DataSet`
        Number density.
    moment : str
        Name of the moment of the distribution to calculate.
    \*\*kwargs : dict
        Keywords for the corresponding moments function.
    
    Returns
    -------
    Sb : `xarray.DataArray`
        Maxwellian entropy
    """
    valid_moms = ('density', 'velocity', 'pressure', 'temperature',
                  'entropy', 'epsilon',
                  'N', 'V', 'P', 'T', 'S', 'e')
    if moment not in valid_moms:
        raise ValueError('Moment {0} is not in {1}'
                         .format(moment, valid_moms)
                         )
    
    if moment in ('density', 'N'):
        func = density
    elif moment in ('velocity', 'V'):
        func = velocity
    elif moment in ('temperature', 'T'):
        func = temperature
    elif moment in ('pressure', 'P'):
        func = pressure
    elif moment in ('entropy', 'S'):
        func = pressure
    elif moment in ('epsilon', 'e'):
        func = pressure
    
    return func(dist, **kwargs)


def precondition(dist, E0=100, E_low=10, scpot=None,
                       low_energy_extrapolation=True,
                       high_energy_extrapolation=True):
    '''
    Before being sent to the integration routine, skymaps are preprocessed
    in the following manner:
      1. f(phi = 0) is repeated as f(phi = 360) to ensure that the periodic
         boundary condition is incorporated to the azimuthal integration.
      2. f(theta=0) = 0 and f(theta=180) = 0 data points are added to ensure
         the polar integration goes from 0 to 180.  The sin(theta)
         dependence of the polar integration force the integrand at
         theta = 0 and theta = 180 to zero regardless of the value of the
         phase space density
      3. f(U = 0) = 0 and f(U=1) =0 data points are added to ensure the
         integration goes from E->0 to E->infinity. V = 0 forces the
         integrand equal to zero regardless of the phase space density. 
    
    Parameters
    ----------
    dist : `metaarray.MetaArray`
        The velocity distribution function (s^3/cm^6) with azimuth, polar,
        and energy dependencies as attributes.
    E0 : float
        Energy value (eV) used when mapping energy bins from range [0,Emax]
        to [0, inf)
    E_low : float
        Energy value (eV) representing the low-energy cut-off
    '''
    J2eV = constants.value('joule-electron volt relationship')
    e = constants.e # C
    
    # pad looks like right approach, but it is still experimental and
    # instead of using the values specified by the constant_values keyword,
    # it always uses np.nan.
    '''
    out = dist.pad(pad_width={'phi', (0, 1),
                              'theta', (1, 1),
                              'e-bin', (1, 1)},
                   mode='constant',
                   constant_values={'phi': (0, 0),
                                    'theta': (0, 0),
                                    'e-bin': (0, 0)}
                   )
    '''
    
    # Append boundary point to make phi periodic
    #   Note that the dimensions must be ordered (time, phi, theta, energy)
    #   for the indexing to work
    try:
        f_phi = dist[:,0,:,:].assign_coords(phi=dist['phi'][0] + 360.0)
        f_out = xr.concat([dist, f_phi], 'phi')
    except ValueError:
        f_phi = dist[:,0,:,:].assign_coords(phi=dist['phi'][:,0] + 360.0)
        f_out = xr.concat([dist, f_phi], 'phi_index')
    
    # Create boundary points to have theta range be [0,180] inclusive.
    # Note that the sin(theta) forces the integrand to be 0 at the
    # boundaries regardless of what the distribution function
    f_theta = xr.DataArray(np.zeros(shape=(2,)),
                           dims='theta',
                           coords={'theta': [0, 180]})
    
    # Append the boundary points to the beginning and end of the
    # array. This will change the order of the dimensions. Set the
    # values at the boundaries to zero (arbitrary) and transpose
    # back to the original shape.
    f_out = xr.concat([f_theta[0], f_out], 'theta')
    f_out = xr.concat([f_out, f_theta[1]], 'theta')

    # This throws an error:
    # ValueError: The truth value of an array with more than one element
    #             is ambiguous. Use a.any() or a.all()
    #
    # f_out = xr.combine_by_coords(f_out, f_theta)
    
    # Adjust for spacecraft potential
    #   - E' = E +- q*Vsc, where + is for ions and - is for electrons
    #   - Make a copy of energy so that the original dest['energy']
    #     does not change
    energy = f_out['energy'].copy()
    if scpot is not None:
#        sign = -1 if dist.attrs['species'] == 'e' else 1
        sign = -1
        energy += (sign * J2eV * e * scpot['Vsc'])
    
    # Low energy integration limit
    #   - Exclude data below the low-energy limit
    #   - xr.DataArray.integrate does not avoid NaNs
    #   - Fill with 0.0 because at front of array and trapezoidal integration
    #     results in zero area.
    mask = energy >= E_low
    energy = energy.where(mask, 0.0)
    f_out = f_out.where(mask, 0.0)
    f_out = f_out.assign_coords({'energy': energy})
    
    # Exclude measurements from below the spacecraft potential
    #   - Same reasoning as for low-energy integration limit
    if scpot is not None:
        mask = energy >= 0
        energy = energy.where(mask, 0.0)
        f_out = f_out.where(mask, 0.0)
        f_out = f_out.assign_coords({'energy': energy})
    
    # Energy extrapolation
    #   - Map the energy to range [0, 1]
    U = f_out['energy'] / (f_out['energy'] + E0)
    U = U.drop_vars('energy')
    
    U_boundaries = xr.DataArray(np.zeros(shape=(f_out.sizes['time'], 2)),
                                dims=('time', 'energy_index'),
                                coords={'time': f_out['time']}
                                )
    U_boundaries[:,-1] = 1.0
    
    # Append the boundary points to the beginning and end of the array.
    U = xr.concat([U_boundaries[:,0], U], 'energy_index')
    U = xr.concat([U, U_boundaries[:,-1]], 'energy_index')
    
    # Create boundary points for the energy at 0 and infinity, essentially
    # extrapolating the distribution to physical limits. Since absolute
    # zero and infinite energies are impossible, set the values in the
    # distribution to zero at those points. This changes the order of the
    # dimensions so they will have to be transposed back.
    f_energy = xr.DataArray(np.zeros((2,)),
                            dims='energy_index',
                            coords={'energy': ('energy_index', [0, np.inf])})
    
    # Append the extrapolated points to the distribution
    f_out = xr.concat([f_energy[0], f_out], 'energy_index')
    f_out = xr.concat([f_out, f_energy[1]], 'energy_index')
    
    # Assign U as another coordinate
    f_out = f_out.assign_coords(U=U)

    # Convert to radians
    f_out = f_out.assign_coords(phi=np.deg2rad(f_out['phi']))
    f_out = f_out.assign_coords(theta=np.deg2rad(f_out['theta']))
    
    # Include metadata
    f_out.attrs['Energy_e0'] = E0
    f_out.attrs['Lower_energy_integration_limit'] = E_low
    f_out.attrs['Upper_energy_integration_limit'] = None
    return f_out


def species_to_mass(species):
    '''
    Return the mass (kg) of the given particle species.
    
    Parameters
    ----------
    species : str
        Particle species: 'i' or 'e'
    
    Returns
    ----------
    mass : float
        Mass of the given particle species
    '''
    if species == 'i':
        mass = constants.m_p
    elif species == 'e':
        mass = constants.m_e
    else:
        raise ValueError(('Unknown species {}. Select "i" or "e".'
                          .format(species))
                         )
    
    return mass


def density(dist, **kwargs):
    '''
    Calculate number density from a time series of 3D distribution function.
    
    Parameters
    ----------
    dist : `xarray.DataArray`
        A time series of 3D distribution functions
    \*\*kwargs : dict
        Keywords accepted by the `precondition` function.
    
    Returns
    -------
    N : `xarray.DataArray`
        Number density
    '''
    mass = species_to_mass(dist.attrs['species'])
    f = precondition(dist, **kwargs)
    
    if dist.attrs['mode'] == 'brst':
        N = xr.concat([density_3D(f1, mass, f.attrs['Energy_e0'])
                       for f1 in f],
                      'time')
    else:
        N = density_4D(f, mass, f.attrs['Energy_e0'])
    
    # Add metadata
    N.name = 'N{}'.format(dist.attrs['species'])
    N.attrs['long_name'] = ('Number density calculated by integrating the '
                            'distribution function.')
    N.attrs['species'] = dist.attrs['species']
    N.attrs['standard_name'] = 'number_density'
    N.attrs['units'] = 'cm^-3'
    
    return N


def entropy(dist, **kwargs):
    '''
    Calculate entropy from a time series of 3D velocity space
    distribution function.
    
    .. [1] Liang, H., Cassak, P. A., Servidio, S., Shay, M. A., Drake,
        J. F., Swisdak, M., … Delzanno, G. L. (2019). Decomposition of
        plasma kinetic entropy into position and velocity space and the
        use of kinetic entropy in particle-in-cell simulations. Physics
        of Plasmas, 26(8), 82903. https://doi.org/10.1063/1.5098888
    
    Parameters
    ----------
    dist : `xarray.DataArray`
        A time series of 3D distribution functions
    \*\*kwargs : dict
        Keywords accepted by the `precondition` function.
    
    Returns
    -------
    S : `xarray.DataArray`
        Entropy
    '''
    mass = species_to_mass(dist.attrs['species'])
    
    f = precondition(dist, **kwargs)
    
    if dist.attrs['mode'] == 'brst':
        S = xr.concat([entropy_3D(f1, mass, f.attrs['Energy_e0'])
                       for f1 in f],
                      'time')
    else:
        S = entropy_4D(f, mass, f.attrs['Energy_e0'])
    
    S.name = 'S{}'.format(dist.attrs['species'])
    S.attrs['long_name'] = 'Velocity space entropy density'
    S.attrs['standard_name'] = 'entropy_density'
    S.attrs['units'] = 'J/K/m^3 ln(s^3/m^6)'
    
    return S


def epsilon(dist, dist_max=None, N=None, V=None, T=None, **kwargs):
    '''
    Calculate epsilon [1]_ from a time series of 3D velocity space
    distribution functions.
    
    .. [1] Greco, A., Valentini, F., Servidio, S., &
        Matthaeus, W. H. (2012). Inhomogeneous kinetic effects related
        to intermittent magnetic discontinuities. Phys. Rev. E,
        86(6), 66405. https://doi.org/10.1103/PhysRevE.86.066405
    
    Parameters
    ----------
    dist : `xarray.DataArray`
        A time series of 3D distribution functions
    dist_max : `xarray.DataArray`
        The maxwellian equivalent of `dist`. If not provided,
        it is calculated
    N : `xarray.DataArray`
        Number density computed from `dist`. If not provided,
        it is calculated
    V : `xarray.DataArray`
        Bulk velocity computed from `dist`. If not provided,
        it is calculated
    T : `xarray.DataArray`
        Scalar temperature computed from `dist`. If not provided,
        it is calculated
    \*\*kwargs : dict
        Keywords accepted by the `precondition` function.
    
    Returns
    -------
    e : `xarray.DataArray`
        Epsilon parameter
    '''
    mass = species_to_mass(dist.attrs['species'])
    if N is None:
        N = density(dist, **kwargs)
    if dist_max is None:
        if V is None:
            V = velocity(dist, N=N, **kwargs)
        if T is None:
            T = temperature(dist, N=N, V=V, **kwargs)
            T = (T[:,0,0] + T[:,1,1] + T[:,2,2]) / 3.0
        dist_max = maxwellian_distribution(dist, N, V, T)
    
    f = precondition(dist, **kwargs)
    f_max = precondition(dist_max, **kwargs)
    
    if dist.attrs['mode'] == 'brst':
        e = xr.concat([epsilon_3D(f1, mass, f.attrs['Energy_e0'], f1_max, n1)
                       for f1, f1_max, n1 in zip(f, f_max, N)],
                      'time')
    else:
        e = epsilon_4D(f, mass, f.attrs['Energy_e0'], f_max, N)
    
    e.name = 'Epsilon{}'.format(dist.attrs['species'])
    e.attrs['long_name'] = 'Non-maxwellian'
    e.attrs['standard_name'] = 'epsilon'
    e.attrs['units'] = '$(s/cm)^{3/2}$'
    
    return e


def pressure(dist, N=None, T=None, **kwargs):
    '''
    Calculate pressure tensor from a time series of 3D velocity space
    distribution function.
    
    Parameters
    ----------
    dist : `xarray.DataArray`
        A time series of 3D distribution functions
    N : `xarray.DataArray`
        Number density computed from `dist`. If not provided,
        it is calculated
    T : `xarray.DataArray`
        Scalar temperature computed from `dist`. If not provided,
        it is calculated
    \*\*kwargs : dict
        Keywords accepted by the `precondition` function.
    
    Returns
    -------
    P : `xarray.DataArray`
        Pressure tensor
    '''
    mass = species_to_mass(dist.attrs['species'])
    if N is None:
        N = density(dist, **kwargs)
    if T is None:
        T = temperature(dist, N=N, **kwargs)
    
    P = pressure_4D(N, T)
    
    P.name = 'P{0}'.format(dist.attrs['species'])
    P.attrs['long_title'] = ('Pressure calculated from d{}s velocity '
                             'distribution function.'
                             .format(dist.attrs['species']))
    P.attrs['units'] = 'nPa'
    
    return P


def temperature(dist, N=None, V=None, **kwargs):
    '''
    Calculate the temperature tensor from a time series of 3D velocity
    space distribution function.
    
    Parameters
    ----------
    dist : `xarray.DataArray`
        A time series of 3D distribution functions
    N : `xarray.DataArray`
        Number density computed from `dist`. If not provided,
        it is calculated
    V : `xarray.DataArray`
        Bulk velocity computed from `dist`. If not provided,
        it is calculated
    \*\*kwargs : dict
        Keywords accepted by the `precondition` function.
    
    Returns
    -------
    T : `xarray.DataArray`
        Temperature tensor
    '''
    mass = species_to_mass(dist.attrs['species'])
    if N is None:
        N = density(dist, **kwargs)
    if V is None:
        V = velocity(dist, N=N, **kwargs)
    
    f = precondition(dist, **kwargs)
    
    if dist.attrs['mode'] == 'brst':
        T = xr.concat([temperature_3D(f1, mass, f.attrs['Energy_e0'], n1, v1)
                       for f1, n1, v1 in zip(f, N, V)],
                      'time')
    else:
        T = temperature_4D(f, mass, f.attrs['Energy_e0'], N, V)
    
    T.name = 'T{0}'.format(dist.attrs['species'])
    T.attrs['species'] = dist.attrs['species']
    T.attrs['long_name'] = ('Temperature calculated from d{}s velocity '
                            'distribution function.'.format(dist.attrs['species']))
    T.attrs['standard_name'] = 'temperature_tensor'
    T.attrs['units'] = 'eV'
    
    return T


def velocity(dist, N=None, **kwargs):
    '''
    Calculate velocity from a time series of 3D velocity space
    distribution functions.
    
    Parameters
    ----------
    dist : `xarray.DataArray`
        A time series of 3D distribution functions
    N : `xarray.DataArray`
        Number density computed from `dist`. If not provided,
        it is calculated
    \*\*kwargs : dict
        Keywords accepted by the `precondition` function.
    
    Returns
    -------
    V : `xarray.DataArray`
        Bulk velocity
    '''
    mass = species_to_mass(dist.attrs['species'])
    if N is None:
        N = density(dist, **kwargs)
    
    f = precondition(dist, **kwargs)
    
    if dist.attrs['mode'] == 'brst':
        V = xr.concat([velocity_3D(f1, mass, f.attrs['Energy_e0'], n1)
                       for f1, n1 in zip(f, N)],
                      'time')
    else:
        V = velocity_4D(f, mass, f.attrs['Energy_e0'], N)
    
    V.name = 'V{}'.format(dist.attrs['species'])
    V.attrs['long_name'] = ('Bulk velocity calculated by integrating the '
                            'distribution function.')
    V.attrs['standard_name'] = 'bulk_velocity'
    V.attrs['units'] = 'km/s'
    
    return V


def vspace_entropy(dist, N=None, s=None, **kwargs):
    '''
    Calculate entropy from a time series of 3D velocity space
    distribution function.
    
    .. [1] Liang, H., Cassak, P. A., Servidio, S., Shay, M. A., Drake,
        J. F., Swisdak, M., … Delzanno, G. L. (2019). Decomposition of
        plasma kinetic entropy into position and velocity space and the
        use of kinetic entropy in particle-in-cell simulations. Physics
        of Plasmas, 26(8), 82903. https://doi.org/10.1063/1.5098888
    
    Parameters
    ----------
    dist : `xarray.DataArray`
        A time series of 3D distribution functions
    N : `xarray.DataArray`
        Number density computed from `dist`
    s : `xarray.DataArray`
        Entropy density computed from `dist`
    \*\*kwargs : dict
        Keywords accepted by the `precondition` function.
    
    Returns
    -------
    sv : `xarray.DataArray`
        Velocity space entropy density
    '''
    mass = species_to_mass(dist.attrs['species'])
    if N is None:
        N = density(dist, **kwargs)
    if s is None:
        s = entropy(dist, **kwargs)
    
    f = precondition(dist, **kwargs)
    
    if dist.attrs['mode'] == 'brst':
        sv = xr.concat([vspace_entropy_3D(f1, mass, f.attrs['Energy_e0'], n1, s1)
                        for f1, n1, s1 in zip(f, N, s)],
                       'time')
    else:
        sv = vspace_entropy_4D(f, mass, f.attrs['Energy_e0'], N, s)
    
    sv.name = 'sv{}'.format(dist.attrs['species'])
    sv.attrs['long_name'] = 'Velocity space entropy density'
    sv.attrs['standard_name'] = 'entropy_density'
    sv.attrs['units'] = 'J/K/m^3 ln(...)'
    
    return sv


def density_3D(f, mass, E0):
    '''
    Calculate number density from a single 3D velocity space
    distribution function.
    
    Notes
    -----
    This is needed because the azimuthal bin locations in the
    burst data FPI distribution functions is time dependent. By
    extracting single distributions, the phi, theta, and energy
    variables become time-independent and easier to work with.
    
    Parameters
    ----------
    f : `xarray.DataArray`
        A preconditioned 3D distribution function
    mass : float
        Mass (kg) of the particle species represented in the distribution.
    E0 : float
        Energy used to normalize the energy bins to [0, inf)
    
    Returns
    -------
    N : `xarray.DataArray`
        Number density
    '''
    eV2J = constants.eV
    
    N = f.integrate('phi')
    N = (np.sin(N['theta']) * N).integrate('theta')
    
    # Integrate over Energy
    y = np.sqrt(N['U']) / (1-N['U'])**(5/2) * N
    y[-1] = 0
    N = (1e6 * np.sqrt(2) * (eV2J * E0 / mass)**(3/2)
         * y.integrate('U')
         )
    
    return N


def entropy_3D(f, mass, E0):
    '''
    Calculate entropy from a single 3D velocity space
    distribution function.
    
    Notes
    -----
    This is needed because the azimuthal bin locations in the
    burst data FPI distribution functions is time dependent. By
    extracting single distributions, the phi, theta, and energy
    variables become time-independent and easier to work with.
    
    Calculation of velocity and kinetic entropy can be found in
    Liang, et al, PoP (2019) [1]_
    
    .. [1] Liang, H., Cassak, P. A., Servidio, S., Shay, M. A., Drake,
        J. F., Swisdak, M., … Delzanno, G. L. (2019). Decomposition of
        plasma kinetic entropy into position and velocity space and the
        use of kinetic entropy in particle-in-cell simulations. Physics
        of Plasmas, 26(8), 82903. https://doi.org/10.1063/1.5098888
    
    Parameters
    ----------
    f : `xarray.DataArray`
        A preconditioned 3D distribution function
    mass : float
        Mass (kg) of the particle species represented in the distribution.
    E0 : float
        Energy used to normalize the energy bins to [0, inf)
    
    Returns
    -------
    S : `xarray.DataArray`
        Velocity space entropy
    '''
    kB = constants.k # J/K
    
    # Integrate over phi and theta
    #   - Measurement bins with zero counts result in a
    #     phase space density of 0
    #   - Photo-electron correction can result in negative
    #     phase space density.
    #   - Log of value <= 0 is nan. Avoid be replacing
    #     with 1 so that log(1) = 0
    S = 1e12 * f
    S = S.where(S > 0, 1)
    S = (S * np.log(S)).integrate('phi')
    S = (np.sin(S['theta']) * S).integrate('theta')
    
    # Integrate over Energy
    y = np.sqrt(S['U']) / (1 - S['U'])**(5/2) * S
    y[-1] = 0
    S = (-kB * np.sqrt(2) * (constants.eV * E0 / mass)**(3/2)
         * y.integrate('U')
         )
    
    return S


def epsilon_3D(f, mass, E0, f_max, N):
    '''
    Calculate the epsilon entropy parameter [1]_ from a single 3D velocity space
    distribution function.
    
    .. [1] Greco, A., Valentini, F., Servidio, S., &
        Matthaeus, W. H. (2012). Inhomogeneous kinetic effects related
        to intermittent magnetic discontinuities. Phys. Rev. E,
        86(6), 66405. https://doi.org/10.1103/PhysRevE.86.066405
    
    Notes
    -----
    This is needed because the azimuthal bin locations in the
    burst data FPI distribution functions is time dependent. By
    extracting single distributions, the phi, theta, and energy
    variables become time-independent and easier to work with.
    
    Parameters
    ----------
    f : `xarray.DataArray`
        A preconditioned 3D distribution function
    mass : float
        Mass (kg) of the particle species represented in the distribution.
    E0 : float
        Energy used to normalize the energy bins to [0, inf)
    f_max : `xarray.DataArray`
        An equivalent, preconditioned Maxwellian distribution
    N : `xarray.DataArray`
        Number density computed from `f`.
    
    Returns
    -------
    epsilon : `xarray.DataArray`
        Epsilon entropy parameter
    '''
    eV2J = constants.eV
    
    # Integrate phi and theta
    df = ((f - f_max)**2).integrate('phi')
    df = (np.sin(df['theta']) * df).integrate('theta')
    
    # Integrate energy
    y = np.sqrt(df['U']) / (1-df['U'])**(5/2) * df
    y[-1] = 0
    
    epsilon = (1e3 * 2**(1/4) * eV2J**(3/4) * (E0 / mass)**(3/2) / N
               * y.integrate('U')
               )
    
    return epsilon
    

def pressure_3D(N, T):
    '''
    Calculate the epsilon entropy parameter [1]_ from a single 3D velocity space
    distribution function.
    
    .. [1] Greco, A., Valentini, F., Servidio, S., &
        Matthaeus, W. H. (2012). Inhomogeneous kinetic effects related
        to intermittent magnetic discontinuities. Phys. Rev. E,
        86(6), 66405. https://doi.org/10.1103/PhysRevE.86.066405
    
    Notes
    -----
    This is needed because the azimuthal bin locations in the
    burst data FPI distribution functions is time dependent. By
    extracting single distributions, the phi, theta, and energy
    variables become time-independent and easier to work with.
    
    Parameters
    ----------
    N : `xarray.DataArray`
        Number density.
    T : `xarray.DataArray`
        Temperature tensor`.
    
    Returns
    -------
    P : `xarray.DataArray`
        Pressure tensor
    '''
    kB = constants.k
    eV2K = constants.value('electron volt-kelvin relationship')
    
    P = 1e15 * N * kB * eV2K * T
    
    return P


def temperature_3D(f, mass, E0, N, V):
    '''
    Calculate the temperature tensor from a single 3D velocity space
    distribution function.
    
    Notes
    -----
    This is needed because the azimuthal bin locations in the
    burst data FPI distribution functions is time dependent. By
    extracting single distributions, the phi, theta, and energy
    variables become time-independent and easier to work with.
    
    Parameters
    ----------
    f : `xarray.DataArray`
        A preconditioned 3D distribution function
    mass : float
        Mass (kg) of the particle species represented in the distribution.
    E0 : float
        Energy used to normalize the energy bins to [0, inf)
    N : `xarray.DataArray`
        Number density computed from `f`.
    V : `xarray.DataArray`
        Bulk velocity computed from `f`.
    
    Returns
    -------
    T : `xarray.DataArray`
        Temperature tensor
    '''
    K2eV = constants.value('kelvin-electron volt relationship')
    eV2J = constants.eV
    kB = constants.k # J/k
    
    # Integrate over phi
    Txx = (np.cos(f['phi'])**2 * f).integrate('phi')
    Tyy = (np.sin(f['phi'])**2 * f).integrate('phi')
    Tzz = f.integrate('phi')
    Txy = (np.cos(f['phi']) * np.sin(f['phi']) * f).integrate('phi')
    Txz = (np.cos(f['phi']) * f).integrate('phi')
    Tyz = (np.sin(f['phi']) * f).integrate('phi')
    
    # Integrate over theta
    Txx = (np.sin(Txx['theta'])**3 * Txx).integrate('theta')
    Tyy = (np.sin(Tyy['theta'])**3 * Tyy).integrate('theta')
    Tzz = (np.cos(Tzz['theta'])**2 * np.sin(Tzz['theta']) * Tzz).integrate('theta')
    Txy = (np.sin(Txy['theta'])**3 * Txy).integrate('theta')
    Txz = (np.cos(Txz['theta']) * np.sin(Txz['theta'])**2 * Txz).integrate('theta')
    Tyz = (np.cos(Tyz['theta']) * np.sin(Tyz['theta'])**2 * Tyz).integrate('theta')
    
    # Combine into tensor
    T = xr.concat([xr.concat([Txx, Txy, Txz], dim='t_index_dim1'),
                   xr.concat([Txy, Tyy, Tyz], dim='t_index_dim1'),
                   xr.concat([Txz, Tyz, Tzz], dim='t_index_dim1'),
                   ], dim='t_index_dim2'
                  )
    T = T.assign_coords(t_index_dim1=['x', 'y', 'z'],
                        t_index_dim2=['x', 'y', 'z'])
    
    # Integrate over energy
    T = T['U']**(3/2) / (1-T['U'])**(7/2) * T
    T[-1,:,:] = 0
    
    coeff = 1e6 * (2/mass)**(3/2) / (N * kB / K2eV) * (E0*eV2J)**(5/2)
    Vij = xr.concat([xr.concat([V[0]*V[0],
                                V[0]*V[1],
                                V[0]*V[2]], dim='t_index_dim1'),
                     xr.concat([V[1]*V[0],
                                V[1]*V[1],
                                V[1]*V[2]], dim='t_index_dim1'),
                     xr.concat([V[2]*V[0],
                                V[2]*V[1],
                                V[2]*V[2]], dim='t_index_dim1')
                     ], dim='t_index_dim2'
                    )
    Vij = Vij.drop('velocity_index')
    
    T = (1e6 * (2/mass)**(3/2) / (N * kB / K2eV) * (E0*eV2J)**(5/2)
         * T.integrate('U')
         - (1e6 * mass / kB * K2eV * Vij)
         )
    
    return T


def velocity_3D(f, mass, E0, N):
    '''
    Calculate the bulk velocity from a single 3D velocity space
    distribution function.
    
    Notes
    -----
    This is needed because the azimuthal bin locations in the
    burst data FPI distribution functions is time dependent. By
    extracting single distributions, the phi, theta, and energy
    variables become time-independent and easier to work with.
    
    Parameters
    ----------
    f : `xarray.DataArray`
        A preconditioned 3D distribution function
    mass : float
        Mass (kg) of the particle species represented in the distribution.
    E0 : float
        Energy used to normalize the energy bins to [0, inf)
    N : `xarray.DataArray`
        Number density computed from `f`.
    
    Returns
    -------
    V : `xarray.DataArray`
        Bulk velocity
    '''
    eV2J = constants.eV
    
    # Integrate over phi
    vx = (np.cos(f['phi']) * f).integrate('phi')
    vy = (np.sin(f['phi']) * f).integrate('phi')
    vz = f.integrate('phi')
    
    # Integrate over theta
    vx = (np.sin(vx['theta'])**2 * vx).integrate('theta')
    vy = (np.sin(vy['theta'])**2 * vy).integrate('theta')
    vz = (np.cos(vz['theta']) * np.sin(vz['theta']) * vz).integrate('theta')
    V = xr.concat([vx, vy, vz], dim='velocity_index')
    V = V.assign_coords({'velocity_index': ['Vx', 'Vy', 'Vz']})
    
    # Integrate over Energy
    E0 = 100
    y = V['U'] / (1 - V['U'])**3 * V
    y[-1,:] = 0
    
    V = (-1e3 * 2 * (eV2J * E0 / mass)**2 / N
         * y.integrate('U')
         )
    return V


def vspace_entropy_3D(f, mass, E0, N, s):
    '''
    Calculate velocity space entropy from a single 3D velocity space
    distribution function. Because the 
    
    Notes
    -----
    This is needed because the azimuthal bin locations in the
    burst data FPI distribution functions is time dependent. By
    extracting single distributions, the phi, theta, and energy
    variables become time-independent and easier to work with.
    
    Calculation of velocity and kinetic entropy can be found in
    Liang, et al, PoP (2019) `[1]`_. The implementation here takes into
    account the fact that the FPI energy bins are not equally spaced.
    
    .. [1]: Liang, H., Cassak, P. A., Servidio, S., Shay, M. A., Drake,
        J. F., Swisdak, M., … Delzanno, G. L. (2019). Decomposition of
        plasma kinetic entropy into position and velocity space and the
        use of kinetic entropy in particle-in-cell simulations. Physics
        of Plasmas, 26(8), 82903. https://doi.org/10.1063/1.5098888
    
    Parameters
    ----------
    f : `xarray.DataArray`
        A preconditioned 3D distribution function
    mass : float
        Mass (kg) of the particle species represented in the distribution.
    E0 : float
        Energy (eV) used to normalize the energy bins to [0, inf)
    
    Returns
    -------
    S : `xarray.DataArray`
        Velocity space entropy
    '''
    kB = constants.k # J/K
    eV2J = constants.eV
    
    # Assume that the azimuth and polar angle bins are equal size
    dtheta = f['theta'].diff(dim='theta').mean().item()
    dphi = f['phi'].diff(dim='phi').mean().item()
    
    # Calculate the factors that associated with the normalized
    # volume element
    #   - U ranges from [0, inf] and np.inf/np.inf = nan
    #   - Set the last element of y along U manually to 0
    #   - log(0) = -inf; Zeros come from theta and y. Reset to zero
    #   - Photo-electron correction can result in negative phase space
    #     density. log(-1) = nan
    coeff = np.sqrt(2) * (eV2J*E0/mass)**(3/2) # m^3/s^3
    y = (np.sqrt(f['U']) / (1 - f['U'])**(5/2)) * (np.sin(f['theta']))
    y[-1,:] = 0
    with np.errstate(invalid='ignore', divide='ignore'):
        lnydy = np.log(y * np.sin(f['theta']) * dtheta * dphi)
    lnydy = lnydy.where(np.isfinite(lnydy), 0)
    
    # Terms in that make up the velocity space entropy density
    sv1 = 1e6 * N * np.log(1e6 * N) # 1/m^3 * ln(1/m^3)
    sv2 = np.log(coeff) * N # 1/m^3 ln(m^3/s^3)
    sv3 = s # J/K/m^3 ln(s^3/m^6) -- Already multiplied by -kB
    
    sv4 = (lnydy * f).integrate('phi')
    sv4 = (np.sin(f['theta']) * sv4).integrate('theta')
    sv4 = coeff * sv4.integrate('U') # 1/m^3
    
    # Velocity space entropy density
    sv = kB * (sv1 - sv2 - sv4) + sv3 # J/K/m^3 * ln(...)
    
    return sv


def density_4D(f, mass, E0):
    '''
    Calculate number density from a time series of 3D velocity space
    distribution functions.
    
    Notes
    -----
    The FPI fast survey velocity distribution functions are time-independent
    (1D) in azimuth and polar angles but time-dependent (2D) in energy. The
    `xarray.DataArray.integrate` function works only with 1D data (phi and
    theta). For energy, we can use `numpy.trapz`.
    
    Parameters
    ----------
    f : `xarray.DataArray`
        A preconditioned 3D time-dependent velocity distribution function
    mass : float
        Mass (kg) of the particle species represented in the distribution.
    E0 : float
        Energy used to normalize the energy bins to [0, inf)
    
    Returns
    -------
    N : `xarray.DataArray`
        Number density
    '''
    eV2J = constants.eV
    
    N = f.integrate('phi')
    N = (np.sin(N['theta'])*N).integrate('theta')
    
    # Integrate over Energy
    #   - U ranges from [0, inf] and np.inf/np.inf = nan
    #   - Set the last element of the energy dimension of y to 0
    E0 = f.attrs['Energy_e0']
    y = np.sqrt(f['U']) / (1-f['U'])**(5/2) * N
    y[:,-1] = 0
    N = (1e6 * np.sqrt(2) * (eV2J * E0 / mass)**(3/2)
         * np.trapz(y, y['U'], axis=y.get_axis_num('energy_index'))
         )
    N = xr.DataArray(N, dims='time', coords={'time': f['time']})
    
    return N


def entropy_4D(f, mass, E0):
    '''
    Calculate entropy [1]_ from a time series of 3D velocity space
    distribution functions.
    
    .. [1] Liang, H., Cassak, P. A., Servidio, S., Shay, M. A., Drake,
        J. F., Swisdak, M., … Delzanno, G. L. (2019). Decomposition of
        plasma kinetic entropy into position and velocity space and the
        use of kinetic entropy in particle-in-cell simulations. Physics
        of Plasmas, 26(8), 82903. https://doi.org/10.1063/1.5098888
    
    Notes
    -----
    The FPI fast survey velocity distribution functions are time-independent
    (1D) in azimuth and polar angles but time-dependent (2D) in energy. The
    `xarray.DataArray.integrate` function works only with 1D data (phi and
    theta). For energy, we can use `numpy.trapz`.
    
    Parameters
    ----------
    f : `xarray.DataArray`
        A preconditioned 3D time-dependent velocity distribution function
    mass : float
        Mass (kg) of the particle species represented in the distribution.
    E0 : float
        Energy used to normalize the energy bins to [0, inf)
    
    Returns
    -------
    S : `xarray.DataArray`
        Velocity space entropy
    '''
    kB = constants.k # J/K
    
    # Integrate over phi and theta
    #   - Measurement bins with zero counts result in a
    #     phase space density of 0
    #   - Photo-electron correction can result in negative
    #     phase space density.
    #   - Log of value <= 0 is nan. Avoid be replacing
    #     with 1 so that log(1) = 0
    S = 1e12 * f
    S = S.where(S > 0, 1)
    S = (S * np.log(S)).integrate('phi')
    S = (np.sin(S['theta']) * S).integrate('theta')
    
    # Integrate over Energy
    y = np.sqrt(S['U']) / (1 - S['U'])**(5/2) * S
    y[:,-1] = 0
    S = (-kB * np.sqrt(2) * (constants.eV * E0 / mass)**(3/2)
         * np.trapz(y, y['U'], axis=y.get_axis_num('energy_index'))
         )
    
    S = xr.DataArray(S, dims='time', coords={'time': f['time']})
    
    return S


def epsilon_4D(f, mass, E0, f_max, N):
    '''
    Calculate the epsilon entropy parameter [1]_ from a time series of 3D
    velocity space distribution functions.
    
    .. [1] Greco, A., Valentini, F., Servidio, S., &
        Matthaeus, W. H. (2012). Inhomogeneous kinetic effects related
        to intermittent magnetic discontinuities. Phys. Rev. E,
        86(6), 66405. https://doi.org/10.1103/PhysRevE.86.066405
    
    Notes
    -----
    The FPI fast survey velocity distribution functions are time-independent
    (1D) in azimuth and polar angles but time-dependent (2D) in energy. The
    `xarray.DataArray.integrate` function works only with 1D data (phi and
    theta). For energy, we can use `numpy.trapz`.
    
    Parameters
    ----------
    f : `xarray.DataArray`
        A preconditioned 3D time-dependent velocity distribution function
    mass : float
        Mass (kg) of the particle species represented in the distribution.
    E0 : float
        Energy used to normalize the energy bins to [0, inf)
    f_max : `xarray.DataArray`
        An equivalent, preconditioned Maxwellian distribution
    N : `xarray.DataArray`
        Number density computed from `f`.
    
    Returns
    -------
    epsilon : `xarray.DataArray`
        Entropy parameter
    '''
    eV2J = constants.eV
    
    df = ((f - f_max)**2).integrate('phi')
    df = (np.sin(df['theta']) * df).integrate('theta')
    
    # Integrate over Energy
    y = np.sqrt(df['U']) / (1-df['U'])**(5/2) * df
    y[:,-1] = 0
    
    epsilon = (1e3 * 2**(1/4) * eV2J**(3/4) * (E0 / mass)**(3/2) / N
               * np.trapz(y, y['U'], axis=y.get_axis_num('energy_index'))
               )
    
    return epsilon


def pressure_4D(N, T):
    '''
    Calculate the pressure tensor from a time series of 3D velocity space
    distribution functions.
    
    Notes
    -----
    The FPI fast survey velocity distribution functions are time-independent
    (1D) in azimuth and polar angles but time-dependent (2D) in energy. The
    `xarray.DataArray.integrate` function works only with 1D data (phi and
    theta). For energy, we can use `numpy.trapz`.
    
    Parameters
    ----------
    N : `xarray.DataArray`
        Number density 
    T : `xarray.DataArray`
        Temperature tensor
    
    Returns
    -------
    P : `xarray.DataArray`
        Pressure tensor
    '''
    kB = constants.k
    eV2K = constants.value('electron volt-kelvin relationship')
    
    P = 1e15 * N * kB * eV2K * T
    
    return P


def temperature_4D(f, mass, E0, N, V):
    '''
    Calculate the temperature tensor from a time series of 3D velocity space
    distribution functions.
    
    Notes
    -----
    The FPI fast survey velocity distribution functions are time-independent
    (1D) in azimuth and polar angles but time-dependent (2D) in energy. The
    `xarray.DataArray.integrate` function works only with 1D data (phi and
    theta). For energy, we can use `numpy.trapz`.
    
    Parameters
    ----------
    f : `xarray.DataArray`
        A preconditioned 3D time-dependent velocity distribution function
    mass : float
        Mass (kg) of the particle species represented in the distribution.
    E0 : float
        Energy used to normalize the energy bins to [0, inf)
    N : `xarray.DataArray`
        Number density computed from `f`.
    V : `xarray.DataArray`
        Bulk velocity computed from `f`.
    
    Returns
    -------
    T : `xarray.DataArray`
        Temperature tensor
    '''
    K2eV = constants.value('kelvin-electron volt relationship')
    eV2J = constants.eV
    kB = constants.k # J/k
    
    # Integrate over phi
    Txx = (np.cos(f['phi'])**2 * f).integrate('phi')
    Tyy = (np.sin(f['phi'])**2 * f).integrate('phi')
    Tzz = f.integrate('phi')
    Txy = (np.cos(f['phi']) * np.sin(f['phi']) * f).integrate('phi')
    Txz = (np.cos(f['phi']) * f).integrate('phi')
    Tyz = (np.sin(f['phi']) * f).integrate('phi')
    
    # Integrate over theta
    Txx = (np.sin(Txx['theta'])**3 * Txx).integrate('theta')
    Tyy = (np.sin(Tyy['theta'])**3 * Tyy).integrate('theta')
    Tzz = (np.cos(Tzz['theta'])**2 * np.sin(Tzz['theta']) * Tzz).integrate('theta')
    Txy = (np.sin(Txy['theta'])**3 * Txy).integrate('theta')
    Txz = (np.cos(Txz['theta']) * np.sin(Txz['theta'])**2 * Txz).integrate('theta')
    Tyz = (np.cos(Tyz['theta']) * np.sin(Tyz['theta'])**2 * Tyz).integrate('theta')
    T = xr.concat([xr.concat([Txx, Txy, Txz], dim='t_index_dim1'),
                   xr.concat([Txy, Tyy, Tyz], dim='t_index_dim1'),
                   xr.concat([Txz, Tyz, Tzz], dim='t_index_dim1'),
                   ], dim='t_index_dim2'
                  )
    
    # Integrate over energy
    E0 = 100
    T = T['U']**(3/2) / (1-T['U'])**(7/2) * T
    T[:,-1,:,:] = 0
    
    coeff = 1e6 * (2/mass)**(3/2) / (N * kB / K2eV) * (E0*eV2J)**(5/2)
    
    Vij = xr.concat([xr.concat([V[:,0]*V[:,0],
                                V[:,0]*V[:,1],
                                V[:,0]*V[:,2]], dim='t_index_dim1'),
                     xr.concat([V[:,1]*V[:,0],
                                V[:,1]*V[:,1],
                                V[:,1]*V[:,2]], dim='t_index_dim1'),
                     xr.concat([V[:,2]*V[:,0],
                                V[:,2]*V[:,1],
                                V[:,2]*V[:,2]], dim='t_index_dim1')
                     ], dim='t_index_dim2'
                    )
    Vij = Vij.drop('velocity_index')
    
    T = ((1e6 * (2/mass)**(3/2) / (N * kB / K2eV) * (E0*eV2J)**(5/2)
          ).expand_dims(['t_index_dim1', 't_index_dim2'], axis=[1,2])
         * np.trapz(T, T['U'].expand_dims(dim=['t_index_dim1', 't_index_dim2'],
                                          axis=[2,3]),
                    axis=T.get_axis_num('energy_index'))
         - (1e6 * mass / kB * K2eV * Vij)
         )
    
    T = T.assign_coords(t_index_dim1=['x', 'y', 'z'],
                        t_index_dim2=['x', 'y', 'z'])
    
    return T


def velocity_4D(f, mass, E0, N):
    '''
    Calculate the bulk velocity from a time series of 3D velocity space
    distribution functions.
    
    Notes
    -----
    The FPI fast survey velocity distribution functions are time-independent
    (1D) in azimuth and polar angles but time-dependent (2D) in energy. The
    `xarray.DataArray.integrate` function works only with 1D data (phi and
    theta). For energy, we can use `numpy.trapz`.
    
    Parameters
    ----------
    f : `xarray.DataArray`
        A preconditioned 3D time-dependent velocity distribution function
    mass : float
        Mass (kg) of the particle species represented in the distribution.
    E0 : float
        Energy used to normalize the energy bins to [0, inf)
    N : `xarray.DataArray`
        Number density computed from `f`.
    
    Returns
    -------
    V : `xarray.DataArray`
        Bulk velocity
    '''
    eV2J = constants.eV
    kB = constants.k # J/K
    
    # Integrate over phi
    vx = (np.cos(f['phi']) * f).integrate('phi')
    vy = (np.sin(f['phi']) * f).integrate('phi')
    vz = f.integrate('phi')
    
    # Integrate over theta
    vx = (np.sin(vx['theta'])**2 * vx).integrate('theta')
    vy = (np.sin(vy['theta'])**2 * vy).integrate('theta')
    vz = (np.cos(vz['theta']) * np.sin(vz['theta']) * vz).integrate('theta')
    V = xr.concat([vx, vy, vz], dim='velocity_index')
    
    # Integrate over Energy
    E0 = 100
    y = V['U'] / (1 - V['U'])**3 * V
    y[:,-1] = 0
    
    V = (-1e3 * 2 * (eV2J * E0 / mass)**2 
         / N.expand_dims(dim='velocity_index', axis=1)
         * np.trapz(y, y['U'].expand_dims(dim='velocity_index', axis=2),
                    axis=y.get_axis_num('energy_index'))
         )
    V = V.assign_coords(velocity_index=['Vx', 'Vy', 'Vz'])
    
    return V


def vspace_entropy_4D(f, mass, E0, N, s):
    '''
    Calculate velocity space entropy density from a time series of 3D
    velocity space distribution functions.
    
    Notes
    -----
    The FPI fast survey velocity distribution functions are time-independent
    (1D) in azimuth and polar angles but time-dependent (2D) in energy. The
    `xarray.DataArray.integrate` function works only with 1D data (phi and
    theta). For energy, we can use `numpy.trapz`.
    
    Parameters
    ----------
    f : `xarray.DataArray`
        A preconditioned 3D time-dependent velocity distribution function
    mass : float
        Mass (kg) of the particle species represented in the distribution.
    E0 : float
        Energy used to normalize the energy bins to [0, inf)
    
    Returns
    -------
    sv : `xarray.DataArray`
        Velocity space entropy density
    '''
    kB = constants.k # J/K
    eV2J = constants.eV
    
    # Assume that the azimuth and polar angle bins are equal size
    dtheta = f['theta'].diff(dim='theta').mean().item()
    dphi = f['phi'].diff(dim='phi').mean().item()
    
    # Calculate the factors that associated with the normalized
    # volume element
    #   - U ranges from [0, inf] and np.inf/np.inf = nan
    #   - Set the last element of y along U manually to 0
    #   - ln(0) = -inf; Zeros come from theta and y. Reset to zero
    #   - Photo-electron correction can result in negative phase space
    #     density. log(-1) = nan
    coeff = np.sqrt(2) * (eV2J*E0/mass)**(3/2) # m^3/s^3
    y = (np.sqrt(f['U']) / (1 - f['U'])**(5/2)) * (np.sin(f['theta']))
    y[:,-1,:] = 0
    with np.errstate(divide='ignore'):
        lnydy = np.log(y * np.sin(f['theta']) * dtheta * dphi)
    lnydy = lnydy.where(np.isfinite(lnydy), 0)
    
    # Terms in that make up the velocity space entropy density
    sv1 = 1e6 * N * np.log(1e6 * N) # 1/m^3 * ln(1/m^3)
    sv2 = np.log(coeff) * N # 1/m^3 ln(m^3/s^3)
    sv3 = s # J/K/m^3 ln(s^3/m^6) -- Already multiplied by -kB
    
    sv4 = (y * lnydy * f).integrate('phi')
    sv4 = (np.sin(f['theta']) * sv4).integrate('theta')
    sv4 = coeff * np.trapz(sv4, axis=sv4.get_axis_num('energy_index')) # 1/m^3
    
    # Velocity space entropy density
    sv = kB * (sv1 - sv2 - sv4) + sv3 # J/K/m^3 * ln(...)
    
    return sv
