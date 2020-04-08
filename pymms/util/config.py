"""
heliopy configuration utility
"""
import configparser
import os
import pymms
from pathlib import Path


def get_config_file():
    """
    Find the configuration file. Locations are{}

    1. <installation folder>/template_config.ini
    2. <installation folder>/config.ini

    Returns
    -------
    config_file : str
        Filepath of the ``pymms`` configuration file
    """
    # Get user configuration location
    module_dir = Path(pymms.__file__)
    config_file_1 = module_dir.parent / 'config.ini'
    config_file_1 = config_file_1.resolve()

    config_file_2 = module_dir.parent / 'config_template.ini'
    config_file_2 = config_file_2.resolve()
    
    for f in [config_file_1, config_file_2]:
        if f.is_file():
            return str(f)


def load_config():
    """
    Read the configuration file. Locations are:

    1. <installation folder>/template_config.ini
    2. <installation folder>/config.ini

    Returns
    -------
    config : dict
        Dictionary containing configuration options.
    """    
    config_location = get_config_file()
    config = configparser.ConfigParser()
    config.read(config_location)
    config_dict = {}
    
    # Set data download directory
    for dirname in ('data_root', 'dropbox_root', 'mirror_root'):
        root = os.path.expanduser(config['DIRS'][dirname])
        if root == 'None':
            config_dict[dirname] = None
            continue
        
        # Make any default directories system-independent
        if os.name == 'nt':
            root = root.replace('/', '\\')
        config_dict[dirname] = root
        
        # Create data download if not created
        if not os.path.isdir(root):
            print('Creating root data directory {}'.format(root))
            os.makedirs(root)

    # login credentials
    for cred in ('username', 'password'):
        config_dict[cred] = config['SDCLOGIN'][cred]
        if config_dict[cred] == 'None':
            config_dict[cred] = None

    return config_dict
