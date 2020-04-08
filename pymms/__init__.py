"""
pymms

Open source Python tools for accessing data from NASA's
Magnetospheric Multiscale (MMS) mission.
"""
from pymms._version import __version__
from pymms.config import load_config
config = load_config()