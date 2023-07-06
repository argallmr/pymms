from setuptools import setup
import os
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

# Read the version file
# https://martin-thoma.com/python-package-versions/
# https://stackoverflow.com/questions/436198/what-is-an-alternative-to-execfile-in-python-3/437857#437857
def execfile(filename, globals=None, locals=None):
    if globals is None:
        globals = sys._getframe(1).f_globals
    if locals is None:
        locals = sys._getframe(1).f_locals
    with open(filename, "rb") as fh:
        exec(compile(fh.read(), filename, 'exec'), globals, locals)

# execute the file
execfile('pymms/_version.py')

setup(
      name = "nasa-pymms",
      version = __version__,
      author = "Matthew R. Argall",
      author_email = "argallmr@gmail.com",
      description = "Access data from the MMS mission via its API.",
      long_description = long_description,
      long_description_content_type = "text/markdown",
      url = "https://github.com/argallmr/pymms",
      license = "MIT",
      classifiers = ["Programming Language :: Python :: 3",
                     "Operating System :: OS Independent",
                     "License :: OSI Approved :: MIT License",
                     "Development Status :: 4 - Beta",
                     "Topic :: Scientific/Engineering :: Physics",
                     "Topic :: Scientific/Engineering :: Astronomy",
                     "Intended Audience :: Science/Research",
                     "Natural Language :: English"],
      keywords = "physics space-physics MMS",
      packages = ["pymms",
                  "pymms.sdc",
                  "pymms.util",
                  "pymms.data",
                  "pymms.sql",
                  "pymms.gls"],
      package_data = {'pymms': ['config_template.ini']},
      include_package_data = True,
      install_requires = ["numpy>=1.24",
                          "requests>=2.22.0",
                          "scipy>=1.4.1", 
                          "tqdm>=4.36.1",
                          "cdflib>=1.0.0",
                          "matplotlib>=3.4"
                          ],
      extras_require = {'data': ["cdflib >= 0.3.18",
                                 "xarray >= 0.16.0"],
                        'gls': ["tensorflow >=1.13.1, <=1.15",
                                "keras >=2.2.4, <=2.3.1"],
                        'sql': [""]},
      python_requires = '>=3.6',
      entry_points={
          'console_scripts': [
              'gls-mp-data = pymms.gls.mp_dl_unh.data:download_from_cmd',
              'gls-mp = pymms.gls.mp_dl_unh.data:main'
          ]
      }
      )
