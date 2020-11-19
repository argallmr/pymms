Changelog
=========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_, and this project adheres to `Semantic Versioning`_.

.. _Keep a Changelog https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning https://semver.org/spec/v2.0.0.html

v0.4.1
------------
Added
^^^^^
* Read FPI omnidirectional energy spectrograms

Fixed
^^^^^
* The data directory now has __init__.py to make it a subpackage

v0.4.0
------
Added
^^^^^
* pymms.data for reading and handling data

Changed
^^^^^^^
* Added sql subpackage to extras

Fixed
^^^^^
* File search no longer returns single remote file when all are local

v0.3.1 (2020-06-12)
-------------------
Changed
^^^^^^^
* Create a git tag without "v" so that Zenodo can process release
* Incorporate Zenodo DOI into README

v0.3.0 (2020-06-12)
-------------------
Added
^^^^^
* Time conversions between datetime, TAI, and TT2000
* Support for GLS models

Fixed
^^^^^
* Handle cases when no file names are returned by the SDC

v0.2.2 (2020-05-22)
-------------------
Added
^^^^^
* Downlink status is now an attribute of SITL burst segments

Changed
^^^^^^^
* SROI not available before orbit 239- use science_roi
* sitl_window not available after orbit 1097

Backwards Incompatible
^^^^^^^^^^^^^^^^^^^^^^
* Ignore case when filtering burst segments

Fixed
^^^^^
* Typo preventing log-in if credentials not provided initially

v0.2.1 (2020-04-24)
-------------------
Added
^^^^^
* `util.tai.py` for converting to/from TAI times.
* This CHANGES file
* Additional testing
* config.py looks in ``~/.pymmsrc/pymmsrc` for configuration settings to make them easier to change when not in development mode.

Fixed
^^^^^
* Convert version numbers to ints in `mrmms_sdc_api.filter_version` to prevent character-by-character comparison (e.g. '53' vs '110').
* Typos in `mrmms_sdc_api.parse_file_name`
* Checked time strings for length incorrectly in `mrmms_sdc_api.parse_time`
* Setting the `files` attribute automatically set `site='public` even for files not on the public site in `mrmms_sdc_api.MrMMS_SDC_API.__getattr__`

v0.2.0 (2020-04-09)
--------------------
Added
^^^^^
* Version number to the pymms module
* Template configuration file

Backward Incompatible
^^^^^^^^^^^^^^^^^^^^^
* Reorganized package content to isolate subpackages and facilitate the use of versioning and configuration files in `setup.py`


v0.1.0 (2020-03-18)
--------------------
* Initial release