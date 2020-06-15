[![DOI](https://zenodo.org/badge/124706809.svg)](https://zenodo.org/badge/latestdoi/124706809)

# PyMMS

## Installation

For development purposes, install the package using
```bash
$ python3 setup.py develop --user
```
This installation will reflect any changes made in the pymms development directory without the need to reinstall the package every single time.

## Scripts

### gls

The `pymms.gls` package includes two user-runnable console commands: `gls-mp` and `gls-mp-data`. Calling `gls-mp` runs the `mp-dl-unh` model to generate predicted SITL selections over a date range.

```
$ gls-mp -h
usage: gls-mp [-h] [-g] [-t] [-c C] [-temp] start end sc

positional arguments:
  start            Start date of data interval, formatted as either '%Y-%m-%d'
                   or '%Y-%m-%dT%H:%M:%S'. Optionally an integer, interpreted
                   as an orbit number.
  end              Start date of data interval, formatted as either '%Y-%m-%d'
                   or '%Y-%m-%dT%H:%M:%S'. Optionally an integer, interpreted
                   as an orbit number.
  sc               Spacecraft IDs ('mms1', 'mms2', 'mms3', 'mms4')

optional arguments:
  -h, --help       show this help message and exit
  -g, -gpu         Enables use of GPU-accelerated model for faster
                   predictions. Requires CUDA installed.
  -t, -test        Runs a test routine on the model.
  -c C, -chunks C  Break up the processing of the date interval in C chunks.
  -temp            If running the job in chunks, deletes the contents of the
                   MMS root data folder after each chunk.
```

Calling `gls-mp-data` generates a CSV file containing data formatted and preprocessed for `gls-mp`. This can be used when training your own version of mp-dl-unh.

```
$ gls-mp-data -h
usage: gls-mp-data [-h] [-is] [-ip] [-v] sc level start end output

positional arguments:
  sc                    Spacecraft IDs ('mms1', 'mms2', 'mms3', 'mms4')
  level                 Data quality level ('l1a', 'l1b', 'sitl', 'l2pre',
                        'l2', 'l3')
  start                 Start date of data interval, formatted as either
                        '%Y-%m-%d' or '%Y-%m-%dT%H:%M:%S'. Optionally an
                        integer, interpreted as an orbit number.
  end                   Start date of data interval, formatted as either
                        '%Y-%m-%d' or '%Y-%m-%dT%H:%M:%S'. Optionally an
                        integer, interpreted as an orbit number.
  output                Path the output CSV file, including the CSV file's
                        name.

optional arguments:
  -h, --help            show this help message and exit
  -is, --include-selections
                        Includes SITL selections in the output data.
  -ip, --include-partials
                        Includes partial magnetopause crossings in SITL
                        selections.
  -v, --verbose         If true, prints out optional information about
                        downloaded variables.
```

If PyMMS is installed with the ``--user`` flag and PyMMS is used from a unix system, you must call:
```bash
$ export PATH=~/.local/bin$PATH
$ source ~/.bash_profile
```
before calling `gls-mp` or `gls-mp-data`.

## Citation

If you make use of this software to analyze MMS use or data, please consider citing the software. Follow the Zenodo DOI at the top for a citation to the most recent release, or head to [Zenodo](https://doi.org/10.5281/zenodo.3765993) to see the citations/DOIs of other releases.
