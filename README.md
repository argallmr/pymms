## Installation

For development purposes, install the package using
```bash
$ python3 setup.py develop --user
```
This installation will reflect any changes made in the pymms development directory without the need to reinstall the package every single time.

Includes a command-line tool to download data needed for the mp-dl-unh pipeline:

```bash
$ mp-dl-unh-data -h
usage: mp-dl-unh-data [-h] [-is] [-ip] [-v] sc level start end output

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

If pymms is installed with the ``--user`` flag and pymms is used from a unix system, you must call:

```bash
$ export PATH=~/.local/bin$PATH
$ source ~/.bash_profile
```

before calling mp-dl-unh-data.