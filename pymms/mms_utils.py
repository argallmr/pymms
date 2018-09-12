#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:34:47 2018

@author: argall
"""
import os
import datetime as dt
import pdb # https://pythonconquerstheuniverse.wordpress.com/2009/09/10/debugging-in-python/

def construct_filename(sc, instr=None, mode=None, level=None, tstart='*', version='*',
                       optdesc=None):
    """
    Construct a file name compliant with MMS file name format guidelines.
    
    MMS file names follow the convention
        sc_instr_mode_level[_optdesc]_tstart_vX.Y.Z.cdf
    
    Arguments:
        sc (str,list,tuple):   Spacecraft ID(s)
        instr   (str,list):    Instrument ID(s)
        mode    (str,list):    Data rate mode(s). Options include slow, fast, srvy, brst
        level   (str,list):    Data level(s). Options include l1a, l1b, l2pre, l2, l3
        tstart  (str,list):    Start time of data file. In general, the format is
                               YYYYMMDDhhmmss for "brst" mode and YYYYMMDD for "srvy"
                               mode (though there are exceptions). If not given, the
                               default is "*".
        version (str,list):    File version, formatted as "X.Y.Z", where X, Y, and Z
                               are integer version numbers.
        optdesc (str,list):    Optional file name descriptor. If multiple parts,
                               they should be separated by hyphens ("-"), not under-
                               scores ("_").
    
    Returns:
        fnames  (str,list);    File names constructed from inputs.
    """
    
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
    
    # Accept tuples, as those returned by MrMMS_Construct_Filename
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
        fnames = ['_'.join((s,i,m,l,t,'v'+v+'.cdf')) for s in sc_ids
                                                     for i in instr
                                                     for m in mode
                                                     for l in level
                                                     for t in tstart
                                                     for v in version]
    else:
        fnames = ['_'.join((s,i,m,l,o,t,'v'+v+'.cdf')) for s in sc_ids
                                                       for i in instr
                                                       for m in mode
                                                       for l in level
                                                       for o in optdesc
                                                       for t in tstart
                                                       for v in version]
    return fnames



def construct_path(sc, instr=None, mode=None, level=None, tstart='*',
                   optdesc=None, root='', files=False):
    """
    Construct a directory structure compliant with MMS path guidelines.
    
    MMS paths follow the convention
        brst: sc/instr/mode/level[/optdesc]/<year>/<month>/<day>
        srvy: sc/instr/mode/level[/optdesc]/<year>/<month>
    
    Arguments:
        sc (str,list,tuple):   Spacecraft ID(s)
        instr   (str,list):    Instrument ID(s)
        mode    (str,list):    Data rate mode(s). Options include slow, fast, srvy, brst
        level   (str,list):    Data level(s). Options include l1a, l1b, l2pre, l2, l3
        tstart  (str,list):    Start time of data file, formatted as a date: '%Y%m%d'.
                               If not given, all dates from 20150901 to today's date are
                               used.
        optdesc (str,list):    Optional file name descriptor. If multiple parts,
                               they should be separated by hyphens ("-"), not under-
                               scores ("_").
        root    (str):         Root directory at which the directory structure begins.
        files   (bool):        If True, file names will be generated and appended to the
                               paths. The file tstart will be "YYYYMMDD*" (i.e. the date
                               with an asterisk) and the version number will be "*".
    
    Returns:
        fnames  (str,list);    File names constructed from inputs.
    """
    
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
    
    # Accept tuples, as those returned by MrMMS_Construct_Filename
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
            paths = [os.path.join(root,s,i,m,l,t[0:4],t[4:6],t[6:8],'_'.join((s,i,m,l,t+'*','v*.cdf'))) if m == 'brst' else
                     os.path.join(root,s,i,m,l,t[0:4],t[4:6],'_'.join((s,i,m,l,t+'*','v*.cdf')))
                     for s in sc_ids
                     for i in instr
                     for m in mode
                     for l in level
                     for t in tstart]
        else:
            paths = [os.path.join(root,s,i,m,l,o,t[0:4],t[4:6],t[6:8],'_'.join((s,i,m,l,o,t+'*','v*.cdf'))) if m == 'brst' else
                     os.path.join(root,s,i,m,l,o,t[0:4],t[4:6],'_'.join((s,i,m,l,o,t+'*','v*.cdf')))
                     for s in sc_ids
                     for i in instr
                     for m in mode
                     for l in level
                     for o in optdesc
                     for t in tstart]
    
    # Paths
    else:
        if optdesc is None:
            paths = [os.path.join(root,s,i,m,l,t[0:4],t[4:6],t[6:8]) if m == 'brst' else
                     os.path.join(root,s,i,m,l,t[0:4],t[4:6]) for s in sc_ids
                                                              for i in instr
                                                              for m in mode
                                                              for l in level
                                                              for t in tstart]
        else:
            paths = [os.path.join(root,s,i,m,l,o,t[0:4],t[4:6],t[6:8]) if m == 'brst' else
                     os.path.join(root,s,i,m,l,o,t[0:4],t[4:6]) for s in sc_ids
                                                                for i in instr
                                                                for m in mode
                                                                for l in level
                                                                for o in optdesc
                                                                for t in tstart]
    
    
    return paths


def filename2path(fnames, root=''):
    """
    Convert an MMS file name to an MMS path.
    
    MMS paths take the form
        
        sc/instr/mode/level[/optdesc]/YYYY/MM[/DD/]
        
    where the optional descriptor [/optdesc] is included if it is also in the
    file name and day directory [/DD] is included if mode='brst'.
    
    Arguments:
        fnames (str,list):    File names to be turned into paths.
        root   (str):     Absolute directory
    
    Returns:
        paths (list):     Path to the data file.
    """
    
    paths = []
    
    # Convert input file names to an array
    if type(fnames) is str:
        fnames = [fnames]
    
    parts = parse_filename(fnames)
    
    for idx, part in enumerate(parts):
        # Create the directory structure
        #   sc/instr/mode/level[/optdesc]/YYYY/MM/
        path = os.path.join(root, *part[0:5], part[5][0:4], part[5][4:6])
        
        # Burst files require the DAY directory
        #   sc/instr/mode/level[/optdesc]/YYYY/MM/DD/
        if part[3] == 'brst':
            path = os.path.join(path, part[5][6:8])
        
        # Append the filename
        path = os.path.join(path, fnames[idx])
        paths.append(path)
    
    return paths


def filter_time(fnames, start_date, end_date):
    """
    Filter files by their start times.
    
    Arguments:
        fnames (str,list):    File names to be filtered.
        start_date (str):     Start date of time interval, formatted as '%Y-%m-%dT%H:%M:%S'
        end_date (str):       End date of time interval, formatted as '%Y-%m-%dT%H:%M:%S'
    
    Returns:
        paths (list):     Path to the data file.
    """
    
    # Output
    files = fnames
    if type(files) is str:
        files = [files]
    
    # Convert date range to datetime objects
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%S')
    
    # Parse the time out of the file name
    parts = parse_filename(fnames)
    fstart = [dt.datetime.strptime(name[-2], '%Y%m%d') if len(name[-2]) == 8 else
              dt.datetime.strptime(name[-2], '%Y%m%d%H%M%S')
              for name in parts]
    
    # Sor the files by start time
    isort = sorted(range(len(fstart)), key=lambda k: fstart[k])
    fstart = [fstart[i] for i in isort]
    files = [files[i] for i in isort]
    
    # End time
    #   - Any files that start on or before END_DATE can be kept
    idx = [i for i, t in enumerate(fstart) if t <= end_date ]
    if len(idx) > 0:
        fstart = [fstart[i] for i in idx]
        files = [files[i] for i in idx]
    else:
        tstart = []
    
    # Start time
    #   - Any file with TSTART <= START_DATE can potentially have data
    #     in our time interval of interest.
    #   - Assume the start time of one file marks the end time of the previous file.
    #   - With this, we look for the file that begins just prior to START_DATE and
    #     throw away any files that start before it.
    idx = [i for i, t in enumerate(fstart) if t.date() < start_date.date()]
    if len(idx) > 0:
        fstart = fstart[idx[-1]:]
        files = files[idx[-1]:]
    
    # The last caveat:
    #   - Our filter may be too lenient. The first file may or may not contain
    #     data within our interval.
    #   - Check if it starts on the same day. If not, toss it
    #   - There may be many files with the same start time, but different
    #     version numbers. Make sure we get all copies OF the first start
    #     time.
    if (len(fstart) > 0) and (fstart[0].date() < start_date.date()):
        idx = [i for i, t in enumerate(fstart) if t.date() != fstart[0].date()]
        if len(idx) > 0:
            fstart = [fstart[i] for i in idx]
            files = [files[i] for i in idx]
        else:
            fstart = []
            files = []
        
    return files


def filter_version(files, latest=None, version=None, min_version=None):
    """
    Filter file names according to their version numbers.
    
    Arguments:
        files (str,list):    File names to be turned into paths.
        latest (bool):       If True, the latest version of each file type is returned.
                             if `version` and `min_version` are not set, this is the
                             default.
        version (str):       Only files with this version are returned.
        min_version (str):   All files with version greater or equal to this are returned.
    
    Returns:
        filtered_files (list):     The files remaining after applying filter conditions.
    """
    
    if version is None and min is None:
        latest = True
    if (version == None + min_version == None + latest == None) > 1:
        ValueError('latest, version, and min are mutually exclusive.')
    
    # Output list
    filtered_files = []
    
    # The latest version of each file type
    if latest:
        # Parse file names and identify unique file types
        #   - File types include all parts of file name except version number
        parts = mms_parse_filename(files)
        bases = ['_'.join(part[0:-2]) for part in parts]
        versions = [part[-1] for part in parts]
        uniq_bases = list(set(bases))
        
        # Filter according to unique file type
        for idx, uniq_base in enumerate(uniq_bases):
            test_idx = [i for i, test_base in bases if test_base == uniq_base]
            file_ref = files[idx]
            vXYZ_ref = versions[idx].split('.')
        
            filtered_files.append(file_ref)
            for i in test_idx:
                vXYZ = versions[i].split('.')
                if ( (vXYZ[0] > vXYZ_ref[0]) or
                     (vXYZ[0] == vXYZ_ref[0] and vXYZ[1] > vXYZ_ref[1]) or
                     (vXYZ[0] == vXYZ_ref[0] and vXYZ[1] == vXYZ_ref[1] and vXYZ[2] > vXYZ_ref[2])
                   ):
                    filtered_files[-1] = files[i]
    
    # All files with version number greater or equal to MIN_VERSION
    elif min_version is not None:
        vXYZ_min = min_version.split('.')
        for idx, v in enumerate(versions):
            vXYZ = v.split('.')
            if ( (vXYZ[0] > vXYZ_min[0]) or 
                 (vXYZ[0] == vXYZ_min[0] and vXYZ[1] > vXYZ_min[1]) or
                 (vXYZ[0] == vXYZ_min[0] and vXYZ[1] == vXYZ_min[1] and vXYZ[2] >= vXYZ_min[2])
               ):
                filtered_files.append(files[idx])
    
    # All files with a particular version number
    elif version is not None:
        vXYZ_ref = min_version.split('.')
        for idx, v in enumerate(versions):
            vXYZ = v.split('.')
            if (vXYZ[0] == vXYZ_ref[0] and
                vXYZ[1] == vXYZ_ref[1] and
                vXYZ[2] == vXYZ_ref[2]
               ):
                filtered_files.append(files[idx])
    
    return filtered_files


def parse_filename(fnames):
    """
    Construct a file name compliant with MMS file name format guidelines.
    
    Arguments:
        fname (str,list): File names to be parsed.
    
    Returns:
        parts (list):     A list of tuples. The tuple elements are:
                          [0]: Spacecraft IDs
                          [1]: Instrument IDs
                          [2]: Data rate modes
                          [3]: Data levels
                          [4]: Optional descriptor (empty string if not present)
                          [5]: Start times
                          [6]: File version number
    """
    
    # Allocate space
    out = []
    
    if type(fnames) is str:
        files = [fnames]
    else:
        files = fnames
    
    # Parse each file
    for file in files:
        # Parse the file names
        parts = os.path.basename(file).split('_')
        
        if len(parts) == 6:
            optdesc = ''
        else:
            optdesc = parts[4]
            
        out.append((*parts[0:4], optdesc, parts[-2], parts[-1][1:-4]))
    
    return out


def parse_time(times):
    """
    Parse the start time of MMS file names.
    
    Arguments:
        times (str,list): Start times of file names.
    
    Returns:
        parts (list):     A list of tuples. The tuple elements are:
                          [0]: Year
                          [1]: Month
                          [2]: Day
                          [3]: Hour
                          [4]: Minute
                          [5]: Second
    """
    
    if isinstance(times, str):
        times = [times]
    
    # Two types: srvy=YYYYMMDD and brst=YYYYMMDDhhmmss
    #   - Accessing "hhmmss" of srvy times returns empty strings, not errors
    parts = [(time[0:4], time[4:6], time[6:8], time[8:10], time[10:12], time[12:14]) for time in times]
    
    return parts


def sort_files(files):
    """
    Sort MMS file names by data product and time.
    
    Arguments:
        files (str,list):   Files to be sorted
    
    Returns:
        sorted (tuple):     Sorted file names. Each tuple element corresponds to
                            a unique data product.
    """
    
    # File types and start times
    parts = parse_filename(files)
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