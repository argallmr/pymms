#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:34:47 2018

@author: argall
"""

def MrMMS_Construct_Filename(sc, instr=None, mode=None, level=None, tstart='*', version='*',
                             optdesc=None):
    """
    Construct a file name compliant with MMS file name format guidelines.
    
    Arguments:
        sc      (str):    Spacecraft ID(s)
        instr   (str):    Instrument ID(s)
        mode    (str):    Data rate mode(s). Options include slow, fast, srvy, brst
        level   (str):    Data level(s). Options include l1a, l1b, l2pre, l2, l3
        tstart  (str):    Start time of data file. In general, the format is
                          YYYYMMDDhhmmss for "brst" mode and YYYYMMDD for "srvy"
                          mode (though there are exceptions). If not given, the
                          default is "*".
        version (str):    File version, formatted as "X.Y.Z", where X, Y, and Z
                          are integer version numbers.
        optdesc (str):    Optional file name descriptor. If multiple parts,
                          they should be separated by hyphens ("-"), not under-
                          scores ("_").
    
    Returns:
        fnames  (str);    File names constructed from inputs.
    """
    
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
        fnames = ['_'.join((s,i,m,l,t,'v'+v)) for s in sc_ids
                                              for i in instr
                                              for m in mode
                                              for l in level
                                              for t in tstart
                                              for v in version]
    else:
        fnames = ['_'.join((s,i,m,l,o,t,'v'+v)) for s in sc_ids
                                                for i in instr
                                                for m in mode
                                                for l in level
                                                for o in optdesc
                                                for t in tstart
                                                for v in version]
    return fnames
