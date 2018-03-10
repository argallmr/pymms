#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 19:16:32 2018

@author: argall
"""

def MrMMS_Parse_Filename(fnames):
    """
    Construct a file name compliant with MMS file name format guidelines.
    
    Arguments:
        fname (str):      File names to be parsed.
    
    Returns:
        parts (list):     A list of tuples. The tuple elements are:
                          [0]: Spacecraft IDs
                          [1]: Instrument IDs
                          [2]: Data rate modes
                          [3]: Data levels
                          [4]: Optional descriptor (empty string if not present)
                          [4]: Start times
                          [5]: File version number
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
        parts = file.split('_')
        
        if len(parts) == 6:
            optdesc = ''
        else:
            optdesc = parts[4]
            
        out.append((*parts[0:4], optdesc, parts[-2], parts[-1][1:-4]))
    
    return out
