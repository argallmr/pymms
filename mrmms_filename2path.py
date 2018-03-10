#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 22:28:28 2018

@author: argall
"""

def MrMMS_Filename2Path(fnames, root=''):
    """
    Convert an MMS file name to an MMS path.
    
    MMS paths take the form
        
        sc/instr/mode/level[/optdesc]/YYYY/MM[/DD/]
        
    where the optional descriptor [/optdesc] is included if it is also in the
    file name and day directory [/DD] is included if mode='brst'.
    
    Arguments:
        fnames (list):    File names to be turned into paths.
        root   (str):     Absolute directory
    
    Returns:
        paths (list):     Path to the data file.
    """
    
    import os
    
    paths = []
    
    parts = MrMMS_Parse_Filename(fnames)
    
    for part, idx in enumerate(parts):
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
