from pymms.sdc import mrmms_sdc_api as api
from pymms.data import util
import datetime as dt
import xarray as xr


def check_spacecraft(sc):
    if sc not in ('mms1', 'mms2', 'mms3', 'mms4'):
        raise ValueError('{} is not a recongized SC ID. '
                         'Must be ("mms1", "mms2", "mms3", "mms4")'
                         .format(sc))

def check_mode(mode, level='l2'):
    
    # Fast and slow mode data are combined into survey data
    if level == 'l2':
        if mode in ('fast', 'slow'):
            mode = 'srvy'
    
    modes = ('brst', 'srvy', 'fast', 'slow')
    if mode not in modes:
        raise ValueError('Mode "{0}" is not in {1}'.format(mode, modes))

    return mode


def check_instr(instr):
    instrs = ('fgm', 'afg', 'dfg')
    if instr not in instrs:
        raise ValueError('Instr "{0}" is not in {1}'.format(instr, instrs))


def check_level(level, instr='fgm'):
    if instr == 'fgm':
        levels = ('l2',)
    else:
        levels = ('l1a', 'l2pre')

    if level not in levels:
        raise ValueError('Level "{0}" is not in {1}'.format(level, levels))


def check_coords(coords, instr='fgm', level='l2'):
    coord_systems = ('dbcs', 'gse')
    if coords not in coord_systems:
        raise ValueError(('Coordinate systems "{0}" is not in {1}'
                          .format(coords, coord_systems)
                          )
                         )


def load_data(sc, mode, start_date, end_date,
              level='l2', coords='gse',
              pd=False, **kwargs):
    """
    Load FPI distribution function data.
    
    Parameters
    ----------
    sc : str
        Spacecraft ID: ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Instrument mode: ('fast', 'brst'). If 'srvy' is given, it is
        automatically changed to 'fast'.
    start_date, end_date : `datetime.datetime`
        Start and end of the data interval.
    
    Returns
    -------
    dist : `metaarray.metaarray`
        Particle distribution function.
    """
    
    # Check the inputs
    check_spacecraft(sc)
    mode = check_mode(mode)
    check_level(level, instr='fgm')
    check_coords(coords)
    
    # File and variable name parameters
    t_vname = 'Epoch'
    b_vname = '_'.join((sc, 'fgm', 'b', coords, mode, level))
    b_labl_vname = '_'.join(('label', 'b', coords))
    
    # Read the data from files
    if pd:
    
        # Download the data
        sdc = api.MrMMS_SDC_API(sc, 'fgm', mode, level,
                                start_date=start_date,
                                end_date=end_date)
        fgm_files = sdc.download_files()
        fgm_files = api.sort_files(fgm_files)[0]
    
        fgm_data = util.cdf_to_df(fgm_files, b_vname)
        util.rename_df_cols(fgm_data, b_vname, ('Bx', 'By', 'Bz', '|B|'))
    else:
    
        # Load the data
        #   - R is concatenated along Epoch, but depends on Epoch_state
        fgm_data = util.load_data(sc=sc, instr='fgm', mode=mode, level=level,
                                  start_date=start_date, end_date=end_date, 
                                  **kwargs)
        
        tr_vname = 'Epoch_state'
        t_delta_vname = '_'.join((sc, 'fgm', 'bdeltahalf', mode, level))
        tr_delta_vname = '_'.join((sc, 'fgm', 'rdeltahalf', mode, level))
        
        r_gse_vname = '_'.join((sc, 'fgm', 'r', 'gse', mode, level))
        r_gsm_vname = '_'.join((sc, 'fgm', 'r', 'gsm', mode, level))
        
        r_gse_labl_vname = '_'.join(('label', 'r', 'gse'))
        r_gsm_labl_vname = '_'.join(('label', 'r', 'gsm'))
        repr_vname = '_'.join(('represent', 'vec', 'tot'))
        
        b_dmpa_vname = '_'.join((sc, 'fgm', 'b', 'dmpa', mode, level))
        b_bcs_vname = '_'.join((sc, 'fgm', 'b', 'bcs', mode, level))
        b_gse_vname = '_'.join((sc, 'fgm', 'b', 'gse', mode, level))
        b_gsm_vname = '_'.join((sc, 'fgm', 'b', 'gsm', mode, level))
        
        b_dmpa_lbl_vname = '_'.join(('label', 'b', 'dmpa'))
        b_bcs_lbl_vname = '_'.join(('label', 'b', 'bcs'))
        b_gse_lbl_vname = '_'.join(('label', 'b', 'gse'))
        b_gsm_lbl_vname = '_'.join(('label', 'b', 'gsm'))
        
        labels = [b_dmpa_lbl_vname, b_bcs_lbl_vname, b_gse_lbl_vname,
                  b_gsm_lbl_vname, r_gse_labl_vname, r_gsm_labl_vname]
        
        # Rename variables
        names = {t_vname: 'time',
                 tr_vname: 'time_r',
                 t_delta_vname: 'time_delta',
                 tr_delta_vname: 'time_r_delta',
                 b_dmpa_vname: 'B_DMPA',
                 b_bcs_vname: 'B_BCS',
                 b_gse_vname: 'B_GSE',
                 b_gsm_vname: 'B_GSM',
                 r_gse_vname: 'r_GSE',
                 r_gsm_vname: 'r_GSM'}

        names = {key:val for key, val in names.items() if key in fgm_data}
        fgm_data = fgm_data.rename(names)
        
        # Standardize labels
        labels = [label for label in labels if label in fgm_data]
        new_labels = {key: ('r_index' if key.startswith('label_r') else 'b_index')
                      for key in labels}
        fgm_data = (fgm_data.assign_coords({'b_index': ['x', 'y', 'z', 't'],
                                            'r_index': ['x', 'y', 'z']})
                            .drop(labels + [repr_vname,])
                            .rename(new_labels)
                    )
        
    return fgm_data


if __name__ == '__main__':
    pass
    