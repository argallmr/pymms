import datetime
import tempfile
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.constants
from cdflib import cdfread, epochs


import pymms
from pymms.sdc import mrmms_sdc_api as api
from pymms.sdc import selections as selections_api


class Model_Data_Downloader:
    """
    Interface with MrMMS_SDC_API to download and format SDC data needed for training, evaluating, and running the
    mp-dl-unh pipeline.

    Interface with the Science Data Center (SDC) API of the
    Magnetospheric Multiscale (MMS) mission.
    https://lasp.colorado.edu/mms/sdc/public/

    Params:
        sc (str,list):              Spacecraft IDs ('mms1', 'mms2', 'mms3', 'mms4')
        level (str,list):           Data quality level ('l1a', 'l1b', 'sitl', 'l2pre', 'l2', 'l3')
        start (str):                Start date of data interval, formatted as either %Y-%m-%d or
                                    %Y-%m-%dT%H:%M:%S.
                                    Optionally can be a single integer, interpreted as an orbit number.
        end (str):                  End date of data interval, formatted as either %Y-%m-%d or
                                    %Y-%m-%dT%H:%M:%S.
                                    Optionally can be a single integer, interpreted as an orbit number.
        include_selections (bool):  If true, includes SITL selections in the combined dataframe.
        include_partials (bool)     If true, includes partial magnetopause crossings in SITL selections.
        verbose (bool):             If true, prints out optional information about downloaded variables.
    """

    def __init__(self,
                 sc,
                 level,
                 start,
                 end,
                 include_selections = True,
                 include_partials = True,
                 verbose = False):
        self.sc = sc
        self.level = level
        self.include_selections = include_selections
        self.verbose = verbose

        if isinstance(start, int):
            sroi = api.mission_events('sroi', start, end, sc=sc)
            self.start_date = sroi['tstart'][0]
            self.end_date = sroi['tend'][-1]
        else:
            self.start_date = validate_date(start)
            self.end_date = validate_date(end)

        if(include_partials and not include_selections):
            raise ValueError("Include_selections must be true in order to include partial selections in the combined dataframe.")

        self.include_selections = include_selections
        self.include_partials = include_partials

        # SITL data is available in the fast-survey region of the orbit.
        # For many instruments, fast- and slow-survey data are combined into a single survey product
        self.mode = 'srvy'

        # This script works only for 'sitl' and 'l2' data
        if level not in ('sitl', 'l2'):
            raise ValueError('Level must be either "sitl" or "l2".')

        # Create an interface to the SDC
        self.mms = api.MrMMS_SDC_API(sc=sc, mode=self.mode, start_date=self.start_date, end_date=self.end_date)

        # Ensure that the log-in information is there.
        #   - If the config file was already set, this step is redundant.
        self.mms._data_root = pymms.config['data_root']
        if self.mode == 'sitl':
            self.mms._session.auth(pymms.config['username'], pymms.config['password'])


    def read_cdf_vars(self, cdf_files, cdf_vars, epoch='Epoch'):
        '''
        Read variables from CDF files into a data frame

        Parameters
        ----------
        cdf_files : str or list
            CDF files to be read
        cdf_vars : str or list
            Names of the variables to be read
        epoch : str
            Name of the time variable that serves as the data frame index

        Returns
        -------
        out : `pandas.DataFrame`
            The data. If a variable is 2D, "_#" is appended, where "#"
            increases from 0 to var.shape[1]-1.
        '''
        tepoch = epochs.CDFepoch()
        if isinstance(cdf_files, str):
            cdf_files = [cdf_files]
        if isinstance(cdf_vars, str):
            cdf_vars = [cdf_vars]
        if epoch not in cdf_vars:
            cdf_vars.append(epoch)

        out = []
        for file in cdf_files:
            file_df = pd.DataFrame()
            cdf = cdfread.CDF(file)

            for var_name in cdf_vars:
                # Read the variable data
                data = cdf.varget(var_name)
                if var_name == epoch:
                    data = tepoch.to_datetime(data, to_np=True)

                # Store as column in data frame
                if data.ndim == 1:
                    file_df[var_name] = data

                # 2D variables get "_#" appended to name for each column
                elif data.ndim == 2:
                    for idx in range(data.shape[1]):
                        file_df['{0}_{1}'.format(var_name, idx)] = data[:, idx]

                # 3D variables gets reshaped to 2D and treated as 2D
                # This includes variables like the pressure and temperature tensors
                elif data.ndim == 3:
                    dims = data.shape
                    data = data.reshape(dims[0], dims[1] * dims[2])
                    for idx in range(data.shape[1]):
                        file_df['{0}_{1}'.format(var_name, idx)] = data[:, idx]
                else:
                    print('cdf_var.ndims > 3. Skipping. {0}'.format(var_name))
                    continue

            # Close the file
            cdf.close()

            # Set the epoch variable as the index
            file_df.set_index(epoch, inplace=True)
            out.append(file_df)

        # Concatenate all of the file data
        out = pd.concat(out)

        # Check that the index is unique
        # Some contiguous low-level data files have data overlap at the edges of the files (e.g., AFG)
        if not out.index.is_unique:
            out['index'] = out.index
            out.drop_duplicates(subset='index', inplace=True, keep='first')
            out.drop(columns='index', inplace=True)

        # File names are not always given in order, so sort the data
        out.sort_index(inplace=True)
        return out

    def rename_df_cols(self, df, old_col, new_cols):
        '''
        Each column of a multi-dimensional CDF variable gets stored as
        its own independent column in the DataFrame, with "_#" appended
        to the original variable name to indicate which column index
        the column was taken from. This function renames those columns.

        Parameters
        ----------
        df : `pandas.DataFrame`
            DataFrame for which the columns are to be renamed
        old_col : str
            Name of the column (sans "_#")
        new_cols : list
            New names to be given to the columns
        '''
        df.rename(columns={'{}_{}'.format(old_col, idx): new_col_name
                           for idx, new_col_name in enumerate(new_cols)},
                  inplace=True)

    def quality_factor(self, data, M=2):
        '''
        Compute a quality factor for burst triggers.

        Parameters
        ----------
        data : `numpy.ndarray`
            One dimensional data array
        M : int
            Smoothing factor

        Returns
        -------
        Q : `numpy.ndarray`
            Burst trigger quality factor
        '''
        smoothed_data = [data[0]]
        for i, value in enumerate(data[1:]):
            smoothed_data.append((smoothed_data[i - 1] * (2 ** M - 1) + value) / 2 ** M)
        return np.subtract(data, smoothed_data)

    def afg_data(self):
        '''
        Downloads, formats, and calculates metafeatures for AFG data from SDC.

        Returns:
            afg_df
        '''

        # There are two magnetometers: AFG and DFG. For L2 data, AFG is
        # used for slow survey and DFG is used for fast survey, but are
        # known by the instrument name FGM. For SITL-level data, the
        # instruments are separate and named as AFG and DFG.
        afg_instr = 'afg'
        if self.level == 'l2':
            afg_instr = 'fgm'

        afg_mode = self.mode

        # The "SITL"-level data for AFG is labeled "ql" for quick-look
        afg_level = self.level
        if self.level == 'sitl':
            afg_level = 'ql'

        afg_optdesc = None

        # Download the data files
        self.mms.instr = afg_instr
        self.mms.mode = afg_mode
        self.mms.level = afg_level
        self.mms.optdesc = afg_optdesc
        afg_files = self.mms.download()
        if self.verbose:
            print(*afg_files, sep='\n')

        """Read the data"""

        # Print the variable names from a sample file
        afg_cdf = cdfread.CDF(afg_files[0])
        info = afg_cdf.cdf_info()
        afg_cdf.close()
        if self.verbose:
            print(*info['zVariables'], sep='\n')

        # Variable names
        t_vname = 'Epoch'
        if afg_level == 'l2':
            b_vname = '_'.join((self.sc, afg_instr, 'b', 'dmpa', afg_mode, afg_level))
        else:
            b_vname = '_'.join((self.sc, afg_instr, afg_mode, 'dmpa'))

        # Read the data
        afg_df = self.read_cdf_vars(afg_files, b_vname, epoch=t_vname)

        # Rename variables
        self.rename_df_cols(afg_df, b_vname, ('Bx', 'By', 'Bz', '|B|'))

        """Compute metafeatures and store data in a dataframe."""

        # Compute metafeatures
        afg_df['P_B'] = afg_df['|B|'] ** 2 / scipy.constants.mu_0
        afg_df['clock_angle'] = np.arctan2(afg_df['By'], afg_df['Bz'])
        afg_df['Q_dBx'] = self.quality_factor(afg_df['Bx'])
        afg_df['Q_dBz'] = self.quality_factor(afg_df['Bz'])

        return afg_df

    def edp_data(self):
        '''
        Downloads, formats, and calculates metafeatures for EDP data from SDC.

        Returns:
            edp_df
        '''

        edp_instr = 'edp'
        edp_optdesc = 'dce'

        # EDP does not have "srvy" data, just "fast" and "slow"
        edp_mode = self.mode
        if self.mode == 'srvy':
            edp_mode = 'fast'

        # The "SITL"-level data for EDP is labeled "ql" for quick-look
        edp_level = self.level
        if self.level == 'sitl':
            edp_level = 'ql'

        # Download the data files
        self.mms.instr = edp_instr
        self.mms.mode = edp_mode
        self.mms.optdesc = edp_optdesc
        edp_files = self.mms.download()
        if self.verbose:
            print(*edp_files, sep='\n')

        """Read the files"""

        # Print the variable names from a sample file
        edp_cdf = cdfread.CDF(edp_files[0])
        info = edp_cdf.cdf_info()
        edp_cdf.close()
        if self.verbose:
            print(*info['zVariables'], sep='\n')

        # Variable names
        if self.level == 'l2':
            t_vname = '_'.join((self.sc, edp_instr, 'epoch', edp_mode, edp_level))
            e_vname = '_'.join((self.sc, edp_instr, edp_optdesc, 'dsl', edp_mode, edp_level))
        else:
            t_vname = '_'.join((self.sc, edp_instr, edp_optdesc, 'epoch'))
            e_vname = '_'.join((self.sc, edp_instr, edp_optdesc, 'xyz', 'dsl'))

        # Read the data
        edp_df = self.read_cdf_vars(edp_files, e_vname, epoch=t_vname)

        # Rename variables
        new_vnames = ('Ex', 'Ey', 'Ez')
        edp_df.rename(columns={'{}_{}'.format(e_vname, idx): vname
                               for idx, vname in enumerate(new_vnames)},
                      inplace=True)

        """Compute metafeatures"""

        edp_df['|E|'] = np.sqrt(edp_df['Ex'] ** 2 + edp_df['Ey'] ** 2 + edp_df['Ez'] ** 2)

        return edp_df

    def dis_data(self):
        '''
        Downloads, formats, and calculates metafeatures for DIS data from SDC.

        Returns:
            dis_df
        '''

        dis_instr = 'fpi'

        # FPI does not have "srvy" data, just "fast" and "slow"
        dis_mode = self.mode
        if self.mode == 'srvy':
            dis_mode = 'fast'

        # The "SITL"-level data for FPI is labeled "ql" for quick-look
        # There is SITL-level data, but it was discontinued early in the mission
        dis_level = self.level
        if self.level == 'sitl':
            dis_level = 'ql'

        dis_optdesc = 'dis'
        if self.level == 'l2':
            dis_optdesc = 'dis-moms'

        # Download the data files
        self.mms.instr = dis_instr
        self.mms.mode = dis_mode
        self.mms.level = dis_level
        self.mms.optdesc = dis_optdesc
        dis_files = self.mms.download()
        if self.verbose:
            print(*dis_files, sep='\n')

        """Read the files"""

        # Print the variable names from a sample file
        dis_cdf = cdfread.CDF(dis_files[0])
        info = dis_cdf.cdf_info()
        if self.verbose:
            print(*info['zVariables'], sep='\n')

        # Print information about the pressure tensor
        # to figure out its dimensions and how the components
        # are stored
        vname = '_'.join((self.sc, 'dis', 'prestensor', 'dbcs', dis_mode))
        var_notes = dis_cdf.attget(attribute='VAR_NOTES', entry=vname)
        if self.verbose:
            print(var_notes['Data'])

        # Close the file
        dis_cdf.close()

        # Variable names
        t_vname = 'Epoch'
        espectr_omni_vname = '_'.join((self.sc, 'dis', 'energyspectr', 'omni', dis_mode))
        n_vname = '_'.join((self.sc, 'dis', 'numberdensity', dis_mode))
        v_vname = '_'.join((self.sc, 'dis', 'bulkv', 'dbcs', dis_mode))
        q_heat_vname = '_'.join((self.sc, 'dis', 'heatq', 'dbcs', dis_mode))
        t_para_vname = '_'.join((self.sc, 'dis', 'temppara', dis_mode))
        t_perp_vname = '_'.join((self.sc, 'dis', 'tempperp', dis_mode))
        t_tens_vname = '_'.join((self.sc, 'dis', 'temptensor', 'dbcs', dis_mode))
        p_tens_vname = '_'.join((self.sc, 'dis', 'prestensor', 'dbcs', dis_mode))

        # Read the data
        dis_df = self.read_cdf_vars(dis_files,
                                    [espectr_omni_vname, n_vname, v_vname,
                                     q_heat_vname, t_para_vname, t_perp_vname,
                                     p_tens_vname, t_tens_vname
                                     ],
                                    epoch=t_vname)

        # Rename variables
        dis_df.rename(columns={n_vname: 'Ni'}, inplace=True)
        dis_df.rename(columns={t_para_vname: 'Ti_para'}, inplace=True)
        dis_df.rename(columns={t_perp_vname: 'Ti_perp'}, inplace=True)
        self.rename_df_cols(dis_df, v_vname, ('Vix', 'Viy', 'Viz'))
        self.rename_df_cols(dis_df, q_heat_vname, ('Qi_xx', 'Qi_yy', 'Qi_zz'))
        self.rename_df_cols(dis_df, t_tens_vname,
                            ('Ti_xx', 'Ti_xy', 'Ti_xz', 'Ti_yx', 'Ti_yy', 'Ti_yz', 'Ti_zx', 'Ti_zy', 'Ti_zz'))
        self.rename_df_cols(dis_df, p_tens_vname,
                            ('Pi_xx', 'Pi_xy', 'Pi_xz', 'Pi_yx', 'Pi_yy', 'Pi_yz', 'Pi_zx', 'Pi_zy', 'Pi_zz'))
        self.rename_df_cols(dis_df, espectr_omni_vname, ['especi_{0}'.format(idx) for idx in range(32)])

        # Drop redundant components of the pressure and temperature tensors
        dis_df.drop(columns=['Ti_xy', 'Ti_xz', 'Ti_yz', 'Pi_xy', 'Pi_xz', 'Pi_yz'], inplace=True)

        """Compute metafeatures"""

        dis_df['Ti_anisotropy'] = (dis_df['Ti_para'] / dis_df['Ti_perp']) - 1
        dis_df['Ti_scalar'] = (dis_df['Ti_para'] + 2 * dis_df['Ti_perp']) / 3.0
        dis_df['Pi_scalar'] = (dis_df['Pi_xx'] + dis_df['Pi_yy'] + dis_df['Pi_zz']) / 3.0
        dis_df['Q_dNi'] = self.quality_factor(dis_df['Ni'])
        dis_df['Q_dViz'] = self.quality_factor(dis_df['Viz'])
        Vi_mag = np.sqrt(dis_df['Vix'] ** 2 + dis_df['Viy'] ** 2 + dis_df['Viz'] ** 2)
        Pi_ram = dis_df['Ni'] * Vi_mag
        dis_df['Q_dPi_ram'] = self.quality_factor(Pi_ram)

        # Drop features that were accidentally excluded
        dis_df.drop(columns=['especi_31', 'Viz', 'Qi_zz'], inplace=True)

        return dis_df

    def des_data(self):
        '''
        Downloads, formats, and calculates metafeatures for DES data from SDC.

        Returns:
            des_df
        '''

        des_instr = 'fpi'

        # FPI does not have "srvy" data, just "fast" and "slow"
        des_mode = self.mode
        if self.mode == 'srvy':
            des_mode = 'fast'

        # The "SITL"-level data for FPI is labeled "ql" for quick-look
        # There is SITL-level data, but it was discontinued early in the mission
        des_level = self.level
        if self.level == 'sitl':
            des_level = 'ql'

        des_optdesc = 'des'
        if self.level == 'l2':
            des_optdesc = 'des-moms'

        # Download the data files
        self.mms.instr = des_instr
        self.mms.mode = des_mode
        self.mms.level = des_level
        self.mms.optdesc = des_optdesc
        des_files = self.mms.download()
        if self.verbose:
            print(*des_files, sep='\n')

        """Read the files"""

        # Print the variable names from a sample file
        des_cdf = cdfread.CDF(des_files[0])
        info = des_cdf.cdf_info()
        if self.verbose:
            print(*info['zVariables'], sep='\n')

        # Print information about the pressure tensor
        # to figure out its dimensions and how the components
        # are stored
        vname = 'mms1_des_prestensor_dbcs_fast'
        var_notes = des_cdf.attget(attribute='VAR_NOTES', entry=vname)
        if self.verbose:
            print(var_notes['Data'])

        # Close the file
        des_cdf.close()

        # Variable names
        t_vname = 'Epoch'
        espectr_omni_vname = '_'.join((self.sc, 'des', 'energyspectr', 'omni', des_mode))
        n_vname = '_'.join((self.sc, 'des', 'numberdensity', des_mode))
        v_vname = '_'.join((self.sc, 'des', 'bulkv', 'dbcs', des_mode))
        q_heat_vname = '_'.join((self.sc, 'des', 'heatq', 'dbcs', des_mode))
        t_para_vname = '_'.join((self.sc, 'des', 'temppara', des_mode))
        t_perp_vname = '_'.join((self.sc, 'des', 'tempperp', des_mode))
        t_tens_vname = '_'.join((self.sc, 'des', 'temptensor', 'dbcs', des_mode))
        p_tens_vname = '_'.join((self.sc, 'des', 'prestensor', 'dbcs', des_mode))

        # Read the data
        des_df = self.read_cdf_vars(des_files,
                                    [espectr_omni_vname, n_vname, v_vname,
                                     q_heat_vname, t_para_vname, t_perp_vname,
                                     p_tens_vname, t_tens_vname
                                     ],
                                    epoch=t_vname)

        # Rename variables
        des_df.rename(columns={n_vname: 'Ne'}, inplace=True)
        des_df.rename(columns={t_para_vname: 'Te_para'}, inplace=True)
        des_df.rename(columns={t_perp_vname: 'Te_perp'}, inplace=True)
        self.rename_df_cols(des_df, v_vname, ('Vex', 'Vey', 'Vez'))
        self.rename_df_cols(des_df, q_heat_vname, ('Qe_xx', 'Qe_yy', 'Qe_zz'))
        self.rename_df_cols(des_df, t_tens_vname,
                            ('Te_xx', 'Te_xy', 'Te_xz', 'Te_yx', 'Te_yy', 'Te_yz', 'Te_zx', 'Te_zy', 'Te_zz'))
        self.rename_df_cols(des_df, p_tens_vname,
                            ('Pe_xx', 'Pe_xy', 'Pe_xz', 'Pe_yx', 'Pe_yy', 'Pe_yz', 'Pe_zx', 'Pe_zy', 'Pe_zz'))
        self.rename_df_cols(des_df, espectr_omni_vname, ['espece_{0}'.format(idx) for idx in range(32)])

        # Drop symmetric, redundant components
        des_df.drop(columns=['Te_xy', 'Te_xz', 'Te_yz', 'Pe_xy', 'Pe_xz', 'Pe_yz'], inplace=True)

        """Compute metafeatures"""

        des_df['Te_anisotropy'] = (des_df['Te_para'] / des_df['Te_perp']) - 1
        des_df['Te_scalar'] = (des_df['Te_para'] + 2 * des_df['Te_perp']) / 3.0
        des_df['Pe_scalar'] = (des_df['Pe_xx'] + des_df['Pe_yy'] + des_df['Pe_zz']) / 3.0
        des_df['Q_dNe'] = self.quality_factor(des_df['Ne'])
        des_df['Q_dVez'] = self.quality_factor(des_df['Vez'])
        Ve_mag = np.sqrt(des_df['Vex']**2 + des_df['Vey']**2 + des_df['Vez']**2)
        Pe_ram = des_df['Ne'] * Ve_mag
        des_df['Q_dPe_ram'] = self.quality_factor(Pe_ram)

        return des_df


    def combined_dataframe(self):
        '''
        Combines all dataframes, downsamples to DES, which as the time index with the longest sampling period (`4.5s`).
        After that, multi-instrument metafeatures are calculated.

        Returns:
            df
        '''

        afg_df = self.afg_data()
        edp_df = self.edp_data()
        dis_df = self.dis_data()
        des_df = self.des_data()

        # Resample data
        afg_df = afg_df.reindex(des_df.index, method='nearest')
        edp_df = edp_df.reindex(des_df.index, method='nearest')
        dis_df = dis_df.reindex(des_df.index, method='nearest')

        # Merge dataframes
        df = des_df
        df = df.join(dis_df, how='outer')
        df = df.join(afg_df, how='outer')
        df = df.join(edp_df, how='outer')

        # Metafeatures
        df['T_ratio'] = df['Ti_scalar'] / df['Te_scalar']
        df['plasma_beta'] = (df['Pe_scalar'] + df['Pi_scalar']) / df['P_B']

        if self.include_selections:
            """Download SITL selections"""

            selections_path = Path(tempfile.gettempdir()) / 'all_selections.csv'
            data = selections_api.selections('sitl+back', self.start_date, self.end_date)
            selections_api.write_csv(selections_path, data)

            """Mark datapoints selected by a SITL as selected"""

            selections = pd.read_csv(selections_path, infer_datetime_format=True, parse_dates=[0, 1])
            selections.dropna()

            if self.include_partials:
                selections = selections[
                    selections['discussion'].str.contains("MP", na=False) | selections['discussion'].str.contains(
                        "magnetopause", na=False)]
            else:
                selections = selections[
                    selections['discussion'].str.contains("MP", na=False) | selections['discussion'].str.contains(
                        "magnetopause", na=False) & ~selections['discussion'].str.contains("magnetopause", na=False)]

            # Create column to denote whether an observation is selected by SITLs
            df['selected'] = False

            # Set selected to be True if the observation is in a date range of a selection
            date_col = df.index
            cond_series = df['selected']
            for start, end in zip(selections['start_time'], selections['stop_time']):
                cond_series |= (start <= date_col) & (date_col <= end)
            if self.verbose:
                print(df.loc[cond_series, 'selected'])
            df.loc[cond_series, 'selected'] = True

        return df


def get_data(sc,
             level,
             start,
             end,
             include_selections = True,
             include_partials = True,
             verbose = False):

    downloader = Model_Data_Downloader(sc, level, start, end, include_selections, include_partials, verbose)
    return downloader.combined_dataframe()


def validate_output_path(path):
    try:
        p = Path(path)
        p.parents[0].mkdir(exist_ok=True)
        return p
    except Exception:
        raise OSError.filename("Output path needs to point to a valid location on disk.")


def validate_date(date):
    if isinstance(date, datetime.datetime):
        return date
    else:
        try:
            return int(date)
        except Exception:
            try:
                return datetime.datetime.strptime(date, '%Y-%m-%dT%H:%M:%S')
            except ValueError:
                try:
                    return datetime.datetime.strptime(date, '%Y-%m-%d')
                except ValueError:
                    raise ValueError("Date input is neither a string nor a datetime.datetime object.")


def download_from_cmd():
    """
    Used for downloading a CSV by calling mp-dl-unh-data from the command line.

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
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("sc", help="Spacecraft IDs ('mms1', 'mms2', 'mms3', 'mms4')")
    parser.add_argument("level", help="Data quality level ('l1a', 'l1b', 'sitl', 'l2pre', 'l2', 'l3')")
    parser.add_argument("start",
                         help="Start date of data interval, formatted as either '%%Y-%%m-%%d' or '%%Y-%%m-%%dT%%H:%%M:%%S'. Optionally an integer, interpreted as an orbit number.",
                         type=validate_date)
    parser.add_argument("end",
                         help="Start date of data interval, formatted as either '%%Y-%%m-%%d' or '%%Y-%%m-%%dT%%H:%%M:%%S'. Optionally an integer, interpreted as an orbit number.",
                         type=validate_date)
    parser.add_argument("output", help="Path the output CSV file, including the CSV file's name.", type=validate_output_path)
    parser.add_argument("-is", "--include-selections", help="Includes SITL selections in the output data.", action="store_true")
    parser.add_argument("-ip", "--include-partials", help="Includes partial magnetopause crossings in SITL selections.", action="store_true")
    parser.add_argument("-v", "--verbose", help="If true, prints out optional information about downloaded variables.", action="store_true")

    args = parser.parse_args()

    if pymms.load_config() is None:
        print("Calling this function requires a valid config.ini so that the program knows where to download the SDC CDFs to.")
        exit(-1)

    sc = args.sc
    level = args.level
    start = args.start
    end = args.end
    include_selections = args.include_selections
    include_partials = args.include_partials
    verbose = args.verbose
    output_path = args.output

    df = get_data(sc, level, start, end, include_selections, include_partials, verbose)
    df.to_csv(output_path)

    print(f"mp-dl-unh data downloaded to {output_path}")


if __name__ == '__main__':
    '''
    Test the downloader by downloading a small sample dataset.
    '''

    sc = 'mms1'
    level = 'sitl'
    start = datetime.datetime(2017, 1, 1)
    end = datetime.datetime(2017, 1, 2)
    include_selections = True
    include_partials = True

    get_data(sc, level, start, end, include_selections, include_partials).to_csv(Path(tempfile.gettempdir()) / Path("mp-dl-unh_data_test.csv"))