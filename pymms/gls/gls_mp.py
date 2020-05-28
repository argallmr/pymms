""" Generate SITL selections from NASA MMS1 spacecraft data.

"""

print("\n-----------------------------------------------------------------------------------------")

import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disables Tensorflow debugging information
import joblib
import sys
import numpy as np
import pandas as pd
import requests
requests.adapters.DEFAULT_RETRIES = 5
import argparse
import tempfile
import shutil
import glob

import pymms
from pymms.sdc import selections as sel
from pymms.gls import gls_mp_data
import pymms.gls.gls_lstm as models

from sklearn.metrics import f1_score

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.expanduser(pymms.config['model_directory'] if pymms.config['model_directory'] is not None else "")
dropbox_dir = pymms.config['dropbox_root'] if pymms.config['dropbox_root'] is not None else ""


def roundTime(dt=None, dateDelta=datetime.timedelta(minutes=1)):
    """Round a datetime object to a multiple of a timedelta
    dt : datetime.datetime object, default now.
    dateDelta : timedelta object, we round to a multiple of this, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
            Stijn Nevens 2014 - Changed to use only datetime objects as variables
    """
    roundTo = dateDelta.total_seconds()

    if dt == None : dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    # // is a floor division, not a comment on following line:
    rounding = (seconds+roundTo/2) // roundTo * roundTo
    return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)


def fix_date_intervals(data_index):
    """
    Temporary workaround for 4.5 second intervals between selection dates.
    """
    dates = []
    for index, date in enumerate(data_index):
        if index % 2 == 0:
            dates.append(date + datetime.timedelta(seconds=1))
        else:
            dates.append(date)
    return dates


def process(start_date, end_date, spacecraft, gpu, test=False):
    # # Define MMS CDF directory location
    # Load model.model
    print(f"Loading model.model. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    model = models.gpu_lstm() if gpu else models.cpu_lstm()
    model.load_weights(os.path.join(model_dir, 'model_weights.h5'))

    # Load data
    print(f"Loading data: | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    data = gls_mp_data.get_data(spacecraft, 'sitl', start_date, end_date, False, False)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.interpolate(method='time', limit_area='inside')

    # Temporary workaround for 4.5 second time cadence of data not working with selections.combine_selections
    data = data.resample("5S").pad()
    data = data.dropna()
    data_index = data.index

    # Scale data
    print(f"Scaling data. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    scaler = joblib.load(open(os.path.join(model_dir, 'scaler.pkl'), 'rb'))
    data = scaler.transform(data)

    # Run data through model.model
    print(f"Generating selection predictions. | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    predictions_list = model.predict(np.expand_dims(data, axis=0))

    # Filter predictions with threshold
    threshold = 0.95
    filtered_output = [0 if x < threshold else 1 for x in predictions_list.squeeze()]

    # Return predictions if testing
    if test:
        return data_index, filtered_output

    # Create selections from predictions
    selections = pd.DataFrame()
    selections.insert(0, "tstart", data_index)
    selections.insert(1, "tstop", data_index)
    selections.insert(2, "prediction", filtered_output)
    selections['FOM'] = "150.0" # This is a placeholder for the FOM
    selections['description'] = "MP crossing (automatically generated)"
    selections['createtime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    selections = selections[selections['prediction'] == 1]

    return selections


def chunk_process(start_date, end_date, spacecraft, gpu, chunks, delete_after_chunk, clear_temp):
    for i, (start, end) in enumerate(chunk_date_range(start_date, end_date, chunks)):
        selections = process(start, end, spacecraft, gpu)
        file_name = f'gls_selections_mp-dl-unh_chunk_{i}.csv'

        print(f"Saving selections to CSV: {dropbox_dir + file_name}, chunk {i} of {chunks}| {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")

        if not selections.empty:
            temp_path = os.path.join(tempfile.gettempdir(), file_name)
            selections.to_csv(temp_path, index=False)
            selections = sel.read_csv(temp_path)
            sel.combine_segments(selections, 5)
            sel.write_csv(dropbox_dir + file_name, selections)

        if delete_after_chunk:
            shutil.rmtree(pymms.config['data_root'])

        if clear_temp:
            files = glob.glob(tempfile.gettempdir() + "/*")
            for f in files:
                os.remove(f)


def chunk_date_range(start, end, interval):
    diff = (end - start) / interval
    s = start
    for i in range(interval):
        yield s, (start + diff * i)
        s = start + diff * i


def test(gpu):
    """
    Test the model through January of 2018.
    """
    validation_data = gls_mp_data.get_data("mms1", 'sitl', "2018-01-01", "2018-01-02", True, True)
    validation_data = validation_data.resample("5s").pad().dropna()
    validation__y = validation_data['selected']
    test_index, test_y = process("2018-01-01", "2018-01-02", "mms1", gpu, True)
    return f1_score(validation__y.astype(int), test_y)


def main():
    """
    Runs the model according to command line arguments.

    usage: gls_mp.py [-h] [-g] [-t] [-c C] [-temp] start end sc

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
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("start",
                        help="Start date of data interval, formatted as either '%%Y-%%m-%%d' or '%%Y-%%m-%%dT%%H:%%M:%%S'. Optionally an integer, interpreted as an orbit number.",
                        type=gls_mp_data.validate_date)
    parser.add_argument("end",
                        help="Start date of data interval, formatted as either '%%Y-%%m-%%d' or '%%Y-%%m-%%dT%%H:%%M:%%S'. Optionally an integer, interpreted as an orbit number.",
                        type=gls_mp_data.validate_date)
    parser.add_argument("sc", help="Spacecraft IDs ('mms1', 'mms2', 'mms3', 'mms4')")
    parser.add_argument("-g", "-gpu", help="Enables use of GPU-accelerated model for faster predictions. Requires CUDA installed.", action="store_true")
    parser.add_argument("-t", "-test", help="Runs a test routine on the model.", action="store_true")
    parser.add_argument("-c", "-chunks", help="Break up the processing of the date interval in C chunks.", type=int)
    parser.add_argument("-temp", help="If running the job in chunks, deletes the contents of the MMS root data folder after each chunk.", action="store_true")

    args = parser.parse_args()

    if pymms.load_config() is None:
        print("Calling this function requires a valid config.ini so that the program knows where to download the SDC CDFs to.")
        exit(-1)

    sc = args.sc
    start = args.start
    end = args.end
    gpu = args.g
    t = args.t
    chunks = args.c
    temp = args.temp

    if sc not in ["mms1", "mms2", "mms3", "mms4"]:
        print("Error: Invalid spacecraft entered.")
        print(f'Expected one of [ "mms1", "mms2", "mms3", "mms4" ], got {sc}.')
        sys.exit(166)

    elif t:
        print(f"Model F1 score: {test(gpu)}")

    elif chunks:
        chunk_process(start, end, sc, gpu, chunks, temp, True)

    else:
        selections = process(start, end, sc, gpu)

        if not selections.empty:
            current_datetime = datetime.datetime.now()
            selections_filetime = current_datetime.strftime('%Y-%m-%d-%H-%M-%S')
            file_name = f'gls_selections_mp-dl-unh_{selections_filetime}.csv'

            # Output selections
            print(f"Saving selections to CSV: {dropbox_dir + file_name} | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
            temp_path = os.path.join(tempfile.gettempdir(), file_name)
            selections.to_csv(temp_path, index=False)
            selections = sel.read_csv(temp_path)
            sel.combine_segments(selections, 5)
            sel.write_csv(dropbox_dir + file_name, selections)

    print(f"Done | {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    sys.exit(0)


if __name__ == '__main__':
    main()
