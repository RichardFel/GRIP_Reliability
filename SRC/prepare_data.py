import pandas as pd
import numpy as np
from SRC.utils import *
from datetime import datetime, timedelta


def downsample_to_frequency(df, current_freq, target_freq):
    # Calculate the ratio of current frequency to target frequency
    ratio = current_freq / target_freq

    # Calculate the interval to downsample
    interval = 1 / (ratio - 1)

    # Initialize an empty list to store the indices to keep
    indices_to_keep = []

    # Iterate over the DataFrame to select appropriate rows
    skip_counter = 0
    for i in range(len(df)):
        if skip_counter >= interval:
            skip_counter = 0  # Reset skip counter
        else:
            indices_to_keep.append(i)
            skip_counter += 1

    # Select the rows based on the calculated indices
    df_downsampled = df.iloc[indices_to_keep].reset_index(drop=True)

    return df_downsampled


def round_time_to_two_hours(dt, mode):
    valid_hours = [6, 8, 10, 12, 14, 16, 18, 20, 22, 0]
    if mode == "begin":
        if dt.hour < 6:
            return dt.replace(hour=6, minute=0, second=0, microsecond=0)
        if dt.hour in valid_hours:
            return dt.replace(hour=dt.hour+2, minute=0, second=0, microsecond=0)
        else:
            return dt.replace(hour=dt.hour+1, minute=0, second=0, microsecond=0)
    else:
        if dt.hour <= 23:
            if dt.hour in valid_hours:
                return dt.replace(hour=dt.hour, minute=0, second=0, microsecond=0)
            else:
                return dt.replace(hour=dt.hour-1, minute=0, second=0, microsecond=0)
        else:
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def split_data(data_df, begintime, endtime, settings):
    data_df['time_2hours'] = ''
    corrected_begintime = round_time_to_two_hours(
        begintime, "begin")
    corrected_endtime = round_time_to_two_hours(endtime, "end")
    two_hours = timedelta(seconds=7200)

    for epoch in range(int((corrected_endtime - corrected_begintime) / two_hours)):
        start = corrected_begintime + two_hours * epoch
        end = corrected_begintime + two_hours * epoch + two_hours
        sample_start = int(
            (start - begintime).seconds * settings['frequency'])
        sample_end = int(
            (end - begintime).seconds * settings['frequency'])
        data_df.loc[sample_start:sample_end, 'time_2hours'] = f'{start.hour}_{end.hour}'
    data_df['time'] = [begintime + timedelta(seconds=i /settings['frequency']) for i in range(len(data_df))]
    return data_df


def clean_data(data_df, settings, all=False):
    if all:
        # Drop the first and last 10 seconds
        samples_per_halfmin = int(settings['frequency'] * 10)
        data_df = data_df.iloc[samples_per_halfmin:len(
            data_df)-samples_per_halfmin, :]

    # Drop data that has more than a minute of inactivity
    not_worn_samples = data_df.loc[data_df['Worn_sensor'] == 0]
    data_df = data_df.loc[data_df['Worn_sensor'] == 1]
    return data_df, not_worn_samples


def prepare_data(file, group, settings):
    # Load raw sensor data from file
    data_df = pd.read_csv(
        f"{settings['DATA_DIR']}/raw_data/{group}/{file}", header=None, skiprows=10,on_bad_lines = 'skip' )
    data_df = data_df.drop(data_df.iloc[:, 4:8], axis=1)
    data_df = data_df.iloc[:, 1:]
    data_df.columns = ['acc_x', 'acc_y', 'acc_z']
    data_df = data_df.dropna()

    # Load timestamps to calculate sampling frequency
    timestamps = pd.read_csv(
        f"{settings['DATA_DIR']}/raw_data/{group}/{file}", skiprows=8, on_bad_lines='skip')
    timestamps = timestamps.dropna(subset='Unnamed: 7')
    start_time = timestamps.iloc[0, -1]
    end_time = timestamps.iloc[-1, -1]
    diff = end_time - start_time
    # Time in seconds
    diff /= 10000
    sampling_freq = len(data_df) / diff

    # resample data to the settingsured frequency
    if sampling_freq > settings['frequency']:
        data_df = downsample_to_frequency(
            data_df, sampling_freq, settings['frequency'])


    # Detect parts in which the sensor was not worn
    data_df['Worn_sensor'] = 1
    samples_per_minute = int(settings['frequency'] * 60)
    samples = int(np.floor(len(data_df) / samples_per_minute))
    for num in range(samples):
        start = samples_per_minute * num
        end = samples_per_minute * num + samples_per_minute
        sample = data_df.iloc[start:end, :]
        std = (np.std(sample['acc_x']) +
               np.std(sample['acc_y']) + np.std(sample['acc_z'])) / 3
        if std < 10 ** (-3):
            data_df.loc[start:end, 'Worn_sensor'] = 0

    enddate = file.split('_')[-2]
    endhour = file.split('_')[-1].split('.')[0]
    endtime = datetime.strptime(f'{enddate} {endhour}', '%Y-%m-%d %H%M%S')
    measurement_duration_seconds = int(len(data_df)/settings['frequency'])
    begintime = endtime - timedelta(seconds=measurement_duration_seconds)
    return data_df, endtime, begintime
