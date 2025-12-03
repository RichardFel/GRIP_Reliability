import pandas as pd
import datetime
import logging
import os

from SRC.utils import *
from SRC.characteristics import characteristics
from SRC.prepare_data import prepare_data, split_data, clean_data
from SRC.predict_data import predict_data
from SRC.barcode_plot import barcodeplot

def process_data(settings):
    # Load config file and settings
    groups = ['CP', 'HP'] # groups to analyse
    final_results = pd.DataFrame()
    for group in groups:
        subjects = os.listdir(f"{settings['DATA_DIR']}/raw_data/{group}")


        # Loop over subjecs
        for subject in subjects:
            if settings['VERBOSE']:
                print(f'Analysing subject {subject}')
            try:
                if subject.endswith('.DS_Store'):
                    continue
                days = os.listdir(f"{settings['DATA_DIR']}/raw_data/{group}/{subject}")
                for file in ['pain_score', '.DS_Store', 'painscore']:
                    if file in days:
                        days.remove(file)
                
                # Loop over days
                for day in days:
                    if day.endswith('.csv'):
                        continue
                    if day.endswith('.DS_Store'):
                        continue
                    try:
                        if settings['VERBOSE']:
                            print(f'Analysing day {day}')
                        results = {}
                        file_path = os.listdir(
                            f"{settings['DATA_DIR']}/raw_data/{group}/{subject}/{day}")[0]
                        file = f"{subject}/{day}/{file_path}"
                        file_name = f"{subject}_{day}"

                        # Load data, downsample and remove not worn
                        data_df, endtime, begintime = prepare_data(
                            file, group, settings)

                        data_df = split_data(
                            data_df, begintime, endtime, settings)
                        
                        # Drop first and last 30 seconds and drop not worn
                        data_df, not_worn_samples = clean_data(
                            data_df, settings, all=True)
                        
                        # Predict data based on a previously trained ML algorithm
                        data = predict_data(data_df, settings,
                                            file_name, 'CNN_model.keras')
                        
                        # visualise activities per chunk size
                        if settings['VISUALISE']:
                            barcodeplot(data, file, file_name)
                            
                        results['subject'] = subject
                        results['group'] = group
                        results['day'] = day
                        results['Samples'] = len(data_df) + len(not_worn_samples)
                        results['not_worn_samples'] = len(not_worn_samples)
                        results['Name'] = file
                        results['begintime'] = begintime
                        results['endtime'] = endtime
                        results = characteristics(
                            results, data['activities [1min]'].values)

                        # data to dataframe
                        results_df = pd.DataFrame.from_dict(
                            results, orient='index').transpose()
                        for column in results_df.columns[8:]:
                            results_df[column] = pd.to_numeric(
                                results_df.loc[:, column]).round(3)
                        final_results = pd.concat((final_results, results_df))

                    except Exception as e:
                        logging.error(f'day: {subject} {day} {e}')
            except Exception as e:
                logging.error(f'subject: {subject} {day} {e}')

    # Replace empty values with 0
    final_results = final_results.fillna(0)
    final_results.to_excel(f'Results/results_per_day.xlsx', index=False)
    final_results.loc[:, 'average_activity_level':].corr().to_excel(
        'Results/correlations_per_day.xlsx')
