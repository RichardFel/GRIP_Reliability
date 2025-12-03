# %%
#
import pandas as pd
import pingouin as pg
import numpy as np

#%%
def clean(data, settings):
    # Minimum duration of 10 hours
    print(f'Data points before duration filter: {len(data)}')
    data_pre = len(data)
    data = data.loc[data['Epochs_of_1minute'] >= settings['MIN_Duration']]
    data = data.loc[data['Epochs_of_1minute'] <= settings['MAX_Duration']]
    data_post = len(data)
    print(f'Data points removed due to duration filter: {data_pre - data_post}')
    # characteristics
    total_time = data['Epochs_of_1minute'].sum()  # in hours
    print(f'Total wear time in hours: {total_time}')

    print(f'Sedentairy time: {data["sedentairy_count"].sum() / total_time}')
    print(f'Light time: {data["light_count"].sum() / total_time}')
    print(f'Moderate time: {data["moderate_count"].sum() / total_time}')
    
    exclude_vars = ['sedentairy_perc', 'std_change', 'Sample_entropy_m2_tau1', 'PLZC_dalay1_dim2',
                'PLZC_dealy2_dim2', 'PLZC_dealy2_dim3', 'PLZC_dealy3_dim3',
                'Sample_entropy_m2_tau1', 'Sample_entropy_m3_tau1', 'Sample_entropy_m4_tau1',
                'Sample_entropy_m2_tau2', 'Sample_entropy_m3_tau2', 'Sample_entropy_m4_tau2',
                'norm_transitions_1_0', 'norm_transitions_2_1',
                'norm_transitions_1_2', 'epochs_moderate_perc',
                'moderate_count', 'light_count', 'sedentairy_count',
                'sedentairy_median_length','light_median_length',
                'moderate_median_length', 'epochs_light_perc']
    for var in exclude_vars:
        if var in data.columns:
            data = data.drop(columns=[var])
            
    data = data.dropna(axis=1)
    data = data.sort_values(by=['day'])



    return data

def correlations(data):
    data = data.loc[:, 'average_activity_level':]
    correlations = data.corr(numeric_only=True)
    correlations.dropna(how='all', inplace=True)
    correlations.dropna(axis = 1, how='all', inplace=True)
    correlations *= 100
    correlations.to_excel(f'Results/correlations.xlsx')

    high_corr_vars = []
    for column in correlations.columns:
        for index, value in correlations[column].items():
            if index != column and abs(value) >= 85:
                high_corr_vars.append((index, column, value))
                print(f'High correlation between {index} and {column}: {value}')

def add_weekend_flag(date):
    pd_date = pd.to_datetime(date)
    if pd_date.weekday() < 5:
        return 0
    else:
        return 1

def ICC_analysis(tmp_data, group, settings):
    subject_counts = tmp_data['subject'].value_counts()
    results = {}
    results_excel = {}
    final_df = pd.DataFrame()
    final_df_excel = pd.DataFrame()
    number_of_days = settings['number_of_days']
    subjects = subject_counts.loc[subject_counts >= number_of_days * 2].index

    for i in range(1, settings['number_of_days']+1):
        subjects_data = tmp_data.loc[tmp_data['subject'].isin(subjects)]

        # Get the first 'i' rows for each subject
        subjects_data = subjects_data.set_index('subject')
        subjects_data = subjects_data.loc[:, 'average_activity_level':]

        # Get the first 'i' rows for each subject and calculate the mean
        first = subjects_data.groupby('subject').head(i)
        mean_values_first = first.groupby('subject').mean()
        # Add a column to indicate 'first' group
        mean_values_first['group_label'] = 1

        # Get the second 'i' rows for each subject (from i to i+7)
        second = subjects_data.groupby('subject').apply(lambda x: x.iloc[i:i+7])
        mean_values_second = second.groupby('subject').mean()
        # Add a column to indicate 'second' group
        mean_values_second['group_label'] = 2

        # Concatenate the mean values of the first and second groups
        final_mean_df = pd.concat(
            [mean_values_first, mean_values_second]).reset_index()
        print(f' \n Aantal dagen: {i}, aantal proefpersonen {len(final_mean_df) / 2} \n')
        for variable in subjects_data.columns:
            # try:
                if ((final_mean_df[variable].mean() == 0 )or np.isnan(final_mean_df[variable].mean())):
                    continue
                icc = pg.intraclass_corr(data=final_mean_df, targets='subject', raters='group_label',
                                        ratings=variable, nan_policy='omit').round(3)
                icc2 = icc.loc[icc['Type'] == 'ICC2']
                CI = icc['CI95%'].loc[1]
                SEM = (np.std(subjects_data[variable]) * np.sqrt(1 - icc2['ICC'].values[0]))
                MDC = (1.96 * SEM * np.sqrt(2))
                # print(f'{variable} MDC: {MDC}')
                results[f'ICC_{variable}'] = f"{icc2['ICC'].values[0]:.2f}"
                results_excel[f'ICC_{variable}'] = f'{icc2['ICC'].values[0]:.2f} [{CI[0]:.2f}-{CI[1]:.2f}] ({MDC:.2f})'
                # results[f'CI_days_{i}_var_{variable}'] = icc2['CI95%'].values[0]
            # except:
            #     continue
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=[i])
        results_df_excel = pd.DataFrame.from_dict(results_excel, orient='index', columns=[i])
        if final_df.empty:
            final_df = results_df
        if final_df_excel.empty:
            final_df_excel = results_df_excel
        else:
            final_df = pd.concat((final_df, results_df), axis=1)
            final_df_excel = pd.concat((final_df_excel, results_df_excel), axis=1)
    final_df.to_excel(f"Results/ICC_waardes_>{settings['MIN_Duration']}_<{settings['MAX_Duration']}_{group}.xlsx")
    final_df_excel.to_excel(f'Results/ICC_waardes_>{settings['MIN_Duration']}_<{settings['MAX_Duration']}_formatted_{group}.xlsx')

def reliability(settings):
    data =  pd.read_excel(f'{settings['RESULTS_DIR']}/results_per_day.xlsx')

    # Highly correlated variables to exclude from reliability analysis
    correlations(data)
    data = clean(data, settings)
    
    # Weekend flag
    data['weekend'] = data.apply(lambda row: add_weekend_flag(row['day']), axis=1)

    for group in data['group'].unique():
        tmp_data = data.loc[data['group'] == group]
        print(f'Group: {group}')
        ICC_analysis(tmp_data, group, settings)


# %%
